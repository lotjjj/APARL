from typing import Tuple
import torch
from torch import nn

from agents.AgentBase import AgentAC


class AgentPPO(AgentAC):
    def __init__(self, config):
        super().__init__(config)
        self.last_observation = None

        self.actor = ActorPPO(self.config.observation_dim, self.config.action_dim, self.config.actor_dims, self.config.is_discrete).to(self.config.device)
        self.critic = CriticPPO(self.config.observation_dim, self.config.action_dim, self.config.critic_dims).to(self.config.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

    def get_action(self, observation: torch.Tensor):
        actions, log_probs = self.actor.get_action(observation)
        return actions, log_probs

    def explore(self, env):

        observations, actions, log_probs, rewards, terminations, truncations = self.build_temp_buffer()

        observation = self.last_observation

        for _ in range(self.config.horizon_len):
            assert type(observation) == torch.Tensor
            action, log_prob = self.get_action(observation)
            np_action = action.cpu().numpy()

            observations[_] = observation
            actions[_] = action
            log_probs[_] = log_prob

            observation, reward, termination, truncation, info = env.step(np_action)

            rewards[_] = torch.from_numpy(reward).to(device=self.config.device)
            terminations[_] = torch.from_numpy(termination).to(device=self.config.device)
            truncations[_] = torch.from_numpy(truncation).to(device=self.config.device)
            observation = torch.from_numpy(observation).to(device=self.config.device)

        self.last_observation = observation.to(self.config.device)
        undone = torch.logical_not(terminations)
        unmasks = torch.logical_not(truncations)
        del terminations, truncations
        return observations, actions, log_probs, rewards, undone, unmasks

    def compute_gae_advantage(self, buffer: Tuple[torch.Tensor, ...]):
        observations, actions, log_probs, rewards, undone, unmasks = buffer
        truncations = torch.logical_not(unmasks)
        # add V to the last state as compensation
        if torch.any(truncations):
            # No compensation
            # rewards[truncations] += self.critic(observations[truncations])
            undone[truncations] = False
        masks = undone * self.config.gamma
        values = self.critic(observations)
        advantages = torch.empty_like(values)
        next_state = self.last_observation.clone()
        next_value = self.critic(next_state)

        advantage = torch.zeros_like(next_value)

        for i in reversed(range(self.config.horizon_len-1,-1,-1)):
            next_value = rewards[i] + masks[i] * next_value
            advantages[i] = advantage = next_value - values[i] + masks[i] * self.config.lambda_gae_adv * advantage
            next_value = values[i]
        return advantages, values

    def update(self, buffer: Tuple[torch.Tensor, ...]):
        with torch.no_grad():
            observations, actions, log_probs, rewards, undone, unmasks = buffer
            advantages, values = self.compute_gae_advantage(buffer)
            reward_sums = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            del rewards, undone, values


        critic_losses = torch.zeros(self.config.num_epochs)
        actor_losses = torch.zeros(self.config.num_epochs)
        entropy_losses = torch.zeros(self.config.num_epochs)

        for _ in range(self.config.num_epochs):
            ids0, ids1 = self.sample_idx()

            observations_batch = observations[ids0,ids1]
            actions_batch = actions[ids0,ids1]
            # unmasks_batch = unmasks[ids0,ids1]
            log_probs_batch = log_probs[ids0,ids1]
            advantages_batch = advantages[ids0,ids1]
            reward_sums_batch = reward_sums[ids0,ids1]

            values_batch = self.critic(observations_batch)

            # actor loss
            new_log_prob_batch, entropy_batch = self.actor.get_log_prob_entropy(observations_batch, actions_batch)

            ratio = torch.exp(new_log_prob_batch - log_probs_batch.detach())
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages_batch
            actor_loss = -torch.min(surr1, surr2).mean()
            entropy_loss = -self.config.entropy_coef * entropy_batch.mean()
            actor_entropy_loss = actor_loss + entropy_loss

            # critic loss
            critic_loss = nn.MSELoss()(values_batch, reward_sums_batch)

            self.optimizer_backward(self.actor_optimizer,  actor_entropy_loss)
            self.optimizer_backward(self.critic_optimizer, critic_loss)
            actor_losses[_] = actor_loss
            critic_losses[_] = critic_loss
            entropy_losses[_] = entropy_loss

        return actor_losses.mean().cpu().item(), entropy_losses.mean().cpu().item(),critic_losses.mean().cpu().item()

    @property
    def _check_point(self):
        check_point = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'epochs': self.epochs,
        }
        return check_point

    def load_model(self, path = None):
        try:
            check_point = super().load_model(path)
            self.actor.load_state_dict(check_point['actor'])
            self.critic.load_state_dict(check_point['critic'])
            self.actor_optimizer.load_state_dict(check_point['actor_optimizer'])
            self.critic_optimizer.load_state_dict(check_point['critic_optimizer'])
            print(f'load model from {path}')
        except Exception as e:
            print(f'load model error: {e}')

    def eval(self):
        self.actor.eval()
        self.critic.eval()

from modules.Extractor import FlattenExtractor
from modules.Head import ContinuousPolicyHead, DiscretePolicyHead

class ActorPPO(nn.Module):
    def __init__(self, observation_dim, action_dim, dims, is_discrete):
        super().__init__()

        dims = [ observation_dim, *dims]

        self.feature_extractor = FlattenExtractor(dims)

        self.is_discrete = is_discrete
        if is_discrete:
            self.policy_head = DiscretePolicyHead([dims[-1], action_dim])
        else:
            self.policy_head = ContinuousPolicyHead([dims[-1], action_dim])


    def forward(self, observation: torch.Tensor):
        feature = self.feature_extractor(observation)
        if self.is_discrete:
            logits = self.policy_head(feature)
            return logits
        else:
            mu, log_std = self.policy_head(feature)
            std = torch.exp(torch.clamp(log_std, min=-20, max=1))
            return self.convert_action(mu), std

    def get_action(self, observation: torch.Tensor):
        if self.is_discrete:
            logits = self.forward(observation)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob
        else:
            mu, std = self.forward(observation)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob.sum(-1)

    def get_log_prob_entropy(self, observation: torch.Tensor, action: torch.Tensor):
        if self.is_discrete:
            logits = self.forward(observation)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(action), dist.entropy()
        else:
            mu, std = self.forward(observation)
            dist = torch.distributions.Normal(mu, std)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return log_prob.sum(-1),entropy.sum(-1)

    @staticmethod
    def convert_action(action):
        return torch.tanh(action)


class CriticPPO(nn.Module):
    def __init__(self, observation_dim, action_dim, dims):
        super().__init__()
        dims = [ observation_dim, *dims]
        self.feature_extractor = FlattenExtractor(dims)
        self.value_head = nn.Linear(dims[-1], 1)

    def forward(self, observation: torch.Tensor):
        feature = self.feature_extractor(observation)
        value = self.value_head(feature)
        return value.squeeze(-1)
