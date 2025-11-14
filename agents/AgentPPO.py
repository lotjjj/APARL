from typing import Tuple

import numpy as np
import torch
from torch import nn

from agents.AgentBase import AgentAC
import torch.nn.functional as F

class AgentPPO(AgentAC):
    def __init__(self, config):
        super().__init__(config)
        self.last_observation = None

        self.actor = ActorPPO(self.config.observation_dim, self.config.action_dim, self.config.actor_dims, self.config.is_discrete).to(self.device)
        self.critic = CriticPPO(self.config.observation_dim, self.config.critic_dims).to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': self.config.actor_lr },
            {'params': self.critic.parameters(), 'lr': self.config.critic_lr }
        ])

    def get_action(self, observation: torch.Tensor, deterministic=False):
        actions, log_probs = self.actor.get_action(observation, deterministic)
        return actions, log_probs

    def explore(self, env):
        """
        Interact with the environment and collect experience.

        Args:
            env: The environment to interact with

        Returns:
            A tuple of (observations, actions, log_probs, rewards, terminations, truncations)
        """
        # Prepare buffers
        observations, actions, log_probs, rewards, terminations, truncations = self.build_temp_buffer()

        # Start interaction
        for step in range(self.config.horizon_len):
            # Get current observation
            observation = self.last_observation

            # Get action and log probability
            action, log_prob = self.get_action(observation)

            # Store observation and action
            observations[step] = observation
            actions[step] = action
            log_probs[step] = log_prob

            # Step in environment
            np_action = action.cpu().numpy()
            next_observation, reward, termination, truncation, info = env.step(np_action)

            # Store reward and done flags
            rewards[step] = torch.from_numpy(reward).to(self.device)
            terminations[step] = torch.from_numpy(termination).to(self.device)
            truncations[step] = torch.from_numpy(truncation).to(self.device)

            # Update last observation
            self.last_observation = torch.from_numpy(next_observation).to(self.device)

        return observations, actions, log_probs, rewards, terminations.logical_not(), truncations.logical_not()

    def compute_gae_advantage(self, buffer: Tuple[torch.Tensor, ...]):
        observations, actions, log_probs, rewards, undone, unmasks = buffer
        truncations = torch.logical_not(unmasks)
        # True if truncated, False if not
        # add V to the last truncated state as compensation
        if torch.any(truncations):
            # compensation
            # ElegantRL -> https://github.com/AI4Finance-Foundation/ElegantRL
            # No clue why do this -> Actually, this *is* the standard way for truncated episodes.
            # When an episode is truncated (e.g., by max_steps), we don't have a terminal state.
            # We use the value estimate V(s_T) as an estimate for the discounted future reward G_T.
            # So, the reward for the final state s_T is R_T + gamma * V(s_{T+1}) = R_T + gamma * V(s_T_truncated)
            # The `undone[truncations] = False` line below ensures the masks stop the advantage calculation.
            rewards[truncations] += self.critic(observations[truncations]).detach()
            undone[truncations] = False  # This makes masks[truncations] = 0, stopping the advantage flow.
        # masks to stop the flow of the advantage
        masks = undone * self.config.gamma  # If undone is False, mask is 0, stopping the advantage calculation.
        # all state values
        values = self.critic(observations).detach()

        # all GAE advantages
        advantages = torch.empty_like(values)

        # s_t+1 (The next state *after* the last step of the collected trajectory)
        next_state = self.last_observation.clone()  # This is the state after the final action of the horizon.
        # V(s_t+1)
        next_value = self.critic(next_state).detach()  # This is V(s_{t+1}), used as the bootstrap value for the *final* step of the trajectory.

        # GAE advantage at time t (Running advantage accumulator)
        advantage = torch.zeros_like(next_value)  # Initialize the GAE accumulator for the *final* step of the trajectory.

        # Iterate backwards from the last step of the trajectory (config.horizon_len - 1) down to 0
        for i in reversed(range(self.config.horizon_len)):
            # GAE_t = delta_t + gamma * lambda * GAE_t+1
            # delta_t = R_{t+1} + gamma * V(s_{t+1}) - V(s_t)
            # Note: rewards[i], values[i] correspond to state s_t and action a_t.
            # masks[i] * next_value represents gamma * V(s_{t+1}) if the episode didn't terminate or truncate at t+1.
            # The `next_value` variable holds V(s_{t+1}) from the previous iteration (or V(s_{T+1}) initially).
            delta = rewards[i] + masks[i] * next_value - values[i]
            # Update the running GAE accumulator for the *next* step (t+1) to be used in the *previous* step (t).
            # advantage = GAE_{t+1} (from previous iteration)
            advantages[i] = advantage = delta + masks[i] * self.config.lambda_gae_adv * advantage
            # Update `next_value` for the *next* iteration (which will be the *previous* step in time).
            # This sets next_value = V(s_t), which will be gamma-multiplied in the next iteration (for step t-1).
            next_value = values[i]

        return advantages, values

    def update(self, buffer: Tuple[torch.Tensor, ...]):

        with torch.no_grad():
            observations, actions, log_probs, rewards, undone, unmasks = buffer
            advantages, values = self.compute_gae_advantage(buffer)
            values_target = advantages + values

            rewards_avg = rewards.mean(dim=0).mean()
            self.logger.add_scalar('reward/one_step_reward_avg', rewards_avg.cpu().item(), self.steps)

            del rewards, undone, values

        critic_losses = []
        actor_losses = []
        entropy_losses = []
        clip_losses = []

        for _ in range(self.config.num_epochs):
            ids0, ids1 = self.shuffle_idx()
            for start in range(0, self.config.horizon_len * self.config.num_envs, self.config.batch_size):
                end = start + self.config.batch_size
                id_horizon = ids0[start:end]
                id_env = ids1[start:end]

                observations_batch = observations[id_horizon, id_env]
                actions_batch = actions[id_horizon, id_env]
                # unmasks_batch = unmasks[id_horizon, id_env]
                log_probs_batch = log_probs[id_horizon, id_env]

                advantages_batch = advantages[id_horizon, id_env]
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

                values_target_batch = values_target[id_horizon, id_env]

                values_batch = self.critic(observations_batch)

                new_log_prob_batch, entropy_batch = self.actor.get_log_prob_entropy(observations_batch, actions_batch)

                ratios = (new_log_prob_batch - log_probs_batch.detach()).exp()

                surr1 = ratios * advantages_batch
                surr2 = torch.clamp(ratios, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages_batch

                # actor loss
                clip_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = - self.config.entropy_coef * torch.mean(entropy_batch)
                actor_loss = clip_loss + entropy_loss

                # critic loss
                critic_loss = self.config.value_coef * F.mse_loss(values_batch, values_target_batch)

                # https://github.com/DLR-RM/stable-baselines3/blob/b018e4bc949503b990c3012c0e36c9384de770e6/stable_baselines3/ppo/ppo.py#L262
                # approx KL divergence
                # early stop
                loss =  critic_loss + actor_loss

                self.optimizer_backward(self.optimizer, loss)

                actor_losses.append(actor_loss.detach().cpu().item())
                clip_losses.append(clip_loss.detach().cpu().item())
                entropy_losses.append(entropy_loss.detach().cpu().item())
                critic_losses.append(critic_loss.detach().cpu().item())

        self.logger.add_scalar('loss/actor_loss', np.mean(actor_losses), self.steps)
        self.logger.add_scalar('loss/clip_loss', np.mean(clip_losses), self.steps)
        self.logger.add_scalar('loss/entropy_loss', np.mean(entropy_losses), self.steps)
        self.logger.add_scalar('loss/critic_loss', np.mean(critic_losses), self.steps)

    def get_objectives(self):
        pass

    def shuffle_idx(self):
        ids =  torch.randperm(self.config.horizon_len * self.config.num_envs, requires_grad=False, device=self.device)
        ids0 = torch.fmod(ids, self.config.horizon_len)
        ids1 = torch.div(ids, self.config.horizon_len, rounding_mode='floor')
        return ids0, ids1

    @property
    def _check_point(self):
        check_point = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
        }
        return check_point

    def load_model(self, path = None):
        try:
            check_point = super().load_model(path)
            self.actor.load_state_dict(check_point['actor'])
            self.critic.load_state_dict(check_point['critic'])
            self.optimizer.load_state_dict(check_point['optimizer'])
            print(f'load model from {path}')
        except Exception as e:
            print(f'load model error: {e}')

from modules.Extractor import FlattenExtractor
from modules.Head import ContinuousPolicyHead, DiscretePolicyHead
from typing import OrderedDict

class CriticPPO(nn.Module):
    def __init__(self, observation_dim, dims):
        super().__init__()
        dims = [ observation_dim, *dims]
        self.feature_extractor = FlattenExtractor(dims)
        self.value_head = nn.Linear(dims[-1], 1)

    def forward(self, observation: torch.Tensor):
        feature = self.feature_extractor(observation)
        value = self.value_head(feature)
        return value.squeeze(-1)

from modules.CustomModule import TopKMoE
class ActorPPO(nn.Module):
    def __init__(self, observation_dim, action_dim, dims, is_discrete, if_moe = False):
        super().__init__()
        self.is_discrete = is_discrete
        dims = [observation_dim, *dims]

        self.net = nn.Sequential(
            OrderedDict(
                {
                    'feature_extractor': FlattenExtractor(dims),
                    'policy_head': DiscretePolicyHead([dims[-1], action_dim]) if is_discrete
                    else ContinuousPolicyHead([dims[-1], action_dim]),
                }
            )
        ) if not if_moe else \
        nn.Sequential(
            OrderedDict(
                {
                    'feature_extractor': FlattenExtractor(dims),
                    'experts': TopKMoE([dims[-1],dims[-1]], 2, 1),
                    'policy_head': DiscretePolicyHead([dims[-1], action_dim]) if is_discrete
                    else ContinuousPolicyHead([dims[-1], action_dim]),
                }
            )
        )

    def forward(self, observation: torch.Tensor):
        if self.is_discrete:
            return self.net(observation)
        else:
            mu, log_std = self.net(observation)
            std = torch.exp(torch.clamp(log_std, min=-20, max=1))
            return self.convert_action(mu), std

    def get_action(self, observation: torch.Tensor, deterministic=False):
        if self.is_discrete:
            logits = self.forward(observation)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = dist.probs.argmax(-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob
        else:
            mu, std = self.forward(observation)
            dist = torch.distributions.Normal(mu, std)
            if deterministic:
                action = mu
            else:
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
            return log_prob.sum(-1), entropy.sum(-1)

    @staticmethod
    def convert_action(action):
        return torch.tanh(action)

