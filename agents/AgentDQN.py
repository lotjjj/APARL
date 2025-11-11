from typing import Tuple

import numpy as np
import torch
from torch import nn

from agents.AgentBase import AgentBase
from modules.Extractor import FlattenExtractor
from modules.ReplayBuffer import ReplayBuffer


class AgentDQN(AgentBase):
    def __init__(self):
        super().__init__(self)
        assert self.config.is_discrete
        self.qnet = QNet()
        self.qnet_target = QNet()
        self.qnet_target.load_state_dict(self.qnet.state_dict())
        self.qnet_target.eval()
        self.qnet_optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.config.lr)
        self.buffer_size = self.config.buffer_size

    def update(self, buffer: ReplayBuffer):
        pass


    def get_action(self, observation: torch.Tensor):
        action = self.qnet.get_action(observation)
        return action

    def explore(self, env):
        observations, actions, rewards, terminations, truncations = self.build_temp_buffer()

        observation = self.last_observation

        for _ in range(self.config.horizon_len):
            observation = torch.from_numpy(observation).to(self.device)
            assert isinstance(observation, torch.Tensor)
            action = self.get_action(observation)
            np_action = action.cpu().numpy()

            observations[_] = observation
            actions[_] = action

            observation, reward, termination, truncation, info = env.step(np_action)

            rewards[_] = reward
            terminations[_] = termination
            truncations[_] = truncation
            observation =observation

        self.last_observation = observation
        undone = np.logical_not(terminations)
        unmasks = np.logical_not(truncations)
        del terminations, truncations
        return observations, actions, rewards, undone, unmasks

    def build_temp_buffer(self) -> Tuple[np.ndarray, ...]:
        observations = np.empty((self.buffer_size, self.config.num_envs, self.config.observation_dim), dtype=np.float32)
        actions = np.empty((self.buffer_size, self.config.num_envs,), dtype=np.float32) if self.config.is_discrete \
            else np.empty((self.buffer_size, self.config.num_envs, self.config.action_dim), dtype=np.float32)
        rewards = np.empty((self.buffer_size, self.config.num_envs), dtype=np.float32)
        unmasks = np.empty((self.buffer_size,self.config.num_envs), dtype=np.bool)  # Not Truncated
        undone = np.empty((self.buffer_size, self.config.num_envs), dtype=np.bool)  # Not Terminated
        return observations, actions, rewards, undone, unmasks

    def _check_point(self):
        check_point = {
            'qnet': self.qnet.state_dict(),
            'qnet_optimizer': self.qnet_optimizer.state_dict(),
            'epochs': self.epochs,
        }
        return check_point


class QNet(nn.Module):
    def __init__(self, observation_dim, action_dim, dims):
        super().__init__()

        dims = [observation_dim, *dims]

        self.feature_extractor = FlattenExtractor(dims)

        self.q_head = nn.Linear(dims[-1], action_dim)

    def forward(self, observation: torch.Tensor):
        features = self.feature_extractor(observation)
        q_values = self.q_head(features)
        return q_values

    def get_action(self, observation: torch.Tensor, deterministic=False):
        q_values = self.forward(observation)
        # epsilon greedy
        if deterministic:
            action = torch.argmax(q_values, dim=-1)
        else:
            action = torch.argmax(q_values, dim=-1) if torch.rand(1) < self.config.epsilon \
                else torch.randint(0, self.config.action_dim, (1,))

        return action
