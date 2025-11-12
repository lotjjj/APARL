from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import nn
import re
from tqdm import tqdm

from modules.logger import Logger


class AgentBase(ABC):
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.last_observation = None
        self.steps = 0
        self.logger = Logger(config)

    @abstractmethod
    def update(self, buffer):
        pass

    @abstractmethod
    def get_action(self, observation: torch.Tensor):
        '''

        :return: action, log_prob
        '''
        pass

    @abstractmethod
    def explore(self, env):
        '''
        Explore a fixed length of steps (config.horizon_len) in the env
        :param env: gymnasium env, should be a vectorized env
        :return: a tuple of observations, actions, log_probs, rewards, terminations, truncations
        '''
        pass

    # I/O
    def save_model(self):
        path = self.config.save_dir / f'{self.config.algorithm}_steps_{self.steps}.pth'

        plist = list(path.parent.glob(f'{self.config.algorithm}_steps_*.pth'))
        if len(plist) > self.config.max_keep:
            plist_sorted = sorted(plist, key=lambda x: int(re.search(r'steps_(\d+).pth', x.name).group(1)))

            for i in range(len(plist_sorted) - self.config.max_keep):
                old_path = plist_sorted[i]
                old_path.unlink()
                tqdm.write(f'\nRemove {old_path}')
        torch.save(self._check_point, path)
        tqdm.write(f'\nSave model to {path}')

    def load_model(self, path):
        check_point = torch.load(path)
        self.steps = check_point['steps']
        return check_point

    def load_model_from_steps(self, steps):
        path = self.config.save_dir / f'{self.config.algorithm}_steps_{steps}.pth'
        return self.load_model(path)

    def eval_mode(self):
        pass

    def train_mode(self):
        pass


    @property
    def _check_point(self):
        return {}

    def optimizer_backward(self, optimizer: torch.optim.Optimizer , loss: torch.Tensor):
        optimizer.zero_grad()
        loss.backward()
        for p_dict in optimizer.param_groups:
            nn.utils.clip_grad_norm_(p_dict['params'], self.config.max_grad_norm)
        optimizer.step()


class AgentAC(AgentBase, ABC):
    def __init__(self, config):
        super(AgentAC, self).__init__(config)
        self.actor = None
        self.critic = None
        self.actor_optimizer = None
        self.critic_optimizer = None

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def build_temp_buffer(self) -> Tuple[torch.Tensor, ...]:
        observations = torch.empty((self.config.horizon_len, self.config.num_envs, self.config.observation_dim), dtype=torch.float32)

        actions =  torch.empty(
            (self.config.horizon_len, self.config.num_envs, self.config.action_dim),
            dtype=torch.float32)\
            if not self.config.is_discrete \
            else torch.empty((self.config.horizon_len, self.config.num_envs,),
                             dtype=torch.int32)

        log_probs = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.float32)
        rewards = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.float32)
        terminations = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.bool)
        truncations = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.bool)
        return observations, actions, log_probs, rewards, terminations, truncations

class AgentQ(AgentBase, ABC):
    def __init__(self, config):
        super(AgentQ, self).__init__(config)
        pass

def build_mlp(dims, activation= nn.ReLU, end_with_activation=False):
    net_list = []
    for i in range(len(dims)-1):
        net_list.extend([nn.Linear(dims[i], dims[i+1]), activation()])
    if not end_with_activation:
        del net_list[-1]
    return nn.Sequential(*net_list)

