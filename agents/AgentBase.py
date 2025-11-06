from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import nn
import re

class AgentBase(ABC):
    def __init__(self, config):
        self.config = config
        self.epochs = 0

    @abstractmethod
    def update(self, buffer: Tuple[torch.Tensor, ...]):
        pass

    @abstractmethod
    def get_action(self, observation: torch.Tensor):
        '''

        :return: action, log_prob
        '''
        pass

    @abstractmethod
    def explore(self, horizon_len: int):
        pass

    def save_model(self, epochs):
        path = self.config.save_dir / f'{self.config.algorithm}_epochs_{epochs}.pth'

        self.epochs = epochs

        torch.save(self._check_point, path)
        plist = list(path.parent.glob(f'{self.config.algorithm}_epochs_*.pth'))
        if len(plist) > self.config.max_keep:
            plist_sorted = sorted(plist, key=lambda x: int(re.search(r'epochs_(\d+).pth', x.name).group(1)))

            for i in range(len(plist_sorted) - self.config.max_keep):
                old_path = plist_sorted[i]
                old_path.unlink()
                print(f'\nRemove {old_path}')
        print(f'\nSave model to {path}')

    def load_model(self, path):
        try:
            check_point = torch.load(path)
            self.epochs = check_point['epochs']

            return check_point
        except Exception as e:
            print(f'No such model')


    @abstractmethod
    def eval(self):
        pass

    @property
    def _check_point(self):
        return {}

    def optimizer_backward(self, optimizer: torch.optim.Optimizer , loss: torch.Tensor):
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], self.config.max_grad_norm)
        optimizer.step()


class AgentAC(AgentBase, ABC):
    def __init__(self, config):
        super(AgentAC, self).__init__(config)
        pass

    def build_temp_buffer(self) -> Tuple[torch.Tensor, ...]:
        observations = torch.empty((self.config.horizon_len, self.config.num_envs, self.config.observation_dim), dtype=torch.float32,device=self.config.device)

        actions =  torch.empty(
            (self.config.horizon_len, self.config.num_envs, self.config.action_dim),
            dtype=torch.float32,device=self.config.device)\
            if not self.config.is_discrete \
            else torch.empty((self.config.horizon_len, self.config.num_envs,),
                             dtype=torch.int32,device=self.config.device)

        log_probs = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.float32,device=self.config.device)
        rewards = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.float32,device=self.config.device)
        terminations = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.bool,device=self.config.device)
        truncations = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.float32,device=self.config.device)
        return observations, actions, log_probs, rewards, terminations, truncations

    def sample_idx(self):
        ids =  torch.randint(0, self.config.horizon_len * self.config.num_envs, size = (self.config.batch_size,), requires_grad=False, device=self.config.device)
        ids0 = torch.fmod(ids, self.config.horizon_len)
        ids1 = torch.div(ids, self.config.horizon_len, rounding_mode='floor')
        return ids0, ids1

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

