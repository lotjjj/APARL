from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch import nn

class AgentBase(ABC):
    def __init__(self, config):
        self.config = config

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

    # @abstractmethod
    # def save_model(self):
    #     pass
    # @abstractmethod
    # def load_model(self):
    #     pass

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
                             dtype=torch.int64,device=self.config.device)

        log_probs = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.float32,device=self.config.device)
        rewards = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.float32,device=self.config.device)
        terminations = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.bool,device=self.config.device)
        truncations = torch.empty((self.config.horizon_len, self.config.num_envs, ), dtype=torch.bool,device=self.config.device)
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

