import torch
from torch import nn

from agents.AgentBase import AgentBase
from modules.ReplayBuffer import ReplayBuffer


class AgentDQN(AgentBase):
    def __init__(self):
        super().__init__(self)
        self.qnet = QNet()
        self.qnet_optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.config.lr)

    def update(self, buffer: ReplayBuffer):
        pass


    def get_action(self, observation: torch.Tensor):
        pass

    def explore(self, env):
        pass

    def _check_point(self):
        check_point = {
            'qnet': self.qnet.state_dict(),
            'qnet_optimizer': self.qnet_optimizer.state_dict(),
            'epochs': self.epochs,
        }
        return check_point


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
