import torch.nn as nn
from agents.AgentBase import build_mlp

class FlattenExtractor(nn.Module):
    def __init__(self, dims, activation=nn.ReLU, end_with_activation=True):
        super(FlattenExtractor, self).__init__()

        self.net = build_mlp(
            dims=dims, activation=activation, end_with_activation=end_with_activation
        )

    def forward(self, x):
        return self.net(x)

