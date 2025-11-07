import torch
import torch.nn as nn
from agents.AgentBase import build_mlp

class ContinuousPolicyHead(nn.Module):
    def __init__(self, dims, activation=nn.ReLU):
        super(ContinuousPolicyHead, self).__init__()

        self.net = build_mlp(dims=dims, activation=activation, end_with_activation=False)
        self.log_std = nn.Linear(dims[0], dims[-1])

    def forward(self, x):
        mu = self.net(x)
        log_std = self.log_std(x)
        return mu, log_std

class DiscretePolicyHead(nn.Module):
    def __init__(self, dims, activation=nn.ReLU):
        super(DiscretePolicyHead, self).__init__()
        self.net = build_mlp(dims=dims, activation=activation, end_with_activation=False)

    def forward(self, x):
        logits = self.net(x)
        return logits



