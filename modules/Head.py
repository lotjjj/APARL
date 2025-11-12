
import torch.nn as nn
from agents.AgentBase import build_mlp

class ContinuousPolicyHead(nn.Module):
    def __init__(self, dims, activation=nn.ReLU):
        super(ContinuousPolicyHead, self).__init__()

        self.net = build_mlp(dims=dims, activation=activation, end_with_activation=False)
        self.log_std = nn.Linear(dims[0], dims[-1])
        self._init_weights()

    def forward(self, x):
        mu = self.net(x)
        log_std = self.log_std(x)
        return mu, log_std

    def _init_weights(self):
        nn.init.normal_(self.log_std.weight, 0, 0.1)
        nn.init.constant_(self.log_std.bias, -0.1)
        nn.init.normal_(self.net[-1].weight, 0, 0.5)
        nn.init.constant_(self.net[-1].bias, 0)

class DiscretePolicyHead(nn.Module):
    def __init__(self, dims, activation=nn.ReLU):
        super(DiscretePolicyHead, self).__init__()
        self.net = build_mlp(dims=dims, activation=activation, end_with_activation=False)
        self._init_weights()

    def forward(self, x):
        logits = self.net(x)
        return logits

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



