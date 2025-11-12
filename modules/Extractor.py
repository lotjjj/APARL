import torch.nn as nn
from agents.AgentBase import build_mlp

class FlattenExtractor(nn.Module):
    def __init__(self, dims, activation=nn.ReLU, end_with_activation=True):
        super(FlattenExtractor, self).__init__()
        self.activation = activation
        self.net = build_mlp(
            dims=dims, activation=activation, end_with_activation=end_with_activation
        )

        self._init_weights()

    def forward(self, x):
        return self.net(x)

    def _init_weights(self):
        if isinstance(self.activation, nn.Tanh):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='tanh')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

