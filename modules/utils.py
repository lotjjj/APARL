import torch.nn as nn

def build_mlp(dims, activation= nn.ReLU, end_with_activation=False):
    net_list = []
    for i in range(len(dims)-1):
        net_list.extend([nn.Linear(dims[i], dims[i+1]), activation()])
    if not end_with_activation:
        del net_list[-1]
    return nn.Sequential(*net_list)