from typing import List, Optional
import torch
import torch.nn as nn
from modules.utils import build_mlp



class AdditionMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, probs: Optional[torch.Tensor] = None) -> torch.Tensor:

        if probs is not None:
            if x.dim() == 2:  # [D_out, K] -> [1, D_out, K]
                x = x.unsqueeze(0)
                probs = probs.unsqueeze(0) if probs.dim() == 1 else probs
            # [B, D_out, K] * [B, 1, K] -> [B, D_out, K] -> [B, D_out]
            return torch.einsum('bok,bk->bo', x, probs)
        else:
            return x.sum(dim=-1)


class QMixMixer(nn.Module):
    pass


class TopKRouter(nn.Module):
    def __init__(self, dims, num_experts, k):
        super().__init__()
        self.k = max(min(k, num_experts), 1)
        self.net = build_mlp([*dims, num_experts], activation=nn.LeakyReLU)
        self._init_weights()

    def forward(self, x):
        logits = self.net(x)
        logits, indices = logits.topk(self.k, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return probs, indices

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class TopKMoE(nn.Module):
    def __init__(self, dims: List[int], num_experts: int = 2, k: int = 1):
        super().__init__()

        self.experts = nn.ModuleList([nn.Linear(dims[0], dims[-1]) for _ in range(num_experts)])
        self.router = TopKRouter(dims, num_experts, k)

        self.k = self.router.k
        self.mixer = AdditionMixer()
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        probs, indices = self.router(x)  # probs: [B, K], indices: [B, K]
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=-1)  # [B, D_out, num_experts]

        # 使用repeat而不是expand，更安全
        expanded_indices = indices.unsqueeze(1).repeat(1, expert_outputs.size(1), 1)  # [B, D_out, K]
        selected_outputs = torch.gather(expert_outputs, dim=-1, index=expanded_indices)  # [B, D_out, K]

        # 使用加权和而不是简单的求和
        output = self.mixer(selected_outputs, probs).squeeze(0)  # [B, D_out]
        return self.activation(output)


class SoftMoE(TopKMoE):
    def __init__(self, dims: List[int], num_experts: int = 2):
        super().__init__(dims, num_experts, num_experts)  # k = num_experts


def test_topkmoe():

    # 模拟build_mlp函数
    def mock_build_mlp(layer_dims, activation):
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i < len(layer_dims) - 2:
                layers.append(activation())
        return nn.Sequential(*layers)

    import sys
    original_build_mlp = sys.modules.get('utils.build_mlp') if 'utils' in sys.modules else None

    import utils
    utils.build_mlp = mock_build_mlp

    try:
        # 测试k=2
        moe_k2 = TopKMoE(dims=[20, 32], num_experts=4, k=2)
        x = torch.randn(20)
        output = moe_k2(x)
        assert output.shape == (32,), f"Expected (32,), got {output.shape}"
        assert len(moe_k2.experts) == 4, "Should have 4 experts"
        assert moe_k2.k == 2, "k should be 2"

    finally:
        # 恢复原始build_mlp
        if original_build_mlp is not None:
            utils.build_mlp = original_build_mlp