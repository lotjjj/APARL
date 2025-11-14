import unittest
from typing import List, Optional
import torch
import torch.nn as nn
from modules.utils import build_mlp
from torch.nn import functional as F


class AdditionMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if probs is not None:
            assert probs.dim() == x.dim()
            if x.dim() == 2:  # [D_out, K] -> [1, D_out, K]
                x = x.unsqueeze(0)
                probs = probs.unsqueeze(0) if probs.dim() == 1 else probs
            # [B, D_out, K] * [B, 1, K] -> [B, D_out, K] -> [B, D_out]
            return torch.einsum('bok,bok->bo', x, probs)
        else:
            return x.sum(dim=-1)


class QMixMixer(nn.Module):
    """
    QMixer: 混合网络，将各智能体的局部 Q 值组合成全局 Q 值。
    保证单调性：dQ_tot / dQ_i >= 0。
    """

    def __init__(self, n_agents, state_dim, hidden_dim=32):
        """
        :param n_agents: 智能体数量
        :param state_dim: 全局状态维度
        :param hidden_dim: 隐藏层维度
        """
        super(QMixMixer, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # 第一层：状态 -> 隐藏层（权重矩阵）
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim)
        )

        # 第二层：状态 -> 标量（偏置项）
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)

        # 输出层：隐藏层 -> 全局 Q 值（标量）
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        :param agent_qs: [batch_size, n_agents] 各智能体的局部 Q 值
        :param states: [batch_size, state_dim] 全局状态
        :return: q_tot: [batch_size, 1] 全局 Q 值
        """
        batch_size = agent_qs.size(0)

        # 第一层：计算权重和偏置
        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.hidden_dim)  # [B, n, h]
        b1 = self.hyper_b1(states).view(-1, 1, self.hidden_dim)  # [B, 1, h]

        # 将 agent_qs 扩展为 [B, n, 1]，然后与 w1 相乘
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # [B, 1, h]

        # 第二层：输出层
        w2 = self.hyper_w2(states).view(-1, self.hidden_dim, 1)  # [B, h, 1]
        b2 = self.hyper_b2(states).view(-1, 1, 1)  # [B, 1, 1]

        q_tot = torch.bmm(hidden, w2) + b2  # [B, 1, 1]

        return q_tot.squeeze(-1)  # [B, 1]

    def get_monotonicity(self, agent_qs, states):
        """
        计算每个智能体对全局 Q 值的梯度，用于验证单调性（可选调试用）
        返回：[batch_size, n_agents] 每个智能体的梯度
        """
        agent_qs.requires_grad_(True)
        q_tot = self.forward(agent_qs, states)
        grad = torch.autograd.grad(
            outputs=q_tot.sum(),
            inputs=agent_qs,
            create_graph=False,
            retain_graph=False
        )[0]
        agent_qs.requires_grad_(False)
        return grad


class TopKRouter(nn.Module):
    def __init__(self, dims, num_experts, k):
        super().__init__()
        self.num_experts = num_experts
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

    def random_expert(self, x, k):
        logits = torch.randn((x.size(0), self.num_experts))
        logits, indices = logits.topk(k, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return probs, indices

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
        return self.expert_fusion(indices, probs, x)

    def expert_fusion(self, indices, probs, x):
        expert_outputs = torch.stack([e(x) for e in self.experts], dim=-1)  # [B, D_out, num_experts]

        expanded_indices = indices.unsqueeze(1).repeat(1, expert_outputs.size(1), 1)  # [B, D_out, K]
        selected_outputs = torch.gather(expert_outputs, dim=-1, index=expanded_indices)  # [B, D_out, K]

        output = self.mixer(selected_outputs, probs).squeeze(0)  # [B, D_out]
        return self.activation(output)

class RandomMoE(TopKMoE):
    def __init__(self, dims: List[int], num_experts: int = 2):
        super().__init__(dims, num_experts, num_experts)  # k = num_experts

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        probs, indices = self.router.random_expert(x, 1)  # probs: [B, K], indices: [B, K]
        return self.expert_fusion(indices, probs, x)

class SoftMoE(TopKMoE):
    def __init__(self, dims: List[int], num_experts: int = 2):
        super().__init__(dims, num_experts, num_experts)
