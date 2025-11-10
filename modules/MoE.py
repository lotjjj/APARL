# This is an experimental implementation of MoE strategy.

import torch
from torch import nn

class MoE(nn.Module):
    def __init__(self, dims, num_experts, num_tokens):
        super().__init__()
        self.num_experts = num_experts
        self.num_tokens = num_tokens

        self.token_embedding = nn.Embedding(num_tokens, dims[-1])
        self.experts = nn.ModuleList([nn.Linear(dims[-1], dims[-1]) for _ in range(num_experts)])
        self.gate = nn.Linear(dims[-1], num_experts)


    def forward(self, x):
        tokens = self.token_embedding(x)
        experts_outputs = [expert(tokens) for expert in self.experts]
        gate_logits = self.gate(tokens)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        return torch.sum(gate_probs * experts_outputs, dim=-1)
