import torch
import torch.nn as nn
from classes.attention import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(embed_dim) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x):
        out = torch.cat([head(x)[0] for head in self.heads], dim=-1)
        return self.linear(out)