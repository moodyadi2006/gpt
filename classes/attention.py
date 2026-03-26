import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / (C ** 0.5)

        # 🔥 Causal mask
        mask = torch.tril(torch.ones(T, T)).to(x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        weights = torch.softmax(scores, dim=-1)

        out = weights @ V

        return out, weights