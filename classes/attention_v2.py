import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, past_k=None, past_v=None):
        B, T, C = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 🔥 Append past KV if exists
        if past_k is not None and past_v is not None:
            K = torch.cat([past_k, K], dim=1)  # (B, T_total, C)
            V = torch.cat([past_v, V], dim=1)

        T_total = K.size(1)

        # Compute attention scores
        scores = Q @ K.transpose(-2, -1) / (C ** 0.5)

        # 🔥 Causal mask (important fix for KV cache)
        mask = torch.tril(torch.ones(T, T_total, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = torch.softmax(scores, dim=-1)
        out = weights @ V

        return out, K, V