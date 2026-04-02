import torch
import torch.nn as nn
from classes.attention_v2 import SelfAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.heads = nn.ModuleList(
            [SelfAttention(embed_dim) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(embed_dim * num_heads, embed_dim)

    def forward(self, x, past_k=None, past_v=None):
        head_outputs = []
        new_Ks = []
        new_Vs = []

        for i, head in enumerate(self.heads):
            pk = past_k[i] if past_k is not None else None
            pv = past_v[i] if past_v is not None else None

            out, K, V = head(x, pk, pv)

            head_outputs.append(out)
            new_Ks.append(K)
            new_Vs.append(V)

        # Concatenate heads
        out = torch.cat(head_outputs, dim=-1)
        out = self.linear(out)

        return out, new_Ks, new_Vs