import torch.nn as nn
from classes.multi_head_attention_v2 import MultiHeadAttention
from classes.feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ffn = FeedForward(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, past_kv=None):
        past_k, past_v = past_kv if past_kv is not None else (None, None)

        # 🔥 Attention with KV cache
        attn_out, K, V = self.attention(x, past_k, past_v)

        x = x + attn_out
        x = self.norm1(x)

        x = x + self.ffn(x)
        x = self.norm2(x)

        return x, (K, V)