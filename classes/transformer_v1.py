import torch.nn as nn
from classes.multi_head_attention_v1 import MultiHeadAttention
from classes.feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        self.ffn = FeedForward(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention + residual + norm
        x = x + self.attention(x)
        x = self.norm1(x)

        # FFN + residual + norm
        x = x + self.ffn(x)
        x = self.norm2(x)

        return x


