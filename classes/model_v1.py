import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=100):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        B, T = x.shape   # Batch, Time (sequence length)
        
        token_emb = self.token_embedding(x)   # (B, T, C)
        
        positions = torch.arange(T)           # [0,1,2,3,...]
        pos_emb = self.position_embedding(positions)  # (T, C)
        
        x = token_emb + pos_emb
        
        return x


