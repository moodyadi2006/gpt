import torch
import torch.nn as nn

"""
I understood that the `pos_offset` is not related to batching or how samples are processed during training, but rather it is used to maintain the correct positional continuity of tokens when dealing with sequences that are processed incrementally, such as during autoregressive generation or streaming. In my current training setup, where each sample in a batch is an independent fixed-length sequence, positional indices always start from 0 to T−1, and therefore offset is not required. However, in scenarios where the sequence grows over time and we do not recompute the entire sequence at every step, the offset ensures that new tokens receive their correct absolute positions instead of always starting from zero. This helps the model understand the true order of tokens across the entire sequence, making `pos_offset` essential for generation but unnecessary for standard batch training.
"""
class SimpleModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_len=100):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x, pos_offset=0):
        B, T = x.shape
        
        token_emb = self.token_embedding(x)  # (B, T, C)
        
        # 🔥 FIX: use position offset
        positions = torch.arange(
            pos_offset, pos_offset + T, device=x.device
        )
        
        pos_emb = self.position_embedding(positions)  # (T, C)
        
        x = token_emb + pos_emb
        
        return x