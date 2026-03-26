import torch.nn as nn

class FinalProjection(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        
        logits = self.linear(x)
        
        # logits: (batch_size, seq_len, vocab_size)
        return logits