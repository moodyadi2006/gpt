import torch.nn as nn
import torch.nn.functional as F

class SoftmaxLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        # logits: (batch_size, seq_len, vocab_size)
        
        probs = F.softmax(logits, dim=-1)
        
        return probs