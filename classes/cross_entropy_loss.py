import torch.nn as nn

class CrossEntropyLossLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        # logits:  (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len)

        B, T, V = logits.shape

        # Flatten for loss computation
        logits = logits.view(B * T, V)
        targets = targets.view(B * T)

        loss = self.loss_fn(logits, targets)

        return loss