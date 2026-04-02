import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokens = tokenizer.encode(text)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]

        return torch.tensor(x), torch.tensor(y)