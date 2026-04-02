import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from classes.dataset_v2 import TextDataset

# ===================== LOAD FUNCTIONS =====================

def load_json_tokens(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_npy_tokens(path):
    return np.load(path)


# ===================== TOKEN PATH CONFIG =====================

TOKEN_PATHS = {
    "char": ("data/encodings/char_tokens.json", "json"),
    "word": ("data/encodings/word_tokens.json", "json"),
    "sentence": ("data/encodings/sentence_tokens.json", "json"),
    "bpe": ("data/encodings/bpe_tokens.json", "json"),
    "gpt_bpe": ("data/encodings/gpt_bpe_tokens.npy", "npy"),
}


# ===================== FACTORY FUNCTION =====================

def create_dataloader(token_type, block_size=16, batch_size=16):
    """
    token_type: one of ["char", "word", "sentence", "bpe", "gpt_bpe"]
    """

    if token_type not in TOKEN_PATHS:
        raise ValueError(f"Unknown token_type: {token_type}")

    path, file_type = TOKEN_PATHS[token_type]

    # Load tokens
    if file_type == "json":
        tokens = load_json_tokens(path)
    else:
        tokens = load_npy_tokens(path)

    # Normalize → torch tensor
    tokens = torch.tensor(tokens, dtype=torch.long)

    dataset = TextDataset(tokens, block_size=block_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataset, dataloader


# ===================== TEST =====================
if __name__ == "__main__":
    token_types = ["char", "word", "sentence", "bpe", "gpt_bpe"]

    for t in token_types:
        print("=" * 50)
        print(f"Testing {t} tokenizer")

        dataset, dataloader = create_dataloader(t)

        x, y = next(iter(dataloader))
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")