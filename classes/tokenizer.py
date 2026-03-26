import json

class CharTokenizer:
    def __init__(self, text=None):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

        if text:
            self.build_vocab(text)

    def build_vocab(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        print(f"Vocab size: {self.vocab_size}")

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, tokens):
        return ''.join([self.itos[int(i)] for i in tokens])

    def save(self, path="data/tokenizer.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "stoi": self.stoi,
                "itos": self.itos
            }, f)

    def load(self, path="tokenizer.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = len(self.stoi)