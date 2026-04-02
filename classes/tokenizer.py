import json
from collections import Counter
import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

"""
In this class, I am treating each individual character as a token.

While building the vocabulary, I extract all unique characters from the dataset, sort them, and assign each character a unique integer ID (stoi).
I also create a reverse mapping (itos) to convert IDs back into characters.
During encoding, I convert each character in the input string into its corresponding index.
During decoding, I reconstruct the original string by mapping indices back to characters and joining them.

👉 This tokenizer is:

Very simple and robust (no unknown token issue)
But highly inefficient, because even small text becomes a large number of tokens
Useful for learning fundamentals, but not practical for real LLMs
"""
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

    def save(self, path="checkpoints/tokenizer.json"):
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

    def save_encoded(self, text, path):
        import json
        tokens = self.encode(text)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokens, f)

"""
In this class, I tokenize text at the word level using whitespace splitting.

While building the vocabulary, I split the entire text into words and store only unique words.
Each word is mapped to an integer ID (stoi), and reverse mapping is stored in itos.
Encoding converts each word in a sentence into its index.
Decoding reconstructs the sentence by joining words with spaces.

👉 This tokenizer:

Reduces number of tokens compared to character-level
But suffers from major issues:
Cannot handle unseen words (no <UNK> handling)
Vocabulary can become extremely large
Not scalable for real-world LLMs
"""
class WordTokenizer:
    def __init__(self, text=None):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

        if text:
            self.build_vocab(text)

    def build_vocab(self, text):
        words = text.split()
        unique_words = sorted(set(words))

        self.vocab_size = len(unique_words)
        self.stoi = {w: i for i, w in enumerate(unique_words)}
        self.itos = {i: w for w, i in self.stoi.items()}

        print(f"Vocab size: {self.vocab_size}")

    def encode(self, s):
        return [self.stoi[w] for w in s.split() if w in self.stoi]

    def decode(self, tokens):
        return ' '.join([self.itos[int(i)] for i in tokens])

    def save(self, path="checkpoints/word_tokenizer.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f)

    def load(self, path="checkpoints/word_tokenizer.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = len(self.stoi)

    def save_encoded(self, text, path):
        import json
        tokens = self.encode(text)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokens, f)

"""
In this class, I implemented a basic version of Byte Pair Encoding (BPE) to create subword tokens.

Initially, I split each word into characters.
Then I repeatedly:
Count the frequency of adjacent character pairs
Merge the most frequent pair into a new token
Add this merged token to the vocabulary
This process continues until the desired vocabulary size is reached.
During encoding, I try to merge character pairs based on learned merges.
Decoding reconstructs text by joining subword tokens.

👉 This tokenizer:

Balances vocabulary size and token efficiency
Can handle unknown words by breaking them into subwords
Mimics how real LLM tokenizers work (though simplified here)

⚠️ Limitations in my implementation:

Greedy merging (not iterative like real GPT tokenizer)
No byte-level handling
No special tokens or advanced normalization
"""
class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.stoi = {}
        self.itos = {}
        self.merges = {}

    def get_stats(self, tokens):
        pairs = Counter()
        for token in tokens:
            for i in range(len(token)-1):
                pairs[(token[i], token[i+1])] += 1
        return pairs

    def merge_tokens(self, tokens, pair):
        new_tokens = []
        bigram = ''.join(pair)

        for token in tokens:
            new_token = []
            i = 0
            while i < len(token):
                if i < len(token)-1 and (token[i], token[i+1]) == pair:
                    new_token.append(bigram)
                    i += 2
                else:
                    new_token.append(token[i])
                    i += 1
            new_tokens.append(new_token)
        return new_tokens

    def build_vocab(self, text):
        tokens = [list(word) for word in text.split()]

        vocab = set(char for word in tokens for char in word)

        while len(vocab) < self.vocab_size:
            pairs = self.get_stats(tokens)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            tokens = self.merge_tokens(tokens, best_pair)
            self.merges[best_pair] = ''.join(best_pair)

            vocab.add(''.join(best_pair))

        vocab = sorted(vocab)
        self.stoi = {t: i for i, t in enumerate(vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}

        print(f"Vocab size: {len(vocab)}")

    def encode(self, s):
        tokens = []
        for word in s.split():
            chars = list(word)
            i = 0
            while i < len(chars):
                if i < len(chars)-1 and (chars[i], chars[i+1]) in self.merges:
                    tokens.append(self.stoi[self.merges[(chars[i], chars[i+1])]])
                    i += 2
                else:
                    tokens.append(self.stoi[chars[i]])
                    i += 1
        return tokens

    def decode(self, tokens):
        return ''.join([self.itos[int(i)] for i in tokens])

    def save(self, path="checkpoints/bpe_tokenizer.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "stoi": self.stoi,
                "itos": self.itos,
                "merges": {str(k): v for k, v in self.merges.items()}
            }, f)

    def load(self, path="checkpoints/bpe_tokenizer.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.merges = {eval(k): v for k, v in data["merges"].items()}

    def save_encoded(self, text, path):
        import json
        tokens = self.encode(text)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokens, f)

"""
In this class, I treat each sentence as a token.

I split the text into sentences using regex (based on punctuation like ., !, ?).
Each unique sentence is assigned an index.
Encoding maps sentences to indices, and decoding joins them back.

👉 This tokenizer:

Produces very few tokens
But is too coarse-grained for language modeling
Cannot be used for next-token prediction properly
Mostly useful for experimentation or understanding tokenization extremes
"""
class SentenceTokenizer:
    def __init__(self, text=None):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

        if text:
            self.build_vocab(text)

    def split_sentences(self, text):
        return re.split(r'(?<=[.!?]) +', text)

    def build_vocab(self, text):
        sentences = self.split_sentences(text)
        unique_sentences = sorted(set(sentences))

        self.vocab_size = len(unique_sentences)
        self.stoi = {s: i for i, s in enumerate(unique_sentences)}
        self.itos = {i: s for s, i in self.stoi.items()}

        print(f"Vocab size: {self.vocab_size}")

    def encode(self, s):
        sentences = self.split_sentences(s)
        return [self.stoi[s] for s in sentences if s in self.stoi]

    def decode(self, tokens):
        return ' '.join([self.itos[int(i)] for i in tokens])

    def save(self, path="checkpoints/sentence_tokenizer.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f)

    def load(self, path="checkpoints/sentence_tokenizer.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.stoi = data["stoi"]
        self.itos = {int(k): v for k, v in data["itos"].items()}
        self.vocab_size = len(self.stoi)

    def save_encoded(self, text, path):
        import json
        tokens = self.encode(text)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tokens, f)

"""
In this class, I implement a GPT-like tokenizer using Byte Pair Encoding (BPE) with a byte-level approach.

I use a production-grade tokenizer library to handle tokenization efficiently.
I first apply byte-level pre-tokenization, which ensures that all text (including spaces, punctuation, and unicode) is properly handled.
Then I train a BPE model that learns subword units by merging frequently occurring byte pairs until the desired vocabulary size is reached.
The tokenizer also supports special tokens like <unk>, <pad>, <bos>, and <eos>.

Encoding converts input text into a sequence of token IDs using the learned BPE merges.
Decoding reconstructs the original text from token IDs using a byte-level decoder.

👉 This tokenizer:

Provides an optimal balance between vocabulary size and token efficiency
Handles unknown and rare words by breaking them into subword units
Works robustly with any kind of text (including unicode and special characters)
Is highly efficient and used in real-world LLMs like GPT

👉 Compared to my custom BPE:

This implementation is more accurate and optimized
Uses byte-level encoding instead of character-level
Applies proper merge rules instead of greedy merging
Suitable for real training pipelines and production use
"""
class GPTBPETokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.tokenizer = None

    def build_vocab(self, text, save_path="checkpoints/gpt_tokenizer.json"):
        # Initialize tokenizer with BPE model
        self.tokenizer = Tokenizer(BPE(unk_token="<unk>"))

        # Byte-level pre-tokenization (like GPT)
        self.tokenizer.pre_tokenizer = ByteLevel()

        # Trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]
        )

        # Train on text (write to temp file since HF expects file)
        with open("temp.txt", "w", encoding="utf-8") as f:
            f.write(text)

        self.tokenizer.train(["temp.txt"], trainer)

        # Decoder
        self.tokenizer.decoder = ByteLevelDecoder()

        # Save tokenizer
        self.tokenizer.save(save_path)
        print(f"Tokenizer saved at {save_path}")

    def load(self, path="checkpoints/gpt_tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def save_encoded(self, text, path):
        import numpy as np
        tokens = self.encode(text)
        np.save(path, np.array(tokens, dtype=np.int32))