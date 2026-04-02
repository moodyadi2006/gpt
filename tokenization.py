import os
from classes.tokenizer import CharTokenizer, WordTokenizer, BPETokenizer, SentenceTokenizer, GPTBPETokenizer

# Create directories
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("data/encodings", exist_ok=True)

# Load dataset
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# ===================== CHAR =====================
print("=" * 50)
print("Char Tokenizer")

char_tokenizer = CharTokenizer(text)
print(f"Vocab Size: {char_tokenizer.vocab_size}")

char_tokenizer.save("checkpoints/char_tokenizer.json")
char_tokenizer.save_encoded(text, "data/encodings/char_tokens.json")

# ===================== WORD =====================
print("=" * 50)
print("Word Tokenizer")

word_tokenizer = WordTokenizer(text)
print(f"Vocab Size: {word_tokenizer.vocab_size}")

word_tokenizer.save("checkpoints/word_tokenizer.json")
word_tokenizer.save_encoded(text, "data/encodings/word_tokens.json")

# ===================== SENTENCE =====================
print("=" * 50)
print("Sentence Tokenizer")

sentence_tokenizer = SentenceTokenizer(text)
print(f"Vocab Size: {sentence_tokenizer.vocab_size}")

sentence_tokenizer.save("checkpoints/sentence_tokenizer.json")
sentence_tokenizer.save_encoded(text, "data/encodings/sentence_tokens.json")

# ===================== CUSTOM BPE =====================
print("=" * 50)
print("Custom BPE Tokenizer")

bpe_tokenizer = BPETokenizer(vocab_size=1000)
bpe_tokenizer.build_vocab(text)
print(f"Vocab Size: {len(bpe_tokenizer.stoi)}")

bpe_tokenizer.save("checkpoints/bpe_tokenizer.json")
bpe_tokenizer.save_encoded(text, "data/encodings/bpe_tokens.json")

# ===================== GPT BPE =====================
print("=" * 50)
print("GPT BPE Tokenizer")

gpt_tokenizer = GPTBPETokenizer(vocab_size=1000)
gpt_tokenizer.build_vocab(text, save_path="checkpoints/gpt_tokenizer.json")
print(f"Vocab Size: {gpt_tokenizer.tokenizer.get_vocab_size()}")

gpt_tokenizer.save_encoded(text, "data/encodings/gpt_bpe_tokens.npy")

print("=" * 50)
print("All tokenizers + encodings saved successfully!")