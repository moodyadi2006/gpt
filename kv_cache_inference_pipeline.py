import torch
import time
from classes.model_v2 import SimpleModel
from classes.transformer_v2 import TransformerBlock
from classes.final_linear_projection import FinalProjection
from classes.tokenizer import CharTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Load tokenizer
# ------------------------
tokenizer = CharTokenizer()
tokenizer.load("checkpoints/char_tokenizer.json")

vocab_size = tokenizer.vocab_size

# ------------------------
# Model config
# ------------------------
embed_dim = 64
num_heads = 2
num_layers = 1
block_size = 16

# ------------------------
# Initialize model
# ------------------------
embedding = SimpleModel(vocab_size, embed_dim).to(device)

transformer_blocks = [
    TransformerBlock(embed_dim, num_heads).to(device)
    for _ in range(num_layers)
]

projection = FinalProjection(embed_dim, vocab_size).to(device)

# ------------------------
# Load weights
# ------------------------
checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)

embedding.load_state_dict(checkpoint["embedding"])

for i, block in enumerate(transformer_blocks):
    block.load_state_dict(checkpoint["blocks"][i])

projection.load_state_dict(checkpoint["projection"])

# ------------------------
# Eval mode
# ------------------------
embedding.eval()
projection.eval()
for block in transformer_blocks:
    block.eval()

# ------------------------
# Generation with KV cache
# ------------------------
def generate(start_text, max_new_tokens=50):
    tokens = tokenizer.encode(start_text)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    past_kvs = [None] * num_layers

    for step in range(max_new_tokens):

        with torch.no_grad():

            if step == 0:
                x = embedding(tokens)
            else:
                x = embedding(tokens[:, -1:], pos_offset=tokens.size(1)-1)

            new_kvs = []

            for i, block in enumerate(transformer_blocks):
                x, kv = block(x, past_kvs[i])
                new_kvs.append(kv)

            past_kvs = new_kvs

            logits = projection(x)
            logits = logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

        # 🔥 Optional: limit KV cache size
        for i in range(len(past_kvs)):
            Ks, Vs = past_kvs[i]

            new_Ks = []
            new_Vs = []

            for k, v in zip(Ks, Vs):
                if k.size(1) > block_size:
                    k = k[:, -block_size:, :]
                    v = v[:, -block_size:, :]

                new_Ks.append(k)
                new_Vs.append(v)

            past_kvs[i] = (new_Ks, new_Vs)

    return tokenizer.decode(tokens[0].tolist())

# ------------------------
# Interactive loop
# ------------------------
while True:
    prompt = input("\nEnter prompt (or 'exit'): ")

    if prompt.lower() == "exit":
        break

    start = time.time()   # ⏱ start

    output = generate(prompt)

    end = time.time()     # ⏱ end

    print("\nGenerated:\n", output)
    print(f"\nTime taken: {end - start:.4f} seconds")