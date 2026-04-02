import torch
import time
from classes.model_v1 import SimpleModel
from classes.transformer_v1 import TransformerBlock
from classes.final_linear_projection import FinalProjection
from classes.tokenizer import CharTokenizer

"""
The inference pipeline is responsible for generating text using the trained transformer model without updating its weights. Unlike training, where the model learns from data using loss and backpropagation, inference simply uses the learned parameters to predict the next token step-by-step. The process begins by loading the **same model architecture and trained weights** that were used during training, along with the tokenizer, which is essential for converting text into token IDs and back. The model is then set to evaluation mode using `.eval()` to disable training-specific behaviors and ensure stable predictions, and computations are performed inside a `torch.no_grad()` block to avoid unnecessary gradient calculations and improve efficiency.

Given an input text, it is first encoded into tokens and passed through the model. These tokens go through embedding layers (token + positional), transformer blocks (self-attention and feed-forward layers), and a final projection layer to produce logits representing scores over the vocabulary. Only the logits corresponding to the **last token** are used, since the task is to predict the next token. These logits are converted into probabilities using softmax, and a sampling strategy is applied to select the next token. Instead of using `argmax` (which always selects the highest probability and leads to repetitive outputs), `multinomial` sampling is used to introduce controlled randomness, allowing the model to generate more natural and diverse text.

The predicted token is then appended to the existing sequence, and the process repeats iteratively, generating one token at a time. To handle long sequences, the input is cropped to a fixed `block_size`, ensuring the model only processes the most recent context within its capacity. This entire process is called **autoregressive generation**, where each new token depends on previously generated tokens. Overall, the inference pipeline transforms a trained transformer into a text generator by repeatedly predicting and appending tokens, balancing probability-driven predictions with randomness to produce coherent and varied outputs.
"""
# ------------------------
# Device
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# Load tokenizer
# ------------------------
tokenizer = CharTokenizer()
tokenizer.load("checkpoints/char_tokenizer.json")

vocab_size = tokenizer.vocab_size

# ------------------------
# Model config (same as training)
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
# Load trained weights
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
# Generation function
# ------------------------
def generate(start_text, max_new_tokens=50):
    tokens = tokenizer.encode(start_text)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):

        # Crop to block_size
        tokens = tokens[:, -block_size:]

        with torch.no_grad():

            # Forward pass
            x = embedding(tokens)

            for block in transformer_blocks:
                x = block(x)

            logits = projection(x)

            # Last token
            logits = logits[:, -1, :]

            probs = torch.softmax(logits, dim=-1)

            # 🔥 Sampling (better than argmax)
            next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())

while True:
    prompt = input("\nEnter prompt (or 'exit'): ")

    if prompt.lower() == "exit":
        break

    start = time.time()   # ⏱ start

    output = generate(prompt)

    end = time.time()     # ⏱ end

    print("\nGenerated:\n", output)
    print(f"\nTime taken: {end - start:.4f} seconds")