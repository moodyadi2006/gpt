import torch
import os
from torch.utils.data import DataLoader

from classes.model import SimpleModel
from classes.transformer import TransformerBlock
from classes.final_linear_projection import FinalProjection
from classes.cross_entropy_loss import CrossEntropyLossLayer
from classes.tokenizer import CharTokenizer
from classes.dataset import TextDataset

"""
Your training pipeline has been significantly improved by adding a set of practical optimizations that make it not just functional, but stable, efficient, and closer to real-world systems. First, you ensured proper **filesystem handling** by creating a `checkpoints/` directory before saving models, preventing runtime errors and enabling organized checkpoint storage. You introduced **device management** by moving both the model and data to the appropriate device (`CPU` or `GPU`) using `.to(device)`, which dramatically improves performance since transformers rely heavily on parallel matrix computations that GPUs handle efficiently. You also set the model to **training mode using `.train()`**, which is essential for correctly activating training-specific behaviors like dropout (even if not currently used, it ensures future correctness).

To improve numerical stability, you added **gradient clipping**, which prevents exploding gradients by limiting their magnitude, ensuring that weight updates remain controlled and training does not diverge. You optimized memory and performance further by using `optimizer.zero_grad(set_to_none=True)`, which is more efficient than the default gradient reset. Additionally, you saved the **tokenizer alongside the model**, which is crucial because the model’s learned weights are meaningless without the exact token-to-index mapping used during training, making this step essential for inference. You also improved checkpointing by saving the **best-performing model based on loss**, ensuring that you retain the most optimal version rather than just the last one. Finally, you identified the need for **causal masking**, which is critical for making the model truly autoregressive (like GPT), preventing it from accessing future tokens during training and ensuring correct learning behavior. Altogether, these optimizations enhance performance, stability, correctness, and usability, transforming your implementation from a basic prototype into a robust and scalable training pipeline.
"""
# ------------------------
# Setup
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs("checkpoints", exist_ok=True)

# ------------------------
# Load Data
# ------------------------
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
tokenizer.save("checkpoints/tokenizer.json")

vocab_size = tokenizer.vocab_size

# ------------------------
# Dataset
# ------------------------
block_size = 16
batch_size = 16

dataset = TextDataset(text, tokenizer, block_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ------------------------
# Model Components
# ------------------------
embed_dim = 64
num_heads = 2
num_layers = 1

embedding = SimpleModel(vocab_size, embed_dim).to(device)

transformer_blocks = [
    TransformerBlock(embed_dim, num_heads).to(device)
    for _ in range(num_layers)
]

projection = FinalProjection(embed_dim, vocab_size).to(device)

# Set training mode
embedding.train()
projection.train()
for block in transformer_blocks:
    block.train()

# ------------------------
# Optimizer
# ------------------------
params = list(embedding.parameters()) + \
         [p for block in transformer_blocks for p in block.parameters()] + \
         list(projection.parameters())

optimizer = torch.optim.Adam(params, lr=1e-3)

loss_fn = CrossEntropyLossLayer()

# ------------------------
# Training Loop
# ------------------------
epochs = 2
best_loss = float("inf")

for epoch in range(epochs):
    total_loss = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # Forward pass
        x = embedding(x)

        for block in transformer_blocks:
            x = block(x)

        logits = projection(x)
        loss = loss_fn(logits, y)

        # Backprop
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping (VERY IMPORTANT)
        torch.nn.utils.clip_grad_norm_(params, 1.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

    # ------------------------
    # Save Best Model
    # ------------------------
    if avg_loss < best_loss:
        best_loss = avg_loss

        torch.save({
            "epoch": epoch,
            "loss": avg_loss,
            "embedding": embedding.state_dict(),
            "projection": projection.state_dict(),
            "blocks": [block.state_dict() for block in transformer_blocks]
        }, "checkpoints/best_model.pth")

        print("✅ Best model saved!")

# ------------------------
# Save Final Model
# ------------------------
torch.save({
    "epoch": epochs,
    "loss": avg_loss,
    "embedding": embedding.state_dict(),
    "projection": projection.state_dict(),
    "blocks": [block.state_dict() for block in transformer_blocks]
}, "checkpoints/final_model.pth")

print("🚀 Training complete. Final model saved!")