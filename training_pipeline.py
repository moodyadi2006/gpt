import torch
from torch.utils.data import DataLoader

from classes.model_v1 import SimpleModel
from classes.transformer_v1 import TransformerBlock
from classes.final_linear_projection import FinalProjection
from classes.cross_entropy_loss import CrossEntropyLossLayer
from classes.tokenizer import CharTokenizer
from classes.dataset_v1 import TextDataset

# ------------------------
# Load Data
# ------------------------
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size

# ------------------------
# Dataset
# ------------------------
# When training a transformer, the process begins with converting raw text into a form the model can learn from. First, the text is passed through a tokenizer, which maps each character or word into numerical tokens. These tokens are then used by the `TextDataset` class to create training examples using a sliding window approach controlled by `block_size`. For each position in the token list, the dataset generates an input sequence `x` of length `block_size` and a corresponding target sequence `y`, which is simply the same sequence shifted by one position to the right. This setup teaches the model next-token prediction—given a sequence of tokens, predict the next token at every position. The total number of such samples is `len(tokens) - block_size`, meaning larger `block_size` reduces the number of training samples but increases the context the model sees.

# Once the dataset is prepared, the `DataLoader` is used to efficiently feed this data into the model during training. It does not create new data or combinations but simply takes these precomputed `(x, y)` samples, optionally shuffles their order to improve generalization, and groups them into batches of size `batch_size`. Each batch is a tensor of shape `(batch_size, block_size)` for both inputs and targets, meaning the model processes multiple sequences in parallel. This batching is crucial because it balances computational efficiency, memory usage, and learning stability. Using the entire dataset as a single batch is theoretically possible but impractical due to memory constraints, slower updates, and poorer generalization, while very small batches can make training noisy and inefficient. Thus, `block_size` determines how much context each training example contains, and `batch_size` determines how many such examples are processed together, making them independent but complementary components in the training pipeline.

block_size = 32
dataset = TextDataset(text, tokenizer, block_size)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ------------------------
# Model Components
# ------------------------
embed_dim = 128
num_heads = 4
num_layers = 2

# Token and positional embeddings together form the foundational input representation for a transformer. First, each token from the vocabulary is mapped to a dense vector using a token embedding layer of size `(vocab_size, embed_dim)`, which captures the semantic meaning of the token. However, since transformers process all tokens in parallel and have no inherent notion of order, positional embeddings are added to encode the position of each token in the sequence. These are learned vectors of size `(max_len, embed_dim)`, where `max_len` defines the maximum sequence length the model can handle, and it must be at least as large as the sequence length (`block_size`) used during training. During the forward pass, token embeddings of shape `(batch_size, block_size, embed_dim)` are combined with positional embeddings of shape `(block_size, embed_dim)` through element-wise addition (via broadcasting), resulting in a final representation that encodes both “what the token is” and “where it is.” This combined embedding allows the transformer to understand sequence order while keeping the dimensionality fixed, making the model efficient and effective for sequence modeling.

embedding = SimpleModel(vocab_size, embed_dim)

# A Transformer block is the core unit that **refines token representations by combining context from other tokens and internal processing**, while keeping the shape of data unchanged. It takes input embeddings of shape `(batch_size, block_size, embed_dim)`, where each token already encodes both meaning (token embedding) and position (positional embedding). The first step inside the block is **Multi-Head Self-Attention (MHA)**, where each token interacts with all other tokens using Query, Key, and Value projections. Multiple heads allow the model to learn different types of relationships (e.g., semantic, syntactic, long-range dependencies) in parallel, as each head has its own parameters and evolves differently through training via gradient updates. The output of attention is then added back to the original input using a **residual connection**, followed by **Layer Normalization** to stabilize training. Next, the data passes through a **Feed Forward Network (FFN)**, which processes each token independently to add non-linearity and deeper feature transformation. Again, a residual connection and LayerNorm are applied. Importantly, the shape `(B, T, C)` remains constant throughout, enabling multiple such blocks to be stacked. Each layer has identical structure but different learned weights, allowing deeper layers to progressively build richer, more contextual and abstract representations of tokens. In essence, a Transformer block alternates between “learning from other tokens” (attention) and “processing internally” (FFN), while residual connections and normalization ensure stable and efficient learning.

transformer_blocks = [
    TransformerBlock(embed_dim, num_heads)
    for _ in range(num_layers)
]

# Logits are the raw output scores produced by the model after the final projection layer, before any normalization is applied. For each token in the sequence, the model generates a vector of size equal to the vocabulary, where each value represents how strongly the model associates that token with each possible next word. These values are not probabilities—they can be any real numbers (positive or negative) and do not sum to one. To convert logits into actual probabilities, a softmax function is applied, which normalizes them into a distribution where all values lie between 0 and 1 and sum to 1. During training, these logits are directly used in the cross-entropy loss function, which internally applies softmax in a numerically stable way. In essence, logits represent the model’s unnormalized confidence about which word should come next, and they are the bridge between learned representations and final predictions.

projection = FinalProjection(embed_dim, vocab_size)

# ------------------------
# Optimizer
# ------------------------
# In your training setup, all learnable components of the model—such as token and positional embeddings, transformer block weights (including attention, FFN, and normalization layers), and the final projection layer—contain parameters that need to be updated during training. Since you are not using a single unified model class, you manually collect these parameters by combining them into one list. This list is then passed to an optimizer (in your case, Adam), which is responsible for updating these parameters. However, the optimizer does not update them immediately; first, a forward pass computes predictions, then a loss is calculated, and backpropagation (`loss.backward()`) computes gradients for each parameter. These gradients are stored in each parameter’s `.grad` attribute. When `optimizer.step()` is called, the optimizer uses these gradients—along with its internal logic (like momentum and adaptive learning rates in Adam)—to adjust the parameters in a way that reduces the loss. In essence, parameter extraction defines *what can be learned*, and the optimizer defines *how it is learned*.

params = list(embedding.parameters()) + \
         [p for block in transformer_blocks for p in block.parameters()] + \
         list(projection.parameters())

optimizer = torch.optim.Adam(params, lr=1e-3)

# Cross Entropy Loss is the function that tells the model how well it is predicting the correct next token. For each token in the batch, the model outputs a vector of logits over the entire vocabulary, which are raw scores representing how strongly each word is considered. These logits are internally converted into probabilities using softmax. The target for each token is not random—it is the actual next token index from the dataset. The loss is then computed as the negative logarithm of the probability assigned to this correct token, meaning if the model assigns a high probability to the correct word, the loss is low, and if it assigns a low probability, the loss is high. Since the model processes multiple tokens across a batch and sequence length, the logits and targets are flattened so that each token prediction is treated as an independent classification task, and the loss is averaged across all tokens. Importantly, the model is not just trying to predict the correct token, but to assign it as high a probability as possible compared to all other vocabulary options. This loss value is then used during backpropagation to adjust the model’s parameters, gradually improving its predictions over time.

loss_fn = CrossEntropyLossLayer()

# ------------------------
# Training Loop
# ------------------------
# An **epoch** represents one complete pass over the entire dataset during training. Since datasets are usually large, they are divided into smaller chunks called **batches**, and each batch is processed sequentially within an epoch. For every batch, the model performs a **forward pass**, where input tokens are converted into embeddings (token + positional), passed through transformer blocks (multi-head attention + feed-forward layers), and finally projected into logits representing scores over the vocabulary. These logits are then compared with the true next-token targets using **cross entropy loss**, which measures how much probability the model assigns to the correct token.

# Once the loss is computed, **backpropagation** (`loss.backward()`) calculates gradients for every parameter in the model, indicating how each weight contributed to the error. These gradients are then used by the **optimizer (Adam)**, which updates each parameter individually using not just the current gradient but also historical information like momentum and adaptive scaling, allowing for stable and efficient learning. This forward → loss → backward → update cycle repeats for all batches in the dataset, completing one epoch. Over multiple epochs, the model repeatedly sees the data, gradually improving its predictions as the loss decreases. Additionally, the training loop often tracks metrics like average loss per epoch and saves the best-performing model, ensuring that the most optimal version of the model is retained. In essence, an epoch encapsulates the entire learning cycle of the model over the dataset, driving progressive improvement through repeated exposure and parameter updates.

"""
ADAM OPTIMIZER STEP:
Step 1: Initialize
m = 0 (momentum)
v = 0 (variance)
Step 2: Update m and v
m = 0.9 × m + 0.1 × grad
  = 0.1 × (-0.682)
  = -0.0682
v = 0.999 × v + 0.001 × grad²
  = 0.001 × (0.682²)
  ≈ 0.001 × 0.465
  ≈ 0.000465
Step 3: Compute update
update = m / sqrt(v)
       ≈ -0.0682 / sqrt(0.000465)
       ≈ -0.0682 / 0.0215
       ≈ -3.17
Step 4: Update weight

Let:

learning rate = 0.001
w = w - lr × update
  = w - (0.001 × -3.17)
  = w + 0.00317
"""
epochs = 5

best_loss = float("inf")

for epoch in range(epochs):
    total_loss = 0

    for x, y in dataloader:
        x = embedding(x)

        for block in transformer_blocks:
            x = block(x)

        logits = projection(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss

        torch.save({
            "embedding": embedding.state_dict(),
            "projection": projection.state_dict(),
            **{f"block_{i}": block.state_dict() for i, block in enumerate(transformer_blocks)}
        }, "checkpoints/best_model.pth")

        print("✅ Best model saved!")
# for epoch in range(epochs):
#     total_loss = 0

#     for x, y in dataloader:

#         # 1. Embedding
#         x = embedding(x)

#         # 2. Transformer stack
#         for block in transformer_blocks:
#             x = block(x)

#         # 3. Final projection
#         logits = projection(x)

#         # 4. Loss
#         loss = loss_fn(logits, y)

#         # 5. Backprop
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(dataloader)
#     print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

#     # 🔥 SAVE MODEL HERE
#     torch.save({
#         "embedding": embedding.state_dict(),
#         "projection": projection.state_dict(),
#         **{f"block_{i}": block.state_dict() for i, block in enumerate(transformer_blocks)}
#     }, f"model_epoch_{epoch+1}.pth")

# torch.save({
#     "embedding": embedding.state_dict(),
#     "projection": projection.state_dict(),
#     **{f"block_{i}": block.state_dict() for i, block in enumerate(transformer_blocks)}
# }, "model_final.pth")