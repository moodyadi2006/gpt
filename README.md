# 🧠 Mini GPT from Scratch (PyTorch)

This project is a **from-scratch implementation of a GPT-style Transformer model** built using PyTorch to deeply understand how Large Language Models (LLMs) work internally.

Instead of relying on APIs or frameworks, this project focuses on **building the complete training and inference pipeline manually**.

---

## 🚀 Motivation

Most developers today use LLMs via APIs (OpenAI, Claude, Gemini) or frameworks like LangChain.  
This project is an attempt to go **beyond usage** and understand:

- How transformers work internally  
- How models are trained  
- How text is generated token by token  

---

## ⚙️ Features

### 🧩 Training Pipeline

- Character-level tokenizer  
- Dataset creation using sliding window (block size)  
- Batch preparation using PyTorch DataLoader  
- Token + positional embeddings  
- Transformer blocks:
  - Self-Attention  
  - Multi-Head Attention  
  - Residual Connections  
  - Layer Normalization  
  - Feed Forward Network (FFN)  
- Final linear projection to logits  
- Cross Entropy Loss  
- Backpropagation  
- Adam Optimizer  
- Gradient clipping  
- Model checkpoint saving  

---

### 🤖 Inference Pipeline

- Load trained model and tokenizer  
- Autoregressive text generation  
- Next token prediction using logits  
- Softmax → probability distribution  
- Multinomial sampling (stochastic decoding)  
- Context window handling (block size)  

---

## 🧠 Key Concepts Learned

- How tokenization works  
- Difference between training and inference  
- Self-attention and multi-head attention  
- Role of embeddings and positional encoding  
- What logits represent  
- Cross entropy loss for language modeling  
- Backpropagation and gradient updates  
- Adam optimizer mechanics  
- Autoregressive generation  
- Sampling strategies (argmax vs multinomial)  

---

## 📂 Project Structure

```

.
├── classes/
│   ├── model.py                  # Token + positional embeddings
│   ├── attention.py              # Self-attention
│   ├── multi_head_attention.py   # Multi-head attention
│   ├── feed_forward.py           # FFN
│   ├── transformer.py            # Transformer block
│   ├── final_linear_projection.py
│   ├── cross_entropy_loss.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── softmax.py
│
├── training_pipeline.py
├── optimized_training_pipeline.py
├── inference_pipeline.py

├── data/
│   ├── input.txt
│   ├── tokenizer.json
│
├── checkpoints/
│   ├── best_model.pth
│   ├── tokenizer.json

````

---

## ▶️ How to Run

### 1. Train the model

```bash
python optimized_training_pipeline.py
````

This will:

* Train the model on the dataset
* Save best model in `checkpoints/`

---

### 2. Run inference (text generation)

```bash
python inference_pipeline.py
```

Then enter prompts like:

```
the king
ROMEO:
I love thee
```

---

## 💻 Hardware Note

This project is trained on a **CPU-only laptop**, so training can be slow.
For faster training, use a GPU and increase:

* `embed_dim`
* `num_layers`
* `block_size`
* `dataset size`

---

## 📈 Example Output

```
Input: the king
Output: the king hath no more words to speak...
```

(Note: Output quality depends on model size and training time)

---

## ⚠️ Limitations

* Character-level model (limited semantic understanding)
* Small model size
* Limited context window
* No advanced sampling (top-k, top-p yet)
* No KV caching or optimization

---

## 🚀 Future Improvements

* Top-k and Top-p sampling
* Temperature scaling
* Word-level or BPE tokenizer
* Larger model architecture
* GPU training
* KV caching for faster inference
* Instruction tuning

---

## 📌 Key Takeaway

This project helped in understanding that:

> LLMs are fundamentally **next-token prediction systems built on transformers**, and everything from ChatGPT to Claude follows this core idea at scale.

---

## 🤝 Contributing

Feel free to:

* Open issues
* Suggest improvements
* Experiment with configurations

---

## ⭐ If you found this helpful

Give it a ⭐ on GitHub and share your feedback!

---

## 📬 Connect

If you're also exploring Agentic AI and LLMs, feel free to connect and discuss ideas 🚀

