# Simple Transformer LLM Implementation in PyTorch

A from-scratch implementation of a GPT-style transformer language model using PyTorch, trained on the tiny Shakespeare dataset using Byte-Pair Encoding (BPE) tokenization.

## Project Overview

This project demonstrates the core components of modern transformer-based large language models:
- **Sinusoidal Positional Encoding** - adds positional information to embeddings
- **Multi-Head Self-Attention** - enables the model to focus on different parts of the sequence
- **Feed-Forward Networks (MLP)** - adds non-linearity and capacity
- **Transformer Blocks** - stacks attention and MLP layers with residual connections
- **Causal Masking** - ensures autoregressive (left-to-right) generation

## Dataset

**Source:** [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)

The dataset is automatically downloaded during the first run and contains approximately 1.1M characters of Shakespeare's works. It's split into 80% training and 20% validation data.

## Tokenization

The project uses **Byte-Pair Encoding (BPE)** tokenization to convert raw text into token sequences.

### Features:
- **Vocabulary Size:** Dynamically calculated as `sqrt(text_length)` to balance compression and granularity
- **Special Tokens:** `[UNK]`, `[PAD]`, `[BOS]`, `[EOS]`
- **Pre-tokenizer:** Whitespace-based splitting
- **Implementation:** Using the `tokenizers` library

### Training Dataset Preparation:
```python
CustomDataset(text, tokenizer, seq_length=128, train=True, train_split=0.8)
```
- Encodes the entire text using the trained tokenizer
- Creates input-target pairs by shifting sequences by 1 token
- Splits data into training (80%) and validation (20%) sets
- Training data is shuffled; validation data is kept in order

## Architecture

### 1. **Input Embedding with Positional Encoding**

```python
class TransformerInputEmbedding(nn.Module)
```

Combines token embeddings with sinusoidal positional encodings.

**Sinusoidal Positional Encoding Formula:**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)$$

Where:
- **pos** = position in sequence (0, 1, 2, ..., max_seq_length - 1)
- **i** = dimension index (0, 1, 2, ..., d_model/2 - 1)
- **d_model** = embedding dimension (embed_dim)

**Parameters:**
- `vocab_size`: Size of vocabulary
- `embed_dim`: Dimension of embeddings
- `max_seq_length`: Maximum sequence length

### 2. **Multi-Head Self-Attention**

```python
class MultiHeadAttention(nn.Module)
```

Implements scaled dot-product attention with multiple representation subspaces.

**Key Features:**
- **Q, K, V Projections:** Linear projections for Query, Key, Value
- **Scaled Dot-Product:** Scores = (Q × K^T) / √(d_k)
- **Causal Masking:** Prevents attending to future tokens (GPT-style)
- **Key Padding Mask:** Masks out padding tokens
- **Multi-Head:** Splits embeddings into multiple heads for parallel attention

**Attention Computation:**
```
Attention(Q, K, V) = softmax(QK^T / √(d_k)) × V
```

**Parameters:**
- `embed_dim`: Embedding dimension (must be divisible by num_heads)
- `num_heads`: Number of attention heads
- `dropout`: Dropout rate

### 3. **Multi-Layer Perceptron (MLP)**

```python
class MLP(nn.Module)
```

Feed-forward network with two linear layers and activation function.

**Architecture:** 
- `Linear(embed_dim → hidden_dim)`
- Activation Function (GELU, ReLU, SiLU, Tanh, LeakyReLU)
- `Linear(hidden_dim → embed_dim)`

**Default:** `hidden_dim = 4 × embed_dim`

### 4. **Transformer Block**

```python
class TransformerBlock(nn.Module)
```

Combines attention and MLP with residual connections and layer normalization.

**Flow:**
```
x → LayerNorm → MultiHeadAttention → + residual → LayerNorm → MLP → + residual → x
```

### 5. **Transformer Language Model**

```python
class TransformerLM(nn.Module)
```

Complete transformer model for language modeling with:
- Input embedding + positional encoding
- Stack of N transformer blocks
- Layer normalization + output projection
- Tied embedding weights (input and output embeddings share parameters)

## Model Configuration

```python
@dataclass
class ModelConfig:
    vocab_size: int = 288              # sqrt(1M characters)
    max_seq_length: int = 128          # Maximum context length
    embed_dim: int = 256               # Embedding dimension
    num_heads: int = 8                 # Number of attention heads
    num_layers: int = 6                # Number of transformer blocks
    dropout: float = 0.1               # Dropout rate
    activation: str = "gelu"           # Activation function
    mlp_hidden_dim: int = 1024         # 4 × embed_dim
    pad_token_id: int = 1              # Padding token
    bos_token_id: int = 2              # Beginning of sequence
    eos_token_id: int = 3              # End of sequence
```

**Trainable Parameters:** ~25.5M

## Training

### Training Loop

```python
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=3e-4,
    checkpoint_dir="checkpoints",
    patience=5,
    device="cuda"
)
```

### Features:

1. **Optimizer:** Adam with learning rate 3e-4
2. **Loss Function:** Cross-entropy loss with padding token ignored
3. **Gradient Clipping:** max_norm=1.0 for stability
4. **Checkpointing:** Saves model weights at each epoch
5. **Early Stopping:** Stops after 5 epochs without validation improvement
6. **Device:** Automatically uses CUDA if available, otherwise CPU

### Checkpoint Structure:
```python
{
    'epoch': int,
    'model_state_dict': dict,
    'optimizer_state_dict': dict,
    'val_loss': float
}
```

### Saving Checkpoints:
- `best_model.pt` - Model with lowest validation loss
- `last_model.pt` - Model from last epoch
- `checkpoint_epoch_N.pt` - Model from epoch N

## Text Generation

### Generation Function

```python
generate_text(
    model,
    tokenizer,
    prompt="To be, or not to be, that is the question:",
    max_new_tokens=50,
    temperature=0.9,
    top_k=50,
    add_bos=False
)
```

**Parameters:**
- `temperature`: Controls randomness (0 = greedy, higher = more random)
- `top_k`: Only samples from top-k most likely tokens (improves quality)
- `add_bos`: Whether to prepend beginning-of-sequence token

**Sampling Strategies:**
- **Greedy:** `temperature=0` (deterministic, always pick argmax)
- **Top-k Sampling:** Restricts selection to top-k candidates by probability
- **Temperature Scaling:** Adjusts probability distribution

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Running the Notebook
```bash
jupyter notebook simple_transformer_implementation_in_pytorch.ipynb
```

### Training
```python
from simple_transformer_implementation_in_pytorch import *

# Initialize model
model = TransformerLM(CFG)

# Train with early stopping
model, train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=50,
    learning_rate=3e-4,
    checkpoint_dir="checkpoints",
    patience=5
)
```

### Loading a Checkpoint
```python
best_model = TransformerLM(CFG)
best_model = load_checkpoint(best_model, "checkpoints/best_model.pt")
```

### Text Generation
```python
prompt = "To be, or not to be, that is the question:"
output = generate_text(
    best_model,
    tokenizer,
    prompt,
    max_new_tokens=40,
    temperature=0.9,
    top_k=50
)
print(output)
```

### Visualization
```python
plot_losses_detailed(
    train_losses=train_losses,
    val_losses=val_losses,
    smoothing_window=3,
    save_path="loss_curves.png"
)
```

## Performance Visualization

The project includes a loss plotting function that displays:
- **Raw Losses:** Unsmoothed training and validation loss curves
- **Smoothed Losses:** Moving average for clearer trends
- **Best Epoch Indicator:** Marks the epoch with lowest validation loss

Output includes:
- Best validation loss and corresponding epoch
- Final training and validation losses
- Saved plot as PNG

## Dependencies

```
torch>=2.0.0
tokenizers>=0.13.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.11.0
```

## Key Design Decisions

1. **Sinusoidal Positional Encoding:** Standard approach that doesn't require training and generalizes to longer sequences
2. **Causal Masking:** Ensures autoregressive generation (each position only sees previous tokens)
3. **Tied Embeddings:** Reduces parameters by sharing weights between input and output embeddings
4. **Batch Normalization → Layer Normalization:** Used LayerNorm for better training stability in transformers
5. **Residual Connections:** Enables deeper networks without degradation
6. **Pre-norm Architecture:** LayerNorm applied before attention/MLP (modern transformer design)

## File Structure

```
Simple Transformer LLM Project/
├── simple_transformer_implementation_in_pytorch.ipynb  # Main implementation
├── README.md                                            # Documentation
├── requirements.txt                                     # Dependencies
├── data/
│   ├── input.txt                                        # Shakespeare dataset
│   └── bpe_tokenizer.json                               # Trained tokenizer
├── checkpoints/
│   ├── best_model.pt                                    # Best checkpoint
│   ├── last_model.pt                                    # Last epoch
│   └── checkpoint_epoch_N.pt                            # Epoch N
└── models/
```

## Training Tips

1. **Start with small models** for quick iteration (embed_dim=128, num_layers=2)
2. **Monitor validation loss** - use early stopping to prevent overfitting
3. **Adjust learning rate** - try 3e-4 to 1e-3 for this model size
4. **Gradient clipping** - helps with stability during training
5. **Batch size** - 32 works well for sequence length 128
6. **Sequence length** - longer sequences capture more context but require more memory

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [Language Models are Unsupervised Multitask Learners](https://arxiv.org/abs/1901.08783) - GPT-2 paper
- [Byte Pair Encoding](https://arxiv.org/abs/1508.07909) - Tokenization method

## License

This project is for educational purposes.

---

**Created:** February 2026
**Framework:** PyTorch
**Dataset:** Tiny Shakespeare
**Implementation Type:** From-scratch transformer LLM
