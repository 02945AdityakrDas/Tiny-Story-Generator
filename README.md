# Tiny-Story-Generator
## 🌟 Overview
This repository contains the complete pipeline to pre-train, tokenize, and run inference with a compact Small Language Model (Gemma3 270 Million). The model is designed to generate creative narratives using simple vocabulary typically understood by 3 to 4-year-olds.
## 🛠️ Architecture
The project implements a decoder-only Transformer architecture inspired by Gemma 3, built from scratch using PyTorch. The architecture is designed for autoregressive language modeling, where the model learns to predict the next token based on previously observed tokens.

The implementation consists of 18 Transformer blocks, with each block containing a Grouped Query Attention mechanism and a gated Feed-Forward Network. The architecture incorporates several important techniques used in modern language models, including Rotary Positional Embeddings (RoPE), RMSNorm, Sliding Window Attention, and residual connections.
### 1. Token Embedding

The input text is first converted into token IDs using the GPT-2 tokenizer (tiktoken). These token IDs are passed through an embedding layer, which maps each token into a dense vector representation of dimension 640.

The resulting embeddings are scaled by the square root of the embedding dimension before being passed through the Transformer blocks.
### 2. Rotary Positional Embeddings (RoPE)

Since Transformer architectures do not inherently understand token order, positional information must be incorporated into the attention mechanism.
This implementation uses Rotary Positional Embeddings (RoPE) to encode positional information directly into the Query and Key representations.
The implementation precomputes sine and cosine positional values and applies rotary transformations to the Query and Key vectors before attention computation.

Two sets of RoPE parameters are maintained:

**Local RoPE:** Uses a base frequency of 10,000 for sliding-window attention.
**Global RoPE:** Uses a base frequency of 1,000,000 for full attention.

This allows the architecture to use different positional encoding configurations for local and global attention layers.
### 3. Grouped Query Attention (GQA)

The attention mechanism uses Grouped Query Attention (GQA) to capture relationships between tokens efficiently.
Unlike standard Multi-Head Attention, which maintains separate Query, Key, and Value heads, GQA uses multiple Query heads that share a smaller number of Key-Value heads.

The input representations are projected into Query, Key, and Value vectors. RoPE is applied to the Query and Key vectors, and the Key-Value representations are expanded across their associated Query heads. Scaled dot-product attention is then computed using a causal attention mask, ensuring that a token cannot access future tokens during training. The resulting attention outputs are concatenated and passed through an output projection layer.
### 4. Sliding Window and Global Attention

One of the important architectural features is the combination of Sliding Window Attention and Global Attention.

Instead of allowing every Transformer layer to attend to the entire preceding sequence, the implementation uses two different attention patterns:

**Sliding Window Attention:** Each token can attend only to a limited number of preceding tokens within a window of 512 positions. This helps capture local contextual relationships.

**Global Attention:** Tokens can attend to all preceding tokens within the available sequence, allowing the model to capture longer-range dependencies.

The 18 Transformer blocks follow a predefined attention pattern consisting of 15 sliding-window attention layers and 3 global attention layers. Global attention is introduced at layers 6, 12, and 18 (1-indexed).

This combination allows the model to learn both local language patterns and broader contextual relationships.
### 5. RMSNorm and Residual Connections

The architecture uses Root Mean Square Layer Normalization (RMSNorm) instead of conventional Layer Normalization.

RMSNorm normalizes hidden representations using their root mean square value, helping maintain numerical stability during training.

Each Transformer block uses four normalization operations:

**Input LayerNorm (RMSNorm) before attention.**

**Post-Attention RMSNorm.**

**Pre-Feed-Forward RMSNorm.**

**Post-Feed-Forward RMSNorm.**

Residual connections are incorporated around both the attention and feed-forward sublayers. These shortcut connections preserve information from earlier representations and improve gradient propagation through the network.
### 6. Gated Feed-Forward Network (SwiGLU-style)

After the attention operation, the representations pass through a gated Feed-Forward Network.

The network uses three linear projection layers:

**Two parallel projections transform the input into the intermediate hidden dimension.**

**The GELU activation function with tanh approximation is applied to one projection.**

**The activated output is multiplied element-wise by the second projection to introduce a gating mechanism.**

**The gated representation is passed through a final linear projection to return to the original embedding dimension.**

The intermediate hidden dimension is 2048, while the output dimension is 640.

This gated architecture provides nonlinear feature transformations and improves the model's ability to learn complex relationships in language.
### 7. Transformer Stack and Language Modeling Head

The complete model consists of 18 sequential Transformer blocks, each containing the attention and feed-forward sublayers described above.

After passing through all Transformer blocks, the hidden representations are processed by a final RMSNorm layer.

A linear language modeling head then projects the final hidden representations into the vocabulary space of 50,257 tokens, producing logits representing the next-token prediction scores.

During training, these logits are compared with the target token IDs using Cross-Entropy Loss.

During inference, the model generates text autoregressively by selecting the next token from the predicted probability distribution. Temperature scaling and Top-K sampling are used to control the generation process.
