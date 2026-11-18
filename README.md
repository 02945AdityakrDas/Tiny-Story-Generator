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

Local RoPE: Uses a base frequency of 10,000 for sliding-window attention.
Global RoPE: Uses a base frequency of 1,000,000 for full attention.

This allows the architecture to use different positional encoding configurations for local and global attention layers.
### 3. Grouped Query Attention (GQA)

The attention mechanism uses Grouped Query Attention (GQA) to capture relationships between tokens efficiently.
Unlike standard Multi-Head Attention, which maintains separate Query, Key, and Value heads, GQA uses multiple Query heads that share a smaller number of Key-Value heads.
The input representations are projected into Query, Key, and Value vectors. RoPE is applied to the Query and Key vectors, and the Key-Value representations are expanded across their associated Query heads. Scaled dot-product attention is then computed using a causal attention mask, ensuring that a token cannot access future tokens during training. The resulting attention outputs are concatenated and passed through an output projection layer.
