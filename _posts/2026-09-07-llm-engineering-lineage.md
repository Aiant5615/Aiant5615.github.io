---
title: "LLM engineering: from RoPE to QLoRA, an 8-paper reading path"
date: 2026-09-07 12:09:00 +0900
overview: true
categories: [paper, llm-engineering]
image: /assets/og-llm-engineering.png
tags: [NLP, DL]
description: "Eight papers on the techniques that turn a Transformer language model into a modern LLM: rotary positions, mixture of experts, low-rank adaptation, compute-optimal scaling, chain-of-thought prompting, IO-aware attention, the LLaMA recipe, and 4-bit fine-tuning. Each has a full review."
---

Eight papers, in order, on the techniques inside current language models. Each link is a review with the equations and figures explained.

| | When | Paper | What it added |
|---|---|---|---|
| **Positions** | Apr 2021 | [RoFormer / RoPE](/blog/roformer/) — Su et al. | Rotary position embedding: relative positions from rotated queries and keys |
| **Sparse capacity** | Jan 2021 | [Switch Transformers](/blog/switch-transformer/) — Fedus, Zoph, Shazeer | Top-1 mixture of experts with a load-balancing loss |
| **Cheap adaptation** | Jun 2021 | [LoRA](/blog/lora/) — Hu et al. | Low-rank weight updates; no inference latency |
| **How much data** | Mar 2022 | [Chinchilla](/blog/chinchilla/) — Hoffmann et al. | Scale tokens with parameters; 20 tokens per parameter |
| **Reasoning by prompting** | Jan 2022 | [Chain-of-Thought Prompting](/blog/chain-of-thought/) — Wei et al. | Worked rationales in the prompt; emergent at scale |
| **Fast exact attention** | May 2022 | [FlashAttention](/blog/flashattention/) — Dao et al. | Tiling and recomputation; linear memory, 2–4× speed |
| **The open recipe** | Feb 2023 | [LLaMA](/blog/llama/) — Touvron et al. | RMSNorm, SwiGLU, RoPE, public data, trained past optimal |
| **Fine-tune anywhere** | May 2023 | [QLoRA](/blog/qlora/) — Dettmers et al. | NF4 quantization + LoRA; 65B on one GPU |

## How to read them

- RoPE, RMSNorm/SwiGLU (in LLaMA) and FlashAttention are the *architecture and kernels* of a 2023+ model; read them to understand a model card.
- Switch and Chinchilla are about *where to spend compute*: on more experts, or on more tokens.
- Chain-of-thought is about spending compute *at inference*; LoRA and QLoRA about spending very little of it to adapt a model.

Reviews link forward and backward and carry a "Part n of 8" navigation.
