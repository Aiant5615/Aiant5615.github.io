---
title: "Multimodal LLMs: from ViT to Qwen2-VL, a 7-paper reading path"
date: 2026-09-07 11:08:00 +0900
overview: true
categories: [paper, multimodal]
tags: [CV, NLP]
description: "Seven papers that explain how images (and video) get into a language model: patches as tokens, contrastive image–text alignment, and four generations of connectors, from gated cross-attention to a linear layer to native-resolution tokens with 3-D positions."
---

Seven papers, in order, on how vision is connected to language models. Each link is a review with the equations and figures explained.

| | When | Paper | What it added |
|---|---|---|---|
| **Images as tokens** | Oct 2020 | [Vision Transformer](/blog/vit/) — Dosovitskiy et al. | Patches as tokens; scale beats inductive bias |
| **Aligning with text** | Feb 2021 | [CLIP](/blog/clip/) — Radford et al. | Contrastive image–text pretraining; zero-shot by prompting |
| **Connecting to an LLM** | Apr 2022 | [Flamingo](/blog/flamingo/) — Alayrac et al. | Frozen backbones, Perceiver Resampler, gated cross-attention |
| | Jan 2023 | [BLIP-2](/blog/blip2/) — Li et al. | Q-Former bottleneck; two-stage pretraining |
| | Apr 2023 | [LLaVA](/blog/llava/) — Liu et al. | Linear projector + GPT-4-generated instruction data |
| | Oct 2023 | [LLaVA-1.5](/blog/llava-15/) — Liu et al. | MLP connector, 336 px, academic VQA data; grid high-res |
| **Any resolution** | Sep 2024 | [Qwen2-VL](/blog/qwen2-vl/) — Wang et al. | Native-resolution tokens, M-RoPE, unified image/video |

## How to read them

- ViT and CLIP are prerequisites: after them, "an image is a sequence of vectors that already know about language".
- The middle four papers answer one question, *how to hand those vectors to a language model*, with decreasing machinery: new cross-attention layers (Flamingo), a query transformer (BLIP-2), a linear layer (LLaVA), an MLP (LLaVA-1.5). Data and resolution, not the connector, turn out to matter most.
- Qwen2-VL removes the fixed token budget and gives positions three dimensions, which is where current open models are.

Reviews link forward and backward and carry a "Part n of 7" navigation.
