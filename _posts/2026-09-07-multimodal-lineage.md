---
title: "Vision LLMs: from ViT to Cosmos-Reason1, a 13-paper reading path"
date: 2026-09-07 11:08:00 +0900
overview: true
categories: [paper, vision-llm]
image: /assets/og-vision-llm.png
tags: [CV, NLP]
description: "Thirteen papers on vision-language models: how images get into a language model (patches as tokens, contrastive alignment, four generations of connectors), and then how such models reason: multimodal chain of thought, R1-style RL, thinking with images, planning in pixels, chain-of-frames video reasoning, and physical common sense."
---

Thirteen papers, in order. The first seven are about how vision is connected to language models; the last six are about making those models reason. Each link is a review with the equations and figures explained.

| | When | Paper | What it added |
|---|---|---|---|
| **Images as tokens** | Oct 2020 | [Vision Transformer](/blog/vit/) — Dosovitskiy et al. | Patches as tokens; scale beats inductive bias |
| **Aligning with text** | Feb 2021 | [CLIP](/blog/clip/) — Radford et al. | Contrastive image–text pretraining; zero-shot by prompting |
| **Connecting to an LLM** | Apr 2022 | [Flamingo](/blog/flamingo/) — Alayrac et al. | Frozen backbones, Perceiver Resampler, gated cross-attention |
| | Jan 2023 | [BLIP-2](/blog/blip2/) — Li et al. | Q-Former bottleneck; two-stage pretraining |
| | Apr 2023 | [LLaVA](/blog/llava/) — Liu et al. | Linear projector + GPT-4-generated instruction data |
| | Oct 2023 | [LLaVA-1.5](/blog/llava-15/) — Liu et al. | MLP connector, 336 px, academic VQA data; grid high-res |
| **Any resolution** | Sep 2024 | [Qwen2-VL](/blog/qwen2-vl/) — Wang et al. | Native-resolution tokens, M-RoPE, unified image/video |
| **Reasoning with vision** | Feb 2023 | [Multimodal-CoT](/blog/multimodal-cot/) — Zhang et al. | Two-stage rationale → answer; fused ViT features cut hallucinated rationales |
| | Mar 2025 | [Vision-R1](/blog/vision-r1/) — Huang et al. | Cold start by modality bridging, GRPO with a progressive length schedule |
| | May 2025 | [DeepEyes](/blog/deepeyes/) — Zheng et al. | Zoom-in as an internal tool inside the chain of thought, trained end-to-end with RL |
| | May 2025 | [Visual Planning](/blog/visual-planning/) — Xu et al. | Plans as image sequences, no text; GRPO with a progress reward |
| | Sep 2025 | [Video models are zero-shot learners](/blog/video-models-zero-shot/) — Wiedemer et al. | Veo 3 does perception, editing and maze solving by prompting; chain of frames |
| **Physical reasoning** | Mar 2025 | [Cosmos-Reason1](/blog/cosmos-reason1/) — NVIDIA | Physical common-sense ontology, SFT + RL with self-supervised video rewards |

## How to read them

- ViT and CLIP are prerequisites: after them, "an image is a sequence of vectors that already know about language".
- The middle four papers answer one question, *how to hand those vectors to a language model*, with decreasing machinery: new cross-attention layers (Flamingo), a query transformer (BLIP-2), a linear layer (LLaVA), an MLP (LLaVA-1.5). Data and resolution, not the connector, turn out to matter most.
- Qwen2-VL removes the fixed token budget and gives positions three dimensions, which is where current open models are.
- The reasoning block asks one question in five ways, *where does the thinking happen?*: in text after reading the image once (Multimodal-CoT, Vision-R1), in text with the model looking again mid-chain (DeepEyes), in images only (Visual Planning), or in generated frames zero-shot (Veo 3). Cosmos-Reason1 fixes the target of the reasoning (physics and embodied decisions) and hands over to the VLA series.

Reviews link forward and backward and carry a "Part n of 13" navigation.
