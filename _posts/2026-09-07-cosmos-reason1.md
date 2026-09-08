---
title: "Cosmos-Reason1: From Physical Common Sense to Embodied Reasoning (NVIDIA, 2025)"
date: 2026-09-07 14:16:00 +0900
categories: [paper, vision-llm]
image: /assets/og-vision-llm.png
tags: [RL, CV, Robotics]
math: true
rating: 4
series: multimodal-llm
series_order: 13
description: "Two ontologies (Space/Time/Fundamental Physics with 16 subcategories; four embodied capabilities across agent types), two models (7B dense on Qwen2.5-VL, 56B hybrid Mamba-MLP-Transformer), and a two-stage recipe: Physical AI SFT on about 4M curated video-text annotations, then GRPO with rule-based MCQ rewards including self-supervised ones (shuffled spatiotemporal puzzles, arrow of time, object permanence). Benchmarks, results, and the async RL framework."
paper:
  title: "Cosmos-Reason1: From Physical Common Sense To Embodied Reasoning"
  published: 2025-03
  authors: NVIDIA
  venue: arXiv
  year: 2025
  link: https://arxiv.org/abs/2503.15558
---

## One-line summary

A VLM that watches a video, thinks in a long chain of thought, and outputs either an explanation of what is physically going on or an embodied decision such as the next action. NVIDIA first *defines* what "Physical AI reasoning" should mean with two ontologies, builds benchmarks from them (604 common-sense questions from 426 videos; 610 embodied questions from 600 videos across humans, robot arms, humanoids and vehicles), and then trains two models, Cosmos-Reason1-7B and 56B, in two stages: supervised fine-tuning on curated physical-AI data (captions, MCQs and DeepSeek-R1-distilled reasoning traces), then [GRPO](/blog/deepseekmath-grpo/) with rule-based, verifiable rewards. SFT gives +7 to +11 points over the backbones; RL adds another +5 on the main benchmarks and +7 on an intuitive-physics benchmark where the model reaches 81.5% while GPT-4o and o1 sit near chance.

## Why it matters

The previous papers in this series push multimodal reasoning on math and puzzles. Cosmos-Reason1 asks what reasoning a robot or a vehicle actually needs, writes it down as a taxonomy, and shows that the R1 recipe (SFT on reasoning traces, then RL with checkable rewards) transfers to that domain when the rewards come from the structure of video itself. It is the reasoning counterpart of the Cosmos world-model program and the direct ancestor of the reasoner inside [Cosmos 3](/blog/cosmos3/).

## Two ontologies (Section 2)

**Physical common sense (Figure 2, Table 1).** Three categories and 16 subcategories, chosen as *capabilities* rather than mechanisms:
- *Space*: Relationship (perspective-aware spatial relations), Plausibility, Affordance, Environment.
- *Time*: Actions (describe, decompose, verify completion), Order, Causality, Camera (position and movement), Planning.
- *Fundamental Physics*: Attributes, States and state change, Object Permanence, Mechanics (statics, kinematics, dynamics), Electromagnetism (optics, electricity, magnetism), Thermodynamics, Anti-Physics (recognizing impossible events).

**Embodied reasoning (Table 2).** Four capabilities, process complex sensory inputs, predict action effects, respect physical constraints, learn from interaction, crossed with agent types (natural agents; robot arms, humanoids, autonomous vehicles). The paper covers the first three with video input, framed as task-completion verification, next-plausible-action prediction, and action affordance.

## Models (Section 3)

Decoder-only, LLaVA-style: vision encoder → projector (two-layer MLP with downsampling) → LLM. **7B**: Qwen2.5-VL with its native dynamic-resolution processing. **56B**: InternViT-300M-V2.5 encoder, 1–12 tiles of 448×448 per image plus a thumbnail, up to 32 video frames at 2 fps, 1,024 tokens per frame reduced to 256 by 2×2 PixelShuffle, and a **hybrid Mamba-MLP-Transformer** backbone (Nemotron-H, 118 layers, Figure 4): mostly alternating Mamba and MLP blocks with a few self-attention blocks interleaved for long-context modeling, trading the quadratic cost of attention for linear-time state-space layers on long video token sequences.

## Training (Sections 4–5)

**Physical AI SFT.** About 4M video–text annotations from two pipelines. For common sense: free-form and multiple-choice VQA over videos, with "understanding" annotations (structured captions of states and actions) and "reasoning" annotations (long CoT traces distilled from DeepSeek-R1 given the captions). For embodied reasoning: subsampled and converted datasets across embodiments (BridgeData V2, RoboVQA, AgiBot, HoloAssist, in-house AV data), each with captions and R1-generated traces for next-subtask prediction, completion verification and affordance. Three **intuitive-physics** tasks are self-supervised by construction (Section 5.1.3): spatial puzzles (shuffle 2×2 patches of a frame among 31 distractor patches and ask for their positions), arrow of time (is the video played forward or backward?), and object permanence (simulated scenes).

**Physical AI RL.** GRPO with the group-normalized advantage

$$
A_i = \frac{R(o_i) - \text{mean}(\mathcal{G})}{\text{std}(\mathcal{G})}, \qquad \mathcal{G} = \{o_1, \dots, o_G\}
$$

and two rule-based rewards: **accuracy** (the `<answer>` string matches the MCQ ground truth) and **format** (thinking inside `<think>`, answer inside `<answer>`, checked by regex). MCQ choices are reshuffled on the fly. Batch 128 questions × 9 samples, max 6,144 tokens, KL coefficient 0.005, 500 iterations. The RL data mixes human-annotated MCQs with the self-supervised puzzle/arrow-of-time/permanence MCQs, so the reward is verifiable *and* about physics.

**Infrastructure (Figure 5).** A fully asynchronous framework: a dispatcher schedules prompts to actor-rollout nodes (DP/PP/TP) and policy-training nodes (5-D parallelism); rewards and advantages are computed on the rollout side; a reference model supplies the KL. Heterogeneous deployment avoids the synchronization stalls of co-located designs (about 160% higher throughput), and the training mesh can drop or add nodes mid-run without restarting.

## Benchmarks and results (Sections 6–7)

- **Physical common sense (Table 7).** Average over Space/Time/Physics: Qwen2.5-VL-7B 47.4 → Cosmos-Reason1-7B 54.3; Nemotron-H-56B 58.2 → Cosmos-Reason1-56B 60.2, slightly above OpenAI o1 (59.9) and GPT-4o (55.6).
- **Embodied reasoning (Table 8).** Six sub-benchmarks (BridgeData V2, RoboVQA, AgiBot, HoloAssist, AV, RoboFail). 7B: 50.8 → 61.8; 56B: 53.5 → 63.7, both above o1 (54.5). RoboFail, hand-curated hard affordance/verification cases, stays hard for everyone.
- **After RL (Table 9).** Cosmos-Reason1-7B average 60.7 → 65.7, with the largest gains on BridgeData V2 (58.8 → 73.5) and AV (55.6 → 67.0); RoboFail unchanged, attributed to a lack of representative training data.
- **Intuitive physics (Table 10).** Arrow of time / spatial puzzle / object permanence: GPT-4o 50.0 / 77.0 / 48.0 and o1 51.0 / 64.0 / 49.0 are at chance on two of three; Cosmos-Reason1-7B after SFT 56.0 / 85.4 / 82.0, after RL 64.5 / 94.0 / 86.0 (average 81.5).
- **Behaviour (Figures 9–12).** After RL the model rejects all options of an ambiguous driving question, reasons through reversed videos while ignoring stationary text, and infers that a disappearing object is not a camera effect.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/cosmosreason1/figure1.png" width="1142" height="527" alt="Figure 1 from NVIDIA (2025)" loading="lazy"><figcaption>Figure 1 of NVIDIA (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (overview).** Model (vision encoder, projector, pre-trained LLM), training (Physical AI SFT, Physical AI RL), ontology and benchmark, each split into physical common sense and embodied reasoning.

<figure class="paper-fig"><img src="/assets/papers/cosmosreason1/figure2.png" width="614" height="614" alt="Figure 2 from NVIDIA (2025)" loading="lazy"><figcaption>Figure 2 of NVIDIA (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (ontology).** A sunburst: Space (Relationship, Plausibility, Affordance, Environment), Time (Actions, Order, Causality, Camera, Planning), Fundamental Physics (Attributes, States, Object Permanence, Mechanics, Electromagnetism, Thermodynamics, Anti-Physics).

<figure class="paper-fig"><img src="/assets/papers/cosmosreason1/figure3.png" width="1147" height="623" alt="Figure 3 from NVIDIA (2025)" loading="lazy"><figcaption>Figure 3 of NVIDIA (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (architecture).** Video → vision encoder → projector → video tokens concatenated with text tokens ("What's the next action?") → dense or hybrid LLM → a `<think>` block describing the robot grabbing an apple, then the answer "put the apple in the right hand into the bag on the table".

<figure class="paper-fig"><img src="/assets/papers/cosmosreason1/figure4.png" width="1157" height="343" alt="Figure 4 from NVIDIA (2025)" loading="lazy"><figcaption>Figure 4 of NVIDIA (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (hybrid backbone).** The 56B stack: blocks of alternating Mamba-MLP (×7, ×9, ×10) separated by Transformer blocks (self-attention + MLP).

<figure class="paper-fig"><img src="/assets/papers/cosmosreason1/figure5.png" width="907" height="606" alt="Figure 5 from NVIDIA (2025)" loading="lazy"><figcaption>Figure 5 of NVIDIA (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5 (RL framework).** Dispatcher, actor rollout (with reward calculation and parameter receive), policy training (5-D parallel, parameter send), and the reference model providing the KL; NCCL communicators between them.

<figure class="paper-fig"><img src="/assets/papers/cosmosreason1/figure9.png" width="1208" height="1566" alt="Figure 9 from NVIDIA (2025)" loading="lazy"><figcaption>Figure 9 of NVIDIA (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 9 (before/after RL).** A driving video with the question "what is the most likely immediate action?": before RL the model talks itself into "change to left lane"; after RL it notes the double yellow lines and parked cars, rules out every option, and says none apply.

## Thoughts

- The ontology is the durable contribution: a checklist of what a physical reasoner should know that others can benchmark against, independent of the model.
- The RL section is the R1 recipe with a twist that matters: the rewards for arrow-of-time, patch puzzles and permanence are *free* (self-supervised from the video), and they are precisely the things frontier VLMs cannot do. Table 10 is the most striking table in the paper.
- Reasoning still ends in text. Whether the next action should be a sentence or an action chunk is the question the VLA series takes up, and Cosmos 3 later folds this reasoner and a generator into one model.

This is the last paper in the Vision LLM series. The [overview post](/blog/multimodal-lineage/) lists all thirteen in order. The story continues in the VLA series with [π*0.6 / RECAP](/blog/pi-star-06-recap/) and [Cosmos 3](/blog/cosmos3/).
