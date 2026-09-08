---
title: "VLA (Robotics): from RT-1 to Cosmos 3, an 8-paper reading path"
date: 2026-09-07 13:06:00 +0900
overview: true
categories: [paper, vla]
image: /assets/og-vla.png
tags: [Robotics]
description: "Eight papers on vision-language-action models: the first large robot Transformer, actions as language tokens, an open cross-robot generalist, an open 7B VLA, a flow-matching action expert, RL from deployment via advantage conditioning, a steerable generalist with subgoal images and metadata prompts, and an omnimodal world model that reasons, generates and acts. Each has a full review."
---

Eight papers, in order, on vision-language-action models for robots. Each link is a review with the equations and figures explained.

| | When | Paper | What it added |
|---|---|---|---|
| **Robot Transformer** | Dec 2022 | [RT-1](/blog/rt1/) — Brohan et al. | 130k demos, 700 tasks, discretized action tokens from a Transformer |
| **Actions as language** | Jul 2023 | [RT-2](/blog/rt2/) — Brohan et al. | Co-fine-tune a 55B VLM; web knowledge transfers to control |
| **Open generalist** | May 2024 | [Octo](/blog/octo/) — Octo Model Team | 800k OXE trajectories, diffusion action head, easy fine-tuning |
| **Open VLA** | Jun 2024 | [OpenVLA](/blog/openvla/) — Kim et al. | 7B open VLM + action tokens; LoRA and int4 for adaptation |
| **Flow-matching actions** | Oct 2024 | [π0](/blog/pi0/) — Black et al. | VLM backbone + action expert, 50-step chunks at 50 Hz |
| **Learning from experience** | Nov 2025 | [π*0.6 / RECAP](/blog/pi-star-06-recap/) — Physical Intelligence | Distributional value function, advantage token in the prompt, 2× throughput |
| **Steerable generalist** | Apr 2026 | [π0.7](/blog/pi07/) — Physical Intelligence | Subgoal images from a world model, episode metadata, CFG; zero-shot cross-embodiment |
| **One model for all** | Jun 2026 | [Cosmos 3](/blog/cosmos3/) — NVIDIA | Mixture-of-Transformers reasoner + generator; actions as a diffusion modality |

## How to read them

- RT-1 sets the interface (images + instruction → action tokens) and the evaluation axes.
- RT-2 and OpenVLA are the *language-model* branch: reuse a VLM, treat actions as words. Octo is the *policy* branch: a small transformer with a generative action head. π0 merges the two.
- Watch three numbers across the series: control frequency (3 Hz → 1–3 Hz → 5–10 Hz → 50 Hz), action representation (256 bins → bins → diffusion chunks → flow-matching chunks), and dataset size (130k episodes → 970k → 10k hours → tens of thousands of hours plus autonomous data).
- The last three papers are about what happens *after* imitation: π*0.6 adds a value function and lets the robot learn from its own rollouts (the [Diffusion RL](/blog/diffusion-rl-lineage/) series explains the CFGRL idea it builds on); π0.7 folds those rollouts, failures and human video into one generalist by describing each episode in the prompt; Cosmos 3 merges the reasoner, the video world model and the policy into one network.

Reviews link forward and backward and carry a "Part n of 8" navigation.
