---
title: "VLA (Robotics): from RT-1 to π0, a 5-paper reading path"
date: 2026-09-07 13:06:00 +0900
overview: true
categories: [paper, vla]
tags: [Robotics]
description: "Five papers on vision-language-action models: the first large robot Transformer, actions as language tokens on a web-scale VLM, an open cross-robot generalist with a diffusion head, an open 7B VLA, and a VLM backbone with a flow-matching action expert. Each has a full review."
---

Five papers, in order, on vision-language-action models for robots. Each link is a review with the equations and figures explained.

| | When | Paper | What it added |
|---|---|---|---|
| **Robot Transformer** | Dec 2022 | [RT-1](/blog/rt1/) — Brohan et al. | 130k demos, 700 tasks, discretized action tokens from a Transformer |
| **Actions as language** | Jul 2023 | [RT-2](/blog/rt2/) — Brohan et al. | Co-fine-tune a 55B VLM; web knowledge transfers to control |
| **Open generalist** | May 2024 | [Octo](/blog/octo/) — Octo Model Team | 800k OXE trajectories, diffusion action head, easy fine-tuning |
| **Open VLA** | Jun 2024 | [OpenVLA](/blog/openvla/) — Kim et al. | 7B open VLM + action tokens; LoRA and int4 for adaptation |
| **Flow-matching actions** | Oct 2024 | [π0](/blog/pi0/) — Black et al. | VLM backbone + action expert, 50-step chunks at 50 Hz |

## How to read them

- RT-1 sets the interface (images + instruction → action tokens) and the evaluation axes.
- RT-2 and OpenVLA are the *language-model* branch: reuse a VLM, treat actions as words. Octo is the *policy* branch: a small transformer with a generative action head. π0 merges the two.
- Watch three numbers across the series: control frequency (3 Hz → 1–3 Hz → 5–10 Hz → 50 Hz), action representation (256 bins → bins → diffusion chunks → flow-matching chunks), and dataset size (130k episodes → 970k → 10k hours).

Reviews link forward and backward and carry a "Part n of 5" navigation.
