---
title: "Diffusion RL: from DDPO to DiffusionNFT, a 5-paper reading path"
date: 2026-09-07 14:06:00 +0900
overview: true
categories: [paper, diffusion-rl]
image: /assets/og-diffusion-rl.png
tags: [RL, Diffusion]
description: "Five papers on reinforcement learning for diffusion and flow models: denoising as an MDP with policy gradients, the same MDP nested inside a robot environment, GRPO on flow models via an ODE-to-SDE conversion, guidance as a policy improvement operator, and likelihood-free RL on the forward process. Each has a full review."
---

Five papers, in order, on using reinforcement learning to improve diffusion and flow-matching models, for images and for robot policies. Each link is a review with the equations and figures explained.

| | When | Paper | What it added |
|---|---|---|---|
| **Denoising as an MDP** | May 2023 | [DDPO](/blog/ddpo/) — Black et al. | Per-step Gaussian policy, exact policy gradient, clipped IS; VLM rewards |
| **Inside a robot MDP** | Sep 2024 | [DPPO](/blog/dppo/) — Ren et al. | Two-layer Diffusion Policy MDP, PPO fine-tuning of Diffusion Policy, sim-to-real |
| **Flow models** | May 2025 | [Flow-GRPO](/blog/flow-grpo/) — Liu et al. | ODE→SDE with the same marginals, GRPO with closed-form KL, 10-step rollouts |
| **Guidance = improvement** | May 2025 | [CFGRL](/blog/cfgrl/) — Frans et al. | Product policies improve; classifier-free guidance samples them; test-time knob |
| **Forward-process RL** | Sep 2025 | [DiffusionNFT](/blog/diffusion-nft/) — Zheng et al. | Implicit positive/negative policies, plain flow-matching loss, any solver, CFG-free |

## How to read them

- DDPO sets the formulation everything else uses or rejects: the reverse chain is an MDP whose per-step transitions are Gaussians, so REINFORCE and PPO apply. Read the RWR comparison carefully, it is the same "exact vs approximate likelihood" argument that reappears in DiffusionNFT.
- DPPO and Flow-GRPO are the two directions that formulation was pushed in: into an environment loop with an outer MDP (robotics), and onto deterministic flow models via an SDE with matching marginals (image generation). Both discover that short, noisy rollouts are enough for the gradient signal.
- CFGRL and DiffusionNFT are the other branch: no likelihoods, no sampler differentiation. Improvement is expressed as guidance between a reference model and an optimality-conditioned (or reward-weighted) model. CFGRL proves it and uses it at test time; DiffusionNFT learns it into the weights online.
- Watch two things across the series: what must be stored per sample (a full trajectory of latents → a clean image and a scalar), and what the sampler must be (first-order SDE → anything). The VLA series continues this thread with [π*0.6 / RECAP](/blog/pi-star-06-recap/), which applies the CFGRL idea to a real robot foundation model.

Reviews link forward and backward and carry a "Part n of 5" navigation.
