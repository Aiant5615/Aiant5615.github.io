---
title: "LLM RL: from PPO to DeepSeek-R1, an 8-paper reading path"
date: 2026-09-07 10:09:00 +0900
overview: true
categories: [paper, llm-rl]
tags: [RL]
description: "Eight papers that explain how reinforcement learning is used to train language models: the optimizer (PPO), learning rewards from human preferences, RLHF on text, AI feedback, DPO's closed form, and RL with verifiable rewards (GRPO, R1). Each has a full review."
---

Eight papers, in order, on reinforcement learning for language models. Each link is a review with the equations and figures explained.

| | When | Paper | What it added |
|---|---|---|---|
| **The optimizer** | Jul 2017 | [Proximal Policy Optimization](/blog/ppo/) — Schulman et al. | The clipped surrogate objective used by all RLHF work |
| **Learning the reward** | Jun 2017 | [Deep RL from Human Preferences](/blog/christiano-preferences/) — Christiano et al. | Reward model from pairwise comparisons, active queries |
| **RLHF on text** | Sep 2019 | [Fine-Tuning LMs from Human Preferences](/blog/ziegler-lm-preferences/) — Ziegler et al. | RLHF on GPT-2, KL penalty, online labels |
| | Sep 2020 | [Learning to Summarize from Human Feedback](/blog/stiennon-summarize/) — Stiennon et al. | Clean RLHF result; reward-model overoptimization measured |
| | Mar 2022 | [InstructGPT](/blog/instructgpt/) — Ouyang et al. | SFT → RM → PPO-ptx on an instruction distribution (reviewed in the LLM-basics series) |
| **AI feedback** | Dec 2022 | [Constitutional AI](/blog/constitutional-ai/) — Bai et al. | Critique/revision and RLAIF from written principles |
| **Closed form** | May 2023 | [Direct Preference Optimization](/blog/dpo/) — Rafailov et al. | The RL step has a closed form; one classification loss |
| **Verifiable rewards** | Feb 2024 | [DeepSeekMath / GRPO](/blog/deepseekmath-grpo/) — Shao et al. | Group-normalized advantages, no value network |
| | Jan 2025 | [DeepSeek-R1](/blog/deepseek-r1/) — DeepSeek-AI | RL on a base model with rule rewards; long reasoning emerges |

## How to read them

- PPO first: it is the update rule inside everything else, and the difference between "KL to the old policy" and "KL to the reference model" is easiest to see there.
- Christiano → Ziegler → Stiennon → InstructGPT is one experiment repeated with better data and bigger models; read Stiennon's Figure 5 (reward-model overoptimization) carefully, it explains every KL term that follows.
- Constitutional AI and DPO each remove one human-in-the-loop component (labels for harmlessness; the RL loop). GRPO and R1 remove the learned reward model entirely for tasks with a checker.

Reviews link forward and backward and carry a "Part n of 8" navigation. The LLM-basics series covers InstructGPT itself; it is included here for order.
