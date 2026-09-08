---
title: "Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017)"
date: 2026-09-07 10:02:00 +0900
categories: [paper, llm-rl]
image: /assets/og-llm-rl.png
tags: [RL]
math: true
rating: 5
series: llm-rl
series_order: 2
description: "The origin of RLHF: learn a reward model from pairwise comparisons of short clips, train the agent against it, and ask for more comparisons where the reward ensemble disagrees. The Bradley–Terry preference model, the loss, the query strategy, and the MuJoCo/Atari figures."
paper:
  title: Deep Reinforcement Learning from Human Preferences
  published: 2017-06
  authors: Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, Dario Amodei (OpenAI, DeepMind)
  venue: NeurIPS
  year: 2017
  link: https://arxiv.org/abs/1706.03741
---

## One-line summary

When no reward function is available, show a human two short video clips of the agent, ask which is better, fit a neural reward predictor to those comparisons, and run RL on the predicted reward. With about 1% of the environment interactions labelled (roughly 700 comparisons for a MuJoCo task, under an hour of human time), the agent learns behaviours as well as with the true reward, and learns things (a backflip) no one can write a reward for.

## Why it matters

This is the paper that turns "RL from human feedback" into a working recipe: pairwise comparisons instead of absolute scores, a learned reward model, asynchronous updating of the reward model while the policy trains, and active selection of what to ask. [Ziegler et al.](/blog/ziegler-lm-preferences/) apply the same recipe to GPT-2 two years later, and InstructGPT's reward model is this one with $$K$$-way rankings.

## The setup

The agent interacts with an environment over trajectory segments $$\sigma = ((o_0, a_0), (o_1, a_1), \dots, (o_{k-1}, a_{k-1}))$$, a clip of 1–2 seconds. A human compares two segments and reports $$\sigma^1 \succ \sigma^2$$, $$\sigma^2 \succ \sigma^1$$, a tie, or "incomparable". Three processes run in parallel (Figure 1):

1. The policy $$\pi$$ interacts with the environment, producing trajectories, and is trained by an RL algorithm (TRPO for MuJoCo, A2C for Atari) using rewards from the predictor $$\hat r$$.
2. Pairs of segments are selected from the trajectories and sent to the human.
3. The reward predictor $$\hat r$$ is fit to the comparisons collected so far.

The human is queried about $$\hat r$$'s inputs, not the policy's, so the labels stay useful as the policy changes.

## The preference model and its loss

Assume the human's preference probability follows the Bradley–Terry / Luce model applied to the *summed* predicted reward over each clip:

$$
\hat P\big[\sigma^1 \succ \sigma^2\big] = \frac{\exp \sum_t \hat r(o^1_t, a^1_t)}{\exp \sum_t \hat r(o^1_t, a^1_t) + \exp \sum_t \hat r(o^2_t, a^2_t)}
$$

Fit $$\hat r$$ by minimizing the cross-entropy between these predictions and the labels, where $$\mu$$ is the human's distribution over the two clips (a one-hot for a clear preference, uniform for a tie):

$$
\operatorname{loss}(\hat r) = -\sum_{(\sigma^1, \sigma^2, \mu) \in D} \Big[\mu(1) \log \hat P[\sigma^1 \succ \sigma^2] + \mu(2) \log \hat P[\sigma^2 \succ \sigma^1]\Big]
$$

This is logistic regression on reward *differences*, so $$\hat r$$ is identifiable only up to an additive constant (and, through the exponent, a scale: the paper normalizes $$\hat r$$ to zero mean and constant variance before feeding it to RL). Section 2.2.3 lists the tricks that made it work: an ensemble of predictors trained on different bootstrap samples, a 10% assumed error rate ($$\hat P$$ is mixed with uniform so that a single inconsistent label cannot blow up the loss), $$\ell_2$$ regularization tuned to keep validation loss between 1.1 and 1.5× training loss, and treating 10% of comparisons as held-out validation.

**Query selection.** Sample many candidate pairs, compute the variance of $$\hat P$$ across ensemble members, and ask about the pairs with the highest disagreement. The paper admits this is crude (it does not account for the expected *value* of information) and that in some tasks it hurt; but it is the first "active RLHF" heuristic.

## Experiments

- **MuJoCo (Section 3.1.1).** Eight tasks (walker, hopper, swimmer, cheetah, ant, reacher, double pendulum, pendulum). Compare RL on the true reward, RL on a reward learned from *synthetic* labels (an oracle that compares clips by true reward, at 350/700/1400 queries), and RL on a reward learned from *real human* labels (700 queries). With 700 human labels the agent nearly matches true-reward RL on most tasks; with 1400 synthetic labels it sometimes does *better* than the true reward, which the authors attribute to the learned reward being better shaped.
- **Atari (Section 3.1.2).** Seven games, 5,500 human queries or 3.3k–10k synthetic ones. Human feedback matches or beats true-reward RL on Pong and BeamRider, is competitive on Seaquest and Qbert, and fails on Enduro (needs exploration) and Qbert with real labels (short clips hard to judge).
- **Novel behaviours (Section 3.2).** A Hopper doing repeated backflips and landing upright from 900 queries in under an hour; a Half-Cheetah moving forward on one leg; Enduro driving level with other cars. None has a hand-written reward.
- **Ablations (Section 3.3).** Removing online collection of labels (all labels from the initial random policy) hurts most; segments of length 1 (single frames) and no regularization also hurt; random queries are only slightly worse than the ensemble-variance strategy.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/christiano/figure1.png" width="483" height="221" alt="Figure 1 from Christiano et al. (2017)" loading="lazy"><figcaption>Figure 1 of Christiano et al. (2017), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (schematic).** The triangle of RL algorithm ↔ environment, reward predictor feeding the RL algorithm, and human feedback feeding the predictor. It is the same diagram as RLHF for language models with "environment" replaced by "prompt".

<figure class="paper-fig"><img src="/assets/papers/christiano/figure2.png" width="1025" height="553" alt="Figure 2 from Christiano et al. (2017)" loading="lazy"><figcaption>Figure 2 of Christiano et al. (2017), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (MuJoCo results).** Eight learning curves; orange is true-reward RL, purple is 700 human labels, blues are synthetic labels at three budgets. Purple tracks orange closely on most tasks; on Ant it beats it.

<figure class="paper-fig"><img src="/assets/papers/christiano/figure3.png" width="1025" height="558" alt="Figure 3 from Christiano et al. (2017)" loading="lazy"><figcaption>Figure 3 of Christiano et al. (2017), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (Atari results).** Seven learning curves in the same colour scheme; note the flat purple line on Qbert and Enduro, the two failures discussed in the text.

<figure class="paper-fig"><img src="/assets/papers/christiano/figure4.png" width="724" height="204" alt="Figure 4 from Christiano et al. (2017)" loading="lazy"><figcaption>Figure 4 of Christiano et al. (2017), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (backflip frames).** Four stills of the Hopper mid-backflip, the paper's most cited image.

<figure class="paper-fig"><img src="/assets/papers/christiano/figure5.jpg" width="982" height="529" alt="Figure 5 from Christiano et al. (2017)" loading="lazy"><figcaption>Figure 5 of Christiano et al. (2017), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5 (ablations on MuJoCo).** Curves with components removed; "no online queries" and "no segments" are the ones that collapse.

## Thoughts

- Almost every design choice here reappears in the language-model papers: comparisons instead of scores (humans are inconsistent on absolute scales), a learned reward model updated during training, and worry about the reward model being exploited. The 10% label-noise assumption and the ensemble are the parts later work quietly dropped.
- The cost argument ("1% of interactions labelled") is what made the approach practical; the LLM version inverts it, since sampling from a language model is cheap and labels are the bottleneck.
- The failure on Enduro is an early example of the reward model not fixing an exploration problem: preferences can only shape behaviour the policy already reaches.

Next: [Fine-Tuning Language Models from Human Preferences](/blog/ziegler-lm-preferences/), the same loop applied to GPT-2.
