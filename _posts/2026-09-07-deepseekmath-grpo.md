---
title: "DeepSeekMath and Group Relative Policy Optimization (Shao et al., 2024)"
date: 2026-09-07 10:07:00 +0900
categories: [paper, llm-rl]
image: /assets/og-llm-rl.png
tags: [RL, NLP]
math: true
rating: 5
series: llm-rl
series_order: 7
description: "GRPO derived next to PPO: the group-normalized advantage that replaces the value network, the clipped objective, the unbiased KL estimator, outcome vs process supervision, iterative RL, and the unified view of SFT/RFT/DPO/PPO/GRPO as one gradient with different coefficients. Plus the 120B-token math corpus."
paper:
  title: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"
  published: 2024-02
  authors: Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y.K. Li, Y. Wu, Daya Guo (DeepSeek)
  venue: arXiv
  year: 2024
  link: https://arxiv.org/abs/2402.03300
---

## One-line summary

Two contributions: a 120B-token web corpus of mathematical text mined from Common Crawl with an iteratively trained classifier, which lifts a 7B model to 51.7% on MATH; and **GRPO**, a PPO variant that samples a *group* of answers per question, uses their mean and standard deviation as the baseline instead of a learned value network, and puts the KL penalty in the loss rather than the reward. GRPO is the algorithm behind [DeepSeek-R1](/blog/deepseek-r1/).

## Why it matters

For LLMs the PPO value network is as large as the policy and hard to train per token when the reward arrives only at the end. GRPO removes it, which halves the memory and matches the structure of the reward-model or rule-based rewards that score whole answers. It also reframes RLHF/RL-fine-tuning methods as one family, which clarifies what each one is really doing.

## The corpus (Sections 2, 3)

Seed with OpenWebMath, train a fastText classifier to recognize math pages, score Common Crawl, keep the top pages, discover math-heavy domains among them, annotate their URL patterns by hand, add newly found pages to the seed, retrain, and iterate four times (Figure 2). Result: 35.5M pages, 120B tokens, 7× Minerva's math data. Trained on it, a 1.3B model outperforms the same model trained on MathPile, OpenWebMath or Proof-Pile-2 on every benchmark curve (Figure 3). DeepSeekMath-Base 7B (initialized from DeepSeek-Coder-Base 7B, 500B tokens with a 56% math-corpus mixture) reaches 36.2% on MATH, above Minerva 540B.

## PPO, written for language models

For a question $$q$$ and a sampled output $$o$$ of length $$\lvert o \rvert$$, PPO maximizes

$$
\mathcal{J}_{PPO}(\theta) = \mathbb{E}\Big[\frac{1}{\lvert o \rvert}\sum_{t=1}^{\lvert o \rvert} \min\Big(\frac{\pi_{\theta}(o_t \mid q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t \mid q, o_{<t})} A_t,\ \operatorname{clip}\Big(\frac{\pi_{\theta}(o_t \mid q, o_{<t})}{\pi_{\theta_{\text{old}}}(o_t \mid q, o_{<t})},\ 1 - \varepsilon,\ 1 + \varepsilon\Big) A_t\Big)\Big]
$$

with per-token advantages $$A_t$$ from GAE using a value model $$V$$, and the per-token reward $$r_t = r_{\varphi}(q, o_{\le t}) - \beta \log \frac{\pi_{\theta}(o_t \mid \cdot)}{\pi_{\text{ref}}(o_t \mid \cdot)}$$, i.e. the reward model's score (only at the last token, in outcome supervision) plus a per-token KL penalty toward the reference model. Four models are in play: policy, reference, reward, value (Figure 4, top).

## GRPO

For each question sample a group of $$G$$ outputs $$\{o_1, \dots, o_G\}$$ from $$\pi_{\theta_{\text{old}}}$$ and maximize

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G} \frac{1}{\lvert o_i \rvert}\sum_{t=1}^{\lvert o_i \rvert}\Big\{\min\big(\rho_{i,t} \hat A_{i,t},\ \operatorname{clip}(\rho_{i,t}, 1 - \varepsilon, 1 + \varepsilon)\, \hat A_{i,t}\big) - \beta\, \mathbb{D}_{KL}\big[\pi_{\theta} \,\Vert\, \pi_{\text{ref}}\big]\Big\}\Big]
$$

where $$\rho_{i,t} = \pi_{\theta}(o_{i,t} \mid q, o_{i,<t}) / \pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})$$ is the usual ratio. Three differences from PPO:

**1. Group-relative advantage (no value model).** With *outcome supervision* the reward model scores each whole output, $$r_i$$, and every token of output $$i$$ gets the same advantage,

$$
\hat A_{i,t} = \frac{r_i - \operatorname{mean}(\{r_1, \dots, r_G\})}{\operatorname{std}(\{r_1, \dots, r_G\})}
$$

The group mean is the baseline that the value network used to provide; because outputs in a group answer the *same* question, it is a good estimate of "how hard this question is". Standardizing by the group's standard deviation matches how reward models are typically trained (on comparisons within a question). With *process supervision* a process reward model scores each reasoning step, the step rewards are normalized across the group, and a token's advantage is the sum of normalized rewards of the steps at or after it.

**2. KL in the loss, with an unbiased estimator.** Instead of subtracting $$\beta \log(\pi_{\theta}/\pi_{\text{ref}})$$ from the reward (which entangles it with the advantage), GRPO adds a KL term to the objective and estimates it with

$$
\mathbb{D}_{KL}\big[\pi_{\theta} \,\Vert\, \pi_{\text{ref}}\big] = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})} - 1
$$

which is nonnegative for every sample (Schulman's "k3" estimator) and has lower variance than the plain log-ratio.

**3. Iterative RL.** Because the reward model is trained on samples from an older policy, GRPO periodically retrains the reward model on new samples from the current policy (10% replay of old data) and continues; Figure 6 shows two rounds of iteration improving GSM8K and MATH.

Algorithm 1 in the paper: for each iteration, set $$\pi_{\text{ref}} \leftarrow \pi_{\theta}$$; for each step, sample a batch of questions, $$G$$ outputs each, score with the reward model, compute group-relative advantages, take $$\mu$$ GRPO gradient steps; then update the reward model.

## A unified view (Section 5.2)

Every fine-tuning method in this series can be written as

$$
\nabla_{\theta} \mathcal{J}_{\mathcal{A}}(\theta) = \mathbb{E}_{(q, o) \sim \mathcal{D}}\Big[\frac{1}{\lvert o \rvert}\sum_{t=1}^{\lvert o \rvert} GC_{\mathcal{A}}(q, o, t, \pi_{rf})\ \nabla_{\theta} \log \pi_{\theta}(o_t \mid q, o_{<t})\Big]
$$

differing in the **data source** $$\mathcal{D}$$ (human-written for SFT, sampled once for offline RFT/DPO, sampled continuously for online RFT/PPO/GRPO), the **reward function** $$\pi_{rf}$$ (rules, a reward model, or none), and the **gradient coefficient** $$GC$$ (1 for SFT; an indicator of correctness for RFT; a sigmoid-weighted log-ratio difference for DPO; the advantage from a value model for PPO; the group-normalized reward for GRPO). Section 5.2.3 concludes that online data beats offline, that a reward model that *grades* (assigns graded scores, or penalizes wrong and long answers) beats an indicator, and that RL mainly improves Maj@K, not Pass@K: it makes the right answer more likely rather than teaching new solutions (Figure 7).

## Results

DeepSeekMath-RL 7B (GRPO on GSM8K + MATH chain-of-thought data, ~144k questions, reward model from the SFT model): **51.7% MATH**, 88.2% GSM8K, up from 46.8% / 82.9% for the SFT model, and improvements on out-of-domain benchmarks (CMATH, MGSM) too. Figure 5 compares GRPO (outcome and process supervision) against RFT and online RFT on the 1.3B instruct model; GRPO with process supervision is best.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/deepseekmath/figure1.png" width="802" height="485" alt="Figure 1 from Shao et al. (2024)" loading="lazy"><figcaption>Figure 1 of Shao et al. (2024), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1.** MATH accuracy of open models over 2023–24, with DeepSeekMath-7B above much larger models.

<figure class="paper-fig"><img src="/assets/papers/deepseekmath/figure2.png" width="1142" height="439" alt="Figure 2 from Shao et al. (2024)" loading="lazy"><figcaption>Figure 2 of Shao et al. (2024), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2.** The iterative data pipeline: fastText seed → recall from Common Crawl → discover domains → annotate URL paths → repeat.

<figure class="paper-fig"><img src="/assets/papers/deepseekmath/figure3.jpg" width="1029" height="974" alt="Figure 3 from Shao et al. (2024)" loading="lazy"><figcaption>Figure 3 of Shao et al. (2024), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3.** Benchmark curves (GSM8K, MATH, CMATH, BBH) vs tokens for four corpora; the DeepSeekMath corpus line is on top.

<figure class="paper-fig"><img src="/assets/papers/deepseekmath/figure4.png" width="1087" height="485" alt="Figure 4 from Shao et al. (2024)" loading="lazy"><figcaption>Figure 4 of Shao et al. (2024), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4.** PPO vs GRPO. Top: policy, reference, reward, value models, GAE, per-token advantage. Bottom: the same without the value model; a group of $$G$$ outputs, $$G$$ rewards, "group computation" producing $$A_1 \dots A_G$$.

<figure class="paper-fig"><img src="/assets/papers/deepseekmath/figure5.png" width="992" height="476" alt="Figure 5 from Shao et al. (2024)" loading="lazy"><figcaption>Figure 5 of Shao et al. (2024), reproduced from the paper for commentary.</figcaption></figure>

<figure class="paper-fig"><img src="/assets/papers/deepseekmath/figure6.png" width="992" height="478" alt="Figure 6 from Shao et al. (2024)" loading="lazy"><figcaption>Figure 6 of Shao et al. (2024), reproduced from the paper for commentary.</figcaption></figure>

<figure class="paper-fig"><img src="/assets/papers/deepseekmath/figure7.png" width="1045" height="502" alt="Figure 7 from Shao et al. (2024)" loading="lazy"><figcaption>Figure 7 of Shao et al. (2024), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5–7.** Training curves for RFT / online RFT / GRPO variants; iterative GRPO; Maj@K and Pass@K of SFT vs RL models.

## Thoughts

- GRPO is "REINFORCE with a per-question baseline, PPO clipping, and a KL regularizer". Its appeal is engineering: two models fewer in memory and no value-function tuning.
- The standard-deviation normalization has been criticized (it up-weights questions where all answers are nearly equal); several 2025 variants drop it. The group mean baseline is the essential part.
- Figure 7's claim (RL sharpens the distribution rather than expanding the set of solvable problems) is the first statement of a debate that R1 reopened.

Next: [DeepSeek-R1](/blog/deepseek-r1/), GRPO with rule-based rewards and no SFT at all.
