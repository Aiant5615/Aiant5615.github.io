---
title: "Learning to Summarize from Human Feedback (Stiennon et al., 2020)"
date: 2026-09-07 10:04:00 +0900
categories: [paper, llm-rl]
image: /assets/og-llm-rl.png
tags: [RL, NLP]
math: true
rating: 5
series: llm-rl
series_order: 4
description: "The clean RLHF result: a 1.3B model trained on human comparisons writes summaries preferred to the human references. The pairwise reward-model loss, the KL-penalized PPO reward, reward-model overoptimization measured directly, and RM scaling laws. Every figure explained."
paper:
  title: Learning to Summarize from Human Feedback
  published: 2020-09
  authors: Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, Paul Christiano (OpenAI)
  venue: NeurIPS
  year: 2020
  link: https://arxiv.org/abs/2009.01325
---

## One-line summary

Collect 65k pairwise comparisons of Reddit post summaries from carefully managed labelers, train a reward model on them, optimize a 1.3B–6.7B GPT-3-style policy with PPO against the reward minus a KL penalty, and get summaries that humans prefer to the human-written reference summaries and to much larger supervised models. Also measure, for the first time, how far you can optimize a reward model before it stops tracking real preferences.

## Why it matters

[Ziegler et al.](/blog/ziegler-lm-preferences/) showed RLHF on text works but produced copying models. This paper fixes the data pipeline (labelers who agree with the researchers 77% of the time, detailed instructions, hands-on onboarding), scales the models, and produces the graph everyone shows for RLHF: preference over references rising with model size. It is [InstructGPT](/blog/instructgpt/) minus the instruction distribution.

## The three steps (Figure 2)

1. **Collect comparisons.** Sample summaries of a Reddit TL;DR post from several policies (the supervised baseline, the current RL policy, human references), show a labeler two, record which is better. 64,832 comparisons in total, collected in batches over the course of training so the reward model sees samples from policies near the current one.
2. **Train the reward model.** Starting from the supervised model, add a scalar head and minimize the pairwise loss below.
3. **Train the policy with PPO** against the reward model, with a KL penalty to the supervised policy.

## Equations

**Reward model.** For a post $$x$$ and summaries $$y_0, y_1$$ where the labeler chose index $$i$$,

$$
\operatorname{loss}(r_{\theta}) = -\mathbb{E}_{(x, y_0, y_1, i) \sim D}\Big[\log \sigma\big(r_{\theta}(x, y_i) - r_{\theta}(x, y_{1-i})\big)\Big]
$$

(the Bradley–Terry model with two candidates). After training, the RM is shifted so that reference summaries score 0 on average. The RM is trained for one epoch to avoid overfitting; its validation accuracy on held-out comparisons is about 65–70%, comparable to human–human agreement (73%).

**Policy.** The per-episode reward is

$$
R(x, y) = r_{\theta}(x, y) - \beta\, \log \frac{\pi_{\phi}^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)}
$$

with a fixed $$\beta$$ (0.05 for the 6.7B model). The KL term is again both an entropy bonus and a guard against reward-model exploitation. The value function is a separate network initialized from the RM. PPO with the usual hyperparameters; the policy starts from the supervised model.

**Supervised baseline.** Fine-tune the pretrained model on the 123k filtered TL;DR posts with their human summaries (the "SFT" model), which also initializes the RM and the RL policy.

## Results

- **TL;DR (Figure 1).** Fraction of times labelers prefer the model's summary to the reference: the 1.3B human-feedback model is preferred about 61% of the time, the 6.7B model about 70%, while the supervised 6.7B model sits at 43% and pretrain-only at 30%. A 1.3B RLHF model beats a 12.9B supervised one.
- **Quality axes (Figure 3).** Labelers rated coverage, accuracy, coherence and overall quality on a 7-point scale; human-feedback summaries beat the references on coverage and overall, and are on par on accuracy and coherence.
- **Transfer (Figure 4).** With no CNN/DailyMail training, the TL;DR-trained RLHF model nearly matches human reference summaries on CNN/DM and beats a supervised T5 fine-tuned on CNN/DM, at short lengths; controlling for length (right panel), the advantage holds.
- **Reward-model overoptimization (Figure 5).** Optimize the policy against a fixed RM to increasing KL from the SFT model (with best-of-N or PPO) and measure *both* the RM's predicted preference and actual human preference. Predicted preference keeps rising; actual preference peaks around KL 10 and then falls, and at KL 250 the summaries are gibberish that the RM loves. This is the empirical definition of "the RM gets exploited" and the reason for the KL penalty.
- **RM scaling (Figure 6).** RM validation accuracy vs model size at four data sizes: doubling the data adds about 1.1 points, doubling the model about 1.8 points.
- **Figure 7.** Optimizing ROUGE with best-of-N *does not* improve human preference beyond a point, while optimizing the RM does; ROUGE is a bad proxy.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/stiennon/figure1.png" width="724" height="526" alt="Figure 1 from Stiennon et al. (2020)" loading="lazy"><figcaption>Figure 1 of Stiennon et al. (2020), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1.** The headline plot: preference over references vs model size for three training methods; only the human-feedback line crosses 0.5.

<figure class="paper-fig"><img src="/assets/papers/stiennon/figure2.png" width="1035" height="591" alt="Figure 2 from Stiennon et al. (2020)" loading="lazy"><figcaption>Figure 2 of Stiennon et al. (2020), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2.** The three-panel procedure (collect, train RM, train policy) with the loss written out on the reward-model panel.

<figure class="paper-fig"><img src="/assets/papers/stiennon/figure3.png" width="427" height="372" alt="Figure 3 from Stiennon et al. (2020)" loading="lazy"><figcaption>Figure 3 of Stiennon et al. (2020), reproduced from the paper for commentary.</figcaption></figure>

<figure class="paper-fig"><img src="/assets/papers/stiennon/figure4.png" width="1008" height="416" alt="Figure 4 from Stiennon et al. (2020)" loading="lazy"><figcaption>Figure 4 of Stiennon et al. (2020), reproduced from the paper for commentary.</figcaption></figure>

- **Figures 3–4.** Bar charts and line plots of the quality axes and the CNN/DM transfer.

<figure class="paper-fig"><img src="/assets/papers/stiennon/figure5.png" width="588" height="371" alt="Figure 5 from Stiennon et al. (2020)" loading="lazy"><figcaption>Figure 5 of Stiennon et al. (2020), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5.** Two curves against KL: dashed (RM prediction) up and up, solid (actual preference) up then down.

<figure class="paper-fig"><img src="/assets/papers/stiennon/figure6.png" width="497" height="363" alt="Figure 6 from Stiennon et al. (2020)" loading="lazy"><figcaption>Figure 6 of Stiennon et al. (2020), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 6.** Four RM accuracy curves, one per data size, each rising log-linearly with parameters; the human-baseline line sits above all of them.

## Thoughts

- Figure 5 is the most important plot in this series. It shows that the reward model is a *proxy* whose validity is local, quantifies the safe region in KL, and explains why every later method has a KL term (or, for DPO, a $$\beta$$ that plays the same role).
- The paper argues that what RLHF really buys is *optimizing what people actually care about* rather than a metric; the ROUGE experiment is the evidence.
- Labeler management gets a whole section. The quality of the preference data, not the algorithm, is what changed between 2019 and 2020.

Next: [Constitutional AI](/blog/constitutional-ai/), which replaces human harmlessness labels with a model reading a list of principles.
