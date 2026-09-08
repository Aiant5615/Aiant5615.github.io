---
title: "Fine-Tuning Language Models from Human Preferences (Ziegler et al., 2019)"
date: 2026-09-07 10:03:00 +0900
categories: [paper, llm-rl]
image: /assets/og-llm-rl.png
tags: [RL, NLP]
math: true
rating: 4
series: llm-rl
series_order: 3
description: "RLHF applied to GPT-2 for the first time: 4-way comparison reward model, PPO with a KL penalty to the pretrained policy, online vs offline label collection, and what the mock-reward, Pareto-frontier and human-evaluation figures show. Also the paper that discovered summarizers learn to copy."
paper:
  title: Fine-Tuning Language Models from Human Preferences
  published: 2019-09
  authors: Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, Geoffrey Irving (OpenAI)
  venue: arXiv
  year: 2019
  link: https://arxiv.org/abs/1909.08593
---

## One-line summary

Take a pretrained 774M-parameter GPT-2, learn a reward model from human choices among four candidate continuations, and fine-tune the language model with PPO against that reward minus a KL penalty that keeps it near the original. Five thousand labels are enough to steer style; sixty thousand produce summaries humans prefer, largely because the model learns to copy.

## Why it matters

This is [Christiano et al.](/blog/christiano-preferences/) transplanted to text and the direct ancestor of [Stiennon et al.](/blog/stiennon-summarize/) and [InstructGPT](/blog/instructgpt/). Most of the machinery those papers use (KL-regularized reward, online data collection, RM initialized from the LM) appears here first, together with the first clear observation of reward-model exploitation.

## The setup

A vocabulary $$\Sigma$$, a language model $$\rho(x_{n+1} \mid x_1 \dots x_n)$$, and a task: given a context $$x \in \Sigma^{\le m}$$ (e.g. an article), produce a continuation $$y \in \Sigma^n$$ (a few sentences, or a summary). The initial policy is $$\pi = \rho$$, and fine-tuning should maximize a reward $$r(x, y)$$ known only through human judgments. The human is shown a context and four samples $$y_0, \dots, y_3$$ and picks the best one, $$b$$.

## Reward model

The reward model $$r$$ is a copy of the language model with a linear head on the final embedding, trained to make the chosen sample most likely under a softmax over the four:

$$
\operatorname{loss}(r) = \mathbb{E}_{(x, \{y_i\}_i, b) \sim S}\Big[\log \frac{e^{r(x, y_b)}}{\sum_i e^{r(x, y_i)}}\Big]
$$

(a 4-way generalization of the pairwise Bradley–Terry model). After training, $$r$$ is normalized so that its mean over samples from $$\rho$$ is zero. The head is trained first, then the whole model, and only one epoch is used because the RM overfits quickly.

## Policy objective

Optimize the policy with PPO against a *penalized* reward,

$$
R(x, y) = r(x, y) - \beta \log \frac{\pi(y \mid x)}{\rho(y \mid x)}
$$

The KL term serves two purposes the paper states explicitly: it acts as an entropy bonus that prevents collapse to a single mode, and it keeps the policy in the region where the reward model, trained on samples near $$\rho$$, is meaningful. $$\beta$$ is set *adaptively* to hit a target KL between $$\pi$$ and $$\rho$$ (the PPO paper's KL-penalty rule, applied to a different KL), because the right $$\beta$$ differs across tasks. For summarization the policy also gets the language-modeling loss on the pretraining data mixed in? No, that is InstructGPT; here the only anchor is the KL.

Two data-collection regimes: **offline** (collect all labels from samples of $$\rho$$, train $$r$$ once, run PPO) and **online** (alternate: collect labels from the *current* policy, retrain $$r$$ on all data, continue PPO). Online matters for summarization, where the policy moves far from $$\rho$$ and an offline RM is exploited.

## Tasks and results

- **Stylistic continuation (Section 4.1):** continue a BookCorpus excerpt with 32 tokens of positive sentiment, or with vivid descriptive language. 5,000 human labels; evaluated by humans against the zero-shot model. The sentiment policy is preferred 88% of the time, descriptiveness 86%. A *mock* sentiment reward (a classifier) is used to study data efficiency (Figure 2) and the reward/KL trade-off (Figure 3).
- **Summarization (Section 4.2):** TL;DR (Reddit) and CNN/Daily Mail, 60,000 human labels each, online collection. The RL-fine-tuned model is preferred over the supervised-fine-tuned model and the zero-shot model, and gets ROUGE comparable to supervised models. But the models "are mostly smart copiers": Figure 5 and Table 7 show they copy whole sentences from the article and often the first three words; humans nonetheless prefer them because copying is *accurate*.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/ziegler/figure1.png" width="616" height="350" alt="Figure 1 from Ziegler et al. (2019)" loading="lazy"><figcaption>Figure 1 of Ziegler et al. (2019), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (training processes).** Two loops. Top: reward model training — a context goes through the policy, four continuations are scored by the RM and labelled by a human, the RM loss is applied. Bottom: policy training — the policy's continuation is scored by the (now fixed) RM and PPO is applied. Online mode interleaves the two.

<figure class="paper-fig"><img src="/assets/papers/ziegler/figure2.png" width="616" height="453" alt="Figure 2 from Ziegler et al. (2019)" loading="lazy"><figcaption>Figure 2 of Ziegler et al. (2019), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (mock-reward learning curves).** Mean mock reward vs episodes for direct optimization of the classifier ("direct") and for reward models trained from 5k, 20k, 60k labels at target KL 8. 60k labels gets within a hair of direct optimization; 5k lags.

<figure class="paper-fig"><img src="/assets/papers/ziegler/figure3.png" width="1186" height="746" alt="Figure 3 from Ziegler et al. (2019)" loading="lazy"><figcaption>Figure 3 of Ziegler et al. (2019), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (KL vs reward).** Reward achieved as a function of the KL between policy and $$\rho$$, for models trained with different label budgets, against the "optimal" curve from direct optimization. With more labels the frontier moves up; the gap widens at high KL because the RM is being exploited there.

<figure class="paper-fig"><img src="/assets/papers/ziegler/figure4.png" width="1246" height="450" alt="Figure 4 from Ziegler et al. (2019)" loading="lazy"><figcaption>Figure 4 of Ziegler et al. (2019), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (human evaluation).** Win rate of offline fine-tuned models against zero-shot for sentiment and descriptiveness, as a function of number of labels; both saturate around 80–90% by a few thousand labels.

<figure class="paper-fig"><img src="/assets/papers/ziegler/figure5.png" width="1221" height="300" alt="Figure 5 from Ziegler et al. (2019)" loading="lazy"><figcaption>Figure 5 of Ziegler et al. (2019), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5 (n-gram novelty).** Percent of 1-, 2-, 3-, 4-grams and sentences in summaries that do not appear in the source, for lead-3, supervised, zero-shot and RL models. The RL models are far less novel than supervised ones: they copy.

## Thoughts

- The paper is candid about failure modes: the summarizers copy, an offline RM can be gamed, and the KL coefficient must be tuned per task. Later work presents cleaner results; this one shows the seams.
- Two ideas that became standard originate here: initializing the RM from the LM, and using online (iterated) label collection to keep the RM valid as the policy moves. Two that did not: the 4-way softmax (InstructGPT uses all pairs) and the adaptive KL (most later work fixes $$\beta$$).
- "Humans prefer copies because they are accurate" is an early warning that preference data rewards *safe* outputs, a theme that returns in the alignment-tax discussion.

Next: [Learning to Summarize from Human Feedback](/blog/stiennon-summarize/), the polished version of this experiment.
