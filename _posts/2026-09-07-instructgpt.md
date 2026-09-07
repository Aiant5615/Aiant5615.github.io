---
title: "Training Language Models to Follow Instructions with Human Feedback (InstructGPT, Ouyang et al., 2022)"
date: 2026-09-07 09:16:00 +0900
categories: [paper, llm-basic]
tags: [NLP, RL]
math: true
rating: 5
series: transformer-lineage
series_order: 16
description: "RLHF in three steps: supervised fine-tuning, a reward model trained on pairwise comparisons, and PPO against that reward with a KL penalty and a pretraining-mix term. Every loss explained, the labeler setup, the win-rate and truthfulness figures, and the alignment tax."
paper:
  title: Training Language Models to Follow Instructions with Human Feedback
  published: 2022-03
  authors: Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. (OpenAI)
  venue: NeurIPS
  year: 2022
  link: https://arxiv.org/abs/2203.02155
---

## One-line summary

Turn a raw [GPT-3](/blog/gpt3/) into a model that does what the user asks by (1) fine-tuning on human-written demonstrations, (2) training a reward model on human rankings of model outputs, and (3) optimizing the language model with reinforcement learning against that reward. Labelers prefer the 1.3B-parameter InstructGPT over the 175B GPT-3, and the model is more truthful and slightly less toxic, with little loss on standard benchmarks.

## Why it matters

Next-token prediction on the web is *misaligned* with "be helpful, honest and harmless to this user": the objective rewards imitating the internet, not following instructions. This paper is the recipe (RLHF) behind ChatGPT and most assistant models, and the source of the terms SFT, reward model, PPO-ptx and alignment tax.

## Data and labelers

Prompts come from two sources: text written by ~40 contracted labelers (plain instructions, few-shot prompts, and prompts for use cases collected from a waitlist), and, after an initial InstructGPT was deployed, real prompts submitted to the OpenAI API (deduplicated, limited to 200 per user, with PII filtered; the paper's Table 1 shows the use-case mix, dominated by generation, open QA and brainstorming). Three datasets: about 13k prompts with demonstrations (SFT), 33k prompts with comparisons (RM), and 31k prompts for RL. Labelers were screened for agreement with researchers and for sensitivity to harmful content; inter-labeler agreement is 72.6 ± 1.5%, and held-out labelers (who wrote no training data) show the same preferences, so the model is not merely fitting its own annotators.

## Step 1: supervised fine-tuning (SFT)

Fine-tune GPT-3 (1.3B, 6B, 175B) on the demonstration data for 16 epochs with cosine decay and residual dropout 0.2. It overfits on validation loss after one epoch, but the paper finds that further epochs still improve *reward-model score and human preference*, so the checkpoint is chosen by RM score, not loss.

## Step 2: reward model (RM)

Start from the 6B SFT model, remove the unembedding layer, and output a scalar $$r_{\theta}(x, y)$$ for a prompt $$x$$ and completion $$y$$. For each prompt, labelers rank $$K$$ completions ($$K$$ from 4 to 9), which yields $$\binom{K}{2}$$ pairwise comparisons. The loss is a pairwise logistic (Bradley–Terry) objective,

$$
\operatorname{loss}(\theta) = -\frac{1}{\binom{K}{2}}\ \mathbb{E}_{(x, y_w, y_l) \sim D}\Big[\log \sigma\big(r_{\theta}(x, y_w) - r_{\theta}(x, y_l)\big)\Big]
$$

where $$y_w$$ is the preferred completion and $$y_l$$ the other. The reward only matters up to a constant, so after training it is shifted so that demonstrations score 0 on average. Two details that matter: all $$\binom{K}{2}$$ pairs from one prompt are put in the **same batch** (one forward pass per completion, and it prevents overfitting to correlated comparisons; training on shuffled pairs overfits within an epoch), and the model is trained for a single epoch. A 6B RM was used for all policy sizes because a 175B RM was unstable.

## Step 3: reinforcement learning (PPO and PPO-ptx)

Treat the language model as a policy $$\pi_{\phi}^{\text{RL}}$$: the prompt is the state, the completion is the action, the episode ends after one response, and the reward is $$r_{\theta}(x, y)$$ from the RM. The optimization is PPO (Schulman et al., 2017) starting from the SFT model, with a value function initialized from the RM, maximizing

$$
\operatorname{objective}(\phi) = \mathbb{E}_{(x, y) \sim D_{\pi_{\phi}^{\text{RL}}}}\Big[r_{\theta}(x, y) - \beta\, \log \frac{\pi_{\phi}^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)}\Big] + \gamma\ \mathbb{E}_{x \sim D_{\text{pretrain}}}\Big[\log \pi_{\phi}^{\text{RL}}(x)\Big]
$$

Read the three parts:

1. $$r_{\theta}(x, y)$$: make the reward model happy.
2. $$-\beta \log \frac{\pi^{\text{RL}}}{\pi^{\text{SFT}}}$$: a per-token KL penalty toward the SFT model. Without it the policy drifts to outputs that exploit the RM (high reward, bad text) and loses diversity; the penalty keeps the policy near a region where the RM's judgments are still valid.
3. $$\gamma\, \mathbb{E}[\log \pi(x)]$$ on pretraining data: the **pretraining-mix** term. Pure PPO improved preference but hurt public NLP benchmarks (SQuAD, DROP, translation), which the paper calls the *alignment tax*. Mixing the pretraining likelihood back in ("PPO-ptx", $$\gamma = 27.8$$ in their units) recovers most of that loss at almost no cost in preference.

## Evaluation

Held-out API prompts are scored by labelers on a 1–7 Likert scale and, mainly, by *win rate against the 175B SFT model*. The paper also tests on public datasets for truthfulness (TruthfulQA), toxicity (RealToxicityPrompts, with human ratings) and bias (Winogender, CrowS-Pairs), and on the standard few-shot NLP suite to measure regressions.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/instructgpt/figure1.png" alt="Figure 1 from Ouyang et al. (2022)" loading="lazy"><figcaption>Figure 1 of Ouyang et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (human evaluation vs model size).** Win rate against SFT-175B for GPT-3, GPT-3 with a carefully constructed prompt, SFT, PPO and PPO-ptx across 1.3B/6B/175B. The 1.3B PPO models beat 175B GPT-3; PPO-ptx and PPO are close; SFT alone is a large step above prompted GPT-3.

<figure class="paper-fig"><img src="/assets/papers/instructgpt/figure2.png" alt="Figure 2 from Ouyang et al. (2022)" loading="lazy"><figcaption>Figure 2 of Ouyang et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (the three steps).** The diagram everyone reproduces: a prompt sampled → labeler writes an answer → SFT; several outputs sampled → labeler ranks → RM; a new prompt → policy answers → RM scores → PPO update.
- **Figure 3 (Likert scores)** and **Figure 4 (metadata).** InstructGPT outputs are rated as following explicit constraints more often, attempting the right instruction more often, and hallucinating less often (21% vs 41% on closed-domain tasks).
- **Figure 5 (held-out labelers, and prompt distributions).** The preference holds for labelers who did not produce training data and for prompts written for GPT-3 rather than for InstructGPT, evidence that the effect generalizes.

<figure class="paper-fig"><img src="/assets/papers/instructgpt/figure6.png" alt="Figure 6 from Ouyang et al. (2022)" loading="lazy"><figcaption>Figure 6 of Ouyang et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 6 (TruthfulQA).** Fraction of truthful and truthful-and-informative answers; PPO models roughly double GPT-3's truthfulness, more so with an "instruction+QA" prompt that permits "I have no comment".
- **Figure 7 (toxicity).** Human-rated toxicity with and without a "respectful" instruction; InstructGPT is less toxic when told to be respectful and about the same when not, and *more* toxic when instructed to be.
- **Appendix figures (alignment tax).** Few-shot scores on public benchmarks for PPO vs PPO-ptx across sizes; the pretraining-mix term closes most of the gap.

## Results

Labelers prefer InstructGPT-175B outputs to GPT-3-175B outputs 85 ± 3% of the time, and to few-shot-prompted GPT-3 71 ± 4%. Truthfulness roughly doubles on TruthfulQA; toxicity drops about 25% when prompted to be respectful; bias is unchanged. Public benchmark regressions are small with PPO-ptx. The model also generalizes beyond the training distribution: it follows instructions in other languages and answers code questions, despite few such examples, though it still makes mistakes (Section 4.3 lists false-premise compliance and over-hedging).

## Limitations the authors emphasize

The model is aligned to ~40 labelers' judgments under instructions written by OpenAI researchers, not to "humans" in general; it is not fully aligned or safe (it can still produce harmful content when asked, and will follow harmful instructions); and the cost of RLHF is a tiny fraction of pretraining (the 175B run used 60 petaflop/s-days vs 3,640 for GPT-3), which they argue makes alignment work cheap relative to scaling.

## Thoughts

- The reward-model loss is Bradley–Terry; the PPO objective is "reward minus KL plus pretraining likelihood". Later methods (DPO) show the KL-regularized objective has a closed form that avoids RL entirely, but the three-step decomposition remains the mental model.
- The paper's own framing, "the objective was wrong, not the model", closes the loop with [Bengio 2003](/blog/nplm-bengio/): twenty years of making $$p(\text{next word})$$ better, and then one paper about the fact that $$p(\text{next word})$$ was never the goal.
- What to read next from here: Constitutional AI (AI feedback instead of human labels), DPO (the closed form), and the scaling-law papers referenced by GPT-3.

This is the last paper in the series. The [index post](/blog/transformer-lineage/) lists all sixteen in order.
