---
title: "Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)"
date: 2026-09-07 10:05:00 +0900
categories: [paper, llm-rl]
tags: [RL, NLP]
math: true
rating: 4
series: llm-rl
series_order: 5
description: "Replace human harmlessness labels with a model that critiques and revises its own outputs against written principles, then RL from AI feedback (RLAIF). The two-stage pipeline, how the preference model is built from principles, the chain-of-thought variant, and what the Elo and revision figures show."
paper:
  title: "Constitutional AI: Harmlessness from AI Feedback"
  published: 2022-12
  authors: Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. (Anthropic)
  venue: arXiv
  year: 2022
  link: https://arxiv.org/abs/2212.08073
---

## One-line summary

Train a harmless assistant without any human labels for harmlessness: (1) have a helpful model critique and revise its own responses to harmful prompts according to a short list of principles (the "constitution"), fine-tune on the revisions; (2) have a model choose the less harmful of two responses according to a principle, train a preference model on those AI labels, and run RL against it. The result is both more harmless and less evasive than RLHF trained on human harmlessness labels.

## Why it matters

[InstructGPT](/blog/instructgpt/)-style RLHF needs tens of thousands of human comparisons per behaviour you want. This paper shows that for harmlessness the comparisons can come from the model itself, guided by natural-language principles, which makes the objective *inspectable* (you can read the constitution) and scalable. It introduced the term RLAIF.

<figure class="paper-fig"><img src="/assets/papers/cai/figure1.png" alt="Figure 1 from Bai et al. (2022)" loading="lazy"><figcaption>Figure 1 of Bai et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

## Stage 1: supervised learning from critiques and revisions (SL-CAI)

Start from a *helpful-only* RLHF model (trained on human helpfulness comparisons, deliberately without harmlessness training so it will comply with harmful requests). For each red-teaming prompt:

1. Sample a response from the helpful model.
2. Ask it, with a randomly chosen principle, to **critique** the response (e.g. "Identify specific ways in which the assistant's last response is harmful, unethical, racist…").
3. Ask it to **revise** the response in light of the critique.
4. Repeat critique–revision several times with new principles.

Fine-tune the pretrained model on the final revisions (plus helpfulness data to keep it helpful). Figure 5 shows harmlessness preference-model scores rising with each revision and Figure 6 that most of the gain comes in the first revision. An ablation finds that revisions *without* the critique step are almost as good; the critique mostly helps smaller models.

## Stage 2: RL from AI feedback (RL-CAI)

Now build a harmlessness preference model from AI labels:

1. Sample **pairs** of responses to harmful prompts from the SL-CAI model.
2. Present the pair to a *feedback model* (a pretrained LM, not fine-tuned) as a multiple-choice question with a principle: "Which of these responses is less harmful? (A) … (B) …", and take the normalized log-probabilities of A and B as a **soft label**.
3. Mix these AI harmlessness comparisons with *human* helpfulness comparisons and train a preference model (PM) on the union, using the usual comparison loss.
4. Run RL (PPO) on the SL-CAI policy against the PM.

The feedback model's judgment can be improved with **chain-of-thought**: a helpful RLHF model is asked to "think step by step" before answering, which makes its labels better calibrated on the HHH evaluations (Figure 4). Because the CoT labels are near 0/1, they are clamped to the 40–60% range to avoid overconfidence.

There is no new loss here; the contribution is *where the labels come from*. The preference model is the same Bradley–Terry-style comparison model as in the RLHF papers, trained on soft targets $$p$$ instead of hard choices.

## Evaluation

Helpfulness and harmlessness are measured with **Elo scores** from crowdworker comparisons between models (Figure 2, Figure 3), and with **HHH multiple-choice evaluations** (Figure 4). Harmlessness is judged with an instruction that penalizes *evasive* answers ("I can't help with that") as well as harmful ones, so the score rewards engaging thoughtfully with a harmful request.

## Results

<figure class="paper-fig"><img src="/assets/papers/cai/figure2.png" alt="Figure 2 from Bai et al. (2022)" loading="lazy"><figcaption>Figure 2 of Bai et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (the Pareto plot).** Harmlessness Elo vs helpfulness Elo for the base, helpful-only RLHF, standard HH RLHF, constitutional SL and constitutional RL (with and without CoT). RL-CAI models lie to the upper right of the standard RLHF line: more harmless *at the same helpfulness*. The paper calls this a Pareto improvement.

<figure class="paper-fig"><img src="/assets/papers/cai/figure3.png" alt="Figure 3 from Bai et al. (2022)" loading="lazy"><figcaption>Figure 3 of Bai et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (scaling).** Helpfulness and harmlessness Elo vs parameters, for helpful RLHF, HH RLHF and RL-CAI; harmlessness of RL-CAI grows fastest with scale, helpfulness tracks the helpful-only model.

<figure class="paper-fig"><img src="/assets/papers/cai/figure4.png" alt="Figure 4 from Bai et al. (2022)" loading="lazy"><figcaption>Figure 4 of Bai et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (HHH evaluations).** Accuracy of preference models on the HHH multiple-choice set vs parameters: a PM trained on human HH feedback, a pretrained LM, and chain-of-thought labels with and without ensembling. CoT labels overtake the human-feedback PM at the largest size.

<figure class="paper-fig"><img src="/assets/papers/cai/figure5.png" alt="Figure 5 from Bai et al. (2022)" loading="lazy"><figcaption>Figure 5 of Bai et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

<figure class="paper-fig"><img src="/assets/papers/cai/figure6.png" alt="Figure 6 from Bai et al. (2022)" loading="lazy"><figcaption>Figure 6 of Bai et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5 / 6 (revisions).** PM scores for harmlessness, helpfulness and combined HH as a function of the number of critique–revision rounds, for several model sizes; harmlessness rises monotonically, helpfulness dips slightly, and one revision captures most of the gain. Figure 6 shows the effect of the number of principles sampled.
- Qualitatively (Section 4.3), RL-CAI answers harmful prompts by explaining *why* it will not help rather than refusing flatly, and is less evasive than the RLHF baseline.

## Thoughts

- The key move is replacing the *harmlessness* labels only; helpfulness still comes from humans. Later RLAIF work replaced those too, with mixed results.
- The constitution is a prompt library, not a training objective; the "objective" that the policy sees is still a learned PM. What is inspectable is the *source* of the labels.
- Compare with DPO (next): both remove a human-in-the-loop component (CAI removes human harmlessness labels; DPO removes the RL loop). They are orthogonal and are often used together.

Next: [Direct Preference Optimization](/blog/dpo/), which shows the RL step has a closed form.
