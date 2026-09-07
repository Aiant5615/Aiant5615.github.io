---
title: "Training Compute-Optimal Large Language Models (Chinchilla, Hoffmann et al., 2022)"
date: 2026-09-07 12:04:00 +0900
categories: [paper, llm-advanced]
tags: [NLP, DL]
math: true
rating: 5
series: llm-advanced
series_order: 4
description: "How to split a compute budget between model size and training tokens: three estimation approaches (envelope, IsoFLOP, parametric loss fit L(N,D)=E+A/N^α+B/D^β), the answer that both should scale equally, and the 70B/1.4T-token Chinchilla that beats 280B Gopher. Every figure explained."
paper:
  title: Training Compute-Optimal Large Language Models
  published: 2022-03
  authors: Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. (DeepMind)
  venue: NeurIPS
  year: 2022
  link: https://arxiv.org/abs/2203.15556
---

## One-line summary

Given a fixed training budget of $$C$$ FLOPs, the loss-minimizing model size $$N$$ and token count $$D$$ grow at the *same* rate, roughly $$N_{\text{opt}} \propto C^{0.5}$$ and $$D_{\text{opt}} \propto C^{0.5}$$; the large models of 2020–21 (GPT-3, Gopher, MT-NLG) were far too big for their token budgets. Chinchilla, 70B parameters on 1.4T tokens, uses the same compute as the 280B Gopher and beats it everywhere.

## Why it matters

Kaplan et al. (2020) had concluded that with 10× more compute one should grow the model 5.5× and the data only 1.8×, which drove the race to ever-larger models. This paper reverses that with more careful experiments and is the reason later models (LLaMA, most 2023+ releases) train small models on many tokens.

## The question

Minimize the final pretraining loss subject to a compute constraint,

$$
N_{\text{opt}}(C),\ D_{\text{opt}}(C) = \operatorname{argmin}_{N, D\ \text{s.t.}\ \text{FLOPs}(N, D) = C} L(N, D), \qquad \text{FLOPs}(N, D) \approx 6\, N D
$$

(each parameter costs about 6 FLOPs per token: 2 in the forward pass, 4 in the backward). Write the optimum as power laws $$N_{\text{opt}} \propto C^{a}$$, $$D_{\text{opt}} \propto C^{b}$$, and estimate $$a, b$$ three ways from over 400 training runs of 70M–16B-parameter models on 5B–500B tokens.

## Three approaches

**1. Fix model sizes, vary tokens (Section 3.1, Figure 2).** For each of many model sizes train with several token budgets and *cosine schedules matched to the run length* (a key correction to Kaplan, whose fixed schedules underestimated the value of more data). For each FLOP count, take the lowest loss across runs (the envelope), and fit power laws through the resulting $$(C, N_{\text{opt}})$$ and $$(C, D_{\text{opt}})$$ points. Result: $$a \approx 0.50$$, $$b \approx 0.50$$.

**2. IsoFLOP profiles (Section 3.2, Figure 3).** Fix nine compute budgets ($$6 \times 10^{18}$$ to $$3 \times 10^{21}$$ FLOPs), vary the model size at each budget (so tokens vary inversely), and fit a parabola to loss vs $$\log N$$; its minimum is the optimal size for that budget. Fitting power laws through the minima gives $$a \approx 0.49$$, $$b \approx 0.51$$.

**3. Parametric loss (Section 3.3, Figure 4).** Model the loss directly as

$$
\hat L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

where $$E$$ is the irreducible loss of natural text, the second term is the penalty for a finite model, the third the penalty for finite data. Fit by minimizing a Huber loss on log-predictions over all runs: $$E = 1.69$$, $$A = 406.4$$, $$B = 410.7$$, $$\alpha = 0.34$$, $$\beta = 0.28$$. Minimizing $$\hat L$$ under $$6ND = C$$ gives closed-form $$N_{\text{opt}}$$ and $$D_{\text{opt}}$$ with exponents $$a = \beta/(\alpha + \beta) \approx 0.46$$, $$b = \alpha/(\alpha + \beta) \approx 0.54$$.

All three agree (Table 2, Figure 1): parameters and tokens should scale in proportion, about **20 tokens per parameter** near current budgets. Applied to Gopher's budget ($$5.76 \times 10^{23}$$ FLOPs), the optimal model is 40–70B parameters trained on 1.3–1.7T tokens, not 280B on 300B.

## Chinchilla

Train a 70B model (same architecture family as Gopher, 80 layers, $$d = 8192$$, 64 heads, RoPE not yet: relative positions as in Gopher) on 1.4T tokens of MassiveText with AdamW, a slightly modified SentencePiece tokenizer, and bfloat16 with float32 master weights, using the same compute as Gopher. Table 4 details the schedule.

## Results (Section 4)

- **Language modeling.** Lower bits-per-byte than Gopher on every subset of The Pile (Figure 5) and on Wikitext103 (perplexity 7.16 vs 7.75).
- **MMLU.** 67.5% average 5-shot vs Gopher's 60.0% (Figure 6), above the 63.4% forecast by experts for June 2023 and the best model at the time; better on 51 of 57 subjects.
- **BIG-bench.** 65.1% vs 54.4% on the 62 tasks reported for Gopher.
- **Reading comprehension and QA.** LAMBADA 77.4%, RACE-h 82.3%, closed-book NaturalQuestions 31.5% (5-shot) vs Gopher 24.5%, TriviaQA 64.6%.
- **Fairness and toxicity** are roughly unchanged; the paper notes that a smaller model is 4× cheaper to fine-tune and serve.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/chinchilla/figure1.png" alt="Figure 1 from Hoffmann et al. (2022)" loading="lazy"><figcaption>Figure 1 of Hoffmann et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (overlaid predictions).** Optimal parameters (left) and tokens (right) vs FLOPs from the three approaches, drawn over the compute budgets of GPT-3, Gopher, MT-NLG and Chinchilla; the existing models sit far above the parameter lines, i.e. they are oversized.

<figure class="paper-fig"><img src="/assets/papers/chinchilla/figure2.png" alt="Figure 2 from Hoffmann et al. (2022)" loading="lazy"><figcaption>Figure 2 of Hoffmann et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (training-curve envelope).** Left: hundreds of loss-vs-FLOPs curves and their lower envelope. Middle and right: the envelope's optimal $$N$$ and $$D$$ vs FLOPs, straight lines on log axes.

<figure class="paper-fig"><img src="/assets/papers/chinchilla/figure3.png" alt="Figure 3 from Hoffmann et al. (2022)" loading="lazy"><figcaption>Figure 3 of Hoffmann et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (IsoFLOP curves).** Left: loss vs parameters at each fixed budget, each a parabola with a clear minimum. Middle/right: the minima's $$N$$ and $$D$$ vs FLOPs.

<figure class="paper-fig"><img src="/assets/papers/chinchilla/figure4.png" alt="Figure 4 from Hoffmann et al. (2022)" loading="lazy"><figcaption>Figure 4 of Hoffmann et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (parametric fit).** Left: contours of the fitted $$\hat L(N, D)$$ with the efficient frontier drawn through them. Right: predicted loss along IsoFLOP slices vs observations.

<figure class="paper-fig"><img src="/assets/papers/chinchilla/figure5.png" alt="Figure 5 from Hoffmann et al. (2022)" loading="lazy"><figcaption>Figure 5 of Hoffmann et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5 (Pile evaluation).** Bits-per-byte improvement of Chinchilla over Gopher per subset; every bar is positive.

## Thoughts

- The practical rule "tokens ≈ 20 × parameters" is *compute*-optimal for training only; when inference cost matters, training a smaller model on even more tokens is better, which is exactly what LLaMA (Part 7) does.
- The parametric form $$E + A/N^{\alpha} + B/D^{\beta}$$ became the template for scaling-law fitting; later papers refine it (data repetition, MoE, distillation) but keep the shape.
- The correction to Kaplan came mostly from matching the learning-rate schedule to the run length, a reminder that scaling laws are only as good as the training recipe underneath them.

Next: [Chain-of-Thought Prompting](/blog/chain-of-thought/), an ability that appears only at scale.
