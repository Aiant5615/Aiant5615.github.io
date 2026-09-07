---
title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Fedus, Zoph & Shazeer, 2021)"
date: 2026-09-07 12:02:00 +0900
categories: [paper, llm-advanced]
tags: [NLP, DL]
math: true
rating: 4
series: llm-advanced
series_order: 2
description: "Mixture-of-experts made simple: route each token to one expert. The router softmax, the top-1 switch layer, expert capacity, the load-balancing auxiliary loss, the training fixes (selective precision, small init, expert dropout), and what the scaling and speed figures show. Background for Mixtral and DeepSeek-V3."
paper:
  title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
  published: 2021-01
  authors: William Fedus, Barret Zoph, Noam Shazeer (Google)
  venue: Journal of Machine Learning Research
  year: 2022
  link: https://arxiv.org/abs/2101.03961
---

## One-line summary

Replace the feed-forward block of a Transformer with $$N$$ copies ("experts") and a tiny router that sends each token to exactly one of them. Parameters grow $$N$$-fold while the compute per token stays constant. With the right load-balancing loss and a few numerical fixes, a Switch model reaches the same quality as T5-Base 7× faster, and the recipe scales to 1.6 trillion parameters.

## Why it matters

Dense scaling ties parameters to FLOPs. Sparse mixture-of-experts breaks the tie, which is how GPT-4-class and DeepSeek-V3-class models get their parameter counts. Switch simplified the earlier MoE recipe (top-2 routing, noisy gating) to the form most later systems use.

## Routing, equation by equation

For a token representation $$x$$, the router is a linear layer $$W_r$$ followed by a softmax over the $$N$$ experts:

$$
h(x) = W_r x, \qquad p_i(x) = \frac{e^{h(x)_i}}{\sum_{j=1}^{N} e^{h(x)_j}}
$$

A classic mixture-of-experts layer sends the token to the top-$$k$$ experts $$\mathcal{T}$$ and combines their outputs weighted by the gate values,

$$
y = \sum_{i \in \mathcal{T}} p_i(x)\, E_i(x)
$$

Shazeer et al. (2017) argued $$k \ge 2$$ was needed so the router receives a gradient that compares experts. The **Switch layer** sets $$k = 1$$: the token goes to the single expert with the largest $$p_i$$, and its output is multiplied by that $$p_i$$ (which is what makes the router differentiable). Benefits: half the routing computation, smaller expert capacity, simpler all-to-all communication.

**Expert capacity.** Experts run as fixed-size batches on separate devices, so each expert accepts at most

$$
\text{expert capacity} = \frac{\text{tokens per batch}}{\text{number of experts}} \times \text{capacity factor}
$$

tokens. If more tokens are routed to an expert than its capacity, the overflow is **dropped**: those tokens skip the expert and pass through the residual connection unchanged (Figure 3). A capacity factor of 1.0–1.25 works; larger factors waste compute on padding.

**Load-balancing loss.** Without pressure, the router collapses onto a few experts. For a batch of $$T$$ tokens, let $$f_i$$ be the fraction of tokens dispatched to expert $$i$$ and $$P_i$$ the fraction of router probability assigned to expert $$i$$,

$$
f_i = \frac{1}{T}\sum_{x \in \mathcal{B}} \mathbb{1}\{\operatorname{argmax} p(x) = i\}, \qquad P_i = \frac{1}{T}\sum_{x \in \mathcal{B}} p_i(x), \qquad \text{loss} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
$$

Both vectors sum to one, so the product is minimized (at $$1/N$$) when routing is uniform; $$f_i$$ is not differentiable but $$P_i$$ is, so the gradient pushes the *probabilities* toward balance. The factor $$N$$ keeps the loss scale constant as experts are added; $$\alpha = 10^{-2}$$ is small enough not to hurt the main objective.

## Training a sparse model stably (Section 2.4)

- **Selective precision.** bfloat16 everywhere except the router, whose softmax is computed in float32 locally (the cast happens *inside* the device, so no extra communication). Fixes divergence at no speed cost.
- **Smaller initialization.** Truncated-normal init scaled down by 10× (scale 0.1 instead of 1.0) reduces early instability.
- **Expert dropout.** Fine-tuning the huge parameter count on small datasets overfits; dropout of 0.4 inside experts and 0.1 elsewhere works best.

## Results

- **Sample efficiency (Figure 1, Table 1).** At equal FLOPs per token, Switch-Base with 128 experts reaches T5-Base's pretraining quality in 1/7 of the steps, and beats T5-Large (3.5× the FLOPs) on speed-to-quality. On C4 with a 2.5k-step budget, negative log perplexity improves monotonically with the number of experts (2 → 256).
- **Scaling (Figure 4).** Quality vs step for 1 to 256 experts: more experts, faster learning, at constant compute. Switch-Base reaches T5-Base's quality with a 7.5× wall-clock speedup on the same hardware (Figure 5).
- **Downstream (Table 5).** Switch-Base and Switch-Large beat T5-Base/Large on GLUE, SuperGLUE, SQuAD, XSum, closed-book QA and reasoning tasks after fine-tuning.
- **Distillation (Table 6).** A Switch-Base can be distilled into a dense T5-Base keeping about 30% of the quality gain with 99% fewer parameters.
- **Multilingual (Figure 7).** On mC4 across 101 languages, speedups over mT5-Base in every language, 4× on average.
- **Trillion scale.** Switch-XXL (395B) and Switch-C (1.571T parameters, 2048 experts) train; Switch-C reaches T5-XXL's quality 4× faster in steps at the same per-token FLOPs, though training instabilities are reported for the FLOP-heavy XXL.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/switch/figure1.png" alt="Figure 1 from Fedus et al. (2021)" loading="lazy"><figcaption>Figure 1 of Fedus et al. (2021), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (scaling and sample efficiency).** Left: perplexity vs parameters at constant FLOPs, improving with sparsity. Right: quality vs training step for T5-Base and Switch-Base with increasing experts.

<figure class="paper-fig"><img src="/assets/papers/switch/figure2.png" alt="Figure 2 from Fedus et al. (2021)" loading="lazy"><figcaption>Figure 2 of Fedus et al. (2021), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (Switch encoder block).** The standard block with the FFN replaced by a router that sends each token ("More", "Parameters") to one of four FFN experts, drawn as parallel boxes.

<figure class="paper-fig"><img src="/assets/papers/switch/figure3.png" alt="Figure 3 from Fedus et al. (2021)" loading="lazy"><figcaption>Figure 3 of Fedus et al. (2021), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (token routing dynamics).** Tokens flowing to experts with capacity factor 1.0 vs 1.5: at 1.0, a red overflow token is dropped; at 1.5, slots are padded (white). The trade-off between dropped tokens and wasted compute.

<figure class="paper-fig"><img src="/assets/papers/switch/figure4.png" alt="Figure 4 from Fedus et al. (2021)" loading="lazy"><figcaption>Figure 4 of Fedus et al. (2021), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (scaling properties).** Left: quality vs experts at fixed steps. Right: quality vs step for 1–256 experts.

<figure class="paper-fig"><img src="/assets/papers/switch/figure5.png" alt="Figure 5 from Fedus et al. (2021)" loading="lazy"><figcaption>Figure 5 of Fedus et al. (2021), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5 (speed advantage).** Quality vs wall-clock time on 32 TPU v3 cores: Switch-Base reaches the T5-Base line in a fraction of the time; T5-Large is slower per step.

## Thoughts

- Top-1 routing with a balancing loss, capacity factors and dropped tokens is the same design in Mixtral (top-2) and DeepSeek-V3 (shared expert + auxiliary-loss-free balancing). The load-balancing loss is the part every successor tweaks.
- MoE buys quality per FLOP at the cost of memory and communication; the paper's own distillation section admits that for deployment the dense model is often what you want.
- Combined with Chinchilla's finding (Part 4) that tokens matter as much as parameters, MoE became the way to add parameters *cheaply* while spending the compute on tokens.

Next: [LoRA](/blog/lora/), adapting a huge model by training two small matrices.
