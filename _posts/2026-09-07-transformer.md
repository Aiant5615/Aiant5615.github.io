---
title: "Attention Is All You Need (Vaswani et al., 2017)"
date: 2026-09-07 09:10:00 +0900
categories: [paper, llm-basics]
tags: [NLP, DL]
math: true
rating: 5
series: transformer-lineage
series_order: 10
description: "The Transformer, every equation explained: scaled dot-product attention and why the scaling, multi-head attention, the position-wise FFN, sinusoidal positions, masking, the learning-rate schedule, and the complexity table. Plus a walk through the architecture figure."
paper:
  title: Attention Is All You Need
  published: 2017-06
  authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google)
  venue: NeurIPS
  year: 2017
  link: https://arxiv.org/abs/1706.03762
---

## One-line summary

Remove the recurrence entirely: an encoder–decoder built only from attention layers (self-attention inside each side, cross-attention from decoder to encoder) and position-wise feed-forward layers, with residual connections and layer normalization, trains in a fraction of the time and sets new translation records.

## Why it matters

Every previous model in this series processed tokens one after another, so the computation at step $$t$$ had to wait for step $$t - 1$$. Self-attention computes all positions at once, which (a) uses GPUs fully and (b) puts every pair of tokens one step apart. This paper is the architecture of BERT, GPT, T5 and everything after.

<svg class="figure" viewBox="0 0 640 300" role="img" aria-label="One Transformer encoder block and one decoder block with residual connections and layer norms">
  <style>.t-b{fill:var(--surface-2);stroke:var(--border)}.t-att{fill:var(--accent-soft);stroke:var(--accent)}.t-n{fill:var(--reading-bg);stroke:var(--reading-fg)}.t-t{font:12px var(--font);fill:var(--text)}.t-m{font:11px var(--font);fill:var(--muted)}.t-a{stroke:var(--muted);fill:none;marker-end:url(#ah3)}.t-r{stroke:var(--accent);fill:none;stroke-dasharray:4 3}</style>
  <defs><marker id="ah3" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/></marker></defs>
  <g transform="translate(40,0)">
    <text class="t-m" x="110" y="292" text-anchor="middle">encoder block ×N</text>
    <text class="t-t" x="110" y="275" text-anchor="middle">x + positional encoding</text>
    <path class="t-a" d="M110,262 L110,232"/>
    <rect class="t-att" x="40" y="200" width="140" height="32" rx="6"/><text class="t-t" x="110" y="221" text-anchor="middle">multi-head self-attn</text>
    <path class="t-r" d="M40,250 L20,250 L20,176 L40,176"/>
    <path class="t-a" d="M110,200 L110,178"/>
    <rect class="t-n" x="40" y="150" width="140" height="28" rx="6"/><text class="t-t" x="110" y="169" text-anchor="middle">Add &amp; LayerNorm</text>
    <path class="t-a" d="M110,150 L110,128"/>
    <rect class="t-b" x="40" y="96" width="140" height="32" rx="6"/><text class="t-t" x="110" y="117" text-anchor="middle">feed-forward (per token)</text>
    <path class="t-r" d="M40,140 L20,140 L20,72 L40,72"/>
    <path class="t-a" d="M110,96 L110,74"/>
    <rect class="t-n" x="40" y="46" width="140" height="28" rx="6"/><text class="t-t" x="110" y="65" text-anchor="middle">Add &amp; LayerNorm</text>
    <path class="t-a" d="M110,46 L110,22"/><text class="t-m" x="110" y="14" text-anchor="middle">encoder output (keys, values)</text>
  </g>
  <g transform="translate(340,0)">
    <text class="t-m" x="110" y="292" text-anchor="middle">decoder block ×N</text>
    <text class="t-t" x="110" y="275" text-anchor="middle">y (shifted right) + positions</text>
    <path class="t-a" d="M110,262 L110,244"/>
    <rect class="t-att" x="40" y="216" width="140" height="28" rx="6"/><text class="t-t" x="110" y="235" text-anchor="middle">masked self-attn + norm</text>
    <path class="t-a" d="M110,216 L110,192"/>
    <rect class="t-att" x="40" y="164" width="140" height="28" rx="6"/><text class="t-t" x="110" y="183" text-anchor="middle">cross-attn (enc K,V) + norm</text>
    <path class="t-a" d="M-120,22 C -60,22 -20,178 40,178" />
    <path class="t-a" d="M110,164 L110,140"/>
    <rect class="t-b" x="40" y="112" width="140" height="28" rx="6"/><text class="t-t" x="110" y="131" text-anchor="middle">feed-forward + norm</text>
    <path class="t-a" d="M110,112 L110,88"/>
    <rect class="t-b" x="40" y="60" width="140" height="28" rx="6"/><text class="t-t" x="110" y="79" text-anchor="middle">linear + softmax</text>
    <path class="t-a" d="M110,60 L110,36"/><text class="t-m" x="110" y="26" text-anchor="middle">next-token probabilities</text>
  </g>
</svg>

## Scaled dot-product attention

Pack queries, keys and values into matrices $$Q \in \mathbb{R}^{n \times d_k}$$, $$K \in \mathbb{R}^{m \times d_k}$$, $$V \in \mathbb{R}^{m \times d_v}$$:

$$
\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\Big(\frac{QK^{\top}}{\sqrt{d_k}}\Big) V
$$

Row $$i$$ of $$QK^{\top}$$ holds the dot products of query $$i$$ with every key; the row-wise softmax turns them into weights; multiplying by $$V$$ takes the weighted average of the values. It is [Luong's dot attention](/blog/luong-attention/) for all queries at once, as one matrix product, which is why it runs fast.

**Why divide by $$\sqrt{d_k}$$.** If the components of $$q$$ and $$k$$ are independent with mean 0 and variance 1, their dot product $$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$ has mean 0 and variance $$d_k$$. With $$d_k = 64$$ the logits would have standard deviation 8, the softmax would saturate on the largest entry, and the gradient through it would be tiny. Dividing by $$\sqrt{d_k}$$ restores unit variance. Footnote 4 of the paper notes that unscaled dot-product attention loses to additive attention at large $$d_k$$ for exactly this reason.

## Multi-head attention

Rather than one attention with $$d_{\text{model}}$$-dimensional queries, project into $$h$$ lower-dimensional subspaces, attend in each, and concatenate:

$$
\operatorname{MultiHead}(Q, K, V) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^{O}, \qquad \text{head}_i = \operatorname{Attention}(Q W_i^{Q},\ K W_i^{K},\ V W_i^{V})
$$

with $$W_i^{Q}, W_i^{K} \in \mathbb{R}^{d_{\text{model}} \times d_k}$$, $$W_i^{V} \in \mathbb{R}^{d_{\text{model}} \times d_v}$$, $$W^{O} \in \mathbb{R}^{h d_v \times d_{\text{model}}}$$. The base model uses $$h = 8$$ heads with $$d_k = d_v = d_{\text{model}}/h = 64$$, so the total cost equals one full-width attention. A single softmax can only express one "mixing pattern" per query; eight heads can simultaneously attend to, say, the previous token, the subject of the sentence, and a coreferent pronoun. Figures 3–5 of the paper visualize exactly such heads.

Attention is used three ways: encoder self-attention ($$Q = K = V$$ = previous layer), decoder self-attention (same, but **masked** so position $$i$$ cannot see $$j > i$$; implemented by setting those logits to $$-\infty$$ before the softmax), and encoder–decoder attention (queries from the decoder, keys and values from the encoder output), which replaces Bahdanau's context vector.

## Position-wise feed-forward network

Each position, independently, passes through the same two-layer network,

$$
\operatorname{FFN}(x) = \max(0,\ x W_1 + b_1)\, W_2 + b_2
$$

with inner size $$d_{ff} = 2048$$ (four times $$d_{\text{model}} = 512$$). This is where most of the parameters are. Attention moves information *between* positions; the FFN transforms it *at* each position. The paper notes it equals two 1×1 convolutions.

## Embeddings and positional encoding

Input and output token embeddings and the pre-softmax linear layer share one weight matrix, and embeddings are multiplied by $$\sqrt{d_{\text{model}}}$$. Because nothing in attention or the FFN knows the order of positions, a position signal is *added* to the embeddings:

$$
PE_{(pos,\,2i)} = \sin\!\Big(\frac{pos}{10000^{2i/d_{\text{model}}}}\Big), \qquad PE_{(pos,\,2i+1)} = \cos\!\Big(\frac{pos}{10000^{2i/d_{\text{model}}}}\Big)
$$

Each dimension pair is a sinusoid with a different wavelength, from $$2\pi$$ up to $$10000 \cdot 2\pi$$; low dimensions oscillate fast, high dimensions slowly, like the bits of a binary counter. The chosen form has a useful property: for any fixed offset $$k$$, $$PE_{pos+k}$$ is a *linear* function of $$PE_{pos}$$ (a rotation in each sine/cosine pair), so the model can learn to attend to "$$k$$ positions back" with a linear map. Learned positional embeddings performed the same on the benchmarks; sinusoids were kept for possible extrapolation to longer inputs.

## Residuals, normalization, regularization

Each sublayer is wrapped as $$\operatorname{LayerNorm}(x + \operatorname{Sublayer}(x))$$ ([ResNet](/blog/resnet/) shortcut, [layer norm](/blog/layer-normalization/)), with dropout (0.1) on the sublayer output and on the summed embeddings. Label smoothing $$\epsilon_{ls} = 0.1$$ hurts perplexity but improves BLEU. $$N = 6$$ blocks per side.

## Training schedule

Adam with $$\beta_1 = 0.9$$, $$\beta_2 = 0.98$$, $$\epsilon = 10^{-9}$$ and the now-famous warmup schedule

$$
\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min\big(\text{step}^{-0.5},\ \text{step} \cdot \text{warmup}^{-1.5}\big)
$$

which increases linearly for the first 4000 steps and then decays as the inverse square root of the step. The base model trained for 100k steps (12 hours on 8 P100s); the big model ($$d_{\text{model}} = 1024$$, $$d_{ff} = 4096$$, 16 heads) for 300k steps (3.5 days).

## Complexity (Table 1)

Per layer, with sequence length $$n$$, dimension $$d$$ and convolution kernel $$k$$:

| layer type | complexity per layer | sequential ops | max path length |
|---|---|---|---|
| self-attention | $$O(n^2 \cdot d)$$ | $$O(1)$$ | $$O(1)$$ |
| recurrent | $$O(n \cdot d^2)$$ | $$O(n)$$ | $$O(n)$$ |
| convolutional | $$O(k \cdot n \cdot d^2)$$ | $$O(1)$$ | $$O(\log_k n)$$ |

Self-attention is cheaper than recurrence whenever $$n < d$$ (true for sentences with $$d = 512$$), fully parallel, and connects any two positions in one step. The quadratic term in $$n$$ is the cost that later "efficient attention" work targets.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/transformer/figure1.png" alt="Figure 1 from Vaswani et al. (2017)" loading="lazy"><figcaption>Figure 1 of Vaswani et al. (2017), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (architecture).** Left tower: embedding + positions → [self-attention → add&norm → FFN → add&norm] × N. Right tower: output embedding (shifted right) → [masked self-attention → add&norm → cross-attention over the encoder output → add&norm → FFN → add&norm] × N → linear → softmax. My drawing above shows one block of each with the residual arcs dashed.

<figure class="paper-fig"><img src="/assets/papers/transformer/figure2.png" alt="Figure 2 from Vaswani et al. (2017)" loading="lazy"><figcaption>Figure 2 of Vaswani et al. (2017), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (left: scaled dot-product; right: multi-head).** MatMul → scale → optional mask → softmax → MatMul, then the multi-head wrapper with $$h$$ parallel linears, concat, and a final linear.

<figure class="paper-fig"><img src="/assets/papers/transformer/figure3.png" alt="Figure 3 from Vaswani et al. (2017)" loading="lazy"><figcaption>Figure 3 of Vaswani et al. (2017), reproduced from the paper for commentary.</figcaption></figure>

<figure class="paper-fig"><img src="/assets/papers/transformer/figure4.png" alt="Figure 4 from Vaswani et al. (2017)" loading="lazy"><figcaption>Figure 4 of Vaswani et al. (2017), reproduced from the paper for commentary.</figcaption></figure>

<figure class="paper-fig"><img src="/assets/papers/transformer/figure5.png" alt="Figure 5 from Vaswani et al. (2017)" loading="lazy"><figcaption>Figure 5 of Vaswani et al. (2017), reproduced from the paper for commentary.</figcaption></figure>

- **Figures 3–5 (attention heads).** In the "making … more difficult" example, several heads at layer 5 connect "making" to "more difficult" across a long distance. In the anaphora example, heads resolve "its" to "the Law". Other heads have visibly syntactic structure. These pictures started the "what do heads do" literature.

## Results

WMT'14 English–German **28.4 BLEU** (big model), more than 2 BLEU above the best prior ensemble; English–French 41.8, a new single-model record at a fraction of the training cost. Table 3 ablates: fewer heads or too many heads hurt; reducing $$d_k$$ hurts; bigger models help; dropout is necessary; learned positions equal sinusoids. Section 6.3 shows the same architecture does English constituency parsing near the state of the art with little tuning.

## Thoughts

- Everything is a matrix product plus a softmax, so the model scales with hardware. That, more than accuracy, is why it won.
- The decoder alone (masked self-attention + FFN) is [GPT](/blog/gpt1/); the encoder alone is [BERT](/blog/bert/); the full pair is [T5](/blog/t5/). The three papers after this one are the three ways of cutting this figure in half.
- The $$O(n^2)$$ attention and the fixed sinusoidal positions are the two design points most revised since: sparse/linear attention and rotary/ALiBi positions.

Next: [GPT](/blog/gpt1/), the decoder stack as a pretrained language model.
