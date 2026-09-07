---
title: "Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau, Cho & Bengio, 2015)"
date: 2026-09-07 09:05:00 +0900
categories: [paper]
tags: [NLP]
math: true
rating: 5
series: transformer-lineage
series_order: 5
description: "Additive attention, derived line by line. Why the fixed-length vector was the bottleneck, how the alignment weights are computed, how to read the alignment heatmaps, and what the BLEU-vs-length plot proves."
paper:
  title: Neural Machine Translation by Jointly Learning to Align and Translate
  authors: Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
  venue: ICLR
  year: 2015
  link: https://arxiv.org/abs/1409.0473
---

## One-line summary

Instead of squeezing the source sentence into one vector, keep every encoder state and let the decoder compute, at each output step, a weighted average of them where the weights are produced by a small network that learns to *align* target words with source words.

## Why it matters

This is where attention enters NLP. Every later model in this series is built on it: Luong's variants, the Transformer's scaled dot-product, and the cross-attention inside T5 all instantiate the three lines below (score, normalize, average).

<svg class="figure" viewBox="0 0 640 210" role="img" aria-label="Attention diagram: encoder states h1..h4 are weighted by alpha and summed into a context c_i used by decoder state s_i">
  <style>.g-box{fill:var(--accent-soft);stroke:var(--accent)}.g-dec{fill:var(--surface-2);stroke:var(--border)}.g-t{font:12px var(--font);fill:var(--text)}.g-m{font:11px var(--font);fill:var(--muted)}.g-a{stroke:var(--muted);fill:none;marker-end:url(#ah2)}.g-w{stroke:var(--accent);fill:none;marker-end:url(#ah2)}</style>
  <defs><marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--muted)"/></marker></defs>
  <g>
    <rect class="g-box" x="30" y="140" width="70" height="36" rx="6"/><text class="g-t" x="65" y="163" text-anchor="middle">h₁</text>
    <rect class="g-box" x="130" y="140" width="70" height="36" rx="6"/><text class="g-t" x="165" y="163" text-anchor="middle">h₂</text>
    <rect class="g-box" x="230" y="140" width="70" height="36" rx="6"/><text class="g-t" x="265" y="163" text-anchor="middle">h₃</text>
    <rect class="g-box" x="330" y="140" width="70" height="36" rx="6"/><text class="g-t" x="365" y="163" text-anchor="middle">h₄</text>
    <text class="g-m" x="215" y="200" text-anchor="middle">bidirectional encoder states (one per source word)</text>
    <path class="g-w" d="M65,140 L245,80" stroke-width="1"/><path class="g-w" d="M165,140 L250,80" stroke-width="3"/><path class="g-w" d="M265,140 L258,80" stroke-width="5"/><path class="g-w" d="M365,140 L268,80" stroke-width="1.5"/>
    <text class="g-m" x="110" y="115" text-anchor="middle">α₁</text><text class="g-m" x="205" y="112" text-anchor="middle">α₂</text><text class="g-m" x="285" y="112" text-anchor="middle">α₃</text><text class="g-m" x="330" y="115" text-anchor="middle">α₄</text>
    <circle cx="258" cy="70" r="14" class="g-dec"/><text class="g-t" x="258" y="74" text-anchor="middle">Σ</text>
    <text class="g-m" x="300" y="66" text-anchor="start">cᵢ = Σⱼ αᵢⱼ hⱼ</text>
    <path class="g-a" d="M258,56 L258,42"/>
    <rect class="g-dec" x="208" y="6" width="100" height="36" rx="6"/><text class="g-t" x="258" y="29" text-anchor="middle">decoder sᵢ</text>
    <path class="g-a" d="M420,24 L308,24"/><text class="g-m" x="430" y="28" text-anchor="start">sᵢ₋₁, yᵢ₋₁</text>
    <text class="g-m" x="440" y="115" text-anchor="start">line width = weight αᵢⱼ,</text><text class="g-m" x="440" y="130" text-anchor="start">recomputed for every output i</text>
  </g>
</svg>

## The problem with a fixed vector

In the [RNN encoder–decoder](/blog/2026/09/07/rnn-encoder-decoder-gru/), the decoder conditions on one vector $$c$$ for the whole output. The authors' Figure 2 shows what that costs: an encoder–decoder trained on sentences up to 30 words (RNNenc-30) degrades steeply on longer test sentences, and even the 50-word version drops after 40 words. A vector of a few thousand numbers cannot hold a 60-word sentence at the fidelity translation needs.

## The model, equation by equation

**Encoder.** A bidirectional RNN (GRUs) reads the source forwards and backwards; the annotation for word $$j$$ concatenates both directions,

$$
h_j = \big[\overrightarrow{h}_j^{\top};\ \overleftarrow{h}_j^{\top}\big]^{\top}
$$

so $$h_j$$ summarizes the whole sentence with a focus on the words around $$x_j$$. Nothing is thrown away: all $$T_x$$ annotations are kept.

**Decoder with a per-step context.** The decoder state and output now use a context vector $$c_i$$ that is *different for every target position* $$i$$:

$$
s_i = f(s_{i-1}, y_{i-1}, c_i), \qquad p(y_i \mid y_1, \dots, y_{i-1}, x) = g(y_{i-1}, s_i, c_i)
$$

**Context as an expected annotation.** The context is a convex combination of the annotations,

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$

with weights that sum to one over the source positions:

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

**The alignment model.** The unnormalized score $$e_{ij}$$ says how well the input around position $$j$$ matches the output at position $$i$$. It is a one-hidden-layer network, evaluated with the *previous* decoder state (because $$s_i$$ is not known yet):

$$
e_{ij} = a(s_{i-1}, h_j) = v_a^{\top} \tanh\big(W_a s_{i-1} + U_a h_j\big)
$$

This is called **additive attention** because the query and key contributions are added before the nonlinearity. $$U_a h_j$$ does not depend on $$i$$ and is precomputed once per sentence, so the extra cost per output step is $$T_x$$ small matrix products.

**Why "soft" alignment.** Traditional alignment is a hard choice of one source word per target word. Here $$\alpha_{ij}$$ is a probability, and the whole thing is differentiable, so the alignment model is trained by the same backpropagation as the rest, with no alignment supervision. The context $$c_i$$ is literally the expected annotation under the distribution $$\alpha_{i\cdot}$$.

**Decoder details (Appendix A).** The decoder GRU takes $$c_i$$ as an extra input in each gate, for example $$z_i = \sigma(W_z E y_{i-1} + U_z s_{i-1} + C_z c_i)$$, and the output is a maxout layer followed by a softmax. Model size: 1000 hidden units per direction, 620-dimensional embeddings, 30k vocabularies on each side, Adadelta, minibatch SGD for about 5 days.

## Figures, explained

- **Figure 1 (the model).** A bidirectional encoder at the bottom with arrows from *every* $$h_j$$ into a summation node labelled with $$\alpha_{t,j}$$, feeding the decoder state $$s_t$$ at the top. The redrawing above follows it.
- **Figure 2 (BLEU vs sentence length).** Four curves. RNNsearch-50 (the attention model trained on up to 50 words) is essentially flat out to 60+ words; the encoder–decoder baselines fall off a cliff. This single plot is the empirical case for attention.
- **Figure 3 (alignments).** Four grey-scale matrices, source words on the x-axis and generated French words on the y-axis, each pixel $$\alpha_{ij}$$. Mostly diagonal (English and French share word order), with the famous off-diagonal block where "European Economic Area" becomes "zone économique européenne", the adjectives reversed, and the model's weights reverse with them. Soft alignment also lets "the man" map onto "l'homme" by looking at both "the" and "man" to choose "l'" over "le".

## Results

On WMT'14 English–French (Table 1): RNNsearch-50 reaches **28.45 BLEU** on all sentences and 36.15 on sentences without unknown words, versus 26.71 / 34.16 for the fixed-vector RNNencdec-50 and 33.30 / 35.63 for Moses (which used far more monolingual data). With the vocabulary limitation removed, the attention model matches the phrase-based system.

## Thoughts

- The three-step recipe (score every key against the query, softmax, weighted sum of values) is the Transformer's attention with a different score function. Here queries are decoder states, keys and values are both $$h_j$$.
- Using $$s_{i-1}$$ rather than $$s_i$$ in the score is a design detail [Luong et al.](/blog/2026/09/07/luong-attention/) change, and it matters for how the computation can be parallelized.
- Attention was sold as a fix for long sentences, but the alignment heatmaps hint at the bigger idea: the model produces an interpretable map of *which input mattered for which output*, which becomes the main tool for analysing Transformers.

Next: [Luong et al.](/blog/2026/09/07/luong-attention/) simplify the scoring, add local windows, and feed attention back into the decoder.
