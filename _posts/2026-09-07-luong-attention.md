---
title: "Effective Approaches to Attention-based Neural Machine Translation (Luong, Pham & Manning, 2015)"
date: 2026-09-07 09:06:00 +0900
categories: [paper, llm-basics]
image: /assets/og-llm-basics.png
tags: [NLP]
math: true
rating: 4
series: transformer-lineage
series_order: 6
description: "Global vs local attention, the three score functions (dot, general, concat), the Gaussian-windowed local model, and input feeding. Each formula explained, plus what the alignment and length figures show."
paper:
  title: Effective Approaches to Attention-based Neural Machine Translation
  published: 2015-08
  authors: Minh-Thang Luong, Hieu Pham, Christopher D. Manning (Stanford)
  venue: EMNLP
  year: 2015
  link: https://arxiv.org/abs/1508.04025
---

## One-line summary

A cleaner and cheaper attention layer on top of a stacked LSTM: compare several ways of scoring source states against the current decoder state, add a *local* variant that attends only to a window, feed the attended vector into the next step, and reach a new state of the art on English–German.

## Why it matters

[Bahdanau's](/blog/bahdanau-attention/) attention used a small MLP as the score and a bidirectional GRU encoder. Luong et al. show that a plain **dot product** between decoder and encoder states works about as well, which is the score function the [Transformer](/blog/transformer/) adopts (with scaling). They also fix the "which decoder state" question by scoring with the *current* top-layer state, and introduce input feeding.

## Global attention, equation by equation

The setup is a multi-layer LSTM encoder and decoder. At target step $$t$$ the decoder's top-layer state is $$h_t$$; the encoder states are $$\bar h_s$$ for source positions $$s$$. **Global** attention considers all of them:

$$
a_t(s) = \operatorname{align}(h_t, \bar h_s) = \frac{\exp\big(\operatorname{score}(h_t, \bar h_s)\big)}{\sum_{s'} \exp\big(\operatorname{score}(h_t, \bar h_{s'})\big)}
$$

Three score functions are compared:

$$
\operatorname{score}(h_t, \bar h_s) =
\begin{cases}
h_t^{\top} \bar h_s & \text{dot} \\
h_t^{\top} W_a \bar h_s & \text{general} \\
v_a^{\top} \tanh\big(W_a [h_t; \bar h_s]\big) & \text{concat}
\end{cases}
$$

*Dot* has no parameters and is a similarity in the shared state space. *General* inserts a learned bilinear map, useful when the two spaces should not be compared directly. *Concat* is Bahdanau's additive form written with a single matrix over the concatenation. The paper also tries a **location-based** score, $$a_t = \operatorname{softmax}(W_a h_t)$$, that ignores the source content and predicts weights from the decoder state alone.

The context vector is the usual weighted average, $$c_t = \sum_s a_t(s)\, \bar h_s$$, and it is combined with the decoder state into an **attentional hidden state**

$$
\tilde h_t = \tanh\big(W_c [c_t; h_t]\big)
$$

which produces the output distribution

$$
p(y_t \mid y_{<t}, x) = \operatorname{softmax}(W_s \tilde h_t)
$$

Compared with Bahdanau's path ($$h_{t-1} \to a_t \to c_t \to h_t$$), this is simpler ($$h_t \to a_t \to c_t \to \tilde h_t$$): the attention is computed *after* the LSTM step from the current state, so the recurrent part and the attention part are cleanly separated.

## Local attention

Global attention costs $$O(T_x)$$ per output step and, for long inputs or paragraphs, wastes effort on irrelevant positions. **Local** attention picks an aligned position $$p_t$$ and attends only inside the window $$[p_t - D, p_t + D]$$ (the paper uses $$D = 10$$). Two ways to choose $$p_t$$:

- **local-m (monotonic):** $$p_t = t$$, assuming source and target advance together.
- **local-p (predictive):** the model predicts the position from the decoder state,

$$
p_t = S \cdot \operatorname{sigmoid}\big(v_p^{\top} \tanh(W_p h_t)\big)
$$

where $$S$$ is the source length, so $$p_t \in [0, S]$$. To make the choice of $$p_t$$ differentiable, the weights inside the window are multiplied by a Gaussian centred on $$p_t$$:

$$
a_t(s) = \operatorname{align}(h_t, \bar h_s)\ \exp\!\Big(-\frac{(s - p_t)^2}{2\sigma^2}\Big), \qquad \sigma = \frac{D}{2}
$$

Positions near $$p_t$$ keep their content-based weight; positions near the edge of the window are damped. Because $$p_t$$ is a real number produced by the network, gradients flow back into $$v_p$$ and $$W_p$$ through the Gaussian term.

## Input feeding

In the global model above, the attention decision at step $$t$$ does not know what was attended at step $$t-1$$, so nothing stops the model from translating the same source word twice. The fix is to concatenate $$\tilde h_t$$ with the next input embedding:

$$
\text{input}_{t+1} = [\,E y_t;\ \tilde h_t\,]
$$

Now the decoder carries a memory of past alignment choices, and the network is effectively deeper (the attentional state is recomputed through the LSTM). Table 1 attributes about +1 BLEU to this alone.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/luong/figure1.png" width="516" height="275" alt="Figure 1 from Luong et al. (2015)" loading="lazy"><figcaption>Figure 1 of Luong et al. (2015), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (stacked LSTM NMT).** The encoder and decoder as one deep LSTM chain reading "A B C D ⟨eos⟩" then emitting "X Y Z ⟨eos⟩"; attention sits on top of the decoder side. It is Sutskever's picture with an attention layer added.

<figure class="paper-fig"><img src="/assets/papers/luong/figure2.png" width="536" height="451" alt="Figure 2 from Luong et al. (2015)" loading="lazy"><figcaption>Figure 2 of Luong et al. (2015), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (global attention).** For one target step: all encoder states, an alignment-weight vector $$a_t$$ over them, the context $$c_t$$, and the combination with $$h_t$$ into $$\tilde h_t$$.

<figure class="paper-fig"><img src="/assets/papers/luong/figure3.png" width="536" height="450" alt="Figure 3 from Luong et al. (2015)" loading="lazy"><figcaption>Figure 3 of Luong et al. (2015), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (local attention).** The same, but the weight vector is drawn only over a window around $$p_t$$, with the Gaussian bump superimposed.

<figure class="paper-fig"><img src="/assets/papers/luong/figure4.png" width="461" height="461" alt="Figure 4 from Luong et al. (2015)" loading="lazy"><figcaption>Figure 4 of Luong et al. (2015), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (input feeding).** The dashed arrow from $$\tilde h_t$$ back down into the next LSTM input.
- **Figure 5 (learning curves).** Test cost vs epochs for the models in Table 1; the attention models separate from the non-attention baseline early and stay below it.

<figure class="paper-fig"><img src="/assets/papers/luong/figure6.png" width="599" height="331" alt="Figure 6 from Luong et al. (2015)" loading="lazy"><figcaption>Figure 6 of Luong et al. (2015), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 6 (BLEU vs sentence length).** Attention models remain strong on 40+ word sentences while the plain model falls, the same shape as Bahdanau's Figure 2.

<figure class="paper-fig"><img src="/assets/papers/luong/figure7.png" width="962" height="726" alt="Figure 7 from Luong et al. (2015)" loading="lazy"><figcaption>Figure 7 of Luong et al. (2015), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 7 (alignment visualizations).** Heatmaps for global, local-m, local-p and the *gold* alignment; local-p looks the sharpest. The paper also scores alignments with **AER** (alignment error rate) against human alignments, where local-p wins.

## Results

WMT'14 English–German: a single attention model reaches **20.9 BLEU** (dot, local-p, input feeding, unknown-word replacement), and an ensemble of 8 reaches **23.0**, +1.0 over the previous best; on WMT'15 the ensemble gets 25.9 and a new state of the art. German–English gains a similar margin. Among score functions, *dot* works best for global attention and *general* for local.

## Thoughts

- "Dot product between states" plus "combine context and state with one matrix" is the shape modern attention layers take; Bahdanau's additive score survives mostly in older tutorials.
- Local attention is an early answer to the quadratic cost of attending everywhere; sparse and windowed Transformers rediscover it.
- Input feeding is the paper's least-cited but most reused trick: any decoder that conditions on its own previous attention output (including a Transformer decoder through its residual stream) is doing a version of it.

Next: [Byte-pair encoding](/blog/bpe-subword/), which makes the vocabulary problem in these models go away.
