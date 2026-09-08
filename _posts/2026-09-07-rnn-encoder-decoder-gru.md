---
title: "Learning Phrase Representations using RNN Encoder–Decoder (Cho et al., 2014)"
date: 2026-09-07 09:03:00 +0900
categories: [paper, llm-basics]
tags: [NLP]
math: true
rating: 4
series: transformer-lineage
series_order: 3
description: "The encoder–decoder recipe and the GRU, equation by equation. Why the gates exist, how the model was used inside a phrase-based SMT system, and what the phrase-embedding plots show."
paper:
  title: Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
  published: 2014-06
  authors: Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio
  venue: EMNLP
  year: 2014
  link: https://arxiv.org/abs/1406.1078
---

## One-line summary

Encode a variable-length source phrase into a fixed vector with one RNN, decode a target phrase from that vector with another RNN, train the pair to maximize conditional log-likelihood, and use a new gated hidden unit (the GRU) so the RNNs can remember across long spans.

## Why it matters

This is the paper that names the **RNN Encoder–Decoder**. [Sutskever et al.](/blog/seq2seq/) scaled the same idea into a full translation system a few months later, and [Bahdanau et al.](/blog/bahdanau-attention/) fixed its bottleneck with attention. The GRU introduced here is still one of the two standard recurrent cells.

## The encoder–decoder, equation by equation

An RNN reads a sequence $$x = (x_1, \dots, x_T)$$ by updating a hidden state,

$$
h_{\langle t \rangle} = f\big(h_{\langle t-1 \rangle}, x_t\big)
$$

and can define a distribution over the next symbol, $$p(x_t \mid x_{t-1}, \dots, x_1) = g(h_{\langle t \rangle})$$. The **encoder** runs this over the source and keeps the final state as a summary $$c = h_{\langle T \rangle}$$. The **decoder** is another RNN whose state is conditioned on the summary and on the previous output:

$$
h_{\langle t \rangle} = f\big(h_{\langle t-1 \rangle}, y_{t-1}, c\big), \qquad P(y_t \mid y_{t-1}, \dots, y_1, c) = g\big(h_{\langle t \rangle}, y_{t-1}, c\big)
$$

Note that $$c$$ enters *every* decoder step, not only the first. Both networks are trained jointly to maximize

$$
\max_{\theta} \frac{1}{N} \sum_{n=1}^{N} \log p_{\theta}\big(y_n \mid x_n\big)
$$

over pairs $$(x_n, y_n)$$. Because the decoder is a conditional language model, the same trained network can *score* a given pair (the use in this paper) or *generate* a target by sampling or beam search.

## The gated recurrent unit

A plain $$\tanh$$ RNN forgets quickly and its gradients vanish. The proposed unit adds two gates per hidden dimension $$j$$. A **reset gate**

$$
r_j = \sigma\big([W_r x]_j + [U_r h_{\langle t-1 \rangle}]_j\big)
$$

decides how much of the previous state to use when proposing new content, and an **update gate**

$$
z_j = \sigma\big([W_z x]_j + [U_z h_{\langle t-1 \rangle}]_j\big)
$$

decides how much of the state to overwrite. The candidate state uses the reset-scaled previous state,

$$
\tilde h_j^{\langle t \rangle} = \tanh\big([Wx]_j + [U(r \odot h_{\langle t-1 \rangle})]_j\big)
$$

and the new state is a per-dimension interpolation:

$$
h_j^{\langle t \rangle} = z_j\, h_j^{\langle t-1 \rangle} + (1 - z_j)\, \tilde h_j^{\langle t \rangle}
$$

Read the last line as a *learned skip connection*: when $$z_j \approx 1$$ the dimension is copied unchanged, so information (and gradient) passes through many steps without being squashed by a $$\tanh$$. When $$r_j \approx 0$$ the unit ignores its past and behaves like the first step of a fresh sequence, which is useful at phrase boundaries. Compared with an LSTM the GRU has no separate memory cell and no output gate, so it has fewer parameters and the same qualitative behaviour. (Later papers swap the roles of $$z$$ and $$1 - z$$; the meaning is unchanged.)

## How it was used

Not as a standalone translator. The authors trained the encoder–decoder on phrase pairs from the WMT'14 English–French phrase table (1000 hidden units, 100-dimensional embeddings, 500-unit maxout output layer) and added its score $$\log p(f \mid e)$$ as an extra feature to a phrase-based SMT system (Moses). The RNN sees each phrase pair once regardless of its corpus frequency, so it learns *linguistic regularity* rather than repeating the translation table's counts.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/cho/figure1.png" alt="Figure 1 from Cho et al. (2014)" loading="lazy"><figcaption>Figure 1 of Cho et al. (2014), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (encoder–decoder).** Two chains of boxes. The left chain consumes $$x_1 \dots x_T$$ and emits $$c$$; the right chain consumes $$c$$ plus its own previous output and emits $$y_1 \dots y_{T'}$$. The arrow from $$c$$ fans out to every decoder step.

<figure class="paper-fig"><img src="/assets/papers/cho/figure2.png" alt="Figure 2 from Cho et al. (2014)" loading="lazy"><figcaption>Figure 2 of Cho et al. (2014), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (the hidden unit).** A single unit drawn with the reset gate $$r$$ on the path from $$h$$ to $$\tilde h$$ and the update gate $$z$$ choosing between $$h$$ and $$\tilde h$$. Compare it with the LSTM diagram in Sutskever et al.: three gates and a cell there, two gates and no cell here.
- **Figure 3 (word embeddings, 2-D).** A Barnes–Hut-SNE projection of the learned word vectors; the insets zoom into clusters such as countries, months and numbers, showing that the encoder learned Word2Vec-like structure as a by-product.

<figure class="paper-fig"><img src="/assets/papers/cho/figure4.png" alt="Figure 4 from Cho et al. (2014)" loading="lazy"><figcaption>Figure 4 of Cho et al. (2014), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (phrase embeddings, 2-D).** The same projection for the encoder's phrase vectors $$c$$. Zoomed regions group phrases by meaning ("at the end of", "before the end of") and by syntax (durations, quantities), which is the evidence for the title's claim about *phrase representations*.

## Results

Adding the RNN score raised BLEU on the WMT'14 En–Fr test set from 33.30 (baseline Moses) to 33.87; combining it with a neural language model (CSLM) reached 34.64. Table 1 also shows that penalizing unknown words helped slightly. Modest numbers, but they proved the encoder–decoder learned something the phrase table did not contain.

## Thoughts

- The fixed-size vector $$c$$ must carry a whole phrase. It works for phrases; the next two papers show it breaks for long sentences, and the fix (attention) keeps every encoder state instead of the last one.
- The gate design is worth memorizing: "interpolate between old and new" appears again as the ResNet identity path and the Transformer's residual connection.

Next: [Sequence to Sequence Learning](/blog/seq2seq/), the same idea with LSTMs, four layers, and a reversed source sentence.
