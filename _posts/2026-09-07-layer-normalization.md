---
title: "Layer Normalization (Ba, Kiros & Hinton, 2016)"
date: 2026-09-07 09:09:00 +0900
categories: [paper, llm-basics]
tags: [DL]
math: true
rating: 4
series: transformer-lineage
series_order: 9
description: "Normalize across the features of one example instead of across the batch. The formulas, the invariance table, the RNN version, why it suits variable-length sequences, and what the convergence figures show."
paper:
  title: Layer Normalization
  published: 2016-07
  authors: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton (Toronto)
  venue: arXiv
  year: 2016
  link: https://arxiv.org/abs/1607.06450
---

## One-line summary

Standardize the summed inputs to the neurons of a layer using the mean and variance computed *over that layer's units for a single example*, so the statistics do not depend on the minibatch, work identically at training and test time, and can be applied at every time step of an RNN.

## Why it matters

The Transformer block is "attention, add, **LayerNorm**, feed-forward, add, LayerNorm". Batch normalization, the standard at the time, is awkward for sequences: the batch statistics differ by time step, long sequences at test time have positions never seen in training, and small batches make the estimates noisy. Layer normalization removes the batch from the equation entirely, which is why it is the normalization in every language model since.

## Batch norm, restated

For unit $$i$$ in a layer, with summed input $$a_i = w_i^{\top} x$$, batch normalization rescales

$$
\bar a_i = \frac{g_i}{\sigma_i}\,(a_i - \mu_i), \qquad \mu_i = \mathbb{E}_{x \sim P(x)}[a_i], \quad \sigma_i = \sqrt{\mathbb{E}_{x \sim P(x)}\big[(a_i - \mu_i)^2\big]}
$$

where the expectation is estimated from the current minibatch, and $$g_i$$ is a learned gain. Each unit gets its own statistics, computed across examples.

## Layer norm, equation by equation

Swap the axis: compute one mean and one standard deviation across all $$H$$ units of the layer, for each example separately,

$$
\mu^{l} = \frac{1}{H} \sum_{i=1}^{H} a_i^{l}, \qquad \sigma^{l} = \sqrt{\frac{1}{H} \sum_{i=1}^{H} \big(a_i^{l} - \mu^{l}\big)^2}
$$

and normalize every unit with the same pair. With gain $$g$$ and bias $$b$$ (one per unit) and nonlinearity $$f$$, the layer output is

$$
h^{l} = f\Big(\frac{g}{\sigma^{l}} \odot \big(a^{l} - \mu^{l}\big) + b\Big)
$$

Nothing here depends on other examples, so there is no train/test discrepancy and no running averages to maintain.

**In an RNN.** With recurrent pre-activations $$a^t = W_{hh} h^{t-1} + W_{xh} x^t$$, apply the same thing at every step:

$$
h^{t} = f\Big(\frac{g}{\sigma^{t}} \odot \big(a^{t} - \mu^{t}\big) + b\Big), \qquad \mu^{t} = \frac{1}{H}\sum_i a_i^{t}, \quad \sigma^{t} = \sqrt{\frac{1}{H}\sum_i (a_i^{t} - \mu^{t})^2}
$$

This keeps the scale of the recurrent state constant over time, which the paper argues prevents the exploding/vanishing hidden-state magnitudes that make RNN training sensitive to initialization and learning rate. For an LSTM, normalization is applied separately to the four gate pre-activations and to the cell before the output gate.

## The invariance table

Table 1 compares what each normalizer is invariant to:

| | weight matrix re-scaling | weight matrix re-centering | single weight re-scaling | data re-scaling | data re-centering | single example re-scaling |
|---|---|---|---|---|---|---|
| Batch norm | yes | no | yes | yes | yes | no |
| Weight norm | yes | no | yes | no | no | no |
| Layer norm | yes | yes | no | yes | no | yes |

Layer norm is invariant to scaling and shifting the *whole weight matrix* and to scaling *one input example*, but not to scaling a single weight vector (since one unit's change alters the shared mean and variance). Batch norm is the mirror image. The "single example re-scaling" invariance is what makes LN robust to inputs with very different magnitudes, including a long sequence's late time steps.

**Geometry (Section 5).** The paper analyses the parameter space with the Fisher information metric and shows that for a normalized layer the metric along the direction of scaling a weight vector shrinks as $$\lVert w\rVert$$ grows, so the effective learning rate for that direction is $$1/\lVert w\rVert^2$$ smaller: growing the weights automatically damps their own updates. This is an early version of the "normalization = implicit learning-rate scheduling" argument.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/layernorm/figure1.png" alt="Figure 1 from Ba et al. (2016)" loading="lazy"><figcaption>Figure 1 of Ba et al. (2016), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (image–sentence ranking, order embeddings).** Recall@K on validation versus training iterations for LN vs baseline GRU encoders; LN converges markedly faster and ends higher.

<figure class="paper-fig"><img src="/assets/papers/layernorm/figure2.png" alt="Figure 2 from Ba et al. (2016)" loading="lazy"><figcaption>Figure 2 of Ba et al. (2016), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (attentive reader).** Validation error curves on the CNN question-answering corpus; LN with 4× the normal learning rate is stable where the baseline diverges.

<figure class="paper-fig"><img src="/assets/papers/layernorm/figure3.png" alt="Figure 3 from Ba et al. (2016)" loading="lazy"><figcaption>Figure 3 of Ba et al. (2016), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (skip-thought vectors).** Downstream sentence-task scores over 1M training iterations; LN skip-thoughts reach the baseline's final score in a fraction of the iterations and surpass it.

<figure class="paper-fig"><img src="/assets/papers/layernorm/figure4.png" alt="Figure 4 from Ba et al. (2016)" loading="lazy"><figcaption>Figure 4 of Ba et al. (2016), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (DRAW image generation).** Test variational bound on MNIST vs epochs for the baseline, weight norm and layer norm; LN converges fastest.

<figure class="paper-fig"><img src="/assets/papers/layernorm/figure5.png" alt="Figure 5 from Ba et al. (2016)" loading="lazy"><figcaption>Figure 5 of Ba et al. (2016), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5 (handwriting sequence generation).** Negative log-likelihood vs updates for LN vs the baseline, train and test; LN reaches the baseline's final score in far fewer updates.

<figure class="paper-fig"><img src="/assets/papers/layernorm/figure6.png" alt="Figure 6 from Ba et al. (2016)" loading="lazy"><figcaption>Figure 6 of Ba et al. (2016), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 6 (MNIST feed-forward, permutation-invariant).** Four panels comparing batch norm and layer norm at batch sizes 128 and 4. With large batches BN and LN are similar; with batch size 4, BN degrades sharply and LN is unaffected. The plot that explains why LN, not BN, suits models trained with small per-device batches.
- **Table on CNNs.** LN does *not* help convolutional networks, where BN remains better; the authors attribute it to the feature-map statistics being very different across positions.

## Results

Consistent speed-ups on six RNN tasks (order embeddings, attentive reader, skip-thoughts, DRAW image generation, handwriting, MNIST), with several new state-of-the-art numbers, and no dependence on batch size. No improvement for ConvNets.

## Thoughts

- The Transformer authors picked LN without much discussion; it was simply the normalization that worked on sequences. RMSNorm (Zhang & Sennrich, 2019), which drops the mean subtraction and keeps only the scale term $$\frac{a}{\text{RMS}(a)} \odot g$$, is the version in LLaMA-style models today.
- Where LN sits relative to the residual add ("post-norm" in Vaswani, "pre-norm" in GPT-2) turned out to matter more than the choice of normalizer.

Next: the paper the series is named for, [Attention Is All You Need](/blog/transformer/).
