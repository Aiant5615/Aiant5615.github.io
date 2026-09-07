---
title: "A Neural Probabilistic Language Model (Bengio et al., 2003)"
date: 2026-09-07 09:01:00 +0900
categories: [paper]
tags: [NLP]
math: true
rating: 4
series: transformer-lineage
series_order: 1
description: "The paper that put word embeddings inside a neural language model. Every equation of the model, the training cost that motivated a decade of tricks, and what its figures show."
paper:
  title: A Neural Probabilistic Language Model
  authors: Yoshua Bengio, Réjean Ducharme, Pascal Vincent, Christian Jauvin
  venue: Journal of Machine Learning Research
  year: 2003
  link: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
---

## One-line summary

Learn a real-valued vector for every word and a neural network that maps the vectors of the previous $$n-1$$ words to a probability distribution over the next word, so that probability mass spreads to sentences that were never seen but are made of *similar* words.

## Why it matters

Before this paper, the standard language model was a smoothed $$n$$-gram: count how often each word follows a short context, and back off to shorter contexts when counts are zero. Counting cannot generalize across words, so "The cat is walking in the bedroom" tells you nothing about "A dog was running in a room". Bengio et al. call this the *curse of dimensionality*: a 10-word sentence over a 100,000-word vocabulary lives in a space with $$10^{50}$$ points, and almost none are observed. Their fix is the distributed representation that every later model in this series inherits: Word2Vec, Seq2Seq, and the Transformer's embedding table are all descendants of the matrix $$C$$ below.

## The model, equation by equation

The goal is a conditional distribution $$P(w_t \mid w_{t-1}, \dots, w_{t-n+1})$$ over a vocabulary $$V$$. The model has two parts.

**1. A shared lookup table.** Each word $$i \in V$$ has a feature vector $$C(i) \in \mathbb{R}^m$$, a row of a matrix $$C \in \mathbb{R}^{|V| \times m}$$. The input to the network is the concatenation of the previous words' vectors:

$$
x = \big(C(w_{t-1}), C(w_{t-2}), \dots, C(w_{t-n+1})\big) \in \mathbb{R}^{(n-1)m}
$$

The same $$C$$ is used at every position, so what the model learns about a word in one context transfers to every other context. This sharing is the whole point.

**2. A one-hidden-layer network with optional skip connections.** The pre-softmax scores for the $$|V|$$ candidate next words are

$$
y = b + Wx + U \tanh(d + Hx)
$$

where $$H \in \mathbb{R}^{h \times (n-1)m}$$ and $$d$$ form the hidden layer of size $$h$$, $$U \in \mathbb{R}^{|V| \times h}$$ maps hidden units to output scores, $$b$$ is the output bias, and $$W \in \mathbb{R}^{|V| \times (n-1)m}$$ is a *direct* connection from the input vectors to the output (set $$W = 0$$ to remove it; the paper finds it speeds up convergence but slightly hurts generalization). The scores become probabilities with a softmax:

$$
P(w_t = i \mid w_{t-1}, \dots, w_{t-n+1}) = \frac{e^{y_i}}{\sum_{j} e^{y_j}}
$$

The full parameter set is $$\theta = (b, d, W, U, H, C)$$ and its size is dominated by the output layer: $$|V|(1 + nm + h) + h(1 + (n-1)m)$$ numbers. With $$|V| = 17{,}000$$, that output layer is why the model is slow.

**3. Training.** Maximize the regularized log-likelihood of the corpus,

$$
L = \frac{1}{T} \sum_t \log f(w_t, w_{t-1}, \dots, w_{t-n+1}; \theta) + R(\theta)
$$

with plain stochastic gradient ascent, $$\theta \leftarrow \theta + \varepsilon \frac{\partial \log P(w_t \mid \cdot)}{\partial \theta}$$. Note that the gradient flows into $$C$$: the embeddings are learned jointly with the predictor, not fixed beforehand. $$R$$ is a weight decay on everything except biases.

**Cost.** One forward pass costs about $$|V| \times (h + nm)$$ multiply-adds because every candidate word needs a score. The authors parallelized over the vocabulary across 40 CPUs and still needed weeks. This bottleneck is exactly what hierarchical softmax and negative sampling attack in [Word2Vec](/blog/2026/09/07/word2vec/).

## Figures, explained

- **Figure 1 (the architecture).** Read it bottom-up. The previous words index into the table $$C$$ (drawn as one shared block with arrows from each word position). The concatenated vectors go up into the $$\tanh$$ layer, then into the softmax over the full vocabulary. The dashed lines are the optional direct connections $$W$$. The picture is the template for every "embedding, then network, then softmax" model that followed.
- **Table 1–2 (perplexity on Brown and AP News).** The rows compare $$n$$-gram baselines (interpolated trigram, Kneser–Ney back-off, class-based) against MLP variants with different $$n$$, $$h$$, $$m$$. The headline: the best neural model reaches a test perplexity about 24% below the best smoothed trigram on Brown, and mixing the neural model with the trigram helps further, because the two make different mistakes.

## Results

On the 1.2M-word Brown corpus (vocabulary 16,383) the neural model reached perplexity 252 versus 312–336 for the best count-based models; on the 14M-word AP News subset the gain was smaller but still clear. The model with $$n = 5$$ (four words of context) beat $$n = 3$$, something count-based models could not exploit because longer contexts are sparser.

## Thoughts

- Everything here is still in a modern LM: a learned embedding matrix, a nonlinear predictor, a softmax over the vocabulary, and maximum likelihood by SGD. What changed is the predictor (RNN, then attention) and the scale.
- The "direct connection" $$W$$ is an early residual-style shortcut, fifteen years before [ResNet](/blog/2026/09/07/resnet/) made it standard.
- The paper's own future-work list already mentions energy-based models, decomposed output layers, and using the representations for other tasks. All three happened.

Next in the series: [Word2Vec](/blog/2026/09/07/word2vec/), which keeps the lookup table and throws away the hidden layer.
