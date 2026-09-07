---
title: "LLM basics: from n-gram LMs to InstructGPT, a 16-paper reading path"
date: 2026-09-07 09:17:00 +0900
overview: true
categories: [paper, llm-basic]
tags: [NLP]
description: "The sixteen papers that get you from the first neural language model to RLHF, grouped by what each one added, with a detailed review of every one."
---

Sixteen papers, read in order, that explain where today's language models come from. Each link goes to a review that walks through the model's equations and figures.

| | Year | Paper | What it added |
|---|---|---|---|
| **Word representations** | 2003 | [A Neural Probabilistic Language Model](/blog/nplm-bengio/) — Bengio et al. | Learned word vectors inside a neural LM |
| | 2013 | [Efficient Estimation of Word Representations](/blog/word2vec/) — Mikolov et al. | Word2Vec: cheap embeddings at scale |
| **RNN Seq2Seq** | 2014 | [RNN Encoder–Decoder](/blog/rnn-encoder-decoder-gru/) — Cho et al. | The encoder–decoder and the GRU |
| | 2014 | [Sequence to Sequence Learning](/blog/seq2seq/) — Sutskever, Vinyals, Le | Deep LSTM seq2seq beats SMT |
| **Attention** | 2015 | [Jointly Learning to Align and Translate](/blog/bahdanau-attention/) — Bahdanau, Cho, Bengio | Additive attention over all encoder states |
| | 2015 | [Effective Approaches to Attention](/blog/luong-attention/) — Luong et al. | Dot-product scores, local attention, input feeding |
| | 2016 | [Subword Units](/blog/bpe-subword/) — Sennrich et al. | BPE tokenization, open vocabulary |
| **Transformer parts** | 2016 | [Deep Residual Learning](/blog/resnet/) — He et al. | Residual connections |
| | 2016 | [Layer Normalization](/blog/layer-normalization/) — Ba, Kiros, Hinton | Per-example normalization for sequences |
| | 2017 | [Attention Is All You Need](/blog/transformer/) — Vaswani et al. | The Transformer |
| **Pretraining** | 2018 | [Generative Pre-Training](/blog/gpt1/) — Radford et al. | GPT: decoder LM, then fine-tune |
| | 2018 | [BERT](/blog/bert/) — Devlin et al. | Masked LM on a bidirectional encoder |
| | 2019 | [Unsupervised Multitask Learners](/blog/gpt2/) — Radford et al. | GPT-2: scale and zero-shot |
| | 2020 | [Unified Text-to-Text Transformer](/blog/t5/) — Raffel et al. | T5: everything is text-to-text |
| | 2020 | [Few-Shot Learners](/blog/gpt3/) — Brown et al. | GPT-3: in-context learning at 175B |
| **Alignment** | 2022 | [Instructions with Human Feedback](/blog/instructgpt/) — Ouyang et al. | InstructGPT: SFT, reward model, PPO |

## How to read them

- The first ten are about *representing* and *moving* information: embeddings (1–2), recurrence (3–4), attention (5–6), tokens (7), and the two pieces that make deep stacks trainable (8–9), assembled in 10.
- The next five are about *what to train on*: a decoder LM (11), a masked encoder (12), more data with no fine-tuning (13), a controlled comparison (14), and scale (15).
- The last one is about *what to optimize for* once the model is good at predicting text (16).

Each review has the same shape: a one-line summary, the equations explained one by one, a section that describes the paper's figures (I can't reproduce them, so I say what each one shows and redraw a few architectures), the headline numbers, and what the paper changed. Reviews link forward and backward, and every review has a "Part n of 16" navigation at the bottom.
