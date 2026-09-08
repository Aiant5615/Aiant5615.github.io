---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022)"
date: 2026-09-07 12:06:00 +0900
categories: [paper, llm-engineering]
image: /assets/og-llm-engineering.png
tags: [DL, Systems]
math: true
rating: 5
series: llm-advanced
series_order: 6
description: "Attention is bounded by memory traffic, not FLOPs. Tiling with the online-softmax rescaling identities, recomputation in the backward pass instead of storing the N×N matrix, the IO-complexity bound, block-sparse extension, and what the speed/memory figures show. Why long contexts became affordable."
paper:
  title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
  published: 2022-05
  authors: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford, Buffalo)
  venue: NeurIPS
  year: 2022
  link: https://arxiv.org/abs/2205.14135
---

## One-line summary

Compute exactly the same attention output, but never write the $$N \times N$$ score matrix to GPU memory: process $$Q, K, V$$ in blocks that fit in on-chip SRAM, keep running softmax statistics so blocks can be combined, and in the backward pass recompute the scores from the saved statistics instead of loading them. Memory goes from quadratic to linear in sequence length, and wall-clock speed improves 2–4× because attention was never compute-bound, it was bandwidth-bound.

## Why it matters

Every Transformer since 2017 paid $$O(N^2)$$ memory for attention, which capped context lengths and made long-sequence training slow. FlashAttention removed the memory wall without approximating anything, is inside essentially every training and inference stack today, and is what allows 8k–128k contexts, long chains of thought, and high-resolution image token sequences.

## The bottleneck (Section 2)

A100 GPUs have about 40–80GB of HBM at 1.5–2 TB/s and 20MB of SRAM at ~19 TB/s per chip. Standard attention (Algorithm 0) does

1. load $$Q, K$$ from HBM, compute $$S = QK^{\top}$$, **write $$S$$ to HBM**;
2. load $$S$$, compute $$P = \operatorname{softmax}(S)$$, **write $$P$$**;
3. load $$P, V$$, compute $$O = PV$$, write $$O$$.

Each step is a memory-bound elementwise or matmul-then-store operation on an $$N \times N$$ array, so runtime is dominated by reading and writing $$O(N^2)$$ entries, not by the $$O(N^2 d)$$ FLOPs. Dropout and masking add more passes. (Figure 1 right: in the GPT-2 profile, matmul is a small slice; softmax, dropout and masking dominate.)

## Tiling with online softmax (Section 3.1)

Softmax couples every column of a row, so a row cannot be split naively. But the softmax of a concatenation can be assembled from per-block statistics. For a row vector $$x = [x^{(1)}\ x^{(2)}]$$, define for each block the max, the shifted exponentials and their sum,

$$
m(x^{(j)}) = \max_i x^{(j)}_i, \qquad f(x^{(j)}) = \big[e^{x^{(j)}_1 - m(x^{(j)})}, \dots\big], \qquad \ell(x^{(j)}) = \sum_i f(x^{(j)})_i
$$

Then the statistics of the full row are

$$
m(x) = \max\big(m(x^{(1)}), m(x^{(2)})\big), \qquad \ell(x) = e^{m(x^{(1)}) - m(x)}\, \ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)}\, \ell(x^{(2)}), \qquad \operatorname{softmax}(x) = \frac{\big[e^{m(x^{(1)}) - m(x)} f(x^{(1)}),\ e^{m(x^{(2)}) - m(x)} f(x^{(2)})\big]}{\ell(x)}
$$

So a block can be processed with its *local* max, and later rescaled by $$e^{m_{\text{old}} - m_{\text{new}}}$$ when a larger max is seen. FlashAttention (Algorithm 1) loops over blocks of $$K, V$$ in the outer loop and blocks of $$Q$$ in the inner loop, keeping in SRAM the current output block $$O_i$$, the running row-max $$m_i$$ and row-sum $$\ell_i$$; for each new $$K_j, V_j$$ block it computes $$S_{ij} = Q_i K_j^{\top}$$, the block's local statistics, updates $$m_i$$ and $$\ell_i$$ with the identities above, and updates $$O_i \leftarrow \operatorname{diag}(\ell_i^{\text{new}})^{-1}\big(\operatorname{diag}(\ell_i)\, e^{m_i - m_i^{\text{new}}} O_i + e^{\tilde m_{ij} - m_i^{\text{new}}} \tilde P_{ij} V_j\big)$$. Block sizes are chosen from the SRAM size $$M$$ (roughly $$B_c = \lceil M / 4d \rceil$$ rows of $$K$$ and $$V$$). Softmax, masking and dropout are all fused into this one kernel, so $$S$$ and $$P$$ never touch HBM.

## Recomputation in the backward pass (Section 3.1)

Backpropagation normally needs $$P$$ ($$N \times N$$). FlashAttention stores only $$O$$ and the per-row logsumexp $$L_i = m_i + \log \ell_i$$ (size $$N$$), then in the backward pass recomputes each block $$S_{ij}, P_{ij}$$ from $$Q_i, K_j$$ and $$L_i$$ on the fly (Algorithm 4). This costs extra FLOPs but *fewer HBM reads*, and is still faster than the standard backward pass. This is the same trade as gradient checkpointing, applied inside one operation.

## IO complexity (Theorem 2)

Standard attention performs $$\Theta(N d + N^2)$$ HBM accesses. FlashAttention performs

$$
\Theta\big(N^2 d^2 M^{-1}\big)
$$

HBM accesses, where $$M$$ is the SRAM size; for typical $$d = 64$$–$$128$$ and $$M \approx 100$$KB this is many times smaller than $$N^2$$, and Proposition 3 proves no exact algorithm can asymptotically do better across all $$M$$. Memory is $$O(N)$$ extra beyond inputs and outputs.

**Block-sparse FlashAttention (Section 3.3).** With a block mask, skip blocks that are all zero; IO drops proportionally to sparsity, giving another 2–4× at long sequences (Figure 2 right).

## Results (Section 4)

- **BERT-large (MLPerf 1.1):** 15% faster end-to-end than the Nvidia record submission.
- **GPT-2 (small/medium on OpenWebText):** up to 3× faster training than HuggingFace and 1.7× faster than Megatron-LM, same perplexity (Table 2, Figure 4 in the appendix shows the loss curves coincide).
- **Long-range arena:** 2.4× faster than standard attention, and Path-X (16k tokens) is solved for the first time by any Transformer (61.4%); block-sparse FlashAttention solves Path-256 (64k tokens).
- **Longer context helps:** GPT-2 with 4k context trained with FlashAttention is faster *and* 0.7 perplexity better than Megatron with 1k (Table 4); document classification improves with longer context (Table 5).
- **Benchmarks (Figure 3):** runtime for forward+backward and memory vs sequence length against PyTorch, Megatron, linear and approximate attention; FlashAttention is fastest up to ~1–2k, block-sparse is fastest beyond, and memory is up to 20× smaller than the baselines and always linear.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/flashattn/figure1.png" width="1018" height="417" alt="Figure 1 from Dao et al. (2022)" loading="lazy"><figcaption>Figure 1 of Dao et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (left: the memory hierarchy and tiling; right: GPT-2 attention profile).** Left shows SRAM/HBM/DRAM bandwidths and the outer/inner loop over $$K, V$$ and $$Q$$ blocks with the output being accumulated. Right: a bar chart of attention time on GPT-2 where the fused FlashAttention kernel is a fraction of the PyTorch stack of matmul, mask, softmax, dropout.

<figure class="paper-fig"><img src="/assets/papers/flashattn/figure2.png" width="544" height="194" alt="Figure 2 from Dao et al. (2022)" loading="lazy"><figcaption>Figure 2 of Dao et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (left: GPT-2 medium runtime vs sequence length; right: block-sparse).** FlashAttention's forward+backward time grows slower than PyTorch's; block-sparse variants are faster still as sparsity increases.

<figure class="paper-fig"><img src="/assets/papers/flashattn/figure3.png" width="975" height="254" alt="Figure 3 from Dao et al. (2022)" loading="lazy"><figcaption>Figure 3 of Dao et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (runtime and memory vs length).** Log–log runtime plots for several attention implementations and a memory plot where FlashAttention is linear while others are quadratic.

## Thoughts

- The paper's real contribution is the *diagnosis*: measure what is memory-bound, not FLOP-bound, and design the algorithm around the memory hierarchy. FlashAttention-2 and 3 are engineering refinements of the same principle for newer GPUs.
- Online softmax was known from earlier work (Milakov & Gimelshein, 2018); combining it with recomputation and a fused kernel is what made exact attention cheap.
- It quietly ended the wave of approximate "efficient attention" methods: if exact attention is fast enough at 16k–64k tokens, approximations that lose accuracy are hard to justify.

Next: [LLaMA](/blog/llama/), what the open recipe looks like with all of the above assembled.
