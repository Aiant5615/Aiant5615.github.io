---
title: "Multimodal Chain-of-Thought Reasoning in Language Models (Zhang et al., 2023)"
date: 2026-09-07 14:11:00 +0900
categories: [paper, vision-llm]
image: /assets/og-vision-llm.png
tags: [CV, NLP]
math: true
rating: 3
series: multimodal-llm
series_order: 8
description: "Why a sub-1B model gets worse at ScienceQA when asked to reason before answering (hallucinated rationales, 56% of errors), and the two-stage fix: generate the rationale with fused ViT patch features and gated cross-attention, then infer the answer from question plus rationale. The fusion equations, the ablations, and the accuracy curves."
paper:
  title: "Multimodal Chain-of-Thought Reasoning in Language Models"
  published: 2023-02
  authors: Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, Alex Smola
  venue: TMLR 2024
  year: 2023
  link: https://arxiv.org/abs/2302.00923
---

## One-line summary

[Chain-of-thought](/blog/chain-of-thought/) prompting works for 100B-scale language models. This paper asks what happens when a small (under 1B) encoder–decoder model is fine-tuned to produce a rationale before the answer on a multimodal science-question benchmark, and finds it gets *worse* (81.6% → 69.3%) because the rationales hallucinate about the image. Two changes fix it: split the problem into a **rationale-generation stage** and an **answer-inference stage** that are trained separately, and inject **ViT patch features** into the language encoder with attention-based fusion. Multimodal-CoT then reaches state-of-the-art on ScienceQA with a 770M T5, beating GPT-3.5 CoT and, at the time, human accuracy.

## Why it matters

It is the first peer-reviewed study of CoT in a vision+language setting and the origin of the "two-stage rationale then answer" recipe. More importantly it documents *why* naive multimodal CoT fails at small scale: the rationale is only helpful if it is grounded in the image, and captions are not enough. Every RL-based multimodal reasoning paper later in this series ([Vision-R1](/blog/vision-r1/), [DeepEyes](/blog/deepeyes/)) is, in one way or another, a way to make the rationale grounded.

## The problem (Section 3)

Setup: ScienceQA, multiple-choice science questions with optional images, annotated with lectures and explanations that serve as gold rationales. Backbone: a 200M T5 variant (FLAN-Alpaca) fine-tuned as text generation. Inputs are the question $$Q$$, context $$C$$ and options $$M$$; outputs are rationale $$R$$ and/or answer $$A$$.

- **One-stage (Table 2).** QCM→A (no CoT) 81.63%; QCM→RA (reason, then answer) 69.32%; QCM→AR (answer, then explain) 69.68%. Generating a rationale first costs 12 points. Output lengths are under 400 tokens, so truncation is not the cause.
- **Two-stage without vision (Table 3).** Train QCM→R and QCMR→A separately. The rationale model gets RougeL 90.73, but answer accuracy is 78.57%, still below the no-CoT baseline. Inspecting 50 errors, 56% are **hallucinated rationales** (Figure 3a): the text-only model has no way to see which magnet pole faces which, so it invents one and then reasons correctly from the wrong premise (Figure 2).
- **Adding vision.** Captions help by only 0.8 points. Feeding ViT patch features into the encoder raises rationale RougeL to 93.46 and answer accuracy to 85.31%, and resolves 60.7% of the previously hallucinated cases (Figure 3b).

## Multimodal-CoT (Section 4)

**Two stages (Figure 4).** Stage 1: $$R = F(X)$$ with $$X = \{X^{1}_{\text{language}}, X_{\text{vision}}\}$$, the language part being the concatenation of question, context and options. Stage 2: append the generated rationale, $$X^{2}_{\text{language}} = X^{1}_{\text{language}} \circ R$$, and infer $$A = F(X')$$ with the same image. Both stages use the same architecture but are trained independently on the annotated pairs $$(X \to R)$$ and $$(XR \to A)$$; at test time stage 2 consumes stage 1's outputs.

**Architecture.** The target text $$Y$$ of length $$N$$ is generated autoregressively,

$$
p(Y \mid X_{\text{language}}, X_{\text{vision}}) = \prod_{i=1}^{N} p_{\theta}\big(Y_i \mid X_{\text{language}}, X_{\text{vision}}, Y_{<i}\big)
$$

with three steps: encoding, interaction, decoding.

*Encoding.* The Transformer encoder gives $$H_{\text{language}} = \text{LanguageEncoder}(X_{\text{language}}) \in \mathbb{R}^{n \times d}$$; a frozen [ViT](/blog/vit/) gives patch-level features, projected by a learned matrix, $$H_{\text{vision}} = W_h \cdot \text{VisionExtractor}(X_{\text{vision}}) \in \mathbb{R}^{m \times d}$$, with $$m$$ patches.

*Interaction.* Single-head attention with the text as queries and the patches as keys/values,

$$
H^{\text{attn}}_{\text{vision}} = \text{Softmax}\Big(\frac{Q K^{\top}}{\sqrt{d_k}}\Big) V, \qquad Q = H_{\text{language}},\ K = V = H_{\text{vision}}
$$

so every text token gathers a vision vector, followed by a **gated fusion**:

$$
\lambda = \sigma\big(W_l H_{\text{language}} + W_v H^{\text{attn}}_{\text{vision}}\big), \qquad H^{\text{fuse}} = (1 - \lambda)\cdot H_{\text{language}} + \lambda \cdot H^{\text{attn}}_{\text{vision}}
$$

The sigmoid gate $$\lambda$$ decides per token and per dimension how much image information to admit (compare the tanh-gated cross-attention in [Flamingo](/blog/flamingo/)).

*Decoding.* The Transformer decoder attends to $$H^{\text{fuse}}$$ and produces $$Y$$.

## Results (Section 5–6)

- **ScienceQA (Table 4).** Multimodal-CoT Large (T5 770M with ViT features) 91.68% overall, above UnifiedQA CoT (74.1%), GPT-3.5 CoT (75.2%) and the human average (88.4%) reported on the leaderboard at release; Base (223M) reaches 85.31% and later versions with stronger backbones push higher.
- **A-OKVQA (Table 5).** Consistent gains over the baselines, showing the recipe is not benchmark-specific.
- **Ablations (Table 6).** Removing vision features or the two-stage split each hurts; both are needed.
- **Convergence (Figure 5).** The two-stage multimodal model starts high and stays high across epochs; the one-stage baselines start low, and the one-stage multimodal variant only slowly catches up. Rationales that are grounded accelerate learning.
- **Vision features (Table 9).** ViT, CLIP, DETR and ResNet features are compared; ViT works best. **Backbones (Table 8)**: UnifiedQA, FLAN-T5 and FLAN-Alpaca all benefit.
- **Error analysis (Figure 6, Section 6.7).** Remaining mistakes are dominated by commonsense (maps, temperature), logical slips, and cases where the rationale was right but the answer was not.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/mmcot/figure1.png" width="661" height="336" alt="Figure 1 from Zhang et al. (2023)" loading="lazy"><figcaption>Figure 1 of Zhang et al. (2023), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (task).** The magnet question: two bar magnets in the image, the language input with question, context and options, and the target output rationale ("the north pole of one magnet is closest to the south pole of the other…") followed by the answer (A).

<figure class="paper-fig"><img src="/assets/papers/mmcot/figure2.png" width="1259" height="455" alt="Figure 2 from Zhang et al. (2023)" loading="lazy"><figcaption>Figure 2 of Zhang et al. (2023), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (hallucination).** The same problem with the gold rationale, then the text-only baseline's rationale (it claims south faces south and answers "repel") beside the vision-fused model's rationale (correct poles, "attract").

<figure class="paper-fig"><img src="/assets/papers/mmcot/figure3.png" width="598" height="314" alt="Figure 3 from Zhang et al. (2023)" loading="lazy"><figcaption>Figure 3 of Zhang et al. (2023), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (error pie).** (a) 56% of two-stage errors are hallucinated rationales; (b) with vision features 60.7% of those are resolved.

<figure class="paper-fig"><img src="/assets/papers/mmcot/figure4.png" width="1207" height="383" alt="Figure 4 from Zhang et al. (2023)" loading="lazy"><figcaption>Figure 4 of Zhang et al. (2023), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (framework).** Stage 1 takes vision + language and generates the rationale; stage 2 takes the original language input with the rationale appended, plus the same image, and outputs the answer.

<figure class="paper-fig"><img src="/assets/papers/mmcot/figure5.png" width="613" height="354" alt="Figure 5 from Zhang et al. (2023)" loading="lazy"><figcaption>Figure 5 of Zhang et al. (2023), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5 (accuracy curves).** Validation accuracy over 10 epochs for one-stage vs two-stage and baseline vs multimodal; the two-stage multimodal curve (blue circles) is highest throughout.

## Thoughts

- The paper's real finding is negative and important: CoT is not free for small models, and an ungrounded rationale is a liability. The measurement (56% hallucination) is more informative than the architecture.
- The fusion is modest, one cross-attention and a gate on a frozen ViT, but placing the fusion in the *rationale* stage is what makes it work: the model needs the image most when it is writing the premises.
- The two-stage design also means the answer model can be trained on gold rationales, a form of teacher forcing that later work replaces by RL on the model's own reasoning; see the next paper.

Next: [Vision-R1](/blog/vision-r1/), which brings the DeepSeek-R1 recipe to multimodal LLMs with a cold start built by modality bridging and a progressive length schedule for GRPO.
