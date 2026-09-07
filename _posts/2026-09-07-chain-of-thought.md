---
title: "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022)"
date: 2026-09-07 12:05:00 +0900
categories: [paper, llm-advanced]
tags: [NLP]
math: true
rating: 4
series: llm-advanced
series_order: 5
description: "Put worked reasoning steps in the few-shot exemplars and large models solve multi-step problems they otherwise cannot. What the prompt looks like, why it is an emergent ability of scale, the ablations that rule out trivial explanations, and the scaling figures explained. The prompting ancestor of R1-style reasoning."
paper:
  title: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
  published: 2022-01
  authors: Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou (Google)
  venue: NeurIPS
  year: 2022
  link: https://arxiv.org/abs/2201.11903
---

## One-line summary

Change nothing about the model. In the few-shot prompt, write each example as ⟨question, a few sentences of reasoning, answer⟩ instead of ⟨question, answer⟩. Models above roughly 100B parameters then produce their own reasoning before answering and jump from 18% to 57% on GSM8K (PaLM 540B), beating a fine-tuned GPT-3 with a verifier; small models get worse.

## Why it matters

It is the first clear demonstration that *how* a model is asked to answer changes what it can do, that reasoning in text is a capability that emerges with scale, and that tokens spent thinking buy accuracy. DeepSeek-R1 (in the RL series) is what happens when a model is trained with RL to produce chains of thought on its own instead of copying them from the prompt.

## The method

Standard few-shot prompting (GPT-3 style) gives $$k$$ input–output pairs and then the test question. Chain-of-thought prompting augments each exemplar with a natural-language rationale that leads from the question to the answer, e.g.

> Q: Roger has 5 tennis balls. He buys 2 more cans of 3 tennis balls each. How many tennis balls does he have now?
> A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11.

Eight such exemplars, hand-written by the authors, are used for all arithmetic benchmarks; the model imitates the format, writes a rationale for the new question, and the final number after "The answer is" is extracted. No fine-tuning, no verifiers, greedy decoding. The paper's argument for why this should help: the rationale decomposes a multi-step problem into steps the model can each do; it allocates more computation (tokens) to harder problems; it is interpretable and debuggable; and it applies to any task humans can explain in words.

## Experiments

**Arithmetic (Section 3).** GSM8K, SVAMP, ASDiv, AQuA, MAWPS, with UL2 20B, LaMDA 137B, GPT-3 175B, Codex, PaLM 8B/62B/540B. Three findings from Figure 4: (1) chain-of-thought is an **emergent ability**: it does not help, and sometimes hurts, models under about 10B parameters, and the gains appear only for the largest ones; (2) gains are largest on the hardest benchmark (GSM8K) and smallest on single-step problems (MAWPS SingleOp); (3) PaLM 540B with chain-of-thought reaches **56.9% on GSM8K**, above the previous best (a fine-tuned GPT-3 175B with a trained verifier, 55%), and 58.6% with an external calculator fixing arithmetic slips. Manual inspection: of 50 correct answers only 2 had accidental correct reasoning; of 50 errors, 46% were "almost correct" (one calculation or step off) and 54% had semantic misunderstandings.

**Ablations (Figure 5).** To rule out shallow explanations: *equation only* (write the equation but no words) helps little on GSM8K, so natural-language steps matter; *variable compute only* (output dots equal in length to the equation) does nothing, so it is not the extra tokens per se; *reasoning after the answer* also fails, so the rationale must come *before* and be used to produce the answer, not post-hoc.

**Robustness (Figure 6).** Chains written by three different annotators, in different styles, and exemplars taken from GSM8K's own training set all give large gains over standard prompting, though with variance across exemplar sets.

**Commonsense (Section 4).** CSQA, StrategyQA, BIG-bench Date and Sports Understanding, SayCan robot planning: chain-of-thought gives PaLM 540B 75.6% on StrategyQA (prior best 69.4%) and 95.4% on Sports Understanding, above an unaided sports enthusiast (84%).

**Symbolic reasoning (Section 5).** Last-letter concatenation and coin-flip state tracking, with in-domain and *longer* out-of-domain test lengths: chain-of-thought makes length generalization possible (near 0% → 95%+ for PaLM 540B on 3–4 word names never seen), standard prompting stays at zero.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/cot/figure1.png" alt="Figure 1 from Wei et al. (2022)" loading="lazy"><figcaption>Figure 1 of Wei et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (the contrast).** Two prompts side by side: standard prompting fails the tennis-ball question, chain-of-thought prompting solves it with the highlighted rationale.

<figure class="paper-fig"><img src="/assets/papers/cot/figure2.png" alt="Figure 2 from Wei et al. (2022)" loading="lazy"><figcaption>Figure 2 of Wei et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (PaLM 540B on GSM8K).** Bars for prior supervised state of the art, standard prompting, and chain-of-thought; the last bar is the new record.

<figure class="paper-fig"><img src="/assets/papers/cot/figure3.png" alt="Figure 3 from Wei et al. (2022)" loading="lazy"><figcaption>Figure 3 of Wei et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (exemplars).** The ⟨input, chain of thought, output⟩ triples used for arithmetic, commonsense and symbolic tasks; worth reading to see how plain the rationales are.

<figure class="paper-fig"><img src="/assets/papers/cot/figure4.png" alt="Figure 4 from Wei et al. (2022)" loading="lazy"><figcaption>Figure 4 of Wei et al. (2022), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (scaling curves).** Accuracy vs model size on GSM8K, SVAMP and MAWPS for LaMDA, GPT-3 and PaLM, standard vs chain-of-thought; the chain-of-thought lines start below the standard lines for small models and cross above them sharply at the largest sizes.
- **Figures 5–6 (ablations and robustness).** Bar charts for the ablation variants and for different annotators/exemplar sources.

## Thoughts

- "Emergent" here means the effect is invisible below a scale threshold; later work (self-consistency, least-to-most, zero-shot "Let's think step by step") lowered the threshold and increased the gains, and instruction tuning made chains appear without exemplars.
- The ablations are the strongest part: they separate *reasoning in text before answering* from "longer outputs" or "seeing equations".
- Reasoning at test time trades compute for accuracy; FlashAttention (next) is about making that compute affordable, and R1 about training models to spend it wisely.

Next: [FlashAttention](/blog/flashattention/), exact attention with far fewer memory reads and writes.
