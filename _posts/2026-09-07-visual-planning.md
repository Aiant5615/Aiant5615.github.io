---
title: "Visual Planning: Let's Think Only with Images (Xu et al., 2025)"
date: 2026-09-07 14:14:00 +0900
categories: [paper, vision-llm]
image: /assets/og-vision-llm.png
tags: [RL, CV]
math: true
rating: 3
series: multimodal-llm
series_order: 11
description: "Planning as a sequence of generated images with no text at all, on a vision-only autoregressive model (LVM-7B). VPRL: stage 1 fine-tunes on random-walk trajectories to keep exploration, stage 2 runs GRPO with a progress reward (+1 toward the goal, 0 for a valid non-progressing move, −5 for an invalid one) parsed by a rules-based dynamics interpreter. FrozenLake/Maze/MiniBehavior exact match 80.6% vs 53.6% for text SFT and 43.7% for Gemini 2.5 Pro."
paper:
  title: "Visual Planning: Let's Think Only with Images"
  published: 2025-05
  authors: Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang, Anna Korhonen, Ivan Vulić
  venue: ICLR 2026
  year: 2025
  link: https://arxiv.org/abs/2505.11409
---

## One-line summary

If a task is about space, why reason about it in words? The paper defines **visual planning**: given a start image $$v_0$$, generate a sequence of images $$\hat{v}_1, \dots, \hat{v}_n$$ that *are* the plan, with no text anywhere, using a Large Vision Model (LVM-7B, pre-trained on image sequences only). Supervised fine-tuning on optimal trajectories (VPFT) roughly matches a text-planning baseline; **VPRL**, a two-stage RL recipe (random-walk initialization, then [GRPO](/blog/deepseekmath-grpo/) with a progress reward), reaches 80.6% average exact match on three grid navigation tasks against 53.6% for a fine-tuned Qwen2.5-VL text planner and 43.7% for Gemini 2.5 Pro with thinking, and it degrades less on larger grids and out-of-distribution sizes.

## Why it matters

The two previous papers keep the reasoning in text and improve how the image is *read*. This one tests the opposite hypothesis, that the reasoning trace itself can be visual, in the cleanest setting available: a model with no language pretraining, so any success cannot be text leaking in. It is also, to my knowledge, the first application of GRPO to image generation for planning, and its reward design (validity plus progress) is reusable.

## The paradigm (Section 2.1, Figure 1)

Three ways to answer a navigation question: direct prompting (text answer), multimodal CoT (verbal thoughts, possibly with images as aids, then a verbal answer), and visual planning (image, image, image). Formally, with a generative vision model $$\pi_{\theta}$$,

$$
\hat{v}_i \sim \pi_{\theta}\big(v_i \mid v_0, \hat{v}_1, \dots, \hat{v}_{i-1}\big)
$$

so actions are never named; they are implied by the transition between consecutive visual states. Each image is a sequence of visual tokens, which is what makes autoregressive generation and token-level RL possible.

## VPRL: two-stage RL for visual planning (Section 2.2, Figure 2)

**Stage 1: policy initialization on random walks.** Do *not* fine-tune on optimal paths first. Instead, collect random-walk trajectories $$(v_0, \dots, v_n)$$, form prefix–next-state pairs $$(v_{\le i}, v_{i+1})$$, and for each prefix pick a random *valid* next state $$\tilde{v}_{i+1}$$ as the target:

$$
L_{\text{VPFT}}(\theta) = -\mathbb{E}_{(v_{\le i},\ \tilde{v}_{i+1})}\big[\log \pi_{\theta}(\tilde{v}_{i+1} \mid v_{\le i})\big]
$$

The point is to make the model produce *valid, diverse* transitions with high entropy, i.e. to be a good explorer, and to produce visually coherent images. (The same loss on gold trajectories is the VPFT baseline.) Figure 6 shows Stage-1 VPRL has much higher action entropy and a lower invalid-action ratio than VPFT, which collapses onto the demonstrated moves.

**Stage 2: GRPO with a progress reward.** For a prefix $$v_{\le i}$$, the old policy samples a group of $$G$$ candidate next states $$\hat{v}^{(1)}_{i+1}, \dots, \hat{v}^{(G)}_{i+1}$$, each scored by $$r(v_i, \hat{v}^{(k)}_{i+1})$$; the advantage is the group-normalized reward $$A^{(k)} = (r^{(k)} - \text{mean})/\text{std}$$, and the objective is the standard clipped one with a KL term:

$$
J_{\text{VPRL}}(\theta) = \mathbb{E}\Big[\frac{1}{G}\sum_{k=1}^{G}\min\big(\rho^{(k)}A^{(k)},\ \text{clip}(\rho^{(k)}, 1-\epsilon, 1+\epsilon)A^{(k)}\big) - \beta\, D_{\text{KL}}(\pi_{\theta}\Vert\pi_{\text{ref}})\Big], \qquad \rho^{(k)} = \frac{\pi_{\theta}(\hat{v}^{(k)}_{i+1} \mid v_{\le i})}{\pi_{\theta_{\text{old}}}(\hat{v}^{(k)}_{i+1} \mid v_{\le i})}
$$

**Reward design.** Images are not decomposable like tokens, so the reward interprets the *transition*. Two components: a **dynamics interpreter** $$\mathcal{D} : \mathcal{V} \times \mathcal{V} \to \mathcal{A} \cup \mathcal{E}$$ that parses a pair of states into a valid action or an invalid one (walking through a wall, hallucinating a new object), and a **progress estimator** $$\mathcal{P} : \mathcal{V} \to \mathbb{N}$$ giving remaining steps to the goal. Partition candidates into

$$
\mathcal{A}_{\text{opt}} = \{a \in \mathcal{A} : \mathcal{P}(\hat{v}_{i+1}) < \mathcal{P}(v_i)\}, \qquad \mathcal{A}_{\text{nopt}} = \{a \in \mathcal{A} : \mathcal{P}(\hat{v}_{i+1}) \ge \mathcal{P}(v_i)\}, \qquad \mathcal{E}_{\text{inv}} = \mathcal{E}
$$

and pay

$$
r(v_i, \hat{v}_{i+1}) = \alpha_{\text{opt}}\,\mathbb{1}[\mathcal{D} \in \mathcal{A}_{\text{opt}}] + \alpha_{\text{nopt}}\,\mathbb{1}[\mathcal{D} \in \mathcal{A}_{\text{nopt}}] + \alpha_{\text{inv}}\,\mathbb{1}[\mathcal{D} \in \mathcal{E}_{\text{inv}}]
$$

with $$\alpha_{\text{opt}} = 1$$, $$\alpha_{\text{nopt}} = 0$$, $$\alpha_{\text{inv}} = -5$$: progress is rewarded, a valid detour is neutral, and breaking the environment's rules is punished hard. In the experiments both $$\mathcal{D}$$ and $$\mathcal{P}$$ are rule-based parsers of the rendered grids; the paper notes they could be learned dynamics models or holistic validators.

## Tasks, models, metrics (Section 3)

- **FrozenLake** (reach the goal without falling into holes), **Maze** (green start to red flag), **MiniBehavior** (reach the printer, pick it up, carry it to the table, drop it: two extra actions). Synthetic data with grid sizes 3–6.
- **Visual planners**: LVM-7B with VPFT or VPRL. **Text planners**: Qwen2.5-VL-Instruct-7B direct, CoT, SFT and GRPO, trained on the same data; Gemini 2.0 Flash and Gemini 2.5 Pro (thinking) zero-shot.
- **Exact match** requires the whole generated trajectory to coincide with one of the shortest optimal trajectories, where two states count as equal if they are reached by the same action from the previous state (a transition-level, not pixel-level, comparison). **Progress rate** is the fraction of consecutive correct steps from the start, maximized over optimal trajectories.

## Results (Tables 1–2, Figures 4–5)

| Model | FrozenLake EM | Maze EM | MiniBehavior EM | Avg EM |
|---|---|---|---|---|
| Gemini 2.5 Pro (think) | 72.0 | 21.5 | 37.6 | 43.7 |
| Qwen2.5-VL-7B SFT (text) | 68.6 | 60.9 | 31.3 | 53.6 |
| LVM-7B VPFT (visual SFT) | 75.4 | 59.0 | 33.8 | 56.1 |
| LVM-7B VPRL | 91.6 | 74.5 | 75.8 | 80.6 |

- Text planning is hard even for frontier models; SFT helps but RL on the text planner (Table 2) does not beat SFT under either the progress reward or the metric-as-reward, which the authors attribute to the modality gap of describing grids in words.
- VPRL gains over 20 points on average over VPFT, and Stage 1 alone is near random, so the gain is from the reward-driven Stage 2.
- Larger grids (Figure 5, Figure 10) hurt text planners sharply and visual planners mildly; on out-of-distribution grid sizes (Table 9) VPRL generalizes better than VPFT; failures are mostly suboptimal-but-valid rather than invalid moves (Table 6), consistent with the −5 penalty.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/visualplanning/figure1.png" width="919" height="509" alt="Figure 1 from Xu et al. (2025)" loading="lazy"><figcaption>Figure 1 of Xu et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (paradigms).** Top: direct prompting, input → verbal output. Middle: multimodal CoT, verbal thoughts with an image in the loop, then a verbal plan that is verbose and wrong ("go straight to the crossing, then turn slightly left…"). Bottom: visual planning, a chain of images of the route with no words.

<figure class="paper-fig"><img src="/assets/papers/visualplanning/figure2.png" width="1020" height="443" alt="Figure 2 from Xu et al. (2025)" loading="lazy"><figcaption>Figure 2 of Xu et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (VPRL).** The LVM takes the image prefix, decodes $$G$$ candidate next states; a parser turns each image pair into an action, which is compared with the optimal action and all valid actions at the current state and scored optimal / non-optimal / invalid; the rewards drive the GRPO policy update.

<figure class="paper-fig"><img src="/assets/papers/visualplanning/figure3.jpg" width="985" height="509" alt="Figure 3 from Xu et al. (2025)" loading="lazy"><figcaption>Figure 3 of Xu et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (tasks and traces).** Generated planning traces on FrozenLake, Maze and MiniBehavior, with examples of optimal moves, valid non-optimal moves, and invalid ones (moving through a wall, entering a table cell).

<figure class="paper-fig"><img src="/assets/papers/visualplanning/figure4.jpg" width="970" height="454" alt="Figure 4 from Xu et al. (2025)" loading="lazy"><figcaption>Figure 4 of Xu et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (one example).** A FrozenLake instance with the VPFT and VPRL image sequences alongside Gemini 2.5 Pro's and the SFT text planner's verbal outputs; the text plans violate constraints, VPRL's images reach the goal.

<figure class="paper-fig"><img src="/assets/papers/visualplanning/figure5.png" width="506" height="356" alt="Figure 5 from Xu et al. (2025)" loading="lazy"><figcaption>Figure 5 of Xu et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5 (difficulty).** Accuracy vs grid size 3–6 on FrozenLake for VPRL, VPFT, Gemini (think), Qwen SFT and Gemini CoT: the visual planners' bars stay high as the grid grows.

## Thoughts

- The experiment is deliberately narrow (rendered grids, rule-based reward parsers, a model that has never seen text), and that narrowness is what makes the result interpretable: on these tasks, a visual trace plus RL beats verbal reasoning by a wide margin.
- The Stage-1 idea, initialize on random walks rather than on demonstrations so that the policy explores, is the most transferable lesson, and it echoes the DPPO observation that RL wants a policy that covers the valid-action manifold.
- Whether this scales to natural images and to tasks without a parser is open; the next paper shows that a large video model already does something similar zero-shot.

Next: [Video models are zero-shot learners and reasoners](/blog/video-models-zero-shot/), where Veo 3 solves mazes and symmetry puzzles by generating frames, a "chain of frames".
