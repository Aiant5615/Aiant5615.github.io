---
title: "DiffusionNFT: Online Diffusion Reinforcement with Forward Process (Zheng et al., 2025)"
date: 2026-09-07 14:05:00 +0900
categories: [paper, diffusion-rl]
image: /assets/og-diffusion-rl.png
tags: [RL, CV, Diffusion]
math: true
rating: 5
series: diffusion-rl
series_order: 5
description: "Online RL for diffusion models without likelihoods or a reverse-process MDP: split samples into positive and negative sets by reward, define the improvement direction as a difference of velocity fields (reinforcement guidance), and optimize a single model through implicit positive and negative policies with a plain flow-matching loss on the forward process. Any solver, clean images only, CFG-free, up to 25× more efficient than Flow-GRPO."
paper:
  title: "DiffusionNFT: Online Diffusion Reinforcement with Forward Process"
  published: 2025-09
  authors: Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, Ming-Yu Liu
  venue: ICLR 2026
  year: 2025
  link: https://arxiv.org/abs/2509.16117
---

## One-line summary

Every earlier online method ([DDPO](/blog/ddpo/), [Flow-GRPO](/blog/flow-grpo/)) discretizes the *reverse* sampler to get per-step Gaussian likelihoods. DiffusionNFT (Negative-aware FineTuning) does RL on the *forward* noising process instead. Collect clean images with any black-box solver, score them with a reward in $$[0, 1]$$ interpreted as the probability of being "positive", and train one velocity model $$v_{\theta}$$ through two implicit models, a positive one $$(1 - \beta)v_{\text{old}} + \beta v_{\theta}$$ and a negative one $$(1 + \beta)v_{\text{old}} - \beta v_{\theta}$$, with the ordinary flow-matching loss weighted by $$r$$ and $$1 - r$$. The optimum moves $$v_{\text{old}}$$ along a provable improvement direction. On SD3.5-Medium without classifier-free guidance, GenEval goes 0.24 → 0.98 in 1k steps, where Flow-GRPO needs over 5k steps plus CFG to reach 0.95.

## Why it matters

This is the paper that makes the [CFGRL](/blog/cfgrl/) view (improvement = guidance) into an online, likelihood-free RL algorithm for generators. It removes the three restrictions of GRPO-style diffusion RL: the sampler must be a first-order SDE, the whole trajectory must be stored, and the model can drift away from a valid forward process. It is also natively off-policy, so sampling and training decouple without importance ratios.

## Background (Section 2)

Forward process $$x_t = \alpha_t x_0 + \sigma_t \epsilon$$ with velocity target $$v = \dot{\alpha}_t x_0 + \dot{\sigma}_t \epsilon$$ and loss $$\mathbb{E}[w(t)\lVert v_{\theta}(x_t, t) - v\rVert^2]$$; sampling follows the ODE $$dx_t/dt = v_{\theta}$$ (flow matching; Euler on it is DDIM). Rectified flow is the case $$\alpha_t = 1 - t,\ \sigma_t = t$$, $$v = \epsilon - x_0$$. Flow-GRPO's SDE with $$g_t = a\sqrt{t/(1-t)}$$ makes each step a Gaussian $$\pi_{\theta}(x_{t - \Delta t} \mid x_t)$$ so that GRPO applies. The paper's three complaints: **forward inconsistency** (optimizing only the reverse chain can turn the model into cascaded Gaussians unrelated to a forward process), **solver restriction** (rollouts must use that first-order SDE), and **complicated CFG integration** (two models, conditional and unconditional, to optimize).

## Problem setup (Section 3.1)

At each iteration sample $$K$$ images $$x_0^{1:K}$$ per prompt from $$\pi_{\text{old}}$$ and evaluate a reward $$r(x_0, c) \in [0, 1]$$ read as an optimality probability $$p(o = 1 \mid x_0, c)$$. This defines two imaginary datasets: each image lands in $$\mathcal{D}^{+}$$ with probability $$r$$, else in $$\mathcal{D}^{-}$$. Their population distributions are Bayes posteriors of the old policy:

$$
\pi^{+}(x_0 \mid c) = \frac{r(x_0, c)}{p_{\pi_{\text{old}}}(o = 1 \mid c)}\,\pi_{\text{old}}(x_0 \mid c), \qquad \pi^{-}(x_0 \mid c) = \frac{1 - r(x_0, c)}{1 - p_{\pi_{\text{old}}}(o = 1 \mid c)}\,\pi_{\text{old}}(x_0 \mid c)
$$

It is immediate that $$\pi^{+} \succ \pi_{\text{old}} \succ \pi^{-}$$ in expected reward, so training on $$\mathcal{D}^{+}$$ alone (rejection fine-tuning) improves. But it wastes the negatives, and the authors find that positive-only training collapses.

**Reinforcement guidance.** Instead of a target *point*, define a target *direction*. Set

$$
v^{*}(x_t, c, t) := v_{\text{old}}(x_t, c, t) + \frac{1}{\beta}\,\Delta(x_t, c, t)
$$

which has the shape of classifier-free guidance: $$\Delta$$ is the guidance direction and $$1/\beta$$ the strength.

## The improvement direction (Theorem 3.1)

Let $$v^{+}, v^{-}, v_{\text{old}}$$ be the velocity fields of $$\pi^{+}, \pi^{-}, \pi_{\text{old}}$$ (the flow models one would get by training on each). Then their differences are proportional:

$$
\Delta := \big(1 - \alpha(x_t)\big)\big[v_{\text{old}} - v^{-}\big] = \alpha(x_t)\big[v^{+} - v_{\text{old}}\big], \qquad \alpha(x_t) = \frac{\pi^{+}_t(x_t \mid c)}{\pi^{\text{old}}_t(x_t \mid c)}\,\mathbb{E}_{\pi_{\text{old}}(x_0 \mid c)}\, r(x_0, c) \in [0, 1]
$$

"Away from the negative model" and "toward the positive model" are the same direction at every noisy point $$x_t$$, up to a scalar. With $$\beta = \alpha(x_t)$$ in the target, $$v^{*} = v^{+}$$ exactly, so following $$\Delta$$ with the right strength reaches the improved policy $$\pi^{+}$$; with other strengths it interpolates or extrapolates along the same line. (This is the diffusion analogue of CFGRL's product policy: guidance toward the optimality-conditioned model.)

## Policy optimization without likelihoods (Theorem 3.2)

Rather than training separate $$v^{+}_{\theta}$$ and $$v^{-}_{\theta}$$, parameterize both *implicitly* through one model:

$$
v^{+}_{\theta} := (1 - \beta)\, v_{\text{old}} + \beta\, v_{\theta} \quad(\text{implicit positive}), \qquad v^{-}_{\theta} := (1 + \beta)\, v_{\text{old}} - \beta\, v_{\theta} \quad(\text{implicit negative})
$$

and minimize the reward-weighted pair of flow-matching losses on the *forward* process:

$$
L(\theta) = \mathbb{E}_{c,\ \pi_{\text{old}}(x_0 \mid c),\ t}\Big[r\,\lVert v^{+}_{\theta}(x_t, c, t) - v\rVert^2 + (1 - r)\,\lVert v^{-}_{\theta}(x_t, c, t) - v\rVert^2\Big]
$$

where $$x_t, v$$ come from noising the clean sample. With unlimited data and capacity the optimum is

$$
v_{\theta^{*}}(x_t, c, t) = v_{\text{old}}(x_t, c, t) + \frac{2}{\beta}\,\Delta(x_t, c, t)
$$

so the learned model is the old model pushed along the improvement direction with strength $$2/\beta$$. Reading the losses: a high-reward image pulls $$v^{+}_{\theta}$$ toward its velocity, which pushes $$v_{\theta}$$ *beyond* $$v_{\text{old}}$$ in that direction (since $$v_{\theta} = v_{\text{old}} + (v^{+}_{\theta} - v_{\text{old}})/\beta$$); a low-reward image pulls $$v^{-}_{\theta}$$ toward its velocity, which pushes $$v_{\theta}$$ the *opposite* way. Negatives are used as actively as positives.

Four properties follow (Section 3.2): **forward consistency** (it is a standard diffusion loss, so the density obeys the Fokker–Planck equation and $$x_t$$ stays correctly coupled to $$x_0$$); **solver flexibility** (rollouts can use ODE or high-order solvers; only $$(c, x_0, r)$$ are stored); **implicit guidance** (the guidance is baked into a single policy so RL can continue iteratively); **likelihood-free** (no variational bound, no reverse discretization, no estimation bias).

## Practical implementation (Section 3.3, Algorithm 1)

- **Optimality reward.** Raw scalar rewards are converted per prompt: $$r = \tfrac{1}{2} + \tfrac{1}{2}\,\text{clip}\big((r_{\text{raw}} - \mathbb{E}_{\pi_{\text{old}}} r_{\text{raw}}) / Z_c,\ -1,\ 1\big)$$, i.e. group-mean-centered and squashed to $$[0, 1]$$, the GRPO baseline in disguise.
- **Soft update of the sampling policy.** Being off-policy, the method uses an EMA $$\theta_{\text{old}} \leftarrow \eta_i \theta_{\text{old}} + (1 - \eta_i)\theta$$; fully on-policy ($$\eta = 0$$) learns fast then collapses, $$\eta \to 1$$ is stable but slow (Figure 8).
- **Adaptive loss weighting.** Replace $$w(t)$$ with a self-normalized $$x_0$$-regression, $$\lVert x_{\theta} - x_0\rVert^2 / \text{sg}(\text{mean}\lvert x_{\theta} - x_0\rvert)$$, borrowed from DMD distillation; faster training (Figure 9).
- **CFG-free.** CFG is interpreted as offline reinforcement guidance (conditional = positive, unconditional = negative), so the RL is run on the conditional model alone. The un-guided SD3.5-M starts at GenEval 0.24 and the RL recovers and surpasses what CFG would have given.
- Training uses LoRA ($$r = 32$$), 48 groups of $$K = 24$$ per epoch, 10-step rollouts for comparisons and 40-step for the multi-reward model, 40-step ODE evaluation.

## Results (Section 4)

- **Multi-reward joint training (Table 1, Figure 1b).** Optimizing GenEval, OCR, PickScore, ClipScore and HPSv2.1 together on the CFG-free 2.5B model: GenEval 0.94, OCR 0.91, PickScore 23.8, and the out-of-domain Aesthetic/ImageReward/UnifiedReward all above CFG-based SD3.5-L (8B) and FLUX.1-Dev (12B).
- **Head-to-head (Figures 1a, 6).** Single-reward runs vs Flow-GRPO in GPU-hours: 25× on GenEval, 24× on OCR, 8× on PickScore, 3× on HPSv2.1, with better final scores.
- **Ablations.** Removing the negative loss collapses training almost immediately; 1st-order SDE, 1st-order ODE and 2nd-order ODE rollouts all work (Figure 7), confirming solver freedom; $$\beta$$ trades speed for stability (Figure 10).

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/diffusionnft/figure1.png" width="831" height="402" alt="Figure 1 from Zheng et al. (2025)" loading="lazy"><figcaption>Figure 1 of Zheng et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (headline).** (a) GenEval vs GPU-hours: the orange DiffusionNFT curve reaches 0.98 within the first few hundred hours, the gray Flow-GRPO curve reaches 0.95 after about 2,500, starting from 0.63 (with CFG) vs 0.24 (without). (b) A radar chart over eight metrics where the CFG-free NFT model encloses SD3.5-M, SD3.5-M with CFG and FLUX.1-Dev.

<figure class="paper-fig"><img src="/assets/papers/diffusionnft/figure2.png" width="820" height="274" alt="Figure 2 from Zheng et al. (2025)" loading="lazy"><figcaption>Figure 2 of Zheng et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (forward vs reverse RL).** Top: GRPO/PPO store the discretized reverse SDE trajectory $$x_T \to \dots \to x_0$$ and compute per-step ratios $$\pi_{\theta}/\pi_{\text{old}}$$. Bottom: NFT samples $$x_0$$ with a black-box solver, re-noises it on the forward process, and compares $$v_{\theta}(x_t)$$ against $$v$$ with reward $$r(x_0)$$.

<figure class="paper-fig"><img src="/assets/papers/diffusionnft/figure3.png" width="525" height="241" alt="Figure 3 from Zheng et al. (2025)" loading="lazy"><figcaption>Figure 3 of Zheng et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (improvement direction).** Distributions $$\mathcal{D}^{-}$$, $$\mathcal{D}$$, $$\mathcal{D}^{+}$$ and their velocities $$v^{-}, v_{\text{old}}, v^{+}$$ on a line: the guidance $$\Delta$$ points from old toward positive and away from negative, with a color bar for $$r$$ from 0 to 1.

<figure class="paper-fig"><img src="/assets/papers/diffusionnft/figure4.png" width="1019" height="241" alt="Figure 4 from Zheng et al. (2025)" loading="lazy"><figcaption>Figure 4 of Zheng et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (dual objectives).** For a prompt and its $$K$$ images with rewards, the noisy $$x_t$$ feed both the implicit positive branch (weighted $$r$$) and the implicit negative branch (weighted $$1 - r$$), each a squared error against $$v$$, all through the single $$v_{\theta}$$ combined with $$v_{\text{old}}$$ by $$\beta$$.

<figure class="paper-fig"><img src="/assets/papers/diffusionnft/figure5.jpg" width="1001" height="638" alt="Figure 5 from Zheng et al. (2025)" loading="lazy"><figcaption>Figure 5 of Zheng et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 5 (qualitative).** SD3.5-M vs Flow-GRPO vs DiffusionNFT on a GenEval prompt (blue pizza, yellow glove), OCR prompts ("Google Research Pizza Cafe", "Street Art Rules"), a red car and a pig-shaped airship; the NFT row renders the text and the colors correctly with the most natural textures.

<figure class="paper-fig"><img src="/assets/papers/diffusionnft/figure6.png" width="1020" height="320" alt="Figure 6 from Zheng et al. (2025)" loading="lazy"><figcaption>Figure 6 of Zheng et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 6 (head-to-head).** OCR, PickScore and HPSv2.1 vs GPU-hours for the two methods with the efficiency multiples annotated.

## Thoughts

- The conceptual shift is from "the sampler is the policy" to "the reward defines a better data distribution, and a diffusion loss can be pointed at it." Once stated, forward-process RL is obviously simpler, and the implicit ± parameterization is a neat way to use negatives without a second network.
- It closes the loop with [CFGRL](/blog/cfgrl/): guidance is improvement, and here the guidance is *learned into the weights* online, iteration after iteration, so no test-time $$w$$ is needed and CFG itself becomes unnecessary.
- What to watch: the theory is for the population optimum; with finite $$K$$, EMA sampling and LoRA, the method is stable in practice but the exact link between $$\beta$$, $$\eta$$ and the effective KL to the reference is empirical. Still, this is the cleanest recipe in the series.

This is the last paper in the diffusion-RL series. The [overview post](/blog/diffusion-rl-lineage/) lists all five in order.
