---
title: "π*0.6: a VLA That Learns From Experience (RECAP, Physical Intelligence, 2025)"
date: 2026-09-07 14:21:00 +0900
categories: [paper, vla]
image: /assets/og-vla.png
tags: [Robotics, RL]
math: true
rating: 5
series: vla
series_order: 6
description: "RECAP: iterated offline RL for a flow-matching VLA. A distributional value function over 201 return bins trained by cross-entropy on all collected data, advantages thresholded into a binary improvement indicator that is fed to the policy as text, and the CFGRL-style argument that the indicator-conditioned policy is a KL-regularized improvement. Demonstrations, autonomous rollouts and human interventions in one recipe; laundry, espresso and box assembly with 2× throughput and half the failures."
paper:
  title: "π*0.6: a VLA That Learns From Experience"
  published: 2025-11
  authors: Physical Intelligence (Ali Amin, Raichelle Aniceto, Ashwin Balakrishna, Kevin Black, Ken Conley, Grace Connors, James Darpinian, Karan Dhabalia, Jared DiCarlo, Danny Driess, Michael Equi, Chelsea Finn, Sergey Levine, et al.)
  venue: arXiv
  year: 2025
  link: https://arxiv.org/abs/2511.14759
---

## One-line summary

Imitation caps a VLA at the quality of its demonstrations. RECAP (RL with Experience and Corrections via Advantage-conditioned Policies) lets the [π0](/blog/pi0/) lineage improve from its own deployments: collect autonomous rollouts labeled with outcomes plus optional human interventions, train a multi-task **value function** on everything collected so far, turn the resulting advantages into a binary "Advantage: positive/negative" token in the policy's prompt, and fine-tune the VLA on *all* the data with that token. At test time you ask for "positive". The model is π*0.6, an RL-adapted π0.6 (Gemma 3 4B backbone, 860M-parameter flow-matching action expert). Two or more iterations of RECAP more than double throughput on the hardest tasks (diverse laundry, espresso) and roughly halve failure rates, and the same recipe beats AWR and a PPO variant on the same on-robot data.

## Why it matters

This is [CFGRL](/blog/cfgrl/) scaled to a foundation-model VLA on real robots. It answers the question DPPO and Flow-GRPO raise for flow-matching policies (how to do RL when the log-likelihood is intractable) by not needing one: policy extraction is supervised learning with an extra input, so it uses off-policy demonstrations, interventions and old rollouts alike. It is also the first VLA paper to show RL improving *throughput* (speed), not only success, on multi-minute tasks.

## Preliminaries (Section III)

Policy $$\pi(a_t \mid o_t)$$, trajectory return $$R(\tau) = \sum_{t=0}^{T} r_t$$ (undiscounted), value $$V^{\pi}(o_t)$$, and an $$n$$-step advantage

$$
A^{\pi}(o_t, a_t) = \mathbb{E}\Big[\sum_{t'=t}^{t+N-1} r_{t'} + V^{\pi}(o_{t+N})\Big] - V^{\pi}(o_t)
$$

The regularized objective $$J(\pi, \pi_{\text{ref}}) = \mathbb{E}[\sum_t \gamma^t r_t] - \beta\, \mathbb{E}[D(\pi \Vert \pi_{\text{ref}})]$$ has the familiar KL solution $$\hat{\pi}(a \mid o) \propto \pi_{\text{ref}}(a \mid o)\exp(A^{\pi_{\text{ref}}}(o, a)/\beta)$$. The paper uses the less familiar cousin from CFGRL: if

$$
\hat{\pi}(a \mid o) \propto \pi_{\text{ref}}(a \mid o)\, p\big(I \mid A^{\pi_{\text{ref}}}(o, a)\big)^{\beta}, \qquad p(I \mid A) = \frac{g(A)}{\int g(A(o, a'))\, da'}
$$

with $$g$$ monotonically increasing, then $$J(\hat{\pi}) \ge J(\pi_{\text{ref}})$$. $$I$$ is an "improvement" event whose probability rises with advantage.

## RECAP (Section IV, Algorithm 1)

Three subroutines, repeated:

1. **Data collection.** Run the VLA on the task; label each episode with an outcome (the reward); optionally let an expert intervene with teleoperated corrections (human-gated DAgger style).
2. **Value function training.** Fit a multi-task, language-conditioned **distributional** value $$p_{\phi}(V \mid o_t, \ell) \in \Delta^{B}$$ over $$B = 201$$ bins of discretized return by cross-entropy on all data collected so far,

$$
\min_{\phi}\ \mathbb{E}_{\tau \in \mathcal{D}}\Big[\sum_{o_t \in \tau} H\big(R^{B}_t(\tau),\ p_{\phi}(V \mid o_t, \ell)\big)\Big]
$$

a Monte Carlo estimate of the *behavior* policy's value, extracted as $$V^{\pi_{\text{ref}}}(o_t, \ell) = \sum_b p_{\phi}(V = b \mid o_t)\, v(b)$$. Rewards are set so the value predicts normalized negative steps-to-success in $$(-1, 0)$$, with 0 at completion (Figure 4). It reuses the VLA architecture with a smaller VLM backbone.
3. **Advantage-conditioned policy training.** Apply Bayes' rule to the improvement event, $$p(I \mid A(o, a)) = \pi_{\text{ref}}(a \mid I, o)/\pi_{\text{ref}}(a \mid o)$$, so that

$$
\hat{\pi}(a \mid o, \ell) \propto \pi_{\text{ref}}(a \mid o, \ell)\Big(\frac{\pi_{\text{ref}}(a \mid I, o, \ell)}{\pi_{\text{ref}}(a \mid o, \ell)}\Big)^{\beta}
$$

and for $$\beta = 1$$, $$\hat{\pi} = \pi_{\text{ref}}(a \mid I, o, \ell)$$: the improved policy is just the reference policy *conditioned on the improvement indicator*. So train one model to represent both the unconditional and the $$I$$-conditioned policy, exactly as classifier-free guidance trains with and without a condition. The indicator is a hard threshold with a task-dependent margin,

$$
p\big(I \mid A^{\pi_{\text{ref}}}(o, a, \ell)\big) = \delta\big(A^{\pi_{\text{ref}}}(o, a, \ell) > \epsilon_{\ell}\big)
$$

and the policy loss is

$$
\min_{\theta}\ \mathbb{E}_{\mathcal{D}^{\pi_{\text{ref}}}}\Big[-\log \pi_{\theta}(a_t \mid o_t, \ell) - \alpha \log \pi_{\theta}(a_t \mid I_t, o_t, \ell)\Big], \qquad I_t = \mathbb{1}\big[A^{\pi_{\text{ref}}}(o_t, a_t, \ell) > \epsilon_{\ell}\big]
$$

Every sample is used, good and bad; the bad ones teach the model what "Advantage: negative" looks like, which is what lets it separate the modes. Human corrections are forced to $$I_t = \text{True}$$. The threshold $$\epsilon_{\ell}$$ replaces CFGRL's test-time weight $$\beta$$: large CFG weights pushed actions to the edges of their support and would not affect the autoregressive part of the model, so the trade-off is set during labelling instead. The advantage itself is $$A = r_t + \dots + V(o_{t+k}) - V(o_t)$$ from the value network (Figure 3).

Pre-training runs steps 2 and 3 on the whole multi-robot demonstration corpus (tens of thousands of hours) to get π*0.6 and a value $$V_{\text{pre}}$$; each task then gets a value $$V^{k}_{\ell}$$ and policy $$\pi^{k}_{\ell}$$ retrained from the pre-trained checkpoints on the growing task dataset after each collection round.

## The model (Section V)

π0.6 is π0.5 with more robot data, a Gemma 3 4B VLM, and an 860M action expert. It outputs a predicted sub-task text $$\hat{\ell}$$, then FAST-tokenized discrete actions $$a^{\ell}_{t:t+H}$$ (Knowledge-Insulation recipe, with a stop-gradient so the flow-matching expert does not perturb the VLM), and continuous 50 Hz action chunks $$a_{t:t+H}$$ by flow matching; the log-likelihood factorizes as $$\log \pi(\hat{\ell} \mid o, \ell) + \log \pi(a^{\ell} \mid o, \ell, \hat{\ell}) + \log \pi(a \mid o, \ell, \hat{\ell})$$. π*0.6 inserts the text "Advantage: positive" or "Advantage: negative" after $$\hat{\ell}$$ and before the actions, so only the action terms are affected. The continuous term is trained by the flow-matching loss, which (via the diffusion lower-bound argument) stands in for the intractable likelihood:

$$
\log \pi_{\theta}(a_{t:t+H}, a^{\ell}_{t:t+H} \mid I_t, o_t, \ell, \hat{\ell}) \ \ge\ \mathbb{E}_{\eta, \omega}\Big[\log p_{\theta}(a^{\ell}_{t:t+H} \mid I_t, o_t, \ell, \hat{\ell}) - \alpha_{\eta}\big\lVert \omega - a_{t:t+H} - f_{\theta}(a^{\eta, \omega}_{t:t+H}, I_t, o_t, \ell, \hat{\ell})\big\rVert^2\Big]
$$

## Results (Section VI)

Tasks: laundry (T-shirts and shorts; 11 diverse item types measured on button-up shirts; a strict single-shirt variant), double-espresso on a professional machine (grind, tamp, lock, extract, serve within 200 s), and factory box assembly (fold, label, place in a crate within 600 s). Baselines: pre-trained π0.5, π0.6 (SFT only), RL-pretrained π*0.6, π*0.6 offline RL + SFT (advantage fixed to True on demonstrations), and the full π*0.6 with on-robot data; plus AWR and a DPPO/FPO-style PPO using the same on-robot data.

- **Throughput and success (Figures 7–8).** Each RECAP stage helps. From offline RL + SFT to the final model, throughput more than doubles on diverse laundry and espresso and failure rates roughly halve; success is above 90% on all tasks but diverse laundry. On box assembly the per-stage breakdown (pick sheet, build, label, place) is highest at every stage.
- **Iterations (Figures 9–10).** Throughput keeps rising over rounds; laundry saturates in success but gets faster; box assembly dips then improves.
- **Policy extraction (Figure 11).** On T-shirt laundry, RECAP's throughput is far above AWR and PPO trained on the same data; success rates are closer.
- **Failure removal (Figure 12).** On the strict single-shirt task, RECAP removes failure modes (collar down, crumpled fold) that SFT keeps repeating.
- Deployments: espresso for 13 hours straight, laundry in a new home for over two hours, boxes used for real packaging.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/pistar06/figure1.jpg" width="1316" height="566" alt="Figure 1 from Physical Intelligence (2025)" loading="lazy"><figcaption>Figure 1 of Physical Intelligence (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (RECAP loop).** Diverse robot data, sub-task commands and web data pre-train the VLA with an advantage input; rollouts and interventions are labeled; the value function is retrained; the VLA is re-conditioned, and the loop repeats.

<figure class="paper-fig"><img src="/assets/papers/pistar06/figure3.png" width="659" height="527" alt="Figure 3 from Physical Intelligence (2025)" loading="lazy"><figcaption>Figure 3 of Physical Intelligence (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (architecture).** The π*0.6 VLA (SigLIP 400M + Gemma 4B backbone, 860M action expert) reads images, prompts and metadata, emits language sub-tasks, discretized actions and continuous actions; a separate value function (SigLIP + Gemma 270M) produces $$V$$, from which $$A = r_t + \dots + V(o_{t+k}) - V(o_t)$$ is binarized as $$A > \epsilon$$ and fed back as the advantage token.

<figure class="paper-fig"><img src="/assets/papers/pistar06/figure4.png" width="1293" height="317" alt="Figure 4 from Physical Intelligence (2025)" loading="lazy"><figcaption>Figure 4 of Physical Intelligence (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 4 (value traces).** Two episodes with value over time: a laundry episode where the value dips when the left arm crumples the folded shirt and recovers after the fold (green), and a failed fridge task where tipping over a water filter produces a drop (red).

<figure class="paper-fig"><img src="/assets/papers/pistar06/figure7.png" width="1284" height="365" alt="Figure 7 from Physical Intelligence (2025)" loading="lazy"><figcaption>Figure 7 of Physical Intelligence (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 7 (throughput).** Successes per hour on four tasks for π0.5, π0.6, RL-pretrained π*0.6, offline RL + SFT, and the final model (yellow), which is tallest everywhere.

<figure class="paper-fig"><img src="/assets/papers/pistar06/figure8.png" width="1286" height="318" alt="Figure 8 from Physical Intelligence (2025)" loading="lazy"><figcaption>Figure 8 of Physical Intelligence (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 8 (success rates).** Same models, absolute success; for box assembly, success per sub-stage.

<figure class="paper-fig"><img src="/assets/papers/pistar06/figure11.png" width="657" height="292" alt="Figure 11 from Physical Intelligence (2025)" loading="lazy"><figcaption>Figure 11 of Physical Intelligence (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 11 (extraction methods).** Throughput and success on T-shirt laundry for pre-trained, offline RL + SFT, RECAP (ours), AWR and PPO; RECAP roughly doubles the next-best throughput.

## Thoughts

- The elegant part is what is *not* here: no importance ratios, no likelihoods of a flow model, no separate residual policy. The RL lives in a value network and one token. That is why the recipe absorbs demonstrations, corrections and rollouts from any earlier policy.
- The threshold instead of a guidance weight is a practical insight worth remembering: with a mixed discrete/continuous output, guidance cannot touch the discrete part, and a labelling-time margin is easier to tune than an inference-time extrapolation.
- The value function is on-policy Monte Carlo, so improvement per iteration is limited to "do more of what worked in this dataset". The authors flag off-policy critics as future work; the next paper takes a different route and distills this RL data into a generalist through richer prompts.

Next: [π0.7](/blog/pi07/), a steerable generalist that trains on demonstrations, failures and RL-specialist rollouts alike by conditioning on episode metadata, subgoal images and detailed language.
