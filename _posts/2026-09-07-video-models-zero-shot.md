---
title: "Video Models Are Zero-Shot Learners and Reasoners (Wiedemer et al., 2025)"
date: 2026-09-07 14:15:00 +0900
categories: [paper, vision-llm]
image: /assets/og-vision-llm.png
tags: [CV, Video]
math: true
rating: 3
series: multimodal-llm
series_order: 12
description: "Prompt Veo 3 with an image and an instruction, and it performs edge detection, segmentation, editing, physics simulation, and early visual reasoning (maze solving, symmetry completion, analogies) without any task-specific training. The four-level capability hierarchy, the best-frame vs last-frame pass@k protocol, the chain-of-frames analogy to chain-of-thought, and the numbers against Veo 2 and Nano Banana."
paper:
  title: "Video Models Are Zero-Shot Learners and Reasoners"
  published: 2025-09
  authors: Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, Robert Geirhos
  venue: arXiv (Google DeepMind)
  year: 2025
  link: https://arxiv.org/abs/2509.20328
---

## One-line summary

LLMs became general-purpose because a large generative model trained on web-scale text can be *prompted* into tasks it was never trained for. The same primitives (large generative model, web-scale video) now exist for vision, so the paper asks whether a video model is a zero-shot vision generalist. The method is deliberately minimal: give Veo 3 an input image as the first frame and a text instruction, get an 8-second 720p video, and check whether some frame solves the task. Across 18,384 generated videos, 62 qualitative tasks and 7 quantitative ones, Veo 3 does edge detection, segmentation, keypoints, de-blurring, physics, editing, and simple visual reasoning like maze solving, symmetry completion and visual analogies, well below bespoke models but far above Veo 2 released six months earlier. Frame-by-frame generation is proposed as a **chain of frames (CoF)**, the visual counterpart of chain-of-thought.

## Why it matters

It is the empirical case for the thesis that [Visual Planning](/blog/visual-planning/) tested in a toy setting: reasoning in the image domain, step by step across frames, is a real capability of large video generators, and it emerges without task-specific training. It also sets up a benchmark protocol (best frame vs last frame, pass@k) that later video-reasoning work uses.

## Method (Section 2)

- **Models.** Veo 3 (`veo-3.0-generate-preview`) and Veo 2 via the Vertex AI API; image-to-video with a text prompt; 16:9, 720p, 24 fps, 8 s. The API applies an LLM prompt rewriter, so the system is treated as one black box; to check that reasoning is not coming from the rewriter, the authors verified that Gemini 2.5 Pro alone cannot reliably solve the key tasks (robot navigation, mazes, symmetry) from the image.
- **Protocol.** Qualitative tasks: 12 samples each, success rate judged by the authors (Figure 1). Quantitative tasks: pass@k over 10 videos, reported for the **best frame** (any frame, a ceiling that is not identifiable in advance) and the **last frame** (predetermined but Veo tends to keep animating after solving), with Nano Banana, an image editing model, as a reference where applicable.

## The capability hierarchy (Section 3, Figure 2)

1. **Perception.** Edge detection, segmentation, keypoint localization, super-resolution, blind deblurring/denoising, low-light enhancement, conjunctive visual search, and interpreting ambiguous images (the dalmatian illusion, a texture–shape cue conflict, Rorschach blots). Apart from denoising, none of these is a training objective of a video model.
2. **Modeling.** Intuitive physics: flammability, rigid and soft bodies, air resistance (Earth vs Moon), buoyancy, refraction and reflection, additive vs subtractive color mixing; Visual Jenga (remove objects in a physically plausible order); which objects fit in a backpack; category distinctions (toys vs a laptop); Omniglot-style pattern recognition and parsing; memory of world state across camera moves.
3. **Manipulation.** Background removal, style transfer, colorization, inpainting/outpainting, text editing, doodle-guided edits, scene composition, novel view synthesis, transfiguration, selfie-to-headshot; simulated dexterous manipulation (opening a jar), affordances, drawing, rolling a burrito.
4. **Reasoning across space and time.** Graph traversal, visual BFS on a tree, sequence completion, color connecting, shape fitting, sorting numbers, tool use, simple Sudoku, mazes and robot navigation, rule extrapolation. The claim: because changes are applied frame by frame, video generation parallels chain-of-thought, hence *chain of frames*.

## Quantitative results (Section 4)

- **Edge detection (Figure 3).** BIPEDv2, 50 images, OIS. Veo 3 0.77 pass@10 (best frame) vs Veo 2 0.57 and task-specific SOTA 0.90; many Veo 3 edge maps are *more* detailed than the ground truth (foliage, tire treads), which the metric penalizes.
- **Segmentation (Figure 4).** Class-agnostic instance segmentation on 50 easy LVIS images, mIoU: Veo 3 0.74 (best frame, green background) vs Nano Banana 0.73 and Veo 2 0.52; a green background beats a white one (0.74 vs 0.66), presumably from green-screen priors.
- **Object extraction (Figure 5).** Line up all the animals in a row; count connected components in the last frame. Veo 3 up to 93% pass@10; Veo 2 near chance.
- **Image editing (Figure 6).** 30 Emu-edit samples rated by humans for fidelity and precision; Veo 3 preserves textures well but tends to add camera motion and animate people.
- **Maze solving (Figure 7).** A red circle must reach a green one along the white path; correctness is checked automatically. 5×5 grids: Veo 3 78% pass@10 vs Veo 2 14%; 7×7: 40% vs 8%; 9×9 falls to 22%. Nano Banana matches or beats Veo 3 on rectangular mazes but fails entirely on irregular ones (0 vs 75%); Gemini 2.5 Pro solves small mazes from ASCII text better than Veo 3 but collapses on 9×9 and on image input.
- **Visual symmetry (Figure 8).** Reflect a pattern across the vertical axis; every cell must be right. Shapes: Veo 3 88% best-frame pass@10 (Nano Banana 37%); random patterns: 100% best frame, 72% last frame; Veo 2 near zero.
- **Visual analogies (Figure 9).** KiVA, four transformations × 50 samples, pass@1 on the last frame: color 95%, resize 67%, reflect 29%, rotate 19%, vs Veo 2's 68/40/23/22; reflection and rotation are still near chance.

Two general trends: a large jump from Veo 2 to Veo 3 across the board, and large gains from $$k = 1$$ to $$k = 10$$, meaning a correct solution exists among a handful of samples even when the first attempt fails.

## Figures, explained

<figure class="paper-fig"><img src="/assets/papers/videomodels/figure1.png" width="1208" height="434" alt="Figure 1 from Wiedemer et al. (2025)" loading="lazy"><figcaption>Figure 1 of Wiedemer et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 1 (62 tasks).** A bar chart of Veo 3's success rate over 12 samples on each qualitative task, grouped into perception (blue), modeling (purple), manipulation (magenta) and reasoning (pink); perception tasks are near 1, reasoning tasks are the most variable.

<figure class="paper-fig"><img src="/assets/papers/videomodels/figure2.jpg" width="1196" height="622" alt="Figure 2 from Wiedemer et al. (2025)" loading="lazy"><figcaption>Figure 2 of Wiedemer et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 2 (examples).** Eight strips: super-resolution and conjunctive search (perception), buoyancy and world-state memory (modeling), 3D-aware reposing and jar opening (manipulation), robot navigation and rule extrapolation (reasoning).

<figure class="paper-fig"><img src="/assets/papers/videomodels/figure3.jpg" width="1208" height="637" alt="Figure 3 from Wiedemer et al. (2025)" loading="lazy"><figcaption>Figure 3 of Wiedemer et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 3 (edges).** An original street photo, Veo 3's generated edge map, the ground truth, and OIS pass@k curves for Veo 3, Veo 2 and Nano Banana, best frame (left) and last frame (right).

<figure class="paper-fig"><img src="/assets/papers/videomodels/figure7.png" width="1208" height="604" alt="Figure 7 from Wiedemer et al. (2025)" loading="lazy"><figcaption>Figure 7 of Wiedemer et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 7 (mazes).** Example 5×5, 7×7, 9×9 and irregular mazes with pass@k curves for Veo 3, Veo 2, Nano Banana, and Gemini 2.5 Pro from image (I2T) and from text (T2T).

<figure class="paper-fig"><img src="/assets/papers/videomodels/figure8.png" width="1208" height="607" alt="Figure 8 from Wiedemer et al. (2025)" loading="lazy"><figcaption>Figure 8 of Wiedemer et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 8 (symmetry).** Half-filled shapes and random patterns with Veo 3's completed frames, and pass@k for best/last frame on both splits.

<figure class="paper-fig"><img src="/assets/papers/videomodels/figure9.png" width="1208" height="250" alt="Figure 9 from Wiedemer et al. (2025)" loading="lazy"><figcaption>Figure 9 of Wiedemer et al. (2025), reproduced from the paper for commentary.</figcaption></figure>

- **Figure 9 (analogies).** The KiVA layout (three panels and a missing fourth), Veo 3's completion, and pass@1 bars per transformation for Veo 3, Veo 2 and chance.

## Thoughts

- This is a capability survey, not a method paper, and it is honest about that: no fine-tuning, no architecture, and a success criterion judged by the authors for most tasks. Its value is the breadth of the probe and the Veo 2 → Veo 3 delta, which is the argument that the trend, not the current number, matters.
- The "chain of frames" framing is the interesting claim. Mazes and symmetry are exactly the tasks where [Visual Planning](/blog/visual-planning/) found visual traces beat verbal ones, and here the same effect appears zero-shot in a web-scale model (and Nano Banana's failure on irregular mazes suggests the *sequential* frames matter).
- What is missing is any account of the mechanism, and any embodied grounding: the videos look physical but nothing checks that the physics is right beyond human judgment. The next paper builds the benchmark and training for exactly that.

Next: [Cosmos-Reason1](/blog/cosmos-reason1/), which trains VLMs for physical common sense and embodied reasoning with SFT and RL on verifiable, self-supervised video questions.
