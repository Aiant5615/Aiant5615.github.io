---
title: "Neural Machine Translation of Rare Words with Subword Units (Sennrich, Haddow & Birch, 2016)"
date: 2026-09-07 09:07:00 +0900
categories: [paper]
tags: [NLP]
math: true
rating: 4
series: transformer-lineage
series_order: 7
description: "Byte-pair encoding for tokenization. The merge algorithm worked through by hand, why it beats a back-off dictionary, joint vs separate vocabularies, and how the rare-word F1 tables should be read."
paper:
  title: Neural Machine Translation of Rare Words with Subword Units
  authors: Rico Sennrich, Barry Haddow, Alexandra Birch (Edinburgh)
  venue: ACL
  year: 2016
  link: https://arxiv.org/abs/1508.07909
---

## One-line summary

Represent words as sequences of frequent character chunks learned by repeatedly merging the most common adjacent symbol pair (byte-pair encoding), so that a fixed vocabulary of 30–90k symbols can encode *any* word, including names, compounds and morphology, and the translation model needs no unknown-word machinery.

## Why it matters

Every model so far had a hard vocabulary cap (30k–160k words) and replaced the rest with an UNK token, then patched the output with dictionaries. Sennrich et al. move the problem to preprocessing. Their BPE is the tokenizer of the [Transformer](/blog/2026/09/07/transformer/) paper (37k joint vocabulary), of [GPT](/blog/2026/09/07/gpt1/), of [GPT-2 and 3](/blog/2026/09/07/gpt2/) in byte-level form, and of most LLMs since. Even T5's SentencePiece is a cousin.

## The algorithm

Byte-pair encoding was a 1994 compression trick: replace the most frequent pair of bytes with an unused byte, repeat. The paper adapts it to *learn a segmentation* instead of compressing:

1. Split the training text into words; count word frequencies. Represent each word as its characters plus an end-of-word marker `</w>` (so "er" at the end of "newer" and "er" inside "error" are different symbols, letting the model restore word boundaries).
2. Count all adjacent symbol pairs, weighted by word frequency.
3. Merge the most frequent pair everywhere it occurs into a new symbol. Record the merge.
4. Repeat for a fixed number of merge operations $$n$$ (the only hyperparameter).

The final vocabulary size is roughly the character set plus $$n$$, and the number of merges controls the trade-off between short sequences (large vocabulary) and good generalization (small vocabulary). The paper's Figure 1 is the whole learning procedure as a dozen lines of Python: a `defaultdict` of pair counts, a `max` to pick the best pair, a regex substitution to apply the merge.

**Worked example (from the paper).** Dictionary: `low` ×5, `lower` ×2, `newest` ×6, `widest` ×3. Each word starts as characters, e.g. `l o w </w>`. Pair counts: `s t` occurs 9 times (6 + 3), the most frequent, so merge → `st`. Next, `st </w>` (9) → `st</w>`. Then `e st</w>` (9) → `est</w>`, then `l o` (7) → `lo`, then `lo w` (7) → `low`, and so on. After training, the word `lowest`, never seen, is segmented deterministically as `low est</w>`, two symbols the model already knows.

**Applying BPE at test time** replays the learned merges in order on each new word, which is why it is reproducible and does not depend on the test corpus. Unlike a fixed dictionary, there is no out-of-vocabulary word: worst case, a word becomes its characters.

## Joint vs separate vocabularies

Learning one BPE code on the concatenation of source and target ("joint BPE") makes the two sides segment shared strings the same way, which helps the model copy names and numbers and lets the attention align subword to subword. For unrelated scripts (English–Russian) the paper first transliterates Cyrillic so the joint code can still share pieces.

## What it replaces

The competing approach was a **back-off dictionary**: translate with a word-level NMT model, and when it outputs UNK, look up the source word aligned by attention in a bilingual dictionary or copy it. That handles names but not morphology (a rare German compound must be *composed*, not copied), and the attention-based lookup is noisy. Character-level models solve this but make sequences 5–10× longer. BPE sits in between.

## Figures and tables, explained

- **Figure 1 (the code).** Learning BPE in Python: `get_stats` counts pairs, `merge_vocab` applies a merge with a regex that respects symbol boundaries. Worth reading closely; a tokenizer is genuinely this small.
- **Table 1 (segmentation statistics).** For each segmentation (word-level, character n-gram variants, BPE with 60k merges, joint BPE 90k), the vocabulary size and the number of unknown types on the test set: BPE variants reach 0 unknowns while keeping sentences only modestly longer.
- **Table 2 (English–German results).** Compares WDict (dictionary back-off), WUnk (no special handling), C2-50k (character bigrams), BPE-60k, and BPE-J90k. BPE-J90k gains up to +1.1 BLEU and +2.1 CHRF3 over the dictionary baseline on newstest2015; the character-bigram variant is close behind, showing that the gain is from subword modeling, not from BPE specifically.
- **Table 3 (English–Russian).** Larger gains (+1.3 BLEU, +2.5 CHRF3) because Russian morphology is richer and dictionary back-off is weaker across scripts.
- **Table 4 (unigram F1 by frequency rank).** F1 on words in frequency bins (top 50k, then rare, then out-of-vocabulary). For rare and OOV words, BPE roughly doubles the dictionary approach's F1; for frequent words all systems tie. This table is the mechanism behind the BLEU gain.
- **Table 5 (examples).** German compounds ("Gesundheitsforschungsinstitute" → `Gesundheits|forsch|ungsin|stitute`) and Russian transliterations that word-level models could not produce.

## Results

On WMT'15 English→German, BPE-J90k reached 22.8 BLEU on newstest2015 (vs 22.0 for the dictionary baseline); on English→Russian, 20.4 vs 19.1. Rare-word unigram F1 improved from 36.8% to 41.8% (En–De) and from 26.5% to 29.7% (En–Ru). Ensembles improved all numbers further.

## Thoughts

- The paper is careful to say character n-grams work about as well; what mattered was the *idea* of open-vocabulary subwords. BPE won on simplicity.
- GPT-2 later applies the same merges to *bytes* rather than Unicode characters, so the base vocabulary is 256 and any string is encodable.
- A lot of LLM behaviour (arithmetic on digits, spelling, tokens that split mid-word) traces back to this frequency-driven segmentation. When a model "cannot count letters", this table of merges is why.

Next, a detour into vision: [ResNet](/blog/2026/09/07/resnet/) and the residual connection every Transformer block uses.
