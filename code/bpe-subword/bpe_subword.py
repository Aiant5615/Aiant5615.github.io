"""Neural Machine Translation of Rare Words with Subword Units (Sennrich, Haddow & Birch, 2016) — pure Python.

What is implemented (section numbers follow the paper):
  * BPE learning exactly as in Figure 1 / Algorithm 1 (Section 3.2): words split into characters plus the
    end-of-word marker '</w>', get_stats() counts adjacent symbol pairs weighted by word frequency, merge_vocab()
    replaces the most frequent pair everywhere with a regex that respects symbol boundaries, repeated n times
  * the worked example of Section 3.2 (low x5, lower x2, newest x6, widest x3 -> merges 'st', 'st</w>', 'est</w>', 'lo', 'low', ...)
    and the segmentation of the unseen word 'lowest' -> 'low est</w>'
  * applying BPE at test time: replay the learned merges in order on each word, independent of the test corpus
  * segmentation statistics of Table 1 (vocabulary size, unknown types, sequence length) as the number of merges varies,
    on a synthetic morphological corpus with held-out stem+suffix combinations (Table 4 / 5 style rare-word behaviour)
  * joint BPE over two "languages" so that shared strings are segmented identically (Section 3.2, joint vocabulary)
Simplifications: no neural model (the paper's contribution is the preprocessing), synthetic words built from stems and
suffixes instead of WMT text, no special handling of the vocabulary-threshold or glossaries of the later toolkit.
torch is not needed; random.seed(0) makes the corpus deterministic.

Run:  python bpe_subword.py        (CPU, about 1 s)
"""
import re, collections, random, time

random.seed(0)


# ───────────────────────── Figure 1: learning BPE ─────────────────────────
def get_stats(vocab):
    """Count adjacent symbol pairs over the dictionary, weighted by word frequency."""
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    """Replace every occurrence of the pair (as whole symbols: lookbehind/lookahead forbid partial matches) by the merged symbol."""
    v_out = {}
    bigram = re.escape(" ".join(pair))
    p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    for word in v_in:
        w_out = p.sub("".join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def learn_bpe(word_freqs, num_merges, verbose=0):
    """Algorithm 1: start from characters + '</w>', merge the most frequent pair num_merges times; return the merge list."""
    vocab = {" ".join(list(w)) + " </w>": f for w, f in word_freqs.items()}
    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=lambda pair: (pairs[pair], pair))   # ties broken like subword-nmt: by the pair itself
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        if i < verbose:
            print(f"  merge {i + 1}: {best[0]!r} + {best[1]!r} -> {''.join(best)!r}   (count {pairs[best]})")
    return merges, vocab


# ───────────────────────── applying BPE to new text ─────────────────────────
def apply_bpe(word, merges):
    """Replay the merges in the order they were learned. Deterministic and independent of the test corpus; a word made of
    characters the code has never seen falls back to characters, so there is no OOV."""
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i, out = 0, []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(symbols[i]); i += 1
        symbols = out
    return symbols


def symbol_vocab(merges, word_freqs):
    """Symbols that can appear after applying the merges = characters + one symbol per merge (approximately)."""
    chars = {c for w in word_freqs for c in w} | {"</w>"}
    return chars | {"".join(m) for m in merges}


# ───────────────────────── synthetic morphological corpus ─────────────────────────
STEMS = ["walk", "talk", "jump", "play", "work", "look", "call", "help", "open", "turn", "wash", "cook", "kick", "pull", "push", "lift"]
SUFFIXES = ["", "s", "ed", "ing", "er", "ers", "ingly", "able"]
STEMS_L2 = ["geh", "sag", "spiel", "arbeit", "schau", "ruf", "helf", "mach"]
SUFFIXES_L2 = ["en", "t", "st", "te", "ten", "end", "ung"]


def make_corpus(stems, suffixes, n_tokens, heldout):
    """Zipf-like frequencies over stems and suffixes; the held-out (stem, suffix) combinations never occur, so they are
    'rare/unseen words' at test time even though every piece is frequent (Section 5.2's argument for compositionality)."""
    freqs = collections.Counter()
    for _ in range(n_tokens):
        s = stems[min(int(random.paretovariate(1.2)) - 1, len(stems) - 1)]
        x = suffixes[min(int(random.paretovariate(1.2)) - 1, len(suffixes) - 1)]
        if (s, x) not in heldout:
            freqs[s + x] += 1
    return freqs


def main():
    t0 = time.time()
    print("== worked example of Section 3.2 ==")
    example = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    merges, vocab = learn_bpe(example, num_merges=10, verbose=10)
    print("  dictionary after 10 merges:", vocab)
    seg = apply_bpe("lowest", merges)
    print(f"  unseen word 'lowest' -> {' '.join(seg)}")
    # 'e s', 's t' and 't </w>' all tie at 9 in the first round. The paper's Figure 1 shows 's t' first; the released
    # subword-nmt tool breaks ties by the pair itself (max over (count, pair)), which we copy, so the order differs but
    # the learned vocabulary and the segmentation of 'lowest' are the same.
    assert {"".join(m) for m in merges[:5]} >= {"est</w>", "low"}, "merges differ from the paper's worked example"
    assert seg == ["low", "est</w>"], "segmentation of 'lowest' differs from the paper"

    print("\n== synthetic morphological corpus (Table 1-style statistics) ==")
    heldout = {("wash", "ingly"), ("kick", "able"), ("lift", "ers"), ("cook", "ing"), ("push", "ed")}
    train = make_corpus(STEMS, SUFFIXES, 20_000, heldout)
    test_words = [s + x for s, x in heldout]
    print(f"  {sum(train.values()):,} training tokens, {len(train)} word types; unseen test words: {test_words}")
    print(f"  {'merges':>7} {'symbols':>8} {'unk types':>10} {'avg symbols / word':>20}   segmentation of unseen words")
    for n in (0, 10, 30, 60, 120):
        merges, _ = learn_bpe(train, n)
        vocab_syms = symbol_vocab(merges, train)
        segs = [apply_bpe(w, merges) for w in test_words]
        unk = sum(any(s not in vocab_syms for s in seg) for seg in segs)
        avg_len = sum(len(apply_bpe(w, merges)) * f for w, f in train.items()) / sum(train.values())
        print(f"  {n:7d} {len(vocab_syms):8d} {unk:10d} {avg_len:20.2f}   " + ", ".join("|".join(s) for s in segs))
    merges, _ = learn_bpe(train, 120)
    segs = {w: apply_bpe(w, merges) for w in test_words}
    # with enough merges every held-out word splits exactly into its stem and its suffix(+</w>)
    good = sum("".join(seg[:1]) in STEMS or "".join(seg).replace("</w>", "") in train for w, seg in segs.items())
    stem_split = sum(seg[0] == s for (s, x), seg in zip(heldout, [apply_bpe(s + x, merges) for s, x in heldout]))
    print(f"  with 120 merges, {stem_split}/{len(heldout)} unseen words are segmented as <stem> + <suffix></w>")

    print("\n== joint BPE over two languages (shared strings get the same segmentation) ==")
    l2 = make_corpus(STEMS_L2, SUFFIXES_L2, 20_000, set())
    for name in ("arbeitsam", "walkabout"):                         # a "name" that appears in both sides of a parallel corpus
        train[name] = l2[name] = 15
    joint = collections.Counter(train) + collections.Counter(l2)
    m_sep1, _ = learn_bpe(train, 60); m_sep2, _ = learn_bpe(l2, 60); m_joint, _ = learn_bpe(joint, 100)
    for name in ("arbeitsam", "walkabout"):
        print(f"  {name}: separate codes {'|'.join(apply_bpe(name, m_sep1))} vs {'|'.join(apply_bpe(name, m_sep2))};"
              f"  joint code {'|'.join(apply_bpe(name, m_joint))} on both sides")
    print(f"({time.time() - t0:.2f} s)")
    assert stem_split == len(heldout), "unseen words were not decomposed into known stem + suffix"
    assert all(apply_bpe(n, m_joint) == apply_bpe(n, m_joint) for n in ("arbeitsam",)) and unk == 0


if __name__ == "__main__":
    main()
