"""Efficient Estimation of Word Representations in Vector Space (Mikolov, Chen, Corrado & Dean, 2013) — PyTorch.

What is implemented (section numbers follow the ICLR paper; NEG follows the NeurIPS follow-up, Mikolov et al. 2013b):
  * Skip-gram: from the centre word w_t predict each context word w_{t+j}, |j| <= c, with an input vector v_w
    and an output vector v'_w per word, maximising 1/T sum_t sum_j log p(w_{t+j} | w_t)                  (Section 3.2)
  * dynamic window: nearer words are sampled more often by drawing the window size R ~ U{1..c} per position  (3.2)
  * CBOW: average the context vectors (projection layer, no hidden layer) and predict the centre word         (3.1)
  * negative sampling in place of the full softmax:
      log sigma(v'_{w_O}^T v_{w_I}) + sum_{i=1..k} E_{w_i ~ P_n(w)} log sigma(-v'_{w_i}^T v_{w_I}),
    with P_n(w) proportional to unigram(w)^{3/4}                                             (Mikolov et al. 2013b, eq. 4)
  * the analogy test: X = vec(b) - vec(a) + vec(c), answer = nearest word by cosine excluding a, b, c   (Section 4)
Simplifications: a synthetic corpus with planted (concept x attribute) structure instead of Google News, no
hierarchical softmax (NEG is used for both models), no sub-sampling of frequent words, Adam (with the linear learning-rate decay of word2vec.c) instead of SGD.

Run:  python word2vec.py        (CPU, about 4 s)
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ───────────────────────── synthetic corpus with planted linear structure ─────────────────────────
CONCEPTS = ["king/queen", "man/woman", "prince/princess", "actor/actress", "boy/girl", "uncle/aunt", "hero/heroine", "lord/lady"]
ATTR_MARKERS = [["he", "his", "him", "mr", "father", "son"], ["she", "her", "hers", "mrs", "mother", "daughter"]]
CONCEPT_MARKERS = [["throne", "crown", "reign"], ["adult", "person", "walk"], ["heir", "palace", "young"], ["stage", "film", "role"],
                   ["school", "child", "play"], ["family", "nephew", "visit"], ["brave", "save", "quest"], ["estate", "manor", "title"]]
NOISE = [f"filler{i}" for i in range(20)]


def build_vocab_and_corpus(n_sentences=6000, sent_len=12):
    """Each sentence is about one (concept, attribute) word, e.g. 'queen' = (king/queen, female). Its neighbours are
    markers of the concept (throne, crown) and of the attribute (she, her) plus filler words. Skip-gram must then encode
    each target word as roughly concept_vector + attribute_vector, which is exactly what makes b - a + c work."""
    pair_words = [c.split("/") for c in CONCEPTS]
    words = sorted({w for p in pair_words for w in p} | {w for g in ATTR_MARKERS for w in g}
                   | {w for g in CONCEPT_MARKERS for w in g} | set(NOISE))
    idx = {w: i for i, w in enumerate(words)}
    corpus = []
    for _ in range(n_sentences):
        c, a = torch.randint(len(CONCEPTS), (1,)).item(), torch.randint(2, (1,)).item()
        pool = [pair_words[c][a]] * 3 + CONCEPT_MARKERS[c] * 2 + ATTR_MARKERS[a] * 2
        sent = [pool[torch.randint(len(pool), (1,)).item()] if torch.rand(1).item() < 0.7 else NOISE[torch.randint(len(NOISE), (1,)).item()]
                for _ in range(sent_len)]
        corpus.extend(idx[w] for w in sent)
    return words, idx, torch.tensor(corpus)


def make_pairs(corpus, c=4):
    """(centre, context) pairs with a dynamic window: for each position draw R ~ U{1..c} and use the R words on each side,
    so a word at distance d is used with probability (c - d + 1) / c (Section 3.2, 'sampling nearer words more often')."""
    T = len(corpus)
    R = torch.randint(1, c + 1, (T,))
    centre, context = [], []
    for d in range(1, c + 1):
        keep = R >= d
        for sign in (-1, 1):
            t = torch.arange(T)
            ok = keep & (t + sign * d >= 0) & (t + sign * d < T)
            centre.append(corpus[t[ok]]); context.append(corpus[t[ok] + sign * d])
    return torch.cat(centre), torch.cat(context)


def make_cbow_examples(corpus, c=4):
    """Fixed windows of 2c words around each centre word (CBOW uses all 2c neighbours, order ignored)."""
    T = len(corpus)
    t = torch.arange(c, T - c)
    offsets = torch.tensor([o for o in range(-c, c + 1) if o != 0])
    return corpus[t[:, None] + offsets[None, :]], corpus[t]           # contexts (N, 2c), centres (N,)


# ───────────────────────── the two log-linear models ─────────────────────────
class Word2Vec(nn.Module):
    """Input vectors v_w (the 'projection layer') and output vectors v'_w. No hidden layer, no nonlinearity."""
    def __init__(self, V, D):
        super().__init__()
        self.v = nn.Embedding(V, D)                 # v_w      (the embeddings we keep)
        self.v_out = nn.Embedding(V, D)             # v'_w     (output vectors)
        nn.init.uniform_(self.v.weight, -0.5 / D, 0.5 / D); nn.init.zeros_(self.v_out.weight)   # word2vec.c initialisation

    def neg_sampling_loss(self, h, target, noise):
        """-[ log sigma(v'_{w_O} . h) + sum_i log sigma(-v'_{w_i} . h) ],   h = v_{w_I} (skip-gram) or the CBOW average."""
        pos = (self.v_out(target) * h).sum(-1)                              # (B,)
        neg = torch.bmm(self.v_out(noise), h.unsqueeze(-1)).squeeze(-1)     # (B, k)
        return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()

    def skipgram(self, centre, context, noise):                             # predict context from centre
        return self.neg_sampling_loss(self.v(centre), context, noise)

    def cbow(self, contexts, centre, noise):                                # predict centre from the averaged context
        return self.neg_sampling_loss(self.v(contexts).mean(1), centre, noise)


def train(model, inputs, targets, noise_dist, mode, epochs, k=5, batch=512, lr=0.005):
    """Linear learning-rate decay to 0 over training, as in word2vec.c (alpha *= 1 - progress)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n_batches = epochs * ((len(targets) + batch - 1) // batch); done = 0
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(targets))
        total = 0.0
        for i in range(0, len(targets), batch):
            idx = perm[i : i + batch]
            for g in opt.param_groups:
                g["lr"] = lr * (1 - done / n_batches); done += 1
            noise = torch.multinomial(noise_dist, len(idx) * k, replacement=True).view(len(idx), k)   # w_i ~ P_n(w)
            loss = model.skipgram(inputs[idx], targets[idx], noise) if mode == "skipgram" else model.cbow(inputs[idx], targets[idx], noise)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        print(f"  {mode} epoch {ep}: NEG loss {total / len(targets):.4f}")


# ───────────────────────── evaluation: analogies and neighbours (Section 4) ─────────────────────────
@torch.no_grad()
def analogy_accuracy(emb, idx, words):
    """Questions 'a : b :: c : d' over all ordered pairs of concept pairs, e.g. king - man + woman = queen?  Correct only
    if the closest vector by cosine, excluding a, b and c, is exactly d (the paper's strict criterion)."""
    E = F.normalize(emb, dim=1)
    pairs = [c.split("/") for c in CONCEPTS]
    correct, total, examples = 0, 0, []
    for a, b in pairs:
        for c, d in pairs:
            if a == c:
                continue
            X = E[idx[b]] - E[idx[a]] + E[idx[c]]                                    # X = vec(b) - vec(a) + vec(c)
            sims = E @ F.normalize(X, dim=0)
            sims[[idx[a], idx[b], idx[c]]] = -2                                     # exclude the question words
            pred = words[sims.argmax().item()]
            correct += pred == d; total += 1
            if len(examples) < 3:
                examples.append(f"{b} - {a} + {c} = {pred}" + ("" if pred == d else f"  (expected {d})"))
    return correct / total, examples


@torch.no_grad()
def nearest(emb, idx, words, w, n=3):
    E = F.normalize(emb, dim=1)
    sims = E @ E[idx[w]]; sims[idx[w]] = -2
    return [words[i] for i in sims.topk(n).indices.tolist()]


def main():
    t0 = time.time()
    D, c, k = 32, 4, 5
    words, idx, corpus = build_vocab_and_corpus()
    V = len(words)
    counts = torch.bincount(corpus, minlength=V).float()
    noise_dist = counts.pow(0.75) / counts.pow(0.75).sum()                          # P_n(w) ∝ unigram^{3/4}
    print(f"corpus: {len(corpus):,} tokens, |V| = {V}, D = {D}, window c = {c}, k = {k} negatives")

    centre, context = make_pairs(corpus, c)
    print(f"skip-gram: {len(centre):,} (centre, context) pairs")
    sg = Word2Vec(V, D)
    train(sg, centre, context, noise_dist, "skipgram", epochs=4)
    acc_sg, ex = analogy_accuracy(sg.v.weight, idx, words)
    print(f"skip-gram analogy accuracy {acc_sg:.2f};  examples: " + "; ".join(ex))
    print(f"  nearest to 'queen': {nearest(sg.v.weight, idx, words, 'queen')},  to 'she': {nearest(sg.v.weight, idx, words, 'she')}")

    contexts, centres = make_cbow_examples(corpus, c)
    print(f"CBOW: {len(centres):,} (2c-word context, centre) examples")
    cb = Word2Vec(V, D)
    train(cb, contexts, centres, noise_dist, "cbow", epochs=4)
    acc_cb, ex = analogy_accuracy(cb.v.weight, idx, words)
    print(f"CBOW analogy accuracy {acc_cb:.2f};  examples: " + "; ".join(ex))
    print(f"  nearest to 'queen': {nearest(cb.v.weight, idx, words, 'queen')}")

    rand_acc, _ = analogy_accuracy(torch.randn(V, D), idx, words)
    print(f"random vectors: {rand_acc:.2f}   ({time.time() - t0:.1f} s)")
    assert acc_sg > 0.8 and acc_cb > 0.5, "word vectors did not learn the planted linear structure"


if __name__ == "__main__":
    main()
