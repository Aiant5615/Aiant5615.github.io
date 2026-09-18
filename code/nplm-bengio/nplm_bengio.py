"""A Neural Probabilistic Language Model (Bengio, Ducharme, Vincent & Jauvin, 2003) — from-scratch PyTorch.

What is implemented (section numbers follow the paper):
  * shared feature-vector table C in R^{|V| x m}, input x = (C(w_{t-1}), ..., C(w_{t-n+1}))       (Section 2, eq. 1)
  * one-hidden-layer network with direct connections  y = b + W x + U tanh(d + H x)              (Section 2, eq. 2)
  * softmax over the vocabulary  P(w_t = i | context) = exp(y_i) / sum_j exp(y_j)                (Section 2)
  * objective L = 1/T sum_t log f(...) + R(theta), weight decay on everything but biases,
    trained with plain stochastic gradient ascent  theta <- theta + eps * d log P / d theta       (Section 2, 3)
  * the W = 0 ablation ("without direct connections") and a smoothed trigram baseline for the
    perplexity comparison of Tables 1–2
Simplifications: |V| = 60 word vocabulary and a toy corpus generated from a hidden word-class Markov chain
(so that "similar words" exist for the model to exploit), minibatches of 32 instead of pure online SGD, no
parallelisation tricks (Section 3).

Run:  python nplm_bengio.py        (CPU, about 4 s)
"""
import math, time
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ───────────────────────── toy corpus ─────────────────────────
def make_corpus(n_classes=6, words_per_class=10, T=30_000):
    """Words belong to hidden classes; the next class depends on the two previous classes, the word inside a class is
    random. Every (w_{t-1}, w_{t-2}) context therefore has thousands of never-seen but *similar* neighbours, which is the
    situation the paper's distributed representation is meant to exploit (Section 1, 'curse of dimensionality')."""
    V = n_classes * words_per_class
    trans = torch.softmax(torch.randn(n_classes, n_classes, n_classes) * 3.0, dim=-1)   # P(class_t | class_{t-1}, class_{t-2})
    within = torch.softmax(torch.randn(n_classes, words_per_class) * 1.0, dim=-1)       # Zipf-ish word frequencies
    cls = [0, 1]
    for _ in range(T - 2):
        cls.append(torch.multinomial(trans[cls[-1], cls[-2]], 1).item())
    words = [c * words_per_class + torch.multinomial(within[c], 1).item() for c in cls]
    return torch.tensor(words), V


def make_ngrams(corpus, n):
    """Context (w_{t-1}, ..., w_{t-n+1}) -> target w_t for every position t >= n-1."""
    ctx = torch.stack([corpus[n - 1 - k : len(corpus) - k] for k in range(1, n)], dim=1)   # (T', n-1), nearest word first
    return ctx, corpus[n - 1 :]


# ───────────────────────── the model (Section 2) ─────────────────────────
class NPLM(nn.Module):
    """y = b + W x + U tanh(d + H x),   x = concat of the n-1 previous words' feature vectors C(w)."""
    def __init__(self, V, n, m, h, direct=True):
        super().__init__()
        self.n, self.m = n, m
        self.C = nn.Embedding(V, m)                                  # feature vectors C(i), shared across all positions
        self.H = nn.Linear((n - 1) * m, h)                           # hidden layer weights H and bias d
        self.U = nn.Linear(h, V, bias=True)                          # hidden-to-output U and output bias b
        self.W = nn.Linear((n - 1) * m, V, bias=False) if direct else None   # direct connections (W = 0 if disabled)

    def forward(self, ctx):                                          # ctx: (B, n-1) integer words
        x = self.C(ctx).view(ctx.size(0), -1)                        # x = (C(w_{t-1}), ..., C(w_{t-n+1}))   eq. 1
        y = self.U(torch.tanh(self.H(x)))                            # b + U tanh(d + H x)
        if self.W is not None:
            y = y + self.W(x)                                        # + W x   (direct input-to-output connections)
        return y                                                     # pre-softmax scores y_i; softmax is in the loss

    def n_params(self):
        """|V|(1 + n m + h) + h(1 + (n-1) m) in the paper's count (their n m counts the direct connections)."""
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def perplexity(model, ctx, tgt):
    """exp( -1/T sum_t log P(w_t | context) ) over the held-out positions."""
    model.eval()
    nll = F.cross_entropy(model(ctx), tgt, reduction="mean")
    model.train()
    return math.exp(nll.item())


def train(model, ctx, tgt, test_ctx, test_tgt, epochs, lr, weight_decay, batch=32, log=True):
    """Stochastic gradient ascent on the regularised log-likelihood (Section 3). R(theta) = weight decay on all
    weights except biases; the update is theta <- theta + eps * d log P / d theta (we minimise -log P)."""
    decay = [p for n_, p in model.named_parameters() if not n_.endswith("bias")]
    no_decay = [p for n_, p in model.named_parameters() if n_.endswith("bias")]
    opt = torch.optim.SGD([{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(tgt))
        for i in range(0, len(tgt), batch):
            idx = perm[i : i + batch]
            loss = F.cross_entropy(model(ctx[idx]), tgt[idx])       # -log P(w_t | w_{t-1..t-n+1}), i.e. -log f
            opt.zero_grad(); loss.backward(); opt.step()
        if log:
            print(f"  epoch {ep}  train nll {loss.item():.3f}  test perplexity {perplexity(model, test_ctx, test_tgt):7.2f}")
    return perplexity(model, test_ctx, test_tgt)


# ───────────────────────── count-based baseline (the competitor in Tables 1–2) ─────────────────────────
def interpolated_trigram_ppl(train_corpus, test_ctx, test_tgt, V, lambdas=(0.6, 0.3, 0.1), delta=0.5):
    """Interpolated add-delta trigram: P = l3 P3 + l2 P2 + l1 P1, each level add-delta smoothed. Simpler than the
    Kneser–Ney / class-based baselines of the paper but the same idea: counts plus back-off to shorter contexts."""
    c = train_corpus.tolist()
    uni, bi, tri = Counter(c), Counter(zip(c[:-1], c[1:])), Counter(zip(c[:-2], c[1:-1], c[2:]))
    bi_ctx, tri_ctx = Counter(c[:-1]), Counter(zip(c[:-2], c[1:-1]))
    nll = 0.0
    for (w1, w2), w in zip(test_ctx.tolist(), test_tgt.tolist()):   # ctx is nearest-first: w1 = w_{t-1}, w2 = w_{t-2}
        p3 = (tri[(w2, w1, w)] + delta) / (tri_ctx[(w2, w1)] + delta * V)
        p2 = (bi[(w1, w)] + delta) / (bi_ctx[w1] + delta * V)
        p1 = (uni[w] + delta) / (len(c) + delta * V)
        nll -= math.log(lambdas[0] * p3 + lambdas[1] * p2 + lambdas[2] * p1)
    return math.exp(nll / len(test_tgt))


def main():
    t0 = time.time()
    n, m, h = 4, 16, 32                                               # order n (n-1 = 3 context words), m = |C(i)|, h hidden
    corpus, V = make_corpus()
    train_corpus, test_corpus = corpus[:24_000], corpus[24_000:]
    ctx, tgt = make_ngrams(train_corpus, n)
    test_ctx, test_tgt = make_ngrams(test_corpus, n)
    seen = set(map(tuple, ctx.tolist()))
    unseen_frac = sum(tuple(c) not in seen for c in test_ctx.tolist()) / len(test_ctx)
    print(f"|V| = {V}, n = {n}: {len(tgt):,} training {n}-grams, {len(test_tgt):,} test; "
          f"{100 * unseen_frac:.0f}% of test contexts never occurred in training")
    print(f"uniform perplexity {V}, interpolated trigram baseline {interpolated_trigram_ppl(train_corpus, test_ctx[:, :2], test_tgt, V):.2f}")

    model = NPLM(V, n, m, h, direct=True)
    print(f"MLP with direct connections: {model.n_params():,} parameters")
    ppl0 = perplexity(model, test_ctx, test_tgt)
    print(f"  before training: test perplexity {ppl0:.2f}")
    ppl = train(model, ctx, tgt, test_ctx, test_tgt, epochs=8, lr=0.1, weight_decay=1e-5)

    model_nodirect = NPLM(V, n, m, h, direct=False)                   # the paper's W = 0 variant
    ppl_nd = train(model_nodirect, ctx, tgt, test_ctx, test_tgt, epochs=6, lr=0.05, weight_decay=1e-5, log=False)
    print(f"after 8 epochs: with direct connections {ppl:.2f}, without {ppl_nd:.2f}   ({time.time() - t0:.1f} s)")

    trigram = interpolated_trigram_ppl(train_corpus, test_ctx[:, :2], test_tgt, V)
    Ctab = model.C.weight
    sim = F.normalize(Ctab, dim=1) @ F.normalize(Ctab, dim=1).T
    same_class = (torch.arange(V) // 10)[:, None] == (torch.arange(V) // 10)[None, :]
    print(f"mean cosine between C(i), C(j): same hidden class {sim[same_class & ~torch.eye(V, dtype=bool)].mean():.2f}, "
          f"different class {sim[~same_class].mean():.2f}")
    assert ppl < 0.7 * ppl0 and ppl < trigram, "neural LM did not beat the count-based baseline"


if __name__ == "__main__":
    main()
