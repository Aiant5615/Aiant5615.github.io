"""Language Models are Few-Shot Learners (GPT-3, Brown et al., 2020) — the paper's central mechanism at toy scale, in PyTorch.

GPT-3 is a scale / evaluation paper with no new model, so this file reproduces its central claim (Sections 1 and 2.1,
Figures 1.1-1.3): an autoregressive LM trained only on next-token prediction learns to perform a *new* task from K
demonstrations placed in its context, with no gradient update, and the more demonstrations (and the larger the
model) the better.

What is implemented:
  * a GPT-2-style decoder-only LM (pre-norm, learned positions, tied embeddings) in two sizes       (2.1, Table 2.1)
  * the "outer loop" (Figure 1.1): every pre-training sequence is one document x_1 y_1 x_2 y_2 ... x_K y_K in which a
    single task f is demonstrated repeatedly, y_i = f(x_i); the task family is all lookup tables f: {0..p-1} -> {0..p-1}
    (a fresh random table for every document, so every task is new), and the loss is next-token prediction of the y tokens
  * the "inner loop" (2.1, Figure 1.2): zero-shot / one-shot / few-shot accuracy on fresh tasks with K = 0, 1, 2, 4, 8
    demonstrations, with the Bayes-optimal accuracy for reference (the query is answerable only if its x was shown)
  * the few-shot vs zero-shot gap for the small and the larger model (Figure 1.3)
Simplifications: d_model 32 / 64 and 1 / 2 layers instead of 96 x 12288; dense attention only (no banded sparse
layers); tasks are lookup tables over integers, so there is no natural-language task description (the "no prompt"
curves of Figure 1.2), and few-shot learning here is the "induction" behaviour of copying the y that followed the same
x earlier in the context; Adam without warm-up or cosine decay.

Run:  python gpt3.py        (CPU, about 25 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
P, K = 4, 9                                    # size of the input/output alphabet; demonstrations per training document
VOCAB, CTX = 2 * P, 2 * K                       # x tokens 0..P-1, y tokens P..2P-1


# ───────────────────────── model: same decoder-only Transformer as GPT-2 (2.1) ─────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.h, self.d_k = h, d_model // h
        self.c_attn, self.c_proj = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = (t.transpose(1, 2) for t in self.c_attn(x).view(B, L, 3, self.h, self.d_k).unbind(2))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), float("-inf"))
        return self.c_proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(B, L, -1))


class Block(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.ln_1, self.attn, self.ln_2 = nn.LayerNorm(d_model), CausalSelfAttention(d_model, h), nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, d_model, h, n_layer, vocab=VOCAB, ctx=CTX):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab, d_model), nn.Embedding(ctx, d_model)
        self.blocks = nn.ModuleList([Block(d_model, h) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1)))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x) @ self.wte.weight.T


# ───────────────────────── the task distribution ─────────────────────────
def make_documents(B):
    """B sequences x_1 y_1 ... x_K y_K, each demonstrating one freshly drawn lookup table f (Figure 1.1)."""
    f = torch.randint(0, P, (B, P))                                        # a random task per document
    x = torch.randint(0, P, (B, K))
    y = f.gather(1, x) + P                                                 # y tokens live in P .. 2P-1
    return torch.stack([x, y], 2).view(B, 2 * K)


def lm_loss(model, docs):
    """Next-token loss on the y positions only: the model reads x_1 y_1 .. x_i and predicts y_i (pre-training)."""
    logits = model(docs[:, :-1])[:, 0::2]                                  # hidden states at x_1, x_2, ..., x_K
    return F.cross_entropy(logits.reshape(-1, VOCAB), docs[:, 1::2].reshape(-1))


@torch.no_grad()
def in_context_accuracy(model, shots, n=4000):
    """Accuracy of predicting y_{K'+1} from K' = `shots` demonstrations of a *new* task, no parameter update (2.1)."""
    docs = make_documents(n)
    pred = model(docs[:, : 2 * shots + 1])[:, -1].argmax(-1)
    return (pred == docs[:, 2 * shots + 1]).float().mean().item()


def bayes_optimal(shots, n=40000):
    """The best any learner can do: if the query x appeared in a demonstration, answer with its y; otherwise guess."""
    docs = make_documents(n)
    seen = (docs[:, 0:2 * shots:2] == docs[:, 2 * shots:2 * shots + 1]).any(1).float().mean().item()
    return seen + (1 - seen) / P


def train(name, model, steps=1500):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    print(f"{name}: {sum(p.numel() for p in model.parameters()):,} parameters")
    for step in range(1, steps + 1):
        loss = lm_loss(model, make_documents(64))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"  step {step:4d}  LM loss {loss.item():.3f}")
    model.eval()
    return model


def main():
    t0 = time.time()
    shots = [0, 1, 2, 4, 8]
    small = train("small model (d=32, 1 layer)", GPT(32, 2, 1))
    large = train("large model (d=64, 2 layers)", GPT(64, 4, 2))
    print(f"\nin-context accuracy on new lookup-table tasks, no gradient updates (Figure 1.2):")
    print("  K shots      " + "".join(f"{k:>8d}" for k in shots))
    rows = {"Bayes-optimal": [bayes_optimal(k) for k in shots],
            "small":         [in_context_accuracy(small, k) for k in shots],
            "large":         [in_context_accuracy(large, k) for k in shots]}
    for name, accs in rows.items():
        print(f"  {name:13s}" + "".join(f"{a:8.2f}" for a in accs))
    gap = {n: rows[n][-1] - rows[n][0] for n in ("small", "large")}
    print(f"few-shot minus zero-shot gap: small {gap['small']:.2f}  large {gap['large']:.2f}  (Figure 1.3: the gap grows with size)   ({time.time() - t0:.1f} s)")
    assert rows["large"][-1] > 0.8 and rows["large"][0] < 0.35 and gap["large"] > gap["small"] + 0.2, \
        "in-context learning did not emerge as expected"


if __name__ == "__main__":
    main()
