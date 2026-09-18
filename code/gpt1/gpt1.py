"""Improving Language Understanding by Generative Pre-Training (GPT, Radford et al., 2018) — a from-scratch PyTorch implementation.

What is implemented (section numbers follow the paper):
  * Transformer decoder LM: h_0 = U W_e + W_p (learned positions), h_l = transformer_block(h_{l-1}),
    P(u) = softmax(h_n W_e^T) with the output softmax tied to W_e; GELU in the FFN            (3.1, eq. 2)
  * Stage 1, unsupervised pre-training: maximise L_1(U) = sum_i log P(u_i | u_{i-k..i-1})     (3.1, eq. 1)
  * Stage 2, supervised fine-tuning: a linear head W_y on the last token's activation h_l^m,
    L_2(C) = sum log P(y | x^1..x^m), combined objective L_3(C) = L_2(C) + lambda L_1(C), lambda = 0.5   (3.2, eqs. 3-5)
  * task-specific input transformation with new, randomly initialised delimiter tokens learned during fine-tuning:
    [start] premise [delim] hypothesis [extract], the head reads the [extract] position       (3.3, Figure 1 right)
  * control run: the same fine-tuning without pre-training (Table 5, "w/o pre-training")
Simplifications: 2 blocks of d_model = 64 instead of 12 x 768; a toy corpus of integer tokens drawn from a hidden Markov
chain instead of BooksCorpus, no BPE; the downstream "entailment" task asks whether the hypothesis is a grammatical
continuation of the premise or random noise; no dropout; constant learning rates instead of warm-up + cosine decay.

Run:  python gpt1.py        (CPU, about 25 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ───────────────────────── model: the Transformer decoder without cross-attention (3.1) ─────────────────────────
class MaskedSelfAttention(nn.Module):
    """Multi-head self-attention in which position i only attends to positions j <= i."""
    def __init__(self, d_model, h):
        super().__init__()
        self.h, self.d_k = h, d_model // h
        self.qkv, self.proj = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = (t.transpose(1, 2) for t in self.qkv(x).view(B, L, 3, self.h, self.d_k).unbind(2))   # (B, h, L, d_k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
        attn = scores.masked_fill(~causal, float("-inf")).softmax(-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, L, -1))


class TransformerBlock(nn.Module):
    """transformer_block: masked self-attention -> LayerNorm -> position-wise FFN with GELU -> LayerNorm (Figure 1 left)."""
    def __init__(self, d_model, h, d_ff):
        super().__init__()
        self.attn, self.ln1 = MaskedSelfAttention(d_model, h), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, h):
        h = self.ln1(h + self.attn(h))
        return self.ln2(h + self.ffn(h))


class GPT(nn.Module):
    def __init__(self, vocab, ctx, d_model=64, h=4, d_ff=256, n_layers=2, n_classes=2):
        super().__init__()
        self.W_e = nn.Embedding(vocab, d_model)      # token embedding matrix, reused by the output softmax (eq. 2)
        self.W_p = nn.Embedding(ctx, d_model)        # learned position embedding (not sinusoidal)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, h, d_ff) for _ in range(n_layers)])
        self.W_y = nn.Linear(d_model, n_classes)     # task head: the only new weights at fine-tuning (3.2)

    def hidden(self, U):
        """eq. 2:  h_0 = U W_e + W_p,   h_l = transformer_block(h_{l-1}) for l = 1..n.   Returns h_n: (B, L, d_model)."""
        h = self.W_e(U) + self.W_p(torch.arange(U.size(1)))
        for block in self.blocks:
            h = block(h)
        return h

    def lm_logits(self, h):
        """eq. 2:  P(u) = softmax(h_n W_e^T)  (tied embeddings)."""
        return h @ self.W_e.weight.T


def L1(model, tokens):
    """Unsupervised objective (eq. 1) as a per-token loss: -mean_i log P(u_i | u_{i-k}, ..., u_{i-1}; Theta)."""
    logits = model.lm_logits(model.hidden(tokens[:, :-1]))
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))


def L2(model, tokens, y):
    """Supervised objective (eqs. 3-4): P(y | x^1..x^m) = softmax(h_l^m W_y), read at the last ([extract]) position."""
    logits = model.W_y(model.hidden(tokens)[:, -1])
    return F.cross_entropy(logits, y), logits


# ───────────────────────── toy data ─────────────────────────
V_TEXT, START, DELIM, EXTRACT = 32, 32, 33, 34       # 32 "word" tokens + the three delimiter tokens of Section 3.3
VOCAB = V_TEXT + 3
# The hidden grammar of the corpus: every word has two likely successors (p = 0.45 each); the rest is uniform noise.
_succ = torch.stack([torch.randperm(V_TEXT)[:2] for _ in range(V_TEXT)])
TRANS = torch.full((V_TEXT, V_TEXT), 0.1 / (V_TEXT - 2)).scatter_(1, _succ, 0.45)


def sample_corpus(B, L, start=None):
    """B token streams of length L from the Markov chain: the unlabeled corpus U = (u_1, ..., u_n)."""
    u = torch.randint(0, V_TEXT, (B,)) if start is None else start
    out = [u]
    for _ in range(L - 1):
        u = torch.multinomial(TRANS[u], 1).squeeze(1)
        out.append(u)
    return torch.stack(out, 1)


def make_task_batch(B, len_p=8, len_h=8):
    """Toy entailment task: is the hypothesis a continuation of the premise under the corpus grammar (y = 1) or random
    noise (y = 0)?  Input transformation (Figure 1, "Entailment"):  [start] premise [delim] hypothesis [extract]."""
    premise = sample_corpus(B, len_p)
    cont = sample_corpus(B, len_h + 1, start=premise[:, -1])[:, 1:]      # the chain continued past the premise
    noise = torch.randint(0, V_TEXT, (B, len_h))
    y = torch.randint(0, 2, (B,))
    hyp = torch.where(y[:, None] == 1, cont, noise)
    tok = lambda t: torch.full((B, 1), t)
    return torch.cat([tok(START), premise, tok(DELIM), hyp, tok(EXTRACT)], 1), y


def finetune(model, steps, lam=0.5, lr=1e-3, tag=""):
    """Stage 2: minimise L_3(C) = L_2(C) + lambda * L_1(C) on the transformed task text (eq. 5); every parameter is updated."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        x, y = make_task_batch(64)
        l2, _ = L2(model, x, y)
        loss = l2 + lam * L1(model, x)                # the LM term on the task sequences acts as the auxiliary objective
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  [{tag}] fine-tune step {step:3d}  L2 {l2.item():.3f}  L3 {loss.item():.3f}")
    x, y = make_task_batch(500)
    with torch.no_grad():
        return (L2(model, x, y)[1].argmax(-1) == y).float().mean().item()


def main():
    t0 = time.time()
    model = GPT(VOCAB, ctx=32)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Stage 1: unsupervised pre-training on the corpus (3.1). The delimiter tokens never occur here, so their embeddings
    # stay at their random initialisation until fine-tuning, exactly as in the paper.
    entropy = -(TRANS * TRANS.log()).sum(1).mean().item()
    print(f"pre-training  (entropy of the hidden chain = {entropy:.3f} nats/token = the best achievable L1)")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)   # the paper: 2.5e-4 peak with warm-up + cosine; a toy trains faster at 1e-3
    for step in range(1, 601):
        loss = L1(model, sample_corpus(32, 32))
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1:
            first = loss.item()
        if step % 150 == 0:
            print(f"  step {step:3d}  L1 {loss.item():.3f}")
    assert loss.item() < entropy + 0.3, "pre-training did not approach the entropy of the chain"
    # Stage 2: supervised fine-tuning with L_3 (3.2), against the same run without pre-training (Table 5).
    print("fine-tuning with L3 = L2 + 0.5 * L1 on [start] premise [delim] hypothesis [extract]")
    acc_pre = finetune(model, 400, tag="pre-trained")            # paper: ~3 epochs at 6.25e-5; the toy needs 1e-3
    acc_scratch = finetune(GPT(VOCAB, ctx=32), 400, tag="no pre-training")
    print(f"held-out accuracy: pre-trained {acc_pre:.2f}   without pre-training {acc_scratch:.2f}   ({time.time() - t0:.1f} s)")
    assert acc_pre > 0.85 and acc_pre > acc_scratch + 0.03, "pre-training did not help the downstream task as expected"


if __name__ == "__main__":
    main()
