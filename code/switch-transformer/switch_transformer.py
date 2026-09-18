"""Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
(Fedus, Zoph & Shazeer, 2021) — a from-scratch PyTorch Switch-FFN layer inside a tiny causal LM.

What is implemented (section numbers follow the paper):
  * router  h(x) = W_r x,  p_i(x) = softmax(h(x))_i, computed in float32 ("selective precision")   (2.1 eq. 1-2, 2.4)
  * top-1 "switch" routing: y = p_{i*}(x) E_{i*}(x), the gate value keeps the router differentiable  (2.1, eq. 3 with k=1)
  * expert capacity = (tokens per batch / number of experts) * capacity factor; overflow tokens are
    DROPPED (skip the expert, pass through the residual unchanged)                                  (2.2, eq. 4, Figure 3)
  * load-balancing auxiliary loss  alpha * N * sum_i f_i * P_i  with f_i (hard dispatch fraction) and
    P_i (mean router probability), alpha = 1e-2                                                     (2.2, eq. 4-6)
  * router z-loss (optional, coefficient 0 by default; it is from the ST-MoE follow-up, not this paper)
  * smaller (0.1x) truncated-normal initialisation, expert dropout                                  (2.4)
Simplifications: N=4 experts on one device (no all-to-all), a 2-layer decoder on a synthetic sequence
  task instead of T5 on C4, bfloat16 everywhere replaced by float32 (the router cast is kept for illustration).

Run:  python switch_transformer.py        (CPU, about 10 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def init_trunc_normal_(w, scale=0.1):
    """Section 2.4: truncated normal with std = sqrt(s / n_in), s = 0.1 (10x smaller than the default of 1.0)."""
    nn.init.trunc_normal_(w, std=math.sqrt(scale / w.size(1)), a=-2 * math.sqrt(scale / w.size(1)), b=2 * math.sqrt(scale / w.size(1)))


class Expert(nn.Module):
    """One FFN expert E_i(x) = W_2 relu(W_1 x), with expert dropout (2.4)."""
    def __init__(self, d, d_ff, dropout):
        super().__init__()
        self.w1, self.w2, self.drop = nn.Linear(d, d_ff), nn.Linear(d_ff, d), nn.Dropout(dropout)
        init_trunc_normal_(self.w1.weight); init_trunc_normal_(self.w2.weight)

    def forward(self, x):
        return self.w2(self.drop(F.relu(self.w1(x))))


class SwitchFFN(nn.Module):
    """The Switch layer: route each token to ONE expert (Section 2.1-2.2, Figure 2)."""
    def __init__(self, d, d_ff, n_experts, capacity_factor=1.25, alpha=1e-2, z_coef=0.0, dropout=0.0):
        super().__init__()
        self.N, self.cf, self.alpha, self.z_coef = n_experts, capacity_factor, alpha, z_coef
        self.router = nn.Linear(d, n_experts, bias=False)                      # W_r
        init_trunc_normal_(self.router.weight)
        self.experts = nn.ModuleList([Expert(d, d_ff, dropout) for _ in range(n_experts)])
        self.aux_loss = self.z_loss = torch.tensor(0.0)
        self.stats = {}

    def forward(self, x):                                   # x: (B, L, d)
        B, L, d = x.shape
        x = x.reshape(-1, d)                                # T = B*L tokens in the batch
        T = x.size(0)
        # --- router in float32 (selective precision, 2.4): h(x) = W_r x, p = softmax(h)   (eq. 1-2)
        logits = self.router(x.float()).float()
        p = logits.softmax(-1)                              # (T, N)
        gate, idx = p.max(-1)                               # top-1: p_{i*}(x) and i* = argmax_i p_i(x)
        # --- expert capacity (eq. 4 in Section 2.2): tokens beyond it are dropped
        capacity = int(math.ceil(T / self.N * self.cf))
        onehot = F.one_hot(idx, self.N).float()             # (T, N) hard dispatch mask  1{argmax p(x) = i}
        position = (onehot.cumsum(0) * onehot).sum(-1)      # order of arrival of each token within its expert (1-based)
        keep = position <= capacity                         # (T,) False = dropped token
        # --- load-balancing loss (eq. 4-6): f_i = fraction dispatched, P_i = mean router prob, loss = alpha N sum f_i P_i
        f = onehot.mean(0)                                  # (N,) not differentiable
        P = p.mean(0)                                       # (N,) differentiable: gradient pushes probabilities to balance
        self.aux_loss = self.alpha * self.N * (f * P).sum()
        self.z_loss = self.z_coef * torch.logsumexp(logits, -1).pow(2).mean()     # optional (ST-MoE), 0 by default
        # --- dispatch: y = p_{i*}(x) E_{i*}(x); dropped tokens get y = 0 so the residual passes x through unchanged
        y = torch.zeros_like(x)
        for i, E in enumerate(self.experts):
            sel = (idx == i) & keep
            if sel.any():
                y[sel] = gate[sel, None] * E(x[sel])
        self.stats = dict(f=f.detach(), dropped=1.0 - keep.float().mean().item(), capacity=capacity)
        return y.reshape(B, L, d)


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.d_k = h, d // h
        self.qkv, self.out = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-2, -1) / math.sqrt(self.d_k)).masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), -1e9)
        return self.out((s.softmax(-1) @ v).transpose(1, 2).reshape(B, L, -1))


class Block(nn.Module):
    """Pre-norm decoder block whose FFN is the Switch layer (Figure 2)."""
    def __init__(self, d, h, d_ff, n_experts, **kw):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.ffn = Attention(d, h), SwitchFFN(d, d_ff, n_experts, **kw)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))                     # dropped tokens: ffn output is 0 -> x unchanged


class SwitchLM(nn.Module):
    def __init__(self, vocab, d=64, h=4, d_ff=128, n_layers=2, n_experts=4, max_len=32, **kw):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h, d_ff, n_experts, **kw) for _ in range(n_layers)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.size(1)))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))

    def aux(self):
        return sum(b.ffn.aux_loss + b.ffn.z_loss for b in self.blocks)


def make_batch(B, L, vocab):
    """Toy language: y_t = (x_t + x_{t-1}) mod (vocab-1) + 1 for t >= 1 (needs attention to the previous token)."""
    x = torch.randint(1, vocab, (B, L))
    y = torch.zeros_like(x)
    y[:, 1:] = (x[:, 1:] + x[:, :-1]) % (vocab - 1) + 1
    return x, y


def train(model, steps, use_aux, tag):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    first = None
    for step in range(1, steps + 1):
        x, y = make_batch(32, 16, 20)
        ce = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss = ce + (model.aux() if use_aux else 0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or ce.item()
        if step % 250 == 0:
            st = model.blocks[-1].ffn.stats
            print(f"  [{tag}] step {step:4d}  ce {ce.item():.3f}  aux {model.aux().item():.4f}  "
                  f"last-layer f_i {[round(v, 2) for v in st['f'].tolist()]}  dropped {100 * st['dropped']:.1f}%  (capacity {st['capacity']})")
    return first, ce.item()


def main():
    t0 = time.time()
    vocab, n_experts = 20, 4
    model = SwitchLM(vocab, n_experts=n_experts, capacity_factor=1.25, alpha=1e-2, dropout=0.1)
    dense_ffn = 2 * (64 * 128 + 128 * 64)
    total = sum(p.numel() for p in model.parameters())
    print(f"parameters: {total:,} total; FFN params {n_experts}x a dense model's ({dense_ffn * n_experts:,} vs {dense_ffn:,}),"
          f" compute per token = one expert")
    print("training WITH the load-balancing loss (alpha=1e-2, capacity factor 1.25):")
    first, last = train(model, 1000, use_aux=True, tag="switch+aux")
    f_bal = torch.stack([b.ffn.stats["f"] for b in model.blocks])            # (layers, N) dispatch fractions
    drop_bal = max(b.ffn.stats["dropped"] for b in model.blocks)

    torch.manual_seed(1)
    model_noaux = SwitchLM(vocab, n_experts=n_experts, capacity_factor=1.25, alpha=1e-2, dropout=0.1)
    print("same model WITHOUT the auxiliary loss (router free to collapse):")
    train(model_noaux, 1000, use_aux=False, tag="switch-noaux")
    f_noaux = torch.stack([b.ffn.stats["f"] for b in model_noaux.blocks])
    drop_noaux = max(b.ffn.stats["dropped"] for b in model_noaux.blocks)

    model.eval()
    with torch.no_grad():
        x, y = make_batch(256, 16, vocab)
        acc = (model(x).argmax(-1)[:, 1:] == y[:, 1:]).float().mean().item()
    print(f"held-out next-token accuracy (with aux): {acc:.3f}   ce {first:.3f} -> {last:.3f}")
    print(f"max expert load f_i: with aux {f_bal.max():.2f} (uniform = {1 / n_experts:.2f}, dropped {100 * drop_bal:.1f}%)"
          f"  |  without aux {f_noaux.max():.2f} (dropped {100 * drop_noaux:.1f}%)   ({time.time() - t0:.1f} s)")
    assert last < 0.5 * first and acc > 0.9, "the Switch LM did not learn the task"
    assert f_bal.max() < 0.5 and drop_bal < 0.1, "load-balancing loss should keep experts balanced and drops rare"
    assert f_noaux.max() > f_bal.max(), "without the auxiliary loss the router should be more imbalanced"


if __name__ == "__main__":
    main()
