"""LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021) — a from-scratch PyTorch implementation.

What is implemented (section numbers follow the paper):
  * h = W_0 x + Delta W x = W_0 x + B A x,  B in R^{d x r} zero-initialised, A in R^{r x k} Gaussian   (4.1, eq. 3)
  * the scaling Delta W x * (alpha / r)                                                                (4.1)
  * frozen pretrained weights: only A, B (and nothing else) receive gradients                         (4.1)
  * LoRA on the attention projections W_q and W_v only, MLP frozen                                    (4.2, Table 5)
  * merging for inference  W = W_0 + B A  (no added latency) and un-merging to switch tasks           (4.1)
  * trainable-parameter count vs full fine-tuning; check that merged == unmerged outputs
Simplifications: a 2-layer, d=64 causal Transformer "pretrained" on a toy task in-file instead of GPT-3;
  the downstream task is a second toy sequence task; r = 4.

Run:  python lora.py        (CPU, about 15 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ───────────────────────── LoRA layer (Section 4.1) ─────────────────────────
class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear W_0 (d x k) with the low-rank update Delta W = B A scaled by alpha / r (eq. 3)."""
    def __init__(self, base: nn.Linear, r=4, alpha=8):
        super().__init__()
        self.base, self.r, self.scaling = base, r, alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)                          # W_0 is frozen and receives no gradient updates
        d, k = base.out_features, base.in_features
        self.A = nn.Parameter(torch.randn(r, k) / math.sqrt(k))   # random Gaussian initialisation for A
        self.B = nn.Parameter(torch.zeros(d, r))                   # zero for B, so Delta W = B A = 0 at the start
        self.merged = False

    def delta_w(self):
        return (self.B @ self.A) * self.scaling              # Delta W (d x k), rank <= r

    def forward(self, x):
        if self.merged:                                      # deployment: a single dense matmul, no extra latency
            return self.base(x)
        return self.base(x) + F.linear(x, self.A) @ self.B.T * self.scaling     # W_0 x + B A x, both branches see x

    @torch.no_grad()
    def merge(self):
        """W = W_0 + B A: store the merged weight in place (the paper's inference trick)."""
        if not self.merged:
            self.base.weight += self.delta_w(); self.merged = True

    @torch.no_grad()
    def unmerge(self):
        """Recover W_0 by subtracting B A (to switch to another task's adapter)."""
        if self.merged:
            self.base.weight -= self.delta_w(); self.merged = False


# ───────────────────────── a tiny causal Transformer to adapt ─────────────────────────
class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.d_k = h, d // h
        self.w_q, self.w_k, self.w_v, self.w_o = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x):
        B, L, _ = x.shape
        split = lambda t: t.view(B, L, self.h, self.d_k).transpose(1, 2)
        q, k, v = split(self.w_q(x)), split(self.w_k(x)), split(self.w_v(x))
        s = (q @ k.transpose(-2, -1) / math.sqrt(self.d_k)).masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), -1e9)
        return self.w_o((s.softmax(-1) @ v).transpose(1, 2).reshape(B, L, -1))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1, self.n2, self.attn = nn.LayerNorm(d), nn.LayerNorm(d), Attention(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class TinyLM(nn.Module):
    def __init__(self, vocab, d=64, h=4, n_layers=2, max_len=32):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.size(1)))
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def add_lora(model, r=4, alpha=8, targets=("w_q", "w_v")):
    """Freeze everything, then wrap W_q and W_v of every layer (Table 5: adapting {W_q, W_v} beats one matrix at 2r)."""
    for p in model.parameters():
        p.requires_grad_(False)
    layers = []
    for b in model.blocks:
        for name in targets:
            lora = LoRALinear(getattr(b.attn, name), r, alpha)
            setattr(b.attn, name, lora); layers.append(lora)
    return layers


# ───────────────────────── toy tasks ─────────────────────────
def make_batch(B, L, vocab, offset):
    """Predict the token `offset` positions back. Pretraining uses offset 1, the downstream task offset 3."""
    x = torch.randint(1, vocab, (B, L))
    y = torch.zeros_like(x); y[:, offset:] = x[:, :-offset]
    return x, y


def run(model, offset, steps, lr, tag, vocab=16):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    first = None
    for step in range(1, steps + 1):
        x, y = make_batch(64, 16, vocab, offset)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or loss.item()
        if step % (steps // 2) == 0:
            print(f"  [{tag}] step {step:4d}  loss {loss.item():.3f}")
    return first, loss.item()


@torch.no_grad()
def accuracy(model, offset, vocab=16):
    x, y = make_batch(256, 16, vocab, offset)
    return (model(x).argmax(-1)[:, offset:] == y[:, offset:]).float().mean().item()


def main():
    t0, vocab = time.time(), 16
    model = TinyLM(vocab)
    n_full = sum(p.numel() for p in model.parameters())
    print("1) 'pretraining' the base model on task A (copy previous token)")
    run(model, offset=1, steps=400, lr=2e-3, tag="pretrain")
    acc_a = accuracy(model, 1)
    w0_snapshot = [b.attn.w_q.weight.clone() for b in model.blocks]
    x, _ = make_batch(8, 16, vocab, 3)
    with torch.no_grad():
        base_out = model(x)

    print("2) LoRA fine-tuning on task B (copy the token 3 back), r=4, alpha=8, on W_q and W_v only")
    loras = add_lora(model, r=4, alpha=8)
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   trainable parameters: full fine-tuning {n_full:,}  vs  LoRA {n_lora:,}  ({n_full / n_lora:.0f}x fewer)")
    with torch.no_grad():
        acc_b_before = accuracy(model, 3)
        assert torch.allclose(model(x), base_out, atol=1e-6), "B = 0 => Delta W = 0: training starts at the pretrained model"
    first, last = run(model, offset=3, steps=400, lr=5e-3, tag="lora")
    acc_b_after = accuracy(model, 3)
    for b, w0 in zip(model.blocks, w0_snapshot):
        assert torch.equal(b.attn.w_q.base.weight, w0), "W_0 must stay frozen"

    print("3) merge W = W_0 + B A for inference and compare outputs")
    model.eval()
    with torch.no_grad():
        out_unmerged = model(x)
        for l in loras: l.merge()
        out_merged = model(x)
        for l in loras: l.unmerge()
        out_back = model(x)
    diff = (out_merged - out_unmerged).abs().max().item()
    rank = torch.linalg.matrix_rank(loras[0].delta_w()).item()
    print(f"   max |merged - unmerged| = {diff:.2e};  rank(Delta W_q, layer 0) = {rank} (r = 4);"
          f"  un-merged restores base exactly: {torch.allclose(out_back, out_unmerged, atol=1e-5)}")
    print(f"task A acc after pretraining {acc_a:.3f} | task B acc: before LoRA {acc_b_before:.3f} -> after {acc_b_after:.3f}"
          f"   loss {first:.3f} -> {last:.3f}   ({time.time() - t0:.1f} s)")
    assert acc_a > 0.95 and acc_b_after > 0.95, "LoRA fine-tuning should solve task B with the frozen base"
    assert diff < 1e-4, "merged and unmerged forward must agree"
    assert n_lora < n_full / 30 and rank <= 4


if __name__ == "__main__":
    main()
