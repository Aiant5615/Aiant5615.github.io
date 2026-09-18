"""RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021) — from-scratch PyTorch.

What is implemented (section numbers follow the paper):
  * the rotation frequencies theta_i = 10000^(-2(i-1)/d) and the block-diagonal R^d_{Theta,m}     (3.2.2, eq. 15-16)
  * the cheap implementation  R x = x * cos + rotate_half(x) * sin  (no matrix is built)            (3.4.2, eq. 34, Fig. 1)
  * numerical check of the defining property <f_q(x_m, m), f_k(x_n, n)> = g(x_m, x_n, m - n)          (3.1 eq. 11, 3.2.1 eq. 13)
  * the long-term decay bound  (1/(d/2)) * sum_i |S_i|,  S_j = sum_{i<j} e^{i (m-n) theta_i}        (3.4.3, eq. 35-37, Fig. 2)
  * a tiny causal LM with RoPE on q,k vs. learned absolute positions on a relative-offset task: shifting every
    position index by a constant leaves the RoPE model's logits unchanged (translation invariance of eq. 13)
Simplifications: 1-layer model, a "copy the token two positions back" task instead of MLM/WMT, and the
  shift-invariance test (plus a length-extrapolation print) instead of the paper's pretraining curves.

Run:  python roformer.py        (CPU, about 6 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ───────────────────────── rotary position embedding ─────────────────────────
def rope_freqs(d, base=10000.0):
    """theta_i = base^(-2(i-1)/d), i = 1..d/2   (eq. 15).  Returns (d/2,)."""
    return base ** (-torch.arange(0, d, 2).float() / d)


def rotate_half(x):
    """(x_1, x_2, x_3, x_4, ...) -> (-x_2, x_1, -x_4, x_3, ...): each adjacent pair rotated by 90 degrees (eq. 34)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x, pos, theta):
    """R^d_{Theta,m} x = x * [cos m th_1, cos m th_1, cos m th_2, ...] + rotate_half(x) * [sin m th_1, sin m th_1, ...]   (eq. 34).
    x: (..., L, d); pos: (L,) integer positions m; theta: (d/2,)."""
    ang = pos[:, None].float() * theta[None, :]                       # (L, d/2): m * theta_i
    cos, sin = ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)   # (L, d)
    return x * cos + rotate_half(x) * sin


def rotation_matrix(m, theta):
    """The explicit block-diagonal R^d_{Theta,m} of eq. 15 (only for the check; never used in the model)."""
    d = 2 * theta.numel()
    R = torch.zeros(d, d)
    for i, th in enumerate(theta):
        c, s = math.cos(m * th), math.sin(m * th)
        R[2 * i: 2 * i + 2, 2 * i: 2 * i + 2] = torch.tensor([[c, -s], [s, c]])
    return R


def decay_bound(rel_dist, theta):
    """Relative upper bound of eq. 37:  (1/(d/2)) sum_{i=1}^{d/2} |S_i|,  S_j = sum_{i=0}^{j-1} exp(i (m-n) theta_i)   (Section 3.4.3)."""
    phases = torch.exp(1j * rel_dist * theta.to(torch.complex64))    # (d/2,) unit complex numbers e^{i (m-n) theta_i}
    S = torch.cumsum(phases, 0)                                      # S_1..S_{d/2}
    return S.abs().mean().item()


# ───────────────────────── tiny causal LM (RoPE or absolute positions) ─────────────────────────
class Attention(nn.Module):
    """Self-attention where q and k are rotated by their positions (eq. 16): only q, k get RoPE, never v."""
    def __init__(self, d, h, rope):
        super().__init__()
        self.h, self.d_k, self.rope = h, d // h, rope
        self.qkv, self.out = nn.Linear(d, 3 * d, bias=False), nn.Linear(d, d, bias=False)
        self.register_buffer("theta", rope_freqs(self.d_k))

    def forward(self, x, pos):
        B, L, _ = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)     # each (B, h, L, d_k)
        if self.rope:
            q, k = apply_rope(q, pos, self.theta), apply_rope(k, pos, self.theta)      # f_q, f_k of eq. 16
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), float("-inf"))
        o = scores.softmax(-1) @ v
        return self.out(o.transpose(1, 2).reshape(B, L, -1))


class TinyLM(nn.Module):
    def __init__(self, vocab, d=32, h=2, rope=True, max_len=256):
        super().__init__()
        self.rope = rope
        self.emb = nn.Embedding(vocab, d)
        self.abs_pos = None if rope else nn.Embedding(max_len, d)   # baseline: learned absolute position embeddings
        self.attn, self.ff = Attention(d, h, rope), nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.n1, self.n2, self.head = nn.LayerNorm(d), nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, idx, shift=0):
        pos = shift + torch.arange(idx.size(1))            # position ids m; `shift` moves the whole sequence
        x = self.emb(idx)
        if self.abs_pos is not None:
            x = x + self.abs_pos(pos)
        x = x + self.attn(self.n1(x), pos)
        x = x + self.ff(self.n2(x))
        return self.head(x)


def make_batch(B, L, vocab, offset=2):
    """Task: predict the token `offset` positions back (a purely relative pattern). Positions < offset predict token 0."""
    x = torch.randint(1, vocab, (B, L))
    y = torch.zeros_like(x)
    y[:, offset:] = x[:, :-offset]
    return x, y


def train_and_eval(rope, vocab=16, L_train=24, shift=100, steps=800):
    """Train at lengths <= L_train starting at position 0; test (a) as trained, (b) the same inputs with every position
    index shifted by `shift`, (c) sequences twice as long. Returns (acc_a, acc_b, acc_c, max |logit_a - logit_b|)."""
    model = TinyLM(vocab, rope=rope)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(1, steps + 1):
        x, y = make_batch(64, int(torch.randint(4, L_train + 1, ())), vocab)     # variable length <= L_train
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 400 == 0:
            print(f"  [{'RoPE' if rope else 'abs '}] step {step:4d}  loss {loss.item():.3f}")
    model.eval()
    acc = lambda logits, y: (logits.argmax(-1)[:, 2:] == y[:, 2:]).float().mean().item()
    with torch.no_grad():
        x, y = make_batch(256, L_train, vocab)
        la, lb = model(x), model(x, shift=shift)
        x2, y2 = make_batch(256, 2 * L_train, vocab)
        return acc(la, y), acc(lb, y), acc(model(x2), y2), (la - lb).abs().max().item()


def main():
    t0 = time.time()
    d, theta = 64, rope_freqs(64)

    # 1. relative-position property (eq. 11 / 13): the score depends on (x_m, x_n, m - n) only
    print("1) <R_m q, R_n k> depends only on m - n:")
    q, k = torch.randn(d), torch.randn(d)
    scores = {(m, n): (apply_rope(q[None], torch.tensor([m]), theta) @ apply_rope(k[None], torch.tensor([n]), theta).T).item()
              for m, n in [(3, 0), (10, 7), (40, 37), (0, 3), (25, 28)]}
    for (m, n), s in scores.items():
        print(f"   m={m:2d} n={n:2d} (m-n={m - n:+d})  score {s:+.5f}")
    same = [scores[(3, 0)], scores[(10, 7)], scores[(40, 37)]]
    assert max(same) - min(same) < 1e-4 and abs(scores[(0, 3)] - scores[(25, 28)]) < 1e-4
    # the cheap form equals the explicit block-diagonal rotation, and R_m^T R_n = R_{n-m}  (eq. 16 / Section 3.2.2)
    assert torch.allclose(apply_rope(q[None], torch.tensor([7]), theta)[0], rotation_matrix(7, theta) @ q, atol=1e-5)
    assert torch.allclose(rotation_matrix(3, theta).T @ rotation_matrix(10, theta), rotation_matrix(7, theta), atol=1e-5)
    print("   cheap implementation == explicit R^d_{Theta,m};  R_m^T R_n == R_{n-m};  ||R_m q|| == ||q||:",
          f"{apply_rope(q[None], torch.tensor([7]), theta).norm().item():.4f} vs {q.norm().item():.4f}")

    # 2. long-term decay (Section 3.4.3, Figure 2)
    dists = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
    bounds = [decay_bound(r, theta) for r in dists]
    print("2) relative upper bound vs |m - n|:", "  ".join(f"{r}:{b:.1f}" for r, b in zip(dists, bounds)))
    near, far = sum(bounds[:3]) / 3, sum(bounds[-3:]) / 3
    assert bounds[0] == max(bounds) and far < 0.5 * near, "the bound should decay with relative distance"

    # 3. tiny LM: RoPE vs learned absolute positions on a relative-offset task
    print("3) tiny causal LM, 'copy the token 2 back', trained at L<=24 with positions 0..L-1")
    r, a = train_and_eval(rope=True), train_and_eval(rope=False)
    print(f"   accuracy            trained | positions +100 | length x2      max|logit diff| under the +100 shift")
    print(f"   RoPE                {r[0]:.3f}   | {r[1]:.3f}          | {r[2]:.3f}          {r[3]:.1e}")
    print(f"   absolute (learned)  {a[0]:.3f}   | {a[1]:.3f}          | {a[2]:.3f}          {a[3]:.1e}    ({time.time() - t0:.1f} s)")
    assert r[0] > 0.95, "RoPE model should solve the relative-offset task"
    assert r[3] < 1e-3 and r[1] == r[0], "RoPE logits must be invariant to shifting all positions (only m-n matters)"
    assert a[1] < r[1] - 0.3, "absolute positions are not shift-invariant"


if __name__ == "__main__":
    main()
