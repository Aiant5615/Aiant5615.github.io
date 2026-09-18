"""Layer Normalization (Ba, Kiros & Hinton, 2016) — a from-scratch PyTorch implementation.

What is implemented (section / equation numbers follow the paper):
  * batch normalisation of the summed inputs, statistics over the minibatch, one pair per unit          (Section 2, eq. 2)
  * layer normalisation: mu^l and sigma^l over the H units of one example, gain g and bias b            (Section 3, eqs. 3-4)
  * the LN recurrent network h^t = f(g/sigma^t * (a^t - mu^t) + b) applied at every time step           (Section 3.1, eq. 5)
    and the LN-LSTM (the four gate pre-activations and the cell normalised separately)                  (Section 3.1 / Suppl.)
  * numerical checks of the invariance table: weight re-scaling, weight re-centering, single-weight re-scaling,
    data re-scaling, data re-centering, single-example re-scaling, for BN and LN                        (Table 1)
  * the geometry result: the gradient of an LN unit w.r.t. its weight vector shrinks as 1/||w||         (Section 5.2)
  * experiments: batch size 1 (where BN is undefined) and an LSTM on a long sequence task with and without LN
Simplifications: a synthetic "sum of marked inputs" sequence task instead of the paper's attentive reader / MNIST /
draw benchmarks; small models trained for a few hundred steps.

Run:  python layer_normalization.py        (CPU, about 30 s)
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ───────────────────────── the two normalisers (eqs. 2-4) ─────────────────────────
def batch_norm(a, g, eps=1e-5):
    """a: (N, H) summed inputs. mu_i, sigma_i over the N examples of the minibatch, one pair per unit i (eq. 2)."""
    mu, var = a.mean(0, keepdim=True), a.var(0, unbiased=False, keepdim=True)
    return g * (a - mu) / (var + eps).sqrt()


def layer_norm(a, g, b, eps=1e-5):
    """a: (..., H). mu^l = mean_i a_i, sigma^l = sqrt(mean_i (a_i - mu^l)^2) over the H units of EACH example (eq. 3);
    output g/sigma^l * (a - mu^l) + b, the same mu, sigma for every unit of the example (eq. 4, before f)."""
    mu = a.mean(-1, keepdim=True)
    sigma = (((a - mu) ** 2).mean(-1, keepdim=True) + eps).sqrt()     # eps inside the sqrt (official code): finite
    return g * (a - mu) / sigma + b                                       # gradient even for an all-zero vector (h^0 = 0)


class LayerNorm(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.g, self.b = nn.Parameter(torch.ones(H)), nn.Parameter(torch.zeros(H))

    def forward(self, a):
        return layer_norm(a, self.g, self.b)


# ───────────────────────── recurrent networks with and without LN (Section 3.1) ─────────────────────────
class LSTM(nn.Module):
    """A plain LSTM, or the LN-LSTM of the paper (supplementary, "Layer normalized LSTM"):
        (f, i, o, g) = LN(W_h h^{t-1}; α_1, β_1) + LN(W_x x^t; α_2, β_2) + b
        c^t = σ(f) ⊙ c^{t-1} + σ(i) ⊙ tanh(g),      h^t = σ(o) ⊙ tanh(LN(c^t; α_3, β_3))
    i.e. the recurrent and the input term are normalised separately over the whole 4H pre-activation vector,
    each with its own gain and bias, and the cell is normalised before the output gate."""
    def __init__(self, d_in, H, ln=False):
        super().__init__()
        self.H, self.ln = H, ln
        self.W_xh, self.W_hh = nn.Linear(d_in, 4 * H, bias=False), nn.Linear(H, 4 * H, bias=False)
        self.b = nn.Parameter(torch.zeros(4 * H))
        with torch.no_grad():                                            # forget-gate bias 1 (the usual LSTM init)
            self.b[H:2 * H].fill_(1.0)
        if ln:
            self.ln_h, self.ln_x, self.ln_cell = LayerNorm(4 * H), LayerNorm(4 * H), LayerNorm(H)

    def forward(self, x):                                                # x: (B, T, d_in) -> h_T (B, H)
        B, T, _ = x.shape
        h = c = x.new_zeros(B, self.H)
        for t in range(T):
            rec, inp = self.W_hh(h), self.W_xh(x[:, t])
            if self.ln:
                rec, inp = self.ln_h(rec), self.ln_x(inp)
            i, f, o, u = (rec + inp + self.b).chunk(4, -1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(u)
            h = torch.sigmoid(o) * torch.tanh(self.ln_cell(c) if self.ln else c)
        return h


def make_sequences(B, T):
    """Sequence task: T steps of (value ~ U(0,1), marker in {0,1}); the target is the sum of the values whose marker is 1
    (two markers per sequence). Long sequences make the recurrent state drift, which LN keeps in check."""
    v = torch.rand(B, T)
    m = torch.zeros(B, T)
    idx = torch.stack([torch.randperm(T)[:2] for _ in range(B)])
    m.scatter_(1, idx, 1.0)
    return torch.stack([v, m], -1), (v * m).sum(1)


def train_lstm(ln, steps=300, T=40, lr=1e-2):
    model, head = LSTM(2, 32, ln=ln), nn.Linear(32, 1)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=lr)
    curve = []
    for step in range(1, steps + 1):
        x, y = make_sequences(32, T)
        loss = F.mse_loss(head(model(x)).squeeze(-1), y)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        curve.append(loss.item())
    return curve


# ───────────────────────── invariance checks (Table 1) and geometry (Section 5.2) ─────────────────────────
def invariance_table():
    """Feed the same summed inputs through BN and LN after each perturbation and report whether the output changed."""
    N, H, D = 8, 6, 5
    x, W = torch.randn(N, D), torch.randn(H, D)
    g, b = torch.ones(H), torch.zeros(H)
    out = lambda x_, W_: (batch_norm(x_ @ W_.T, g), layer_norm(x_ @ W_.T, g, b))
    bn0, ln0 = out(x, W)
    same = lambda a, a0: torch.allclose(a, a0, atol=1e-4)
    x_one = x.clone(); x_one[0] *= 3.0                                   # re-scale a single example
    W_one = W.clone(); W_one[0] *= 3.0                                   # re-scale a single weight vector (one unit)
    cases = [("weight matrix re-scaling", x, 2.5 * W), ("weight matrix re-centering", x, W + 0.7),
             ("single weight re-scaling", x, W_one), ("data re-scaling", 4.0 * x, W),
             ("data re-centering", x + 1.3, W), ("single example re-scaling", x_one, W)]
    paper = {"weight matrix re-scaling": ("yes", "yes"), "weight matrix re-centering": ("no", "yes"),
             "single weight re-scaling": ("yes", "no"), "data re-scaling": ("yes", "yes"),
             "data re-centering": ("yes", "no"), "single example re-scaling": ("no", "yes")}
    print(f"  {'perturbation':28s} {'BN invariant':>13s} {'LN invariant':>13s}   (paper: BN / LN)")
    ok = True
    for name, x_, W_ in cases:
        bn, ln = out(x_, W_)
        got = ("yes" if same(bn, bn0) else "no", "yes" if same(ln, ln0) else "no")
        ok &= got == paper[name]
        print(f"  {name:28s} {got[0]:>13s} {got[1]:>13s}   ({paper[name][0]} / {paper[name][1]})")
    return ok


def gradient_vs_weight_norm():
    """Section 5.2: for a normalised layer the gradient w.r.t. the incoming weights scales like 1/||w||, so a larger weight
    vector receives smaller relative updates (the 'implicit learning-rate' effect)."""
    x, W0, downstream = torch.randn(64, 5), torch.randn(6, 5) * 0.4, torch.randn(64, 6)
    norms, grads = [1.0, 2.0, 4.0, 8.0], []
    for s in norms:                                                      # the same weights scaled by s: LN's output is unchanged
        W = (s * W0).requires_grad_(True)
        h = layer_norm(x @ W.T, torch.ones(6), torch.zeros(6))
        (h * downstream).sum().backward()                                # a fixed arbitrary downstream loss
        grads.append(W.grad.norm().item())
    return norms, grads


def main():
    t0 = time.time()
    print("== Table 1: invariance properties (numerical) ==")
    ok = invariance_table()
    norms, grads = gradient_vs_weight_norm()
    print("== Section 5.2: ||dL/dW|| as ||W|| grows by 2x each step:", " ".join(f"{g:.3f}" for g in grads), "(halves each time: ∝ 1/||W||)")
    # batch size 1: BN has no statistics (sigma = 0 -> the normalised activation is 0/0), LN is unaffected
    a = torch.randn(1, 6)
    print(f"== batch size 1: BN output {batch_norm(a, torch.ones(6)).abs().max().item():.3f} (degenerate: every unit becomes 0), "
          f"LN output std {layer_norm(a, torch.ones(6), torch.zeros(6)).std().item():.3f} (fine)")
    print("== LSTM vs LN-LSTM on the marked-sum task, T = 40 (Figures 1-3 style convergence comparison)")
    plain, ln = train_lstm(False), train_lstm(True)
    for k in (50, 100, 200, 300):
        print(f"  step {k:3d}  loss  plain {sum(plain[k-25:k])/25:.4f}   LN {sum(ln[k-25:k])/25:.4f}")
    print(f"({time.time() - t0:.1f} s)")
    assert ok, "invariance table does not match Table 1"
    assert all(abs(grads[k] / grads[k + 1] - 2) < 0.05 for k in range(3)), "gradient should scale as 1/||W||"
    assert sum(ln[-25:]) < 0.5 * sum(plain[-25:]), "the LN-LSTM should converge faster than the plain LSTM"


if __name__ == "__main__":
    main()
