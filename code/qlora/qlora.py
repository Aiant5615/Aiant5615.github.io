"""QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023) — a from-scratch PyTorch implementation.

What is implemented (section numbers follow the paper):
  * block-wise absmax quantization: X_int = round(c X) with one constant c per block of 64 values,
    dequant(c, X_int) = X_int / c                                                                   (Section 2, eqs. 1-2)
  * the 4-bit NormalFloat data type: 2^{k-1} negative and 2^{k-1}+1 positive levels taken from quantiles of N(0, 1)
    and normalised to [-1, 1] so that 0 is exact; checked against the constants of bitsandbytes    (Section 3, eq. 3)
  * double quantization: the FP32 block constants are themselves quantized to Int8 in blocks of 256 (after removing
    their mean, as in the official code), bringing the overhead from 0.5 to ~0.127 bits/parameter          (Section 3)
  * the QLoRA linear layer, eq. 5: Y = X doubleDequant(c1, c2, W_NF4) + X L1 L2 with the base weights frozen in 4 bits
    and gradients flowing only into the 16-bit LoRA factors; LoRA on ALL linear layers             (Section 3, eq. 5)
  * Figure 3: NF4 has a lower quantization error than Int4 and FP4 (E2M1) on normally distributed weights
  * Table 3: 4-bit without fine-tuning loses accuracy on the base task; QLoRA on the downstream task matches
    16-bit LoRA and 16-bit full fine-tuning, while adapters on only some layers do not
Simplifications: the "pretrained LLM" is a 4-layer MLP trained on a synthetic classification task; computations run
in FP32 instead of BF16; the paged optimizer (an NVIDIA unified-memory feature) has no CPU-only counterpart and is
not implemented.

Run:  python qlora.py        (CPU, about 15 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
NORMAL = torch.distributions.Normal(0.0, 1.0)


# ───────────────────────── data types: Int4, FP4 and NormalFloat4 (Section 3) ─────────────────────────
def nf4_levels(offset=0.9677083):
    """NF4: the 16 levels are quantiles of N(0, 1) (eq. 3), 8 on the negative side and 9 on the positive side (with 0)
    so that zero is represented exactly, then normalised to [-1, 1]. `offset` is the outermost quantile, the constant
    used by bitsandbytes' create_normal_map."""
    pos = NORMAL.icdf(torch.linspace(offset, 0.5, 9)[:-1])                 # 8 positive levels (eq. 3 with 2^{k-1}+1 bins)
    neg = -NORMAL.icdf(torch.linspace(offset, 0.5, 8)[:-1])                # 7 negative levels
    levels = torch.cat([pos, torch.zeros(1), neg]).sort().values
    return levels / levels.abs().max()


NF4 = nf4_levels()
INT4 = torch.linspace(-1, 1, 16)                                              # 16 evenly spaced levels
FP4 = torch.tensor([s * m for s in (-1, 1) for m in (0, 0.5, 1, 1.5, 2, 3, 4, 6)]).unique() / 6   # E2M1: 15 distinct values
BNB_NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                        -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                        0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                        0.7229568362236023, 1.0])                             # the table hard-coded in bitsandbytes


# ───────────────────────── block-wise quantization and double quantization (Sections 2-3) ─────────────────────────
def quantize_blockwise(x, levels, block=64):
    """Eq. 1 with blocks: each block of 64 values is rescaled by its absmax to [-1, 1] and every value is snapped to
    the nearest level. Returns (uint8 indices, FP32 absmax per block)."""
    flat = x.reshape(-1, block)
    absmax = flat.abs().amax(1).clamp(min=1e-12)
    idx = (flat / absmax[:, None])[..., None].sub(levels).abs().argmin(-1)
    return idx.to(torch.uint8), absmax


def dequantize_blockwise(idx, absmax, levels, shape):
    """Eq. 2: dequant(c, X) = X / c, i.e. level[idx] * absmax."""
    return (levels[idx.long()] * absmax[:, None]).reshape(shape)


def double_quantize(absmax, block=256):
    """Second quantization: the FP32 constants c2 (one per 64 weights) are centred and quantized to Int8 in blocks of
    256 with their own FP32 constant c1, as bitsandbytes does with compress_statistics=True."""
    offset = absmax.mean()
    centred = F.pad(absmax - offset, (0, -absmax.numel() % block)).reshape(-1, block)   # pad to whole blocks
    c1 = centred.abs().amax(1).clamp(min=1e-12)
    c2 = torch.round(centred / c1[:, None] * 127).to(torch.int8)
    return c2, c1, offset


def double_dequantize(c2, c1, offset, n):
    return (c2.float() / 127 * c1[:, None]).reshape(-1)[:n] + offset


def bits_per_parameter(double):
    """Storage per weight: 4 bits + the block constants. 4 + 32/64 = 4.5 without, 4 + 8/64 + 32/(64*256) = 4.127 with
    double quantization (the paper's 0.5 -> 0.127 bits of overhead)."""
    return 4 + (8 / 64 + 32 / (64 * 256) if double else 32 / 64)


# ───────────────────────── the QLoRA linear layer (eq. 5) ─────────────────────────
class QLoRALinear(nn.Module):
    """Y = X doubleDequant(c1, c2, W_NF4) + X L1 L2. The base weight is stored as NF4 indices + double-quantized
    constants (buffers, no gradient); L1 (r x d_in, Gaussian) and L2 (d_out x r, zero) are the only parameters."""
    def __init__(self, linear, r=8, alpha=16, levels=NF4):
        super().__init__()
        W = linear.weight.detach()
        self.shape, self.scale = W.shape, alpha / r
        self.register_buffer("levels", levels)                               # NF4 (or Int4 / FP4 for Figure 3)
        idx, absmax = quantize_blockwise(W, levels)
        c2, c1, offset = double_quantize(absmax)
        self.register_buffer("w_nf4", idx); self.register_buffer("c2", c2); self.register_buffer("c1", c1)
        self.register_buffer("offset", offset); self.register_buffer("bias", linear.bias.detach().clone())
        self.L1 = nn.Parameter(torch.randn(r, W.shape[1]) / math.sqrt(W.shape[1]))
        self.L2 = nn.Parameter(torch.zeros(W.shape[0], r))

    def dequantized_weight(self):
        return dequantize_blockwise(self.w_nf4, double_dequantize(self.c2, self.c1, self.offset, self.w_nf4.shape[0]), self.levels, self.shape)

    def forward(self, x):
        return F.linear(x, self.dequantized_weight(), self.bias) + self.scale * (x @ self.L1.T) @ self.L2.T


class LoRALinear(nn.Module):
    """16-bit LoRA for comparison: the same adapters on a frozen full-precision weight."""
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.base, self.scale = linear, alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.L1 = nn.Parameter(torch.randn(r, linear.in_features) / math.sqrt(linear.in_features))
        self.L2 = nn.Parameter(torch.zeros(linear.out_features, r))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.L1.T) @ self.L2.T


# ───────────────────────── the "pretrained model" and the two tasks ─────────────────────────
D_IN, D, N_CLASSES = 32, 128, 8
TEACHER = nn.Sequential(nn.Linear(D_IN, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh()).requires_grad_(False)
HEAD_BASE = torch.randn(64, N_CLASSES)
HEAD_NEW = nn.Sequential(nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, N_CLASSES)).requires_grad_(False)


def make_data(n, head):
    """Inputs x ~ N(0, I); the label is the argmax of a readout of a hidden teacher network's features. The base task
    (linear readout) and the downstream task (an extra nonlinear layer) share the teacher, so the pretrained features
    transfer, but the downstream task needs changes deeper than one layer."""
    x = torch.randn(n, D_IN)
    with torch.no_grad():
        feats = TEACHER(x)
        return x, (feats @ head if torch.is_tensor(head) else head(feats)).argmax(1)


def make_model():
    return nn.Sequential(nn.Linear(D_IN, D), nn.GELU(), nn.Linear(D, D), nn.GELU(), nn.Linear(D, D), nn.GELU(), nn.Linear(D, N_CLASSES))


def train(model, data, steps, lr):
    x, y = data
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        idx = torch.randint(0, x.size(0), (128,))
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def accuracy(model, data):
    x, y = data
    return (model(x).argmax(1) == y).float().mean().item()


def adapt(model, wrap, layers):
    """Replace the chosen Linear layers of a copy of `model` by 4-bit or 16-bit LoRA layers; everything else frozen."""
    m = make_model(); m.load_state_dict(model.state_dict())
    for p in m.parameters():
        p.requires_grad_(False)
    for i in layers:
        m[i] = wrap(m[i])
    return m


def main():
    t0 = time.time()
    # 1) the data type
    print(f"NF4 levels vs bitsandbytes table: max |difference| = {(NF4 - BNB_NF4).abs().max():.1e}")
    assert torch.allclose(NF4, BNB_NF4, atol=1e-6), "NF4 construction should reproduce the official constants"
    w = torch.randn(1 << 16) * 0.02                                        # pretrained weights: roughly N(0, 0.02^2)
    print("Figure 3 proxy: relative quantization error of Gaussian weights, block size 64")
    for name, levels in (("Int4", INT4), ("FP4 (E2M1)", FP4), ("NF4", NF4)):
        err = (dequantize_blockwise(*quantize_blockwise(w, levels), levels, w.shape) - w).pow(2).mean() / w.pow(2).mean()
        print(f"  {name:12s} relative MSE {err.item():.4f}")
        if name != "NF4":
            assert err > (dequantize_blockwise(*quantize_blockwise(w, NF4), NF4, w.shape) - w).pow(2).mean() / w.pow(2).mean()
    # 2) double quantization
    idx, absmax = quantize_blockwise(w, NF4)
    c2, c1, off = double_quantize(absmax)
    rel = ((double_dequantize(c2, c1, off, absmax.numel()) - absmax).abs() / absmax).max().item()
    print(f"double quantization: {absmax.numel()} FP32 constants -> Int8 + {c1.numel()} FP32; max relative error {rel:.4f}; "
          f"{bits_per_parameter(False):.3f} -> {bits_per_parameter(True):.3f} bits/parameter")
    assert rel < 0.02
    # 3) fine-tuning (Table 3)
    base = make_model()
    task_a, task_b = make_data(6000, HEAD_BASE), make_data(6000, HEAD_NEW)
    test_a, test_b = make_data(20000, HEAD_BASE), make_data(2000, HEAD_NEW)
    train(base, task_a, 1500, 1e-3)
    acc16 = accuracy(base, test_a)
    q_acc = {name: accuracy(adapt(base, lambda l: QLoRALinear(l, levels=lv), [0, 2, 4, 6]), test_a)   # L2 = 0: the plain 4-bit model
             for name, lv in (("Int4", INT4), ("FP4", FP4), ("NF4", NF4))}
    acc4 = q_acc["NF4"]
    print(f"base task without fine-tuning (Figure 3 / Table 3 'no fine-tuning'): 16-bit {acc16:.4f}   " +
          "   ".join(f"{k} {v:.4f}" for k, v in q_acc.items()))
    n_base = sum(p.numel() for p in base.parameters())
    runs = {
        "16-bit full fine-tuning": (adapt(base, lambda l: l, []), 1e-3),
        "16-bit LoRA (all layers)": (adapt(base, lambda l: LoRALinear(l), [0, 2, 4, 6]), 3e-3),
        "QLoRA (all layers)": (adapt(base, lambda l: QLoRALinear(l), [0, 2, 4, 6]), 3e-3),
        "QLoRA (last layer only)": (adapt(base, lambda l: QLoRALinear(l), [6]), 3e-3),
    }
    for m in runs["16-bit full fine-tuning"][0].parameters():
        m.requires_grad_(True)
    print(f"downstream task (new nonlinear readout), 600 steps each:      trainable params   accuracy")
    res = {}
    for name, (m, lr) in runs.items():
        train(m, task_b, 600, lr)
        res[name] = accuracy(m, test_b)
        n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"  {name:28s} {n_tr:12,d} / {n_base:,}   {res[name]:.3f}")
    print(f"({time.time() - t0:.1f} s)")
    assert acc4 <= acc16 and q_acc["NF4"] >= max(q_acc["Int4"], q_acc["FP4"]) - 0.005, "NF4 should be the most faithful 4-bit type"
    assert res["QLoRA (all layers)"] > res["16-bit full fine-tuning"] - 0.03, "QLoRA should match full fine-tuning"
    assert abs(res["QLoRA (all layers)"] - res["16-bit LoRA (all layers)"]) < 0.03, "QLoRA should match 16-bit LoRA"
    assert res["QLoRA (last layer only)"] < res["QLoRA (all layers)"] - 0.03, "adapters on all layers should matter"


if __name__ == "__main__":
    main()
