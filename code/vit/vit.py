"""An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT, Dosovitskiy et al., ICLR 2021).

What is implemented (section / equation numbers follow the paper):
  * image -> N = HW/P^2 flattened P x P patches and the linear patch embedding E            (Section 3.1, eq. 1)
  * learnable [class] token x_class prepended, learned 1-D position embeddings E_pos         (eq. 1)
  * pre-norm Transformer encoder:  z'_l = MSA(LN(z_{l-1})) + z_{l-1},  z_l = MLP(LN(z'_l)) + z'_l   (eq. 2, 3)
  * image representation y = LN(z_L^0) (final state of the class token) + MLP head          (eq. 4)
  * fine-tuning at higher resolution: 2-D interpolation of E_pos, zero-initialised linear head   (Section 3.2)
  * the position-embedding similarity check of Figure 7 (centre): does E_pos recover the 2-D grid?
Simplifications: 16x16 one-channel synthetic images with a planted shape (square / plus / cross / ring) instead of
ImageNet, P = 4 so N = 16 patches, a 3-layer, 64-wide encoder, Adam, no dropout, no warm-up or weight decay.

Run:  python vit.py        (CPU, about 15 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ───────────────────────── synthetic images with planted shapes ─────────────────────────
def make_shapes(s=6):
    """Four s x s stencils: filled square, plus, cross (X) and ring. The class of an image is which stencil it contains."""
    sq = torch.ones(s, s)
    plus = torch.zeros(s, s); plus[s // 2 - 1 : s // 2 + 1, :] = 1; plus[:, s // 2 - 1 : s // 2 + 1] = 1
    cross = ((torch.eye(s) + torch.eye(s).flip(1)) > 0).float()
    ring = torch.ones(s, s); ring[1:-1, 1:-1] = 0
    return torch.stack([sq, plus, cross, ring])


SHAPES = make_shapes()


def make_batch(B, res=16, scale=1, noise=0.4):
    """(B, 1, res, res) images: one stencil (scaled by `scale`, nearest neighbour) at a random position, plus noise."""
    st = F.interpolate(SHAPES[:, None], scale_factor=scale, mode="nearest")[:, 0] if scale != 1 else SHAPES
    s, y = st.size(-1), torch.randint(0, 4, (B,))
    x = torch.zeros(B, 1, res, res)
    r, c = torch.randint(0, res - s + 1, (B,)), torch.randint(0, res - s + 1, (B,))
    for i in range(B):
        x[i, 0, r[i] : r[i] + s, c[i] : c[i] + s] = st[y[i]]
    return x + noise * torch.randn_like(x), y


# ───────────────────────── the Vision Transformer ─────────────────────────
class MSA(nn.Module):
    """Multi-head self-attention, exactly as in the Transformer (Appendix A, eq. 5-7)."""
    def __init__(self, D, heads):
        super().__init__()
        self.h, self.d_h = heads, D // heads
        self.qkv, self.proj = nn.Linear(D, 3 * D), nn.Linear(D, D)

    def forward(self, z):
        B, N, D = z.shape
        q, k, v = self.qkv(z).view(B, N, 3, self.h, self.d_h).permute(2, 0, 3, 1, 4)     # each (B, h, N, d_h)
        A = (q @ k.transpose(-2, -1) / math.sqrt(self.d_h)).softmax(-1)                   # eq. 6
        return self.proj((A @ v).transpose(1, 2).reshape(B, N, D))                        # eq. 7


class Block(nn.Module):
    """One pre-norm encoder layer:  z' = MSA(LN(z)) + z  (eq. 2);   z = MLP(LN(z')) + z'  (eq. 3)."""
    def __init__(self, D, heads, mlp_dim):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.msa = MSA(D, heads)
        self.mlp = nn.Sequential(nn.Linear(D, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, D))   # two layers with GELU

    def forward(self, z):
        z = self.msa(self.ln1(z)) + z
        return self.mlp(self.ln2(z)) + z


class ViT(nn.Module):
    def __init__(self, res=16, P=4, C=1, D=64, L=3, heads=4, mlp_dim=128, n_classes=4):
        super().__init__()
        self.P, self.grid = P, res // P
        N = self.grid ** 2                                                     # N = HW / P^2
        self.E = nn.Linear(P * P * C, D)                                       # patch embedding E in R^{(P^2 C) x D}
        self.x_class = nn.Parameter(torch.zeros(1, 1, D))                     # learnable [class] token
        self.E_pos = nn.Parameter(0.02 * torch.randn(1, N + 1, D))            # learned 1-D position embeddings
        self.blocks = nn.ModuleList([Block(D, heads, mlp_dim) for _ in range(L)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Sequential(nn.Linear(D, mlp_dim), nn.Tanh(), nn.Linear(mlp_dim, n_classes))  # MLP head (3.1)

    def patchify(self, x):
        """(B, C, H, W) -> (B, N, P^2 C): the only image-specific operation in the model."""
        B, C, H, W = x.shape; P = self.P
        x = x.unfold(2, P, P).unfold(3, P, P)                                  # (B, C, H/P, W/P, P, P)
        return x.reshape(B, C, -1, P * P).transpose(1, 2).reshape(B, -1, C * P * P)

    def forward(self, x):
        B = x.size(0)
        z = torch.cat([self.x_class.expand(B, -1, -1), self.E(self.patchify(x))], 1) + self.E_pos   # eq. 1
        for blk in self.blocks:
            z = blk(z)                                                         # eq. 2, 3 for l = 1..L
        y = self.ln(z[:, 0])                                                   # y = LN(z_L^0)      eq. 4
        return self.head(y)

    @torch.no_grad()
    def interpolate_pos(self, new_res):
        """Section 3.2: keep P, so the sequence gets longer; 2-D interpolate the pretrained patch positions to the new grid."""
        g, D = self.grid, self.E_pos.size(-1)
        cls, pos = self.E_pos[:, :1], self.E_pos[:, 1:].reshape(1, g, g, D).permute(0, 3, 1, 2)   # (1, D, g, g)
        self.grid = new_res // self.P
        pos = F.interpolate(pos, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        self.E_pos = nn.Parameter(torch.cat([cls, pos.permute(0, 2, 3, 1).reshape(1, -1, D)], 1))


def accuracy(model, res, scale, n=512):
    model.eval()
    with torch.no_grad():
        x, y = make_batch(n, res, scale)
        acc = (model(x).argmax(-1) == y).float().mean().item()
    model.train()
    return acc


def pos_embedding_grid_check(model):
    """Figure 7 (centre): cosine similarity of each patch position embedding to its row/column neighbours vs the rest."""
    g = model.grid
    pos = F.normalize(model.E_pos[0, 1:].detach(), dim=-1)
    sim = pos @ pos.T
    idx = torch.arange(g * g)
    r, c = idx // g, idx % g
    neigh = ((r[:, None] - r[None]).abs() + (c[:, None] - c[None]).abs()) == 1       # 4-neighbourhood on the grid
    far = ~neigh & ~torch.eye(g * g, dtype=torch.bool)
    return sim[neigh].mean().item(), sim[far].mean().item()


def main():
    t0 = time.time()
    model = ViT()
    print(f"ViT: P=4, N={model.grid ** 2} patches (+1 class token), parameters {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first_loss = None
    for step in range(1, 801):                                                  # "pre-training" at 16 px
        x, y = make_batch(64)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        first_loss = first_loss or loss.item()
        if step % 200 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}  test acc @16px {accuracy(model, 16, 1):.3f}")
    acc16 = accuracy(model, 16, 1)
    near, far = pos_embedding_grid_check(model)
    print(f"E_pos cosine similarity: grid neighbours {near:.3f} vs non-neighbours {far:.3f}  (Figure 7 centre: 2-D structure learned from 1-D ids)")

    # ── Section 3.2: fine-tune at higher resolution (24 px, shapes 1.5x larger, 36 patches instead of 16) ──
    model.interpolate_pos(24)
    print(f"interpolated E_pos to a {model.grid}x{model.grid} grid; acc @24px before fine-tuning {accuracy(model, 24, 1.5):.3f}")
    model.head = nn.Linear(64, 4); nn.init.zeros_(model.head.weight); nn.init.zeros_(model.head.bias)   # zero-init D x K head
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)              # SGD with momentum for fine-tuning (App. B.1.1)
    for step in range(1, 301):
        x, y = make_batch(64, 24, 1.5)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"fine-tune step {step:3d}  loss {loss.item():.3f}  test acc @24px {accuracy(model, 24, 1.5):.3f}")
    acc24 = accuracy(model, 24, 1.5)
    print(f"final: acc @16px {acc16:.3f}, acc @24px after fine-tuning {acc24:.3f}   ({time.time() - t0:.1f} s)")
    assert loss.item() < first_loss * 0.5 and acc16 > 0.9 and acc24 > 0.85, "ViT did not learn the shapes"


if __name__ == "__main__":
    main()
