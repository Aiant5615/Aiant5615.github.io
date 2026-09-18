"""Flamingo: a Visual Language Model for Few-Shot Learning (Alayrac et al., NeurIPS 2022) — the bridge, from scratch.

What is implemented (section / figure numbers follow the paper):
  * a frozen vision encoder producing a grid of features per image                                    (Section 2.1)
  * the Perceiver Resampler: R learned latent queries cross-attend to the flattened features, with the
    latents concatenated to the keys/values, then FFW; output is a fixed R visual tokens per image        (Section 2.1, Figure 5)
  * gated cross-attention-dense layers inserted before every frozen LM block:
        y <- y + tanh(a_xattn) * Attention(q = LN(y), kv = X);   y <- y + tanh(a_dense) * FFW(LN(y))    (Section 2.2, Figure 4)
    with both gates a initialised to 0, so the LM is untouched at step 0; gate growth is printed        (Figure 6)
  * interleaved <image>/text sequences and the per-image cross-attention mask: a text token attends only to
    the visual tokens of the single image that precedes it                                             (Section 2.3, Figure 7)
  * the training loss  -sum_l log p(y_l | y_<l, x_<=l), trained on the new layers only (LM + vision frozen)   (Section 2.4)
Simplifications: the "pretrained" LM is a 2-layer decoder pretrained here on text-only captions; the vision encoder is a frozen
random patch projection; images are 16x16 synthetic coloured shapes; captions are "<image> red square ." over a 13-word vocab.

Run:  python flamingo.py        (CPU, about 10 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

WORDS = ["<pad>", "<bos>", "<image>", ".", "red", "green", "blue", "square", "plus", "cross", "ring", "a", "<eos>"]
TOK = {w: i for i, w in enumerate(WORDS)}
IMG, PAD = TOK["<image>"], TOK["<pad>"]


def make_stencils(s=6):
    sq = torch.ones(s, s)
    plus = torch.zeros(s, s); plus[2:4, :] = 1; plus[:, 2:4] = 1
    cross = ((torch.eye(s) + torch.eye(s).flip(1)) > 0).float()
    ring = torch.ones(s, s); ring[1:-1, 1:-1] = 0
    return torch.stack([sq, plus, cross, ring])


STENCILS = make_stencils()


def make_images(B, res=16, noise=0.2):
    color, shape = torch.randint(0, 3, (B,)), torch.randint(0, 4, (B,))
    x = torch.zeros(B, 3, res, res)
    r, c = torch.randint(0, 11, (B,)), torch.randint(0, 11, (B,))
    for i in range(B):
        x[i, color[i], r[i] : r[i] + 6, c[i] : c[i] + 6] = STENCILS[shape[i]]
    return x + noise * torch.randn_like(x), color, shape


def make_interleaved(B, n_img=2):
    """Section 2.3: an M3W-style sequence "<bos> <image> a red square . <image> a blue ring . <eos>" with n_img images."""
    imgs, colors, shapes = make_images(B * n_img)
    imgs = imgs.view(B, n_img, 3, 16, 16); colors, shapes = colors.view(B, n_img), shapes.view(B, n_img)
    seq = torch.full((B, 1 + 5 * n_img + 1), PAD)
    seq[:, 0] = TOK["<bos>"]
    for j in range(n_img):
        seq[:, 1 + 5 * j] = IMG; seq[:, 2 + 5 * j] = TOK["a"]
        seq[:, 3 + 5 * j] = 4 + colors[:, j]; seq[:, 4 + 5 * j] = 7 + shapes[:, j]; seq[:, 5 + 5 * j] = TOK["."]
    seq[:, -1] = TOK["<eos>"]
    return imgs, seq


# ───────────────────────── frozen backbones ─────────────────────────
class VisionEncoder(nn.Module):
    """Stands in for the frozen NFNet: a (frozen, random) linear patch projection -> 16 feature vectors per image."""
    def __init__(self, P=4, d_v=48):
        super().__init__()
        self.P, self.proj = P, nn.Linear(3 * P * P, d_v)

    def forward(self, x):                                   # (B, 3, 16, 16) -> (B, 16, d_v)
        B, C, H, W = x.shape; P = self.P
        patches = x.unfold(2, P, P).unfold(3, P, P).reshape(B, C, -1, P * P).transpose(1, 2).reshape(B, -1, C * P * P)
        return self.proj(patches)


class LMBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ffw = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, y):
        L = y.size(1); causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        y = y + self.attn(self.ln1(y), self.ln1(y), self.ln1(y), attn_mask=causal, need_weights=False)[0]
        return y + self.ffw(self.ln2(y))


class LM(nn.Module):
    """A small decoder-only LM (stands in for Chinchilla); pretrained on text only, then frozen."""
    def __init__(self, vocab, d=64, h=4, n_layers=2, max_len=32):
        super().__init__()
        self.emb, self.pos = nn.Embedding(vocab, d), nn.Parameter(0.02 * torch.randn(1, max_len, d))
        self.blocks = nn.ModuleList([LMBlock(d, h) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, seq):
        y = self.emb(seq) + self.pos[:, : seq.size(1)]
        for blk in self.blocks:
            y = blk(y)
        return self.head(self.ln(y))


# ───────────────────────── the trainable bridge ─────────────────────────
def attention(q, k, v, mask=None):
    """Plain multi-head attention on (B, h, L, d) tensors; rows with nothing to attend to return zeros."""
    s = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is None:
        return s.softmax(-1) @ v
    out = s.masked_fill(~mask, -1e9).softmax(-1) @ v
    return out * mask.any(-1, keepdim=True)                  # a text token before the first image sees no image


class PerceiverAttention(nn.Module):
    """Figure 5 pseudocode: attention_i(q = latents, kv = concat([x_f, latents]))."""
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln_x, self.ln_l = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, x_f, latents):
        B, R, d = latents.shape
        x_f, latents = self.ln_x(x_f), self.ln_l(latents)
        q = self.q(latents).view(B, R, self.h, -1).transpose(1, 2)
        k, v = self.kv(torch.cat([x_f, latents], 1)).chunk(2, -1)                  # keys/values: features AND latents
        k, v = (t.view(B, -1, self.h, d // self.h).transpose(1, 2) for t in (k, v))
        return self.o(attention(q, k, v).transpose(1, 2).reshape(B, R, d))


class PerceiverResampler(nn.Module):
    def __init__(self, d_v, d, R=8, n_layers=2, h=4):
        super().__init__()
        self.proj, self.latents = nn.Linear(d_v, d), nn.Parameter(torch.randn(R, d) * 0.5)    # R learned latent queries
        self.attn = nn.ModuleList([PerceiverAttention(d, h) for _ in range(n_layers)])
        self.ffw = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(n_layers)])

    def forward(self, x_f):                                  # (B, T*S, d_v) flattened features -> (B, R, d)
        x_f, x = self.proj(x_f), self.latents.expand(x_f.size(0), -1, -1)
        for attn, ffw in zip(self.attn, self.ffw):
            x = x + attn(x_f, x)                             # x = x + attention_i(q=x, kv=concat([x_f, x]))
            x = x + ffw(x)                                   # x = x + ffw_i(x)
        return x


class GatedXAttnDense(nn.Module):
    """Figure 4:  y += tanh(a_xattn) * attn(q=LN(y), kv=X);  y += tanh(a_dense) * ffw(LN(y)).  Gates start at 0."""
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln1, self.ln2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.kv, self.o = nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)
        self.ffw = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.a_xattn, self.a_dense = nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(()))   # initialised to 0

    def forward(self, y, X, media_mask):                     # y: (B, L, d) text; X: (B, n_img*R, d); media_mask: (B, L, n_img*R)
        B, L, d = y.shape
        q = self.q(self.ln1(y)).view(B, L, self.h, -1).transpose(1, 2)
        k, v = (t.view(B, -1, self.h, d // self.h).transpose(1, 2) for t in self.kv(X).chunk(2, -1))
        out = self.o(attention(q, k, v, media_mask[:, None]).transpose(1, 2).reshape(B, L, d))
        y = y + torch.tanh(self.a_xattn) * out
        return y + torch.tanh(self.a_dense) * self.ffw(self.ln2(y))


class Flamingo(nn.Module):
    def __init__(self, vision, lm, d=64, R=8):
        super().__init__()
        self.vision, self.lm, self.R = vision, lm, R
        for p in list(vision.parameters()) + list(lm.parameters()):
            p.requires_grad_(False)                          # both backbones frozen; only the bridge is trained
        self.resampler = PerceiverResampler(48, d, R)
        self.xattn = nn.ModuleList([GatedXAttnDense(d, 4) for _ in lm.blocks])

    @staticmethod
    def media_mask(seq, n_img, R):
        """Figure 7: token i may attend only to the R visual tokens of image  (#<image> tokens at positions <= i) - 1."""
        img_idx = (seq == IMG).long().cumsum(1) - 1                                       # -1 before the first image
        vis_idx = torch.arange(n_img).repeat_interleave(R)                                # which image each visual token belongs to
        return img_idx[:, :, None] == vis_idx[None, None, :]                              # (B, L, n_img*R)

    def forward(self, imgs, seq):
        B, n_img = imgs.shape[:2]
        with torch.no_grad():
            feats = self.vision(imgs.flatten(0, 1))                                       # (B*n_img, 16, d_v)
        X = self.resampler(feats).view(B, n_img * self.R, -1)                             # (B, n_img*R, d) visual tokens
        mask = self.media_mask(seq, n_img, self.R)
        y = self.lm.emb(seq) + self.lm.pos[:, : seq.size(1)]
        for gated, blk in zip(self.xattn, self.lm.blocks):
            y = blk(gated(y, X, mask))                                                    # gated xattn-dense, then frozen LM block
        return self.lm.head(self.lm.ln(y))


def lm_loss(logits, seq):
    """-sum_l log p(y_l | y_<l, x_<=l): next-token cross-entropy on the text, padding ignored."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), seq[:, 1:].reshape(-1), ignore_index=PAD)


@torch.no_grad()
def caption_accuracy(model, n=800):
    """Accuracy of the colour and shape tokens after each of the two <image> tokens (teacher forcing)."""
    imgs, seq = make_interleaved(n)
    pred = model(imgs, seq)[:, :-1].argmax(-1)
    ok = (pred == seq[:, 1:])                                                             # pred[t] predicts seq[t+1]
    slots = lambda j: ok[:, 2 + 5 * j : 4 + 5 * j].all(1).float().mean().item()          # colour+shape of image j both right
    return slots(0), slots(1)


def main():
    t0 = time.time()
    lm = LM(len(WORDS))
    opt = torch.optim.Adam(lm.parameters(), lr=2e-3)
    for step in range(300):                                                              # text-only pretraining of the LM
        _, seq = make_interleaved(64)
        loss = lm_loss(lm(seq), seq); opt.zero_grad(); loss.backward(); opt.step()
    print(f"frozen LM pretrained on text only: loss {loss.item():.3f}  (floor without seeing images: {(math.log(3) + math.log(4)) * 2 / 10:.3f})")
    model = Flamingo(VisionEncoder(), lm)
    frozen = {k: v.clone() for k, v in lm.state_dict().items()}
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable bridge parameters: {n_train:,} of {sum(p.numel() for p in model.parameters()):,}")
    with torch.no_grad():
        imgs, seq = make_interleaved(16)
        assert torch.allclose(model(imgs, seq), lm(seq), atol=1e-5), "zero gates: Flamingo must equal the LM at init"
    print("at init (gates = 0) the Flamingo output equals the frozen LM output exactly")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / 600)                 # linear decay to 0
    for step in range(1, 601):
        imgs, seq = make_interleaved(64)
        loss = lm_loss(model(imgs, seq), seq)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 150 == 0:
            gates = [f"{abs(math.tanh(g.a_xattn.item())):.2f}/{abs(math.tanh(g.a_dense.item())):.2f}" for g in model.xattn]
            a0, a1 = caption_accuracy(model)
            print(f"step {step:3d}  loss {loss.item():.3f}  |tanh gate| xattn/dense per layer {gates}  caption acc img1 {a0:.2f} img2 {a1:.2f}")
    assert all(torch.equal(frozen[k], v) for k, v in lm.state_dict().items()), "the LM must stay frozen"
    a0, a1 = caption_accuracy(model)
    # few-shot style prompt (Section 2.4): "<image> a red square . <image> a" -> decode the second caption greedily
    imgs, seq = make_interleaved(1)
    prompt = seq[:, :8].clone()
    for _ in range(2):
        prompt = torch.cat([prompt, model(imgs, prompt)[:, -1:].argmax(-1)], 1)
    print("prompt:", " ".join(WORDS[t] for t in seq[0, :8]), "-> generated:", " ".join(WORDS[t] for t in prompt[0, 8:]),
          "| truth:", " ".join(WORDS[t] for t in seq[0, 8:10]), f"  ({time.time() - t0:.1f} s)")
    assert a0 > 0.9 and a1 > 0.9, "captions after each <image> should be predicted from that image"
    assert abs(math.tanh(model.xattn[-1].a_xattn.item())) > 0.05, "the deepest gate should have opened (Figure 6)"


if __name__ == "__main__":
    main()
