"""Learning Transferable Visual Models From Natural Language Supervision (CLIP, Radford et al., ICML 2021).

What is implemented (section / figure numbers follow the paper):
  * an image encoder (tiny ViT) and a text encoder (causal Transformer, [EOS] activation as the text feature)  (Section 2.4)
  * linear projections W_i, W_t into the joint space and l2 normalisation                                     (Figure 3)
  * logits = (I_e T_e^T) * exp(t) with a learned temperature t, initialised to log(1/0.07), scale clipped at 100  (Section 2.5)
  * the symmetric cross-entropy over the N x N similarity matrix (rows: image->text, columns: text->image)      (Section 2.3, Figure 3)
  * zero-shot classification by embedding class names inside prompt templates, and prompt ensembling         (Section 3.1.4)
Simplifications: 16x16 synthetic images of a coloured shape (3 colours x 4 shapes) with template captions over a
14-word vocabulary instead of 400M web pairs; batch 64 instead of 32,768; a few hundred Adam steps.

Run:  python clip.py        (CPU, about 15 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ───────────────────────── synthetic (image, caption) pairs ─────────────────────────
WORDS = ["<pad>", "<bos>", "<eos>", "a", "photo", "of", "picture", "drawing", "red", "green", "blue", "square", "plus", "cross", "ring"]
TOK = {w: i for i, w in enumerate(WORDS)}
COLORS, SHAPES = ["red", "green", "blue"], ["square", "plus", "cross", "ring"]
TEMPLATES = ["a photo of a {} {}", "a picture of a {} {}", "a drawing of a {} {}", "{} {}"]
MAX_LEN = 8


def make_stencils(s=6):
    sq = torch.ones(s, s)
    plus = torch.zeros(s, s); plus[2:4, :] = 1; plus[:, 2:4] = 1
    cross = ((torch.eye(s) + torch.eye(s).flip(1)) > 0).float()
    ring = torch.ones(s, s); ring[1:-1, 1:-1] = 0
    return torch.stack([sq, plus, cross, ring])


STENCILS = make_stencils()


def tokenize(text):
    ids = [TOK["<bos>"]] + [TOK[w] for w in text.split()] + [TOK["<eos>"]]
    return ids + [TOK["<pad>"]] * (MAX_LEN - len(ids))


def make_batch(B, res=16, noise=0.3, template=None):
    """Images (B,3,res,res): shape stencil painted into one colour channel at a random place; captions from a random template."""
    color, shape = torch.randint(0, 3, (B,)), torch.randint(0, 4, (B,))
    x = torch.zeros(B, 3, res, res)
    r, c = torch.randint(0, res - 6 + 1, (B,)), torch.randint(0, res - 6 + 1, (B,))
    caps = []
    for i in range(B):
        x[i, color[i], r[i] : r[i] + 6, c[i] : c[i] + 6] = STENCILS[shape[i]]
        t = TEMPLATES[torch.randint(0, len(TEMPLATES), (1,)).item()] if template is None else template
        caps.append(tokenize(t.format(COLORS[color[i]], SHAPES[shape[i]])))
    return x + noise * torch.randn_like(x), torch.tensor(caps), color * 4 + shape        # class id in 0..11


# ───────────────────────── the two encoders (Section 2.4) ─────────────────────────
class ImageEncoder(nn.Module):
    """A tiny ViT: patch embedding, class token, learned positions, 2 pre-norm blocks; output = class token (LN'd)."""
    def __init__(self, res=16, P=4, C=3, D=64, L=2):
        super().__init__()
        N = (res // P) ** 2
        self.P, self.embed = P, nn.Linear(P * P * C, D)
        self.cls, self.pos = nn.Parameter(torch.zeros(1, 1, D)), nn.Parameter(0.02 * torch.randn(1, N + 1, D))
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 2 * D, 0.0, "gelu", batch_first=True, norm_first=True), L, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(D)

    def forward(self, x):
        B, C, H, W = x.shape; P = self.P
        patches = x.unfold(2, P, P).unfold(3, P, P).reshape(B, C, -1, P * P).transpose(1, 2).reshape(B, -1, C * P * P)
        z = torch.cat([self.cls.expand(B, -1, -1), self.embed(patches)], 1) + self.pos
        return self.ln(self.blocks(z)[:, 0])


class TextEncoder(nn.Module):
    """Transformer with masked (causal) self-attention over BPE-like tokens; the [EOS] token's final activation is the feature."""
    def __init__(self, vocab, D=64, L=2, max_len=MAX_LEN):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, D), nn.Parameter(0.02 * torch.randn(1, max_len, D))
        self.blocks = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 2 * D, 0.0, "gelu", batch_first=True, norm_first=True), L, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(D)

    def forward(self, ids):
        L = ids.size(1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        h = self.ln(self.blocks(self.tok(ids) + self.pos[:, :L], mask=causal))
        eos_pos = (ids == TOK["<eos>"]).float().argmax(1)                              # first [EOS] per sequence
        return h[torch.arange(ids.size(0)), eos_pos]


class CLIP(nn.Module):
    """Figure 3: I_e = l2norm(I_f W_i), T_e = l2norm(T_f W_t), logits = I_e T_e^T * exp(t), symmetric cross-entropy."""
    def __init__(self, d_e=32):
        super().__init__()
        self.image_encoder, self.text_encoder = ImageEncoder(), TextEncoder(len(WORDS))
        self.W_i, self.W_t = nn.Linear(64, d_e, bias=False), nn.Linear(64, d_e, bias=False)
        self.t = nn.Parameter(torch.tensor(math.log(1 / 0.07)))                          # learned temperature (Section 2.5)

    def encode_image(self, x):
        return F.normalize(self.W_i(self.image_encoder(x)), dim=-1)

    def encode_text(self, ids):
        return F.normalize(self.W_t(self.text_encoder(ids)), dim=-1)

    def forward(self, x, ids):
        I_e, T_e = self.encode_image(x), self.encode_text(ids)
        scale = self.t.exp().clamp(max=100.0)                                             # clipped so the scale never exceeds 100
        return I_e @ T_e.T * scale                                                        # (N, N) scaled pairwise cosine similarities


def clip_loss(logits):
    """loss = (cross_entropy(logits, labels, axis=0) + cross_entropy(logits, labels, axis=1)) / 2  with labels = arange(N)."""
    labels = torch.arange(logits.size(0))
    loss_i = F.cross_entropy(logits, labels)            # each image picks its own text among the N texts   (rows)
    loss_t = F.cross_entropy(logits.T, labels)          # each text picks its own image among the N images  (columns)
    return (loss_i + loss_t) / 2


# ───────────────────────── zero-shot transfer (Section 3.1.4) ─────────────────────────
@torch.no_grad()
def zero_shot_classifier(model, templates):
    """One text embedding per class: embed the class name inside each template, average over templates, re-normalise."""
    names = [(c, s) for c in COLORS for s in SHAPES]                                      # class id = color*4 + shape
    weights = []
    for tpl in templates:
        ids = torch.tensor([tokenize(tpl.format(c, s)) for c, s in names])
        weights.append(model.encode_text(ids))
    return F.normalize(torch.stack(weights).mean(0), dim=-1)                              # (12, d_e), ensemble of prompts


@torch.no_grad()
def zero_shot_accuracy(model, classifier, n=600):
    x, _, y = make_batch(n)
    pred = (model.encode_image(x) @ classifier.T).argmax(-1)                              # nearest class-text embedding
    return (pred == y).float().mean().item()


def main():
    t0 = time.time()
    model = CLIP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-6)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}   initial exp(t) = {model.t.exp().item():.1f}")
    first_loss = None
    for step in range(1, 701):
        x, ids, _ = make_batch(64)
        loss = clip_loss(model(x, ids))
        opt.zero_grad(); loss.backward(); opt.step()
        first_loss = first_loss or loss.item()
        if step % 175 == 0:
            zs = zero_shot_accuracy(model, zero_shot_classifier(model, TEMPLATES[:1]), n=300)
            print(f"step {step:4d}  contrastive loss {loss.item():.3f}  exp(t) {model.t.exp().item():5.1f}  zero-shot acc {zs:.3f}")
    model.eval()
    acc_single = zero_shot_accuracy(model, zero_shot_classifier(model, ["a photo of a {} {}"]))
    acc_bare = zero_shot_accuracy(model, zero_shot_classifier(model, ["{} {}"]))
    acc_ens = zero_shot_accuracy(model, zero_shot_classifier(model, TEMPLATES))
    print(f"zero-shot 12-way accuracy: bare class name {acc_bare:.3f} | 'a photo of a ...' {acc_single:.3f} | 4-template ensemble {acc_ens:.3f}")
    x, ids, y = make_batch(64, template="a photo of a {} {}")
    keep = torch.tensor([i for i in range(64) if y[i] not in y[:i]][:8])                # 8 pairs with distinct classes
    sim = model.encode_image(x[keep]) @ model.encode_text(ids[keep]).T
    print("image->text retrieval on 8 held-out pairs (diagonal should win):", (sim.argmax(1) == torch.arange(8)).tolist())
    print(f"done in {time.time() - t0:.1f} s")
    assert loss.item() < first_loss * 0.5 and acc_ens > 0.85, "CLIP did not align images and captions"


if __name__ == "__main__":
    main()
