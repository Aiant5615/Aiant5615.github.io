"""Visual Instruction Tuning (LLaVA, Liu et al., 2023) — toy implementation.

What is implemented (section / figure numbers follow the paper):
  * the architecture of Figure 1: a frozen image encoder g gives grid features Z_v = g(X_v); a single trainable
    projection W maps them into the LLM's word-embedding space, H_v = W Z_v; the visual tokens are fed to the LLM
    together with the embedded instruction; the LLM generates the answer                          (Section 4, eq. 1)
  * GPT-4-style data generation from SYMBOLIC descriptions only (Section 3): a program that never looks at the pixels
    writes conversation, detailed-description and reasoning questions from the caption + object list of each image
  * training (Section 4.1): the autoregressive loss only on the assistant's answer tokens (eq. 2); at the first turn
    the image tokens go before or after the question at random; two stages:
      stage 1 feature alignment  - caption data, only W is trained (encoder and LLM frozen)
      stage 2 end-to-end        - instruction data, W and the LLM are trained, the encoder stays frozen
  * evaluation: answer accuracy per question type, against the text-only LLM given the true description
    (the upper bound the GPT-4 judge uses) and against a stage-1-only model (frozen LLM + linear connector)
Simplifications: 12x36 images with up to three coloured shapes instead of COCO; a small conv net pretrained on shape
classification stands in for CLIP ViT-L/14; a 2-layer Transformer pretrained on text-only scene Q&A stands in for
Vicuna; the "GPT-4" data generator is rule-based.

Run:  python llava.py        (CPU, about 40 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ───────────────────────── the world: images and their symbolic descriptions (captions + boxes) ─────────────────────────
COLOURS, SHAPES, SLOTS = ["red", "green", "blue", "yellow"], ["square", "cross"], ["left", "middle", "right"]
RGB = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=torch.float)
S = 12                                                            # each slot is an S x S patch


def draw(scene):
    """scene: list of (slot, colour, shape) or None per slot -> image (3, S, 3S) with a little noise."""
    img = 0.05 * torch.rand(3, S, 3 * S)
    for slot, obj in enumerate(scene):
        if obj is None:
            continue
        c, sh = obj
        y0, x0 = torch.randint(2, 4, (2,)).tolist()
        cell = torch.zeros(S, S)
        if sh == 0:
            cell[y0:y0 + 6, x0:x0 + 6] = 1
        else:
            cell[y0:y0 + 6, x0 + 2:x0 + 4] = 1; cell[y0 + 2:y0 + 4, x0:x0 + 6] = 1
        img[:, :, slot * S:(slot + 1) * S] += RGB[c][:, None, None] * cell
    return img.clamp(0, 1)


def make_scene():
    return [None if torch.rand(()) < 0.3 else (torch.randint(4, ()).item(), torch.randint(2, ()).item()) for _ in range(3)]


# ───────────────────────── vocabulary and the "GPT-4" data generator (Section 3) ─────────────────────────
WORDS = COLOURS + SHAPES + SLOTS + ["0", "1", "2", "3", "none", "yes", "no", "empty", ";",
         "describe", "what-colour", "what-shape", "how-many", "how-many-objects", "is-there", "which-side", "left-of",
         "<img>", "<q>", "<a>", "<end>", "<pad>"]
TOK = {w: i for i, w in enumerate(WORDS)}
IMG, QS, AS, END, PAD = (TOK[w] for w in ["<img>", "<q>", "<a>", "<end>", "<pad>"])
VOCAB, N_IMG = len(WORDS), 3                                      # 3 visual tokens, one per slot


def caption(scene):
    """The image's caption (the symbolic description GPT-4 gets): 'red square ; empty ; blue cross'."""
    out = []
    for obj in scene:
        out += ["empty"] if obj is None else [COLOURS[obj[0]], SHAPES[obj[1]]]
        out.append(";")
    return out[:-1]


def gpt4_questions(scene):
    """Questions written from the description alone (the generator never sees pixels): conversation-style recognition
    questions, the detailed description, and reasoning questions (counting, existence, relative position)."""
    qa = []
    for slot, obj in enumerate(scene):                                                    # conversation
        qa.append((["what-colour", SLOTS[slot]], [COLOURS[obj[0]]] if obj else ["none"]))
        qa.append((["what-shape", SLOTS[slot]], [SHAPES[obj[1]]] if obj else ["none"]))
    qa.append((["describe"], caption(scene)))                                             # detailed description
    for c in range(4):                                                                    # reasoning
        n = sum(1 for o in scene if o and o[0] == c)
        qa.append((["how-many", COLOURS[c]], [str(n)]))
        qa.append((["is-there", COLOURS[c]], ["yes" if n else "no"]))
        if n == 1:
            qa.append((["which-side", COLOURS[c]], [SLOTS[[o is not None and o[0] == c for o in scene].index(True)]]))
    qa.append((["how-many-objects"], [str(sum(o is not None for o in scene))]))
    for sh in range(2):
        idx = [i for i, o in enumerate(scene) if o and o[1] == sh]
        if len(idx) == 1 and idx[0] > 0 and scene[idx[0] - 1]:
            qa.append((["left-of", SHAPES[sh]], [COLOURS[scene[idx[0] - 1][0]]]))
    return qa


def encode(words):
    return [TOK[w] for w in words]


def build_sequence(prefix, q, a, image_first):
    """One training sequence. The image tokens go before or after the question at random (Section 4.1); the loss mask
    covers only the answer tokens (eq. 2)."""
    img = [IMG] * N_IMG
    q_toks = [QS] + encode(q)
    body = (img + q_toks if image_first else q_toks + img) + [AS]
    ans = encode(a) + [END]
    toks = prefix + body + ans
    mask = [0] * (len(prefix) + len(body)) + [1] * len(ans)
    return toks, mask


def make_batch(B, kind, with_image):
    """kind='caption': stage-1 data. kind='instruct': the GPT-4-style Q&A. with_image=False gives the text-only
    version in which the caption replaces the image (used to pretrain the LLM, and as the oracle)."""
    seqs, masks, imgs = [], [], []
    for _ in range(B):
        scene = make_scene()
        if kind == "caption":
            q, a = ["describe"], caption(scene)
        else:
            qa = gpt4_questions(scene); q, a = qa[torch.randint(len(qa), ()).item()]
        prefix = [] if with_image else encode(caption(scene) + [";"])
        toks, mask = build_sequence(prefix, q, a, image_first=bool(torch.rand(()) < 0.5))
        seqs.append(toks); masks.append(mask); imgs.append(draw(scene) if with_image else torch.zeros(3, S, 3 * S))
    L = max(map(len, seqs))
    pad = lambda xs, v: torch.tensor([x + [v] * (L - len(x)) for x in xs])
    return pad(seqs, PAD), pad(masks, 0), torch.stack(imgs)


# ───────────────────────── vision encoder g (frozen CLIP stand-in) ─────────────────────────
class VisionEncoder(nn.Module):
    """A conv net whose output is a grid of 3 slot features Z_v (d_v each). Pretrained on classifying the colour and
    shape in every slot, then frozen; like LLaVA we keep the features, not the classifier head."""
    def __init__(self, d_v=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, d_v, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 3)))
        self.head = nn.Linear(d_v, 5 + 3)                        # colour (4 + none) and shape (2 + none) per slot

    def forward(self, img):                                       # -> Z_v (B, 3, d_v)
        return self.net(img).squeeze(2).transpose(1, 2)


def pretrain_encoder(enc, steps=300):
    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    for _ in range(steps):
        scenes = [make_scene() for _ in range(64)]
        imgs = torch.stack([draw(s) for s in scenes])
        col = torch.tensor([[4 if o is None else o[0] for o in s] for s in scenes])
        shp = torch.tensor([[2 if o is None else o[1] for o in s] for s in scenes])
        out = enc.head(enc(imgs))
        loss = F.cross_entropy(out[..., :5].reshape(-1, 5), col.reshape(-1)) + F.cross_entropy(out[..., 5:].reshape(-1, 3), shp.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    enc.requires_grad_(False).eval()
    with torch.no_grad():
        acc = (out[..., :5].argmax(-1) == col).float().mean().item()
    return acc


# ───────────────────────── the LLM f_phi and the projection W ─────────────────────────
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln1, self.ln2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool))
        att = (q @ k.transpose(-1, -2) / math.sqrt(d // self.h)).masked_fill(~mask, float("-inf")).softmax(-1)
        x = x + self.out((att @ v).transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.ln2(x))


class LLM(nn.Module):
    """Decoder-only Transformer over word embeddings; `forward` accepts embeddings so visual tokens can be spliced in."""
    def __init__(self, d=64, h=4, n_layers=2, ctx=32):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB, bias=False)

    def forward(self, emb):
        x = emb + self.pos(torch.arange(emb.shape[1]))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


class LLaVA(nn.Module):
    """H_v = W Z_v (eq. 1): the projected grid features replace the <img> placeholder embeddings."""
    def __init__(self, encoder, llm, d_v=32):
        super().__init__()
        self.encoder, self.llm = encoder, llm
        self.W = nn.Linear(d_v, llm.tok.embedding_dim)

    def forward(self, toks, imgs):
        emb = self.llm.tok(toks)
        H_v = self.W(self.encoder(imgs))                                             # (B, 3, d)
        is_img = toks == IMG
        emb = emb.clone(); emb[is_img] = H_v.reshape(-1, H_v.shape[-1])
        return self.llm(emb)


def answer_loss(logits, toks, mask):
    """eq. 2: cross-entropy on the answer tokens only (mask = 1), teacher forcing."""
    lp = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), toks[:, 1:].reshape(-1), reduction="none")
    m = mask[:, 1:].reshape(-1).float()
    return (lp * m).sum() / m.sum()


def train(model, params, kind, with_image, steps, lr, tag):
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(1, steps + 1):
        toks, mask, imgs = make_batch(64, kind, with_image)
        logits = model(toks, imgs) if isinstance(model, LLaVA) else model(model.tok(toks))
        loss = answer_loss(logits, toks, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % (steps // 2) == 0:
            print(f"  [{tag}] step {step:4d}  answer loss {loss.item():.3f}")


@torch.no_grad()
def accuracy(model, with_image, n=400):
    """Greedy decoding of the answer; exact match of the whole answer, by question type."""
    hits = {}
    for _ in range(n):
        scene = make_scene()
        q, a = gpt4_questions(scene)[torch.randint(len(gpt4_questions(scene)), ()).item()]
        prefix = [] if with_image else encode(caption(scene) + [";"])
        toks, _ = build_sequence(prefix, q, [], image_first=True)
        toks = torch.tensor(toks[:-1])[None]                                       # drop the <end> of the empty answer
        img = draw(scene)[None]
        out = []
        for _ in range(len(a) + 1):
            logits = model(toks, img) if isinstance(model, LLaVA) else model(model.tok(toks))
            nxt = logits[0, -1].argmax()
            if nxt.item() == END:
                break
            out.append(nxt.item()); toks = torch.cat([toks, nxt.view(1, 1)], 1)
        hits.setdefault(q[0], []).append(float(out == encode(a)))
    return {k: sum(v) / len(v) for k, v in hits.items()}, sum(map(sum, hits.values())) / n


def main():
    t0 = time.time()
    enc = VisionEncoder()
    print(f"vision encoder pretrained (colour accuracy per slot {pretrain_encoder(enc):.2f}), now frozen")
    llm = LLM()
    print("pretraining the text-only LLM on scene Q&A given the caption (stands in for Vicuna)")
    train(llm, llm.parameters(), "instruct", with_image=False, steps=800, lr=2e-3, tag="LLM")
    oracle = accuracy(llm, with_image=False)[1]
    print(f"  text-only LLM with the true caption: {oracle:.2f} (the reference the GPT-4 judge would use)")
    model = LLaVA(enc, llm)
    print("stage 1, feature alignment: caption data, only W trained")
    train(model, model.W.parameters(), "caption", with_image=True, steps=300, lr=3e-3, tag="stage 1")
    stage1 = accuracy(model, with_image=True)
    print(f"  after stage 1 (frozen LLM + linear connector): instruction accuracy {stage1[1]:.2f}")
    print("stage 2, end-to-end: instruction data, W and the LLM trained, encoder frozen")
    train(model, list(model.W.parameters()) + list(model.llm.parameters()), "instruct", with_image=True, steps=600, lr=1e-3, tag="stage 2")
    by_type, overall = accuracy(model, with_image=True)
    print(f"  LLaVA accuracy {overall:.2f} = {overall / oracle:.0%} of the text-only reference")
    for k, v in sorted(by_type.items()):
        print(f"    {k:18s} {v:.2f}")
    scene = make_scene()
    toks, _ = build_sequence([], ["describe"], [], True); toks = torch.tensor(toks[:-1])[None]; img = draw(scene)[None]
    out = []
    with torch.no_grad():
        for _ in range(9):
            nxt = model(toks, img)[0, -1].argmax()
            if nxt.item() == END: break
            out.append(WORDS[nxt.item()]); toks = torch.cat([toks, nxt.view(1, 1)], 1)
    print(f"  e.g. image [{' '.join(caption(scene))}] -> describe -> {' '.join(out)}   ({time.time() - t0:.1f} s)")
    assert oracle > 0.9, "the text-only LLM should master the Q&A given the caption"
    assert overall > 0.8 and overall > stage1[1] + 0.1, "end-to-end fine-tuning should transfer the LLM's skill to images"


if __name__ == "__main__":
    main()
