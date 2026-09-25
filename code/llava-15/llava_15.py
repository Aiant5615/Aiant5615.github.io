"""Improved Baselines with Visual Instruction Tuning (LLaVA-1.5, Liu et al., 2023) — toy implementation.

What is implemented (section / figure / table numbers follow the paper):
  * the LLaVA recipe (frozen encoder -> connector -> LLM, loss on answer tokens, projector-only stage then end-to-end)
  * change 1, response-format prompting (Section 3.1, Table 1): conversation data has long answers, VQA data has
    short answers and carries the instruction "answer with a single word"; the same model then produces either style
    on demand, and short-answer accuracy jumps when the instruction is appended at test time
  * change 2, the MLP connector (Section 3.1): a two-layer MLP with GELU instead of the linear W, compared to linear
  * change 3, academic-task (short-answer VQA) data added to the conversation data (Section 3.1)
  * LLaVA-1.5-HD (Section 3.2, Figure 2): a higher-resolution image the encoder cannot take is split into a grid of
    crops encoded independently, their features concatenated with a downsampled global view; a fine-detail question
    (where is the small dot?) is answered with the grid and not with the global view alone
Simplifications: the LLaVA toy world (12x36 images with up to three coloured shapes, a conv encoder pretrained on
shape classification as the CLIP stand-in, a 2-layer Transformer as the LLM); HD images are 24x72 with a 1-pixel
dot; the LLM-choice and data-fraction ablations are not reproduced.

Run:  python llava_15.py        (CPU, about 45 s)
"""
import copy, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ───────────────────────── the world ─────────────────────────
COLOURS, SHAPES, SLOTS = ["red", "green", "blue", "yellow"], ["square", "cross"], ["left", "middle", "right"]
RGB = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=torch.float)
S = 12                                                            # the encoder's input is (3, S, 3S)


def draw(scene, scale=1, dot=None):
    """scene: (colour, shape) or None per slot. scale=2 draws the 24x72 HD image; dot=(slot) adds a 1-px white dot."""
    s = S * scale
    img = 0.05 * torch.rand(3, s, 3 * s)
    for slot, obj in enumerate(scene):
        if obj is None:
            continue
        c, sh = obj
        y0, x0 = (torch.randint(2, 4, (2,)) * scale).tolist()
        cell = torch.zeros(s, s)
        if sh == 0:
            cell[y0:y0 + 6 * scale, x0:x0 + 6 * scale] = 1
        else:
            cell[y0:y0 + 6 * scale, x0 + 2 * scale:x0 + 4 * scale] = 1; cell[y0 + 2 * scale:y0 + 4 * scale, x0:x0 + 6 * scale] = 1
        img[:, :, slot * s:(slot + 1) * s] += RGB[c][:, None, None] * cell
    if dot is not None:
        y, x = torch.randint(1, s - 1, (2,)).tolist()
        img[:, y, dot * s + x] = 1.0                                          # a single white pixel
    return img.clamp(0, 1)


def make_scene():
    return [None if torch.rand(()) < 0.3 else (torch.randint(4, ()).item(), torch.randint(2, ()).item()) for _ in range(3)]


WORDS = COLOURS + SHAPES + SLOTS + ["0", "1", "2", "3", "none", "yes", "no", "empty", ";", "the", "object", "is", "there", "are",
         "describe", "what-colour", "what-shape", "how-many", "how-many-objects", "is-there", "which-side", "where-dot",
         "<short>", "<img>", "<q>", "<a>", "<end>", "<pad>"]
TOK = {w: i for i, w in enumerate(WORDS)}
SHORT, IMG, QS, AS, END, PAD = (TOK[w] for w in ["<short>", "<img>", "<q>", "<a>", "<end>", "<pad>"])
VOCAB = len(WORDS)


def caption(scene):
    out = []
    for obj in scene:
        out += ["empty"] if obj is None else [COLOURS[obj[0]], SHAPES[obj[1]]]
        out.append(";")
    return out[:-1]


def questions(scene):
    """(question, short answer, long answer): the conversation style answers in a sentence, VQA style in one word."""
    qa = []
    for slot, obj in enumerate(scene):
        a = [COLOURS[obj[0]]] if obj else ["none"]
        qa.append((["what-colour", SLOTS[slot]], a, ["the", SLOTS[slot], "object", "is"] + a))
        a = [SHAPES[obj[1]]] if obj else ["none"]
        qa.append((["what-shape", SLOTS[slot]], a, ["the", SLOTS[slot], "object", "is"] + a))
    for c in range(4):
        n = sum(1 for o in scene if o and o[0] == c)
        qa.append((["how-many", COLOURS[c]], [str(n)], ["there", "are", str(n), COLOURS[c]]))
        qa.append((["is-there", COLOURS[c]], ["yes" if n else "no"], ["yes" if n else "no", ";", "there", "are", str(n)]))
        if n == 1:
            s = SLOTS[[o is not None and o[0] == c for o in scene].index(True)]
            qa.append((["which-side", COLOURS[c]], [s], ["the", COLOURS[c], "object", "is", s]))
    n = str(sum(o is not None for o in scene))
    qa.append((["how-many-objects"], [n], ["there", "are", n]))
    return qa


def encode(words):
    return [TOK[w] for w in words]


def build_sequence(q, a, n_img, short):
    """[<img> x n_img] <q> question [<short>] <a> answer <end>; loss mask on the answer only."""
    body = [IMG] * n_img + [QS] + encode(q) + ([SHORT] if short else []) + [AS]
    ans = encode(a) + [END]
    return body + ans, [0] * len(body) + [1] * len(ans)


def make_batch(B, mix, n_img=3):
    """mix: 'conversation' (long answers, no format prompt), 'vqa' (short answers + prompt), 'caption', 'dot' (HD)."""
    seqs, masks, imgs = [], [], []
    for _ in range(B):
        scene = make_scene()
        kind = mix if mix != "conversation+vqa" else ("vqa" if torch.rand(()) < 0.5 else "conversation")
        if kind == "caption":
            q, a, short, img = ["describe"], caption(scene), False, draw(scene)
        elif kind == "dot":
            d = torch.randint(3, ()).item()
            q, a, short, img = ["where-dot"], [SLOTS[d]], True, draw(scene, scale=2, dot=d)
        else:
            q, s_ans, l_ans = questions(scene)[torch.randint(len(questions(scene)), ()).item()]
            short = kind == "vqa"
            q, a, img = q, (s_ans if short else l_ans), draw(scene)
        toks, mask = build_sequence(q, a, n_img, short)
        seqs.append(toks); masks.append(mask); imgs.append(img)
    L = max(map(len, seqs))
    pad = lambda xs, v: torch.tensor([x + [v] * (L - len(x)) for x in xs])
    return pad(seqs, PAD), pad(masks, 0), torch.stack(imgs)


# ───────────────────────── encoder, connectors, LLM ─────────────────────────
class VisionEncoder(nn.Module):
    """Fixed-resolution conv encoder (the CLIP stand-in): (3, 12, 36) -> 3 slot features. Pretrained on colour, shape
    and 'is there a dot in this slot', then frozen."""
    def __init__(self, d_v=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, d_v, 3, padding=1), nn.ReLU(),
                                 nn.AdaptiveMaxPool2d((1, 3)))
        self.head = nn.Linear(d_v, 5 + 3 + 2)

    def forward(self, img):
        return self.net(img).squeeze(2).transpose(1, 2)


def pretrain_encoder(enc, steps=300):
    opt = torch.optim.Adam(enc.parameters(), lr=3e-3)
    for _ in range(steps):
        scenes = [make_scene() for _ in range(64)]
        dots = [None if torch.rand(()) < 0.5 else torch.randint(3, ()).item() for _ in scenes]
        imgs = torch.stack([draw(s, dot=d) for s, d in zip(scenes, dots)])
        col = torch.tensor([[4 if o is None else o[0] for o in s] for s in scenes])
        shp = torch.tensor([[2 if o is None else o[1] for o in s] for s in scenes])
        has_dot = torch.tensor([[int(d == k) for k in range(3)] for d in dots])
        out = enc.head(enc(imgs))
        loss = (F.cross_entropy(out[..., :5].reshape(-1, 5), col.reshape(-1)) + F.cross_entropy(out[..., 5:8].reshape(-1, 3), shp.reshape(-1))
                + F.cross_entropy(out[..., 8:].reshape(-1, 2), has_dot.reshape(-1)))
        opt.zero_grad(); loss.backward(); opt.step()
    enc.requires_grad_(False).eval()


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
    def __init__(self, d=64, h=4, n_layers=2, ctx=48):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB, bias=False)

    def forward(self, emb):
        x = emb + self.pos(torch.arange(emb.shape[1]))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


def connector(kind, d_v, d):
    """LLaVA's linear W, or LLaVA-1.5's two-layer MLP with GELU (Section 3.1)."""
    return nn.Linear(d_v, d) if kind == "linear" else nn.Sequential(nn.Linear(d_v, d), nn.GELU(), nn.Linear(d, d))


def hd_features(encoder, img):
    """LLaVA-1.5-HD (Figure 2): split the 24x72 image into 2x2 crops of the encoder's resolution, encode each crop
    independently, concatenate the features, and append a downsampled global view encoded separately."""
    B, _, H, W = img.shape
    crops = [img[:, :, i * S:(i + 1) * S, j * 3 * S:(j + 1) * 3 * S] for i in range(H // S) for j in range(W // (3 * S))]
    glob = F.avg_pool2d(img, H // S)
    return torch.cat([encoder(c) for c in crops] + [encoder(glob)], 1)


class LLaVA(nn.Module):
    def __init__(self, encoder, llm, kind, d_v=32):
        super().__init__()
        self.encoder, self.llm, self.proj = encoder, llm, connector(kind, d_v, llm.tok.embedding_dim)

    def visual_tokens(self, imgs):
        if imgs.shape[-1] == 3 * S:
            return self.encoder(imgs)
        return hd_features(self.encoder, imgs) if self.hd else self.encoder(F.avg_pool2d(imgs, imgs.shape[-2] // S))

    def forward(self, toks, imgs):
        emb = self.llm.tok(toks).clone()
        H_v = self.proj(self.visual_tokens(imgs))
        emb[toks == IMG] = H_v.reshape(-1, H_v.shape[-1])
        return self.llm(emb)


def answer_loss(logits, toks, mask):
    lp = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), toks[:, 1:].reshape(-1), reduction="none")
    m = mask[:, 1:].reshape(-1).float()
    return (lp * m).sum() / m.sum()


def train(model, params, mix, steps, lr, n_img=3):
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        toks, mask, imgs = make_batch(48, mix, n_img)
        loss = answer_loss(model(toks, imgs), toks, mask)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


@torch.no_grad()
def short_accuracy(model, short_prompt, n=150, mix="vqa", n_img=3):
    """Exact match against the single-word reference (what a VQA benchmark scores), with or without the format prompt."""
    hits = 0
    for _ in range(n):
        toks, mask, img = make_batch(1, mix, n_img)
        prompt_len = int((mask[0] == 0).sum())
        gold = toks[0, prompt_len:].tolist(); gold = gold[: gold.index(END)]
        seq = toks[:, :prompt_len - 1]                                            # up to (not including) <a>
        seq = torch.cat([seq[seq != SHORT][None] if not short_prompt else seq, torch.tensor([[AS]])], 1) if mix != "dot" else toks[:, :prompt_len]
        out = []
        for _ in range(6):
            nxt = model(seq, img)[0, -1].argmax()
            if nxt.item() == END:
                break
            out.append(nxt.item()); seq = torch.cat([seq, nxt.view(1, 1)], 1)
        hits += out == gold
    return hits / n


def main():
    t0 = time.time()
    enc = VisionEncoder(); pretrain_encoder(enc)
    llm = LLM()
    # text-only pretraining is folded into the conversation data here; stage 1 = caption data, connector only
    results = {}
    for kind in ("linear", "mlp"):
        torch.manual_seed(1)
        model = LLaVA(enc, copy.deepcopy(llm), kind); model.hd = False
        train(model, model.proj.parameters(), "caption", 200, 3e-3)
        train(model, model.parameters(), "conversation+vqa", 500, 2e-3)
        results[kind] = (short_accuracy(model, short_prompt=False), short_accuracy(model, short_prompt=True))
        print(f"{kind:6s} connector, trained on conversation + VQA data: short-answer accuracy "
              f"without the format prompt {results[kind][0]:.2f}, with 'answer with a single word' {results[kind][1]:.2f}")
        if kind == "mlp":
            best = model
    torch.manual_seed(1)
    conv_only = LLaVA(enc, copy.deepcopy(llm), "mlp"); conv_only.hd = False
    train(conv_only, conv_only.proj.parameters(), "caption", 200, 3e-3)
    train(conv_only, conv_only.parameters(), "conversation", 500, 2e-3)
    acc_conv = short_accuracy(conv_only, short_prompt=True)
    print(f"mlp    connector, conversation data only: {acc_conv:.2f} even with the prompt (Table 1: academic VQA data matters)")
    # LLaVA-1.5-HD
    print("HD: 24x72 images with a 1-pixel dot; 'where is the dot?' — global view only vs 2x2 crops + global (15 tokens)")
    hd = {}
    for use_grid in (False, True):
        m = copy.deepcopy(best); m.hd = use_grid
        train(m, m.parameters(), "dot", 300, 1e-3, n_img=15 if use_grid else 3)
        hd[use_grid] = short_accuracy(m, True, mix="dot", n_img=15 if use_grid else 3)
    print(f"  global view only {hd[False]:.2f}   grid of crops + global view {hd[True]:.2f}   ({time.time() - t0:.1f} s)")
    assert results["mlp"][1] > results["mlp"][0] + 0.3, "the format prompt should switch the model to short answers"
    assert results["mlp"][1] > 0.8 and results["mlp"][1] >= results["linear"][1] - 0.03, "the MLP connector should be at least as good"
    assert acc_conv < results["mlp"][1] - 0.2, "without short-answer data the prompt cannot be followed"
    assert hd[True] > hd[False] + 0.3, "the grid of crops should recover the fine detail"


if __name__ == "__main__":
    main()
