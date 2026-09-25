"""Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution (Wang et al., 2024) — toy implementation.

What is implemented (section / figure numbers follow the paper):
  * naive dynamic resolution (Section 2.1, Figure 2): a ViT WITHOUT absolute position embeddings that uses 2D-RoPE
    (axial rotary embedding over patch row / column, as in the official code), so it accepts any patch grid; after
    the encoder a PatchMerger (LayerNorm -> MLP over each 2x2 group of neighbouring patches) cuts the token count by
    four; <vision_start> / <vision_end> delimit the visual tokens; the number of tokens follows the image size
  * M-RoPE (Section 2.1, Figure 3): the LLM's rotary dimensions are split into three sections carrying temporal,
    height and width ids; text tokens have all three ids equal to their 1-D index; an image has a constant temporal
    id and (row, column) ids of the merged grid; a video's temporal id advances per frame; the next modality starts
    at the previous maximum id + 1 (the get_rope_index logic of the official code)
  * unified image and video: frames go through the same encoder, merger and position scheme
  * checks: text tokens get exactly the base LLM's 1-D RoPE; attention logits depend only on relative (dt, dh, dw)
  * the encoder trained on small grids is tested on larger unseen grids: 2D-RoPE generalises, learned absolute
    positions do not; the full model trained on small images and 2-frame videos answers at unseen larger sizes
Not reproduced at toy scale: relational spatial questions (which blob is above which) — two tiny layers did not learn
them from RoPE-only positions in a minute; the position checks above are exact instead.
Simplifications: pixel "images" of 2x2-pixel patches with coloured blobs instead of photographs; one frame per
temporal patch instead of 3-D convolutions over two frames; batches contain one image size at a time instead of
packed variable-size sequences; tiny ViT and LLM.

Run:  python qwen2_vl.py        (CPU, about 25 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ───────────────────────── data: images and videos of any size, and questions about them ─────────────────────────
COLOURS = ["red", "green", "blue", "yellow"]
RGB = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=torch.float)
P = 2                                                     # patch size in pixels
WORDS = COLOURS + ["yes", "no", "is-there",
                   "<vs>", "<ve>", "<a>", "<end>", "<pad>"]
TOK = {w: i for i, w in enumerate(WORDS)}
VS, VE, AS, END, PAD = (TOK[w] for w in ["<vs>", "<ve>", "<a>", "<end>", "<pad>"])
VOCAB = len(WORDS)


def blob(img, r, q, c):
    """Paint a 2x2-patch (4x4-pixel) blob of colour c at merged-grid cell (r, q)."""
    img[:, 2 * r * P:(2 * r + 2) * P, 2 * q * P:(2 * q + 2) * P] = RGB[c][:, None, None]


def make_image(h, w, n_obj):
    """h x w patch grid (even h, w) with n_obj blobs of distinct colours at distinct cells of the merged h/2 x w/2 grid.
    Returns (pixels, {colour: (row, col) on the merged grid})."""
    img = 0.05 * torch.rand(3, h * P, w * P)
    hm, wm = h // 2, w // 2
    cells = torch.randperm(hm * wm)[:n_obj].tolist(); cols = torch.randperm(4)[:n_obj].tolist()
    objs = {}
    for cell, c in zip(cells, cols):
        r, q = divmod(cell, wm)
        blob(img, r, q, c); objs[c] = (r, q)
    return img, objs


def image_question(objs):
    """'is there a <colour> blob?' — the answer needs the question token to find a matching visual token, wherever
    it is and however many visual tokens the input produced."""
    c = torch.randint(4, ()).item()
    return ["is-there", COLOURS[c]], ["yes" if c in objs else "no"]


def make_video(h, w, T=2):
    """T frames, each with one blob of a different colour at a random cell, asked the same 'is there' question; the
    frames go through the same encoder, merger and (temporal, height, width) position scheme as images."""
    hm, wm = h // 2, w // 2
    cols = torch.randperm(4)[:T].tolist()
    frames = []
    for t in range(T):
        f = 0.05 * torch.rand(3, h * P, w * P)
        cell = torch.randint(hm * wm, ()).item()
        blob(f, cell // wm, cell % wm, cols[t])
        frames.append(f)
    c = torch.randint(4, ()).item()
    return torch.stack(frames), ["is-there", COLOURS[c]], ["yes" if c in cols else "no"]


# ───────────────────────── rotary embeddings: 2D-RoPE for the ViT, M-RoPE for the LLM ─────────────────────────
def rotate_half(x):
    x1, x2 = x.chunk(2, -1)
    return torch.cat([-x2, x1], -1)


def rope_angles(pos_ids, sections, head_dim, base=10000.0, axial=False):
    """pos_ids: (n_groups, L) integer ids; sections: how many of the head_dim/2 frequency bands each group owns.
    M-RoPE (axial=False): the LLM's bands keep their usual frequencies and band j uses the id of the group it belongs
    to (the official `mrope_section` recomposition: temporal = highest frequencies, width = lowest).
    2D-RoPE for the ViT (axial=True): each axis gets its own copy of the full frequency range over head_dim/4 bands
    (the official Qwen2VLVisionRotaryEmbedding with dim = head_dim // 2). Returns cos, sin of shape (L, head_dim)."""
    group = torch.repeat_interleave(torch.arange(len(sections)), torch.tensor(sections))
    if axial:
        n = head_dim // 4
        inv_freq = (base ** (-torch.arange(0, 2 * n, 2).float() / (2 * n))).repeat(len(sections))
    else:
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)         # (head_dim/2,)
    freqs = pos_ids[group].T.float() * inv_freq                                        # (L, head_dim/2)
    freqs = torch.cat([freqs, freqs], -1)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):                                                           # x: (B, h, L, head_dim)
    return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.qkv, self.out = h, nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x, cos, sin, causal):
        B, L, d = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = q @ k.transpose(-1, -2) / math.sqrt(d // self.h)
        if causal:
            att = att.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), float("-inf"))
        return self.out((att.softmax(-1) @ v).transpose(1, 2).reshape(B, L, d))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2, self.attn = nn.LayerNorm(d), nn.LayerNorm(d), Attention(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cos, sin, causal):
        x = x + self.attn(self.ln1(x), cos, sin, causal)
        return x + self.mlp(self.ln2(x))


class ViT(nn.Module):
    """Patch embedding + Transformer with 2D-RoPE (half of the rotary bands rotate by the patch row, half by the
    column) and no absolute position embedding, so any h x w grid is accepted. `absolute=True` is the fixed-size
    baseline with learned positions for a maximum grid."""
    def __init__(self, d=32, h=2, n_layers=2, absolute=False, max_grid=12):
        super().__init__()
        self.d, self.h, self.absolute = d, h, absolute
        self.patch = nn.Linear(3 * P * P, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)
        if absolute:
            self.pos = nn.Embedding(max_grid * max_grid, d); self.max_grid = max_grid

    def forward(self, img):                                                            # (B, 3, hP, wP) -> (B, h*w, d)
        B, _, H, W = img.shape
        hp, wp = H // P, W // P
        x = self.patch(img.unfold(2, P, P).unfold(3, P, P).permute(0, 2, 3, 1, 4, 5).reshape(B, hp * wp, -1))
        rows, cols = torch.meshgrid(torch.arange(hp), torch.arange(wp), indexing="ij")
        if self.absolute:
            x = x + self.pos((rows * self.max_grid + cols).reshape(-1))
            cos, sin = rope_angles(torch.zeros(2, hp * wp, dtype=torch.long), [self.d // self.h // 4] * 2, self.d // self.h, axial=True)
        else:
            cos, sin = rope_angles(torch.stack([rows.reshape(-1), cols.reshape(-1)]), [self.d // self.h // 4] * 2, self.d // self.h, axial=True)
        for blk in self.blocks:
            x = blk(x, cos, sin, causal=False)
        return self.ln(x)


class PatchMerger(nn.Module):
    """LayerNorm, then an MLP over the 4 concatenated features of each 2x2 group of patches -> one LLM-dim token."""
    def __init__(self, d_vit, d_llm):
        super().__init__()
        self.ln = nn.LayerNorm(d_vit)
        self.mlp = nn.Sequential(nn.Linear(4 * d_vit, 4 * d_vit), nn.GELU(), nn.Linear(4 * d_vit, d_llm))

    def forward(self, x, hp, wp):                                                       # (B, hp*wp, d) -> (B, hp/2*wp/2, d_llm)
        B = x.shape[0]
        x = self.ln(x).view(B, hp // 2, 2, wp // 2, 2, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, (hp // 2) * (wp // 2), -1)
        return self.mlp(x)


def rope_index(n_vis_per_item, grids, n_text_before, n_text_after, absolute=False):
    """M-RoPE position ids (3, L) for a sequence [text | vision item 1 | vision item 2 ... | text]:
    text tokens: t = h = w = running index; a vision item with merged grid (T, H, W): t = start + frame,
    h = start + row, w = start + col; the next segment starts at the previous maximum id + 1.
    absolute=True gives plain 1-D ids in all three groups (the fixed-size baseline)."""
    ids, nxt = [], 0
    ids.append(torch.arange(n_text_before).repeat(3, 1)); nxt = n_text_before
    for (T, H, W) in grids:
        t, h, w = torch.meshgrid(torch.arange(T), torch.arange(H), torch.arange(W), indexing="ij")
        if absolute:
            ids.append(torch.arange(nxt, nxt + T * H * W).repeat(3, 1)); nxt += T * H * W
        else:
            ids.append(torch.stack([t.reshape(-1), h.reshape(-1), w.reshape(-1)]) + nxt); nxt = ids[-1].max().item() + 1
    ids.append(torch.arange(nxt, nxt + n_text_after).repeat(3, 1))
    return torch.cat(ids, 1)


class QwenLM(nn.Module):
    """Decoder-only LLM whose rotary bands are split into three sections (temporal, height, width) = M-RoPE."""
    def __init__(self, d=48, h=2, n_layers=2, mrope_section=(4, 4, 4)):
        super().__init__()
        self.h, self.section = h, list(mrope_section)
        self.tok = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB, bias=False)

    def forward(self, emb, pos_ids):
        cos, sin = rope_angles(pos_ids, self.section, emb.shape[-1] // self.h)
        x = emb
        for blk in self.blocks:
            x = blk(x, cos, sin, causal=True)
        return self.head(self.ln(x))


class Qwen2VL(nn.Module):
    def __init__(self, absolute=False):
        super().__init__()
        self.absolute = absolute
        self.vit, self.llm = ViT(absolute=absolute), QwenLM()
        self.merger = PatchMerger(self.vit.d, self.llm.tok.embedding_dim)

    def forward(self, frames, q, a):
        """frames: (B, T, 3, hP, wP). Sequence: <vs> visual tokens <ve> question <a> answer. Returns logits, targets."""
        B, T, _, H, W = frames.shape
        hp, wp = H // P, W // P
        vis = self.merger(self.vit(frames.reshape(B * T, 3, H, W)), hp, wp).view(B, T * (hp // 2) * (wp // 2), -1)
        text_after = torch.cat([torch.full((B, 1), VE), q, torch.full((B, 1), AS), a], 1)
        emb = torch.cat([self.llm.tok(torch.full((B, 1), VS)), vis, self.llm.tok(text_after)], 1)
        pos = rope_index(None, [(T, hp // 2, wp // 2)], 1, text_after.shape[1], self.absolute)
        logits = self.llm(emb, pos)
        return logits[:, -a.shape[1] - 1:-1], a


def batch(kind, size, B=32):
    """One batch of one size (h, w in patches, even numbers) — a different size every batch (dynamic resolution)."""
    h, w = size
    frames, qs, ans = [], [], []
    for _ in range(B):
        if kind == "image":
            img, objs = make_image(h, w, torch.randint(1, 4, ()).item())
            q, a = image_question(objs); frames.append(img[None])
        else:
            v, q, a = make_video(h, w); frames.append(v)
        qs.append(q); ans.append(a)
    q = torch.tensor([[TOK[t] for t in x] + [PAD] * (3 - len(x)) for x in qs])
    a = torch.tensor([[TOK[t] for t in x] + [END] + [PAD] * (3 - len(x)) for x in ans])
    return torch.stack(frames), q, a


TRAIN_SIZES = [(4, 4), (4, 6), (6, 4), (6, 6)]
TEST_SIZES = [(8, 8), (6, 10), (10, 6)]


def vit_task(h, w, B=32):
    """Stage-1-style supervision for the encoder: for every merged token, which colour it holds (4 labels) and whether
    another blob lies to its LEFT in the same row (needs the relative column offset between tokens)."""
    imgs, labels = [], []
    for _ in range(B):
        img, objs = make_image(h, w, torch.randint(1, 4, ()).item())
        lab = torch.zeros(h // 2, w // 2, 5)
        for c, (r, q) in objs.items():
            lab[r, q, c] = 1
            lab[r, q, 4] = float(any(r2 == r and q2 < q for (r2, q2) in objs.values()))
        imgs.append(img); labels.append(lab.view(-1, 5))
    return torch.stack(imgs), torch.stack(labels)


def train_vit(vit, merger, head, steps=500, lr=2e-3):
    opt = torch.optim.Adam(list(vit.parameters()) + list(merger.parameters()) + list(head.parameters()), lr=lr)
    for _ in range(steps):
        h, w = TRAIN_SIZES[torch.randint(len(TRAIN_SIZES), ()).item()]
        imgs, labels = vit_task(h, w)
        loss = F.binary_cross_entropy_with_logits(head(merger(vit(imgs), h, w)), labels)
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def vit_accuracy(vit, merger, head, sizes, n=4):
    hits = []
    for h, w in sizes:
        for _ in range(n):
            imgs, labels = vit_task(h, w)
            pred = (head(merger(vit(imgs), h, w)) > 0).float()
            has_blob = labels[..., :4].sum(-1) > 0                                     # score the tokens that hold a blob
            hits.append((pred[..., 4] == labels[..., 4])[has_blob].float().mean().item())
    return sum(hits) / len(hits)


def stage2(model, steps, lr=2e-3):
    """LLM training with the vision tower (ViT + merger) frozen, as in the paper's instruction-tuning stage: images
    and videos of a different size every batch, one question each. (Unfreezing the encoder here, before the LLM can
    read its tokens, wrecks the toy encoder — the paper only unfreezes it in its second, much longer stage.)"""
    opt = torch.optim.Adam(model.llm.parameters(), lr=lr)
    for step in range(1, steps + 1):
        kind = "video" if step % 3 == 0 else "image"
        frames, q, a = batch(kind, TRAIN_SIZES[torch.randint(len(TRAIN_SIZES), ()).item()])
        logits, tgt = model(frames, q, a)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


@torch.no_grad()
def qa_accuracy(model, sizes, n=6):
    acc = {}
    for kind in ("image", "video"):
        hits = []
        for size in sizes:
            for _ in range(n):
                frames, q, a = batch(kind, size)
                hits.append((model(frames, q, a)[0].argmax(-1)[:, 0] == a[:, 0]).float().mean().item())
        acc[kind] = sum(hits) / len(hits)
    return acc


def main():
    t0 = time.time()
    # 1) dynamic resolution: the token count follows the input
    m = Qwen2VL()
    counts = {}
    for name, frames in (("4x6 image", torch.rand(1, 1, 3, 8, 12)), ("8x8 image", torch.rand(1, 1, 3, 16, 16)), ("2-frame 4x4 video", torch.rand(1, 2, 3, 8, 8))):
        B, T, _, H, W = frames.shape
        counts[name] = m.merger(m.vit(frames.reshape(B * T, 3, H, W)), H // P, W // P).view(B, -1, 48).shape[1]
    print("Figure 2: visual tokens per input ->", counts)
    assert counts == {"4x6 image": 6, "8x8 image": 16, "2-frame 4x4 video": 8}
    # 2) M-RoPE ids and two properties
    print("M-RoPE ids for [<vs>, 2x2 image, <ve>, question ...] (rows: temporal, height, width):")
    print("  " + str(rope_index(None, [(1, 2, 2)], 1, 3).tolist()))
    print("M-RoPE ids for a 2-frame 1x2 video:  " + str(rope_index(None, [(2, 1, 2)], 1, 2).tolist()))
    text_ids = torch.arange(10).repeat(3, 1)
    c3, s3 = rope_angles(text_ids, [4, 4, 4], 24); c1, s1 = rope_angles(text_ids[:1], [12], 24)
    assert torch.allclose(c3, c1) and torch.allclose(s3, s1), "text tokens must get exactly the base LLM's 1-D RoPE"
    attn = Attention(48, 2).eval(); x = torch.randn(1, 2, 48)
    ids = torch.tensor([[1, 2], [3, 5], [2, 2]])
    logit = lambda ids: (lambda cos, sin: (apply_rope(attn.qkv(x).view(1, 2, 3, 2, 24).permute(2, 0, 3, 1, 4)[0], cos, sin)
                         @ apply_rope(attn.qkv(x).view(1, 2, 3, 2, 24).permute(2, 0, 3, 1, 4)[1], cos, sin).transpose(-1, -2)))(*rope_angles(ids, [4, 4, 4], 24))
    assert torch.allclose(logit(ids), logit(ids + torch.tensor([[7], [3], [11]])), atol=1e-4), "attention must depend only on relative (dt, dh, dw)"
    print("checked: text ids reproduce plain 1-D RoPE; attention logits are invariant to a common (dt, dh, dw) shift")
    # 3) the encoder at unseen resolutions: 2D-RoPE vs learned absolute positions
    print("encoder trained on 4x4..6x6 grids, tested on 8x8, 6x10, 10x6 (label: 'is there a blob to my left in my row?')")
    res = {}
    for name, absolute in (("2D-RoPE", False), ("absolute positions", True)):
        torch.manual_seed(0)
        vit, merger, head = ViT(absolute=absolute), PatchMerger(32, 48), nn.Linear(48, 5)
        train_vit(vit, merger, head)
        res[name] = (vit_accuracy(vit, merger, head, TRAIN_SIZES), vit_accuracy(vit, merger, head, TEST_SIZES))
        print(f"  {name:20s} trained sizes {res[name][0]:.2f}   unseen sizes {res[name][1]:.2f}")
    # 4) end to end: one model for images and videos of any size
    print("full model (2D-RoPE ViT -> merger -> M-RoPE LLM): vision tower trained, then the LLM on 4x4..6x6 inputs, 'is there a <colour> blob?'")
    torch.manual_seed(0)
    model = Qwen2VL()
    train_vit(model.vit, model.merger, nn.Linear(48, 5))
    loss = stage2(model, 600)
    tr, te = qa_accuracy(model, TRAIN_SIZES), qa_accuracy(model, TEST_SIZES)
    print(f"  loss {loss:.3f}   trained sizes: image {tr['image']:.2f} video {tr['video']:.2f}   unseen larger sizes: image {te['image']:.2f} video {te['video']:.2f}")
    print(f"({time.time() - t0:.1f} s)")
    assert res["2D-RoPE"][1] > 0.9 and res["2D-RoPE"][1] > res["absolute positions"][1] + 0.15, "2D-RoPE should generalise to unseen grids"
    assert min(tr.values()) > 0.9 and min(te.values()) > 0.85, "the model should answer at trained and at unseen sizes"


if __name__ == "__main__":
    main()
