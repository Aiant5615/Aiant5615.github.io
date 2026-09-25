"""Multimodal Chain-of-Thought Reasoning in Language Models (Zhang et al., 2023) — toy implementation.

What is implemented (section / figure / table numbers follow the paper):
  * an encoder-decoder (T5-style) model that generates text from question + context + options            (Section 4)
  * the one-stage variants of Table 2: QCM->A (no rationale), QCM->RA (rationale, then answer)
  * the two-stage framework of Figure 4: stage 1 QCM->R (rationale generation), stage 2 QCMR->A (answer inference),
    trained separately on gold pairs; at test time stage 2 consumes stage 1's generated rationale
  * vision features: a frozen patch extractor, projected by W_h; single-head attention with the text as queries and
    the patches as keys/values; gated fusion lambda = sigmoid(W_l H + W_v H_attn), H_fuse = (1-lambda) H + lambda H_attn,
    fed to the decoder (Section 4.2, the gate_dense / mha_layer of the official code)
  * the hallucination analysis of Figure 3: how often the generated rationale states a premise that contradicts the
    image, text-only vs with vision features; and the accuracy of every variant (Tables 2-3, 6)
Simplifications: a synthetic "ScienceQA": two bar magnets in a 8x24 image whose facing poles are coloured; the
question is attract-or-repel; half the problems also state the poles in the text context (answerable without the
image), half do not; the rationale is "left <pole> ; right <pole> ; same/opposite"; a conv net stands in for ViT.

Run:  python multimodal_cot.py        (CPU, about 50 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ───────────────────────── data: magnets, questions, rationales ─────────────────────────
WORDS = ["N", "S", "left", "right", ";", "same", "opposite", "attract", "repel", "question", "context", "options", "unknown",
         "<bos>", "<end>", "<pad>"]
TOK = {w: i for i, w in enumerate(WORDS)}
BOS, END, PAD = TOK["<bos>"], TOK["<end>"], TOK["<pad>"]
VOCAB = len(WORDS)
POLE_RGB = {0: torch.tensor([1.0, 0.2, 0.2]), 1: torch.tensor([0.2, 0.2, 1.0])}     # N red, S blue


def make_problem():
    """Two magnets; the pole of the left magnet facing right and of the right magnet facing left decide the answer.
    With probability 1/2 the context states both poles (text-answerable), otherwise only the image shows them."""
    pl, pr = torch.randint(2, ()).item(), torch.randint(2, ()).item()
    img = 0.6 * torch.ones(3, 8, 24) + 0.05 * torch.rand(3, 8, 24)
    img[:, :, 10:14] = 1.0                                                          # the gap between the magnets
    img[:, 2:6, 7:10] = POLE_RGB[pl][:, None, None]                                  # facing end of the left magnet
    img[:, 2:6, 14:17] = POLE_RGB[pr][:, None, None]                                 # facing end of the right magnet
    in_text = bool(torch.rand(()) < 0.5)
    ctx = ["left", WORDS[pl], "right", WORDS[pr]] if in_text else ["unknown"]
    x = ["question", "context"] + ctx + ["options", "attract", "repel"]
    rationale = ["left", WORDS[pl], ";", "right", WORDS[pr], ";", "same" if pl == pr else "opposite"]
    answer = ["repel" if pl == pr else "attract"]
    return img, x, rationale, answer, in_text, (pl, pr)


def encode(words):
    return [TOK[w] for w in words]


def make_batch(B, mode):
    """mode: 'A' (QCM->A), 'RA' (QCM->RA), 'R' (QCM->R), 'RtoA' (QCMR->A with the GOLD rationale appended)."""
    imgs, xs, ys, meta = [], [], [], []
    for _ in range(B):
        img, x, r, a, in_text, poles = make_problem()
        if mode == "RtoA":
            x = x + [";"] + r
        y = {"A": a, "RA": r + [";"] + a, "R": r, "RtoA": a}[mode]
        imgs.append(img); xs.append(encode(x)); ys.append([BOS] + encode(y) + [END]); meta.append((in_text, poles))
    Lx, Ly = max(map(len, xs)), max(map(len, ys))
    pad = lambda seqs, L: torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])
    return torch.stack(imgs), pad(xs, Lx), pad(ys, Ly), meta


# ───────────────────────── vision extractor (frozen) ─────────────────────────
class PatchExtractor(nn.Module):
    """Stands in for the frozen ViT: 4x4 patches -> 2x6 = 12 patch features. Pretrained to classify each patch
    (background / gap / N / S), then frozen."""
    def __init__(self, d_v=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, d_v, 4, stride=4))
        self.head = nn.Linear(d_v, 4)

    def forward(self, img):                                                          # (B, 3, 8, 24) -> (B, 12, d_v)
        return self.net(img).flatten(2).transpose(1, 2)


def pretrain_extractor(ext, steps=200):
    opt = torch.optim.Adam(ext.parameters(), lr=3e-3)
    for _ in range(steps):
        imgs, labels = [], []
        for _ in range(32):
            img, _, _, _, _, (pl, pr) = make_problem()
            lab = torch.zeros(2, 6, dtype=torch.long); lab[:, 2:4] = 1                # background 0, gap 1
            lab[:, 1] = 2 + pl; lab[:, 4] = 2 + pr                                  # patches holding the poles (rows 0-1 both touch them)
            imgs.append(img); labels.append(lab.view(-1))
        loss = F.cross_entropy(ext.head(ext(torch.stack(imgs))).reshape(-1, 4), torch.stack(labels).reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    ext.requires_grad_(False).eval()


# ───────────────────────── the encoder-decoder with gated vision fusion (Section 4.2) ─────────────────────────
class MHA(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.q, self.kv, self.out = h, nn.Linear(d, d), nn.Linear(d, 2 * d), nn.Linear(d, d)

    def forward(self, x, mem, causal=False, mem_pad=None):
        B, L, d = x.shape
        q = self.q(x).view(B, L, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(mem).view(B, mem.shape[1], 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        att = q @ k.transpose(-1, -2) / math.sqrt(d // self.h)
        if causal:
            att = att.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), float("-inf"))
        if mem_pad is not None:
            att = att.masked_fill(mem_pad[:, None, None, :], float("-inf"))
        return self.out((att.softmax(-1) @ v).transpose(1, 2).reshape(B, L, d))


class EncLayer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2, self.attn = nn.LayerNorm(d), nn.LayerNorm(d), MHA(d, h)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(self, x, pad):
        x = x + self.attn(self.ln1(x), self.ln1(x), mem_pad=pad)
        return x + self.ff(self.ln2(x))


class DecLayer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn, self.cross, self.ff = MHA(d, h), MHA(d, h), nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(self, y, mem, mem_pad):
        y = y + self.self_attn(self.ln1(y), self.ln1(y), causal=True)
        y = y + self.cross(self.ln2(y), mem, mem_pad=mem_pad)
        return y + self.ff(self.ln3(y))


class MMCoT(nn.Module):
    """T5-style encoder-decoder. With `vision`, the encoder output H_language is fused with the projected patch
    features: H_attn = softmax(Q K^T / sqrt(d)) V with Q = H_language, K = V = W_h * patches (single head), then
    lambda = sigmoid(W_l H + W_v H_attn), H_fuse = (1 - lambda) H + lambda H_attn (eqs. of Section 4.2)."""
    def __init__(self, extractor=None, d=64, h=4, n_layers=2, d_v=32, ctx=32):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(ctx, d)
        self.enc = nn.ModuleList([EncLayer(d, h) for _ in range(n_layers)])
        self.dec = nn.ModuleList([DecLayer(d, h) for _ in range(n_layers)])
        self.ln_enc, self.ln_dec, self.head = nn.LayerNorm(d), nn.LayerNorm(d), nn.Linear(d, VOCAB, bias=False)
        self.extractor = extractor
        if extractor is not None:
            self.W_h = nn.Linear(d_v, d)                                             # image_dense
            self.mha = MHA(d, 1)                                                     # mha_layer, one head
            self.gate = nn.Linear(2 * d, d)                                          # gate_dense: [W_l, W_v]

    def encode(self, x, img):
        pad = x == PAD
        H = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        for layer in self.enc:
            H = layer(H, pad)
        H = self.ln_enc(H)
        if self.extractor is not None:
            H_vision = self.W_h(self.extractor(img))
            H_attn = self.mha(H, H_vision)
            lam = torch.sigmoid(self.gate(torch.cat([H, H_attn], -1)))
            H = (1 - lam) * H + lam * H_attn
        return H, pad

    def decode(self, y, mem, pad):
        h = self.tok(y) + self.pos(torch.arange(y.shape[1]))
        for layer in self.dec:
            h = layer(h, mem, pad)
        return self.head(self.ln_dec(h))

    def forward(self, x, img, y):
        mem, pad = self.encode(x, img)
        return self.decode(y[:, :-1], mem, pad)

    @torch.no_grad()
    def generate(self, x, img, max_len=10):
        mem, pad = self.encode(x, img)
        y = torch.full((x.shape[0], 1), BOS)
        for _ in range(max_len):
            nxt = self.decode(y, mem, pad)[:, -1].argmax(-1, keepdim=True)
            y = torch.cat([y, nxt], 1)
        return y[:, 1:]


def train(model, mode, steps, lr=1e-3):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        img, x, y, _ = make_batch(32, mode)
        logits = model(x, img, y)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); opt.step()


def strip(seq):
    seq = seq.tolist()
    return seq[: seq.index(END)] if END in seq else seq


@torch.no_grad()
def evaluate(stage1, stage2, mode, n=400):
    """Answer accuracy overall and on the image-only problems, plus the hallucination rate: the fraction of generated
    rationales on image-only problems whose stated poles contradict the image (Figure 3)."""
    img, x, y, meta = make_batch(n, "A")
    gold = y[:, 1]
    halluc, image_only = [], torch.tensor([not m[0] for m in meta])
    if mode == "A":
        out = stage1.generate(x, img, 2)[:, 0]
    else:
        r = stage1.generate(x, img, 9)
        if mode == "RA":
            out = torch.tensor([(s + [PAD] * 9)[8] for s in map(strip, r)])              # the token after the rationale
        else:
            x2 = torch.cat([x, torch.full((n, 1), TOK[";"]), r[:, :7]], 1)
            out = stage2.generate(x2, img, 2)[:, 0]
        for s, (in_text, (pl, pr)) in zip(r.tolist(), meta):
            if not in_text:
                halluc.append(float(s[1] != TOK[WORDS[pl]] or s[4] != TOK[WORDS[pr]]))
    correct = (out == gold).float()
    return correct.mean().item(), correct[image_only].mean().item(), (sum(halluc) / len(halluc)) if halluc else float("nan")


def main():
    t0 = time.time()
    ext = PatchExtractor(); pretrain_extractor(ext)
    steps = 400
    rows = []
    print(f"{'model':40s} {'accuracy':>9s} {'image-only acc':>15s} {'hallucinated rationales':>24s}")
    for vision in (False, True):
        make = lambda: MMCoT(ext if vision else None)
        tag = "with vision features" if vision else "text only"
        torch.manual_seed(1); m = make(); train(m, "A", steps); rows.append((f"QCM->A  one-stage, {tag}", evaluate(m, None, "A")))
        torch.manual_seed(1); m = make(); train(m, "RA", steps); rows.append((f"QCM->RA one-stage, {tag}", evaluate(m, None, "RA")))
        torch.manual_seed(1); s1, s2 = make(), make(); train(s1, "R", steps); train(s2, "RtoA", steps)
        rows.append((f"QCM->R, QCMR->A two-stage, {tag}", evaluate(s1, s2, "two")))
        for name, (acc, acc_img, hal) in rows[-3:]:
            print(f"{name:40s} {acc:9.2f} {acc_img:15.2f} {hal:24.2f}")
    print(f"({time.time() - t0:.1f} s)")
    res = dict(rows)
    text_two, vis_two = res["QCM->R, QCMR->A two-stage, text only"], res["QCM->R, QCMR->A two-stage, with vision features"]
    assert text_two[2] > 0.4 and text_two[1] < 0.7, "a text-only model must hallucinate the poles it cannot see (Figure 3a)"
    assert vis_two[2] < 0.1 and vis_two[0] > 0.9, "vision features should ground the rationale and fix the answers (Figure 3b)"
    assert res["QCM->A  one-stage, text only"][1] < 0.7, "without the image the direct answer is a guess on image-only problems"


if __name__ == "__main__":
    main()
