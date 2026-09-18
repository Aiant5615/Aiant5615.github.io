"""BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and LLMs (Li et al., ICML 2023).

What is implemented (section / figure numbers follow the paper):
  * the Q-Former: N_q learned queries and text tokens sharing self-attention layers; queries cross-attend to the
    frozen image encoder's features; separate feed-forward nets for the two sides                        (Section 3.1, Figure 2)
  * stage 1, three objectives trained jointly with three self-attention masks                          (Section 3.2, Figure 2 right)
      - ITC: max over queries of the query-text [CLS] similarity, symmetric contrastive loss, *uni-modal* mask
      - ITG: caption generation from the queries with the *multimodal causal* mask ([DEC] starts the text)
      - ITM: binary match head on every query output, averaged, with hard negatives mined from the ITC similarities,
             *bi-directional* mask
  * stage 2: a fully-connected layer projects the query outputs to the LLM width; they are prepended to the caption as
    soft visual prompts; language-modelling loss with the LLM frozen (decoder-only case)                 (Section 3.3, Figure 3)
  * the Figure 5 ablation: stage 2 with vs without stage-1 representation learning (measured as how fast the frozen LLM
    becomes usable: caption accuracy after 25 and after 600 generative steps)
Simplifications: frozen image encoder = a patch projection + one Transformer layer pre-trained here on colour/shape
classification (in place of ViT-g); the "LLM" is a
2-layer decoder pretrained here on text-only captions; 8 queries instead of 32; 16x16 synthetic coloured-shape images with
captions "a red square" over a 12-word vocabulary; both stages run for a few hundred steps.

Run:  python blip2.py        (CPU, about 25 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

WORDS = ["<pad>", "[CLS]", "[DEC]", "<bos>", "<eos>", "a", "red", "green", "blue", "square", "plus", "cross", "ring"]
TOK = {w: i for i, w in enumerate(WORDS)}
NQ, D, D_IMG = 8, 64, 48


def make_stencils(s=6):
    sq = torch.ones(s, s)
    plus = torch.zeros(s, s); plus[2:4, :] = 1; plus[:, 2:4] = 1
    cross = ((torch.eye(s) + torch.eye(s).flip(1)) > 0).float()
    ring = torch.ones(s, s); ring[1:-1, 1:-1] = 0
    return torch.stack([sq, plus, cross, ring])


STENCILS = make_stencils()


def make_batch(B, res=16, noise=0.2):
    """Images (B,3,16,16) and captions "a {colour} {shape}" as token ids (B, 3): [a, colour, shape]."""
    color, shape = torch.randint(0, 3, (B,)), torch.randint(0, 4, (B,))
    x = torch.zeros(B, 3, res, res)
    r, c = torch.randint(0, 11, (B,)), torch.randint(0, 11, (B,))
    for i in range(B):
        x[i, color[i], r[i] : r[i] + 6, c[i] : c[i] + 6] = STENCILS[shape[i]]
    cap = torch.stack([torch.full((B,), TOK["a"]), 6 + color, 9 + shape], 1)
    return x + noise * torch.randn_like(x), cap


class FrozenImageEncoder(nn.Module):
    """Stands in for the frozen ViT-g: patch projection + one Transformer layer. It is pre-trained once on an image
    classification task (the paper's encoder is a CLIP/EVA ViT) and then frozen for both BLIP-2 stages."""
    def __init__(self, P=4):
        super().__init__()
        self.P, self.proj = P, nn.Linear(3 * P * P, D_IMG)
        self.layer = nn.TransformerEncoderLayer(D_IMG, 4, 2 * D_IMG, 0.0, batch_first=True, norm_first=True)

    def forward(self, x):
        B, C, H, W = x.shape; P = self.P
        patches = x.unfold(2, P, P).unfold(3, P, P).reshape(B, C, -1, P * P).transpose(1, 2).reshape(B, -1, C * P * P)
        return self.layer(self.proj(patches))                                             # (B, 16, D_IMG)

    def pretrain_and_freeze(self, steps=200):
        """Supervised pre-training on (colour, shape) classes, mean-pooled features -> 12-way head; then frozen."""
        head, opt = nn.Linear(D_IMG, 12), torch.optim.Adam(self.parameters(), lr=2e-3)
        opt.add_param_group({"params": head.parameters()})
        for _ in range(steps):
            x, cap = make_batch(64)
            loss = F.cross_entropy(head(self(x).mean(1)), (cap[:, 1] - 6) * 4 + (cap[:, 2] - 9))
            opt.zero_grad(); loss.backward(); opt.step()
        for p in self.parameters():
            p.requires_grad_(False)
        return loss.item()


# ───────────────────────── the Q-Former (Section 3.1) ─────────────────────────
def attention_mask(kind, L):
    """Figure 2 (right). Returns the nn.MultiheadAttention convention: True = may NOT attend. Sequence = [queries ; text]."""
    n = NQ + L
    allowed = torch.zeros(n, n, dtype=torch.bool)
    allowed[:NQ, :NQ] = True                                     # queries always attend to each other, never to text
    if kind == "unimodal":                                       # ITC: no query-text interaction
        allowed[NQ:, NQ:] = True
    elif kind == "multimodal_causal":                            # ITG: text sees all queries and earlier text
        allowed[NQ:, :NQ] = True
        allowed[NQ:, NQ:] = torch.tril(torch.ones(L, L, dtype=torch.bool))
    elif kind == "bidirectional":                                # ITM: everything attends to everything
        allowed[:] = True
    return ~allowed


class QFormerLayer(nn.Module):
    def __init__(self, h=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(D, h, batch_first=True)                    # shared by queries and text
        self.cross_attn = nn.MultiheadAttention(D, h, batch_first=True, kdim=D_IMG, vdim=D_IMG)   # queries -> image
        self.ln_sa, self.ln_ca, self.ln_fq, self.ln_ft = (nn.LayerNorm(D) for _ in range(4))
        self.ffn_q = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))   # image-transformer FFN
        self.ffn_t = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))   # text-transformer FFN

    def forward(self, q, t, img, mask):
        x = torch.cat([q, t], 1)
        h = self.ln_sa(x)
        x = x + self.self_attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        q, t = x[:, :NQ], x[:, NQ:]
        q = q + self.cross_attn(self.ln_ca(q), img, img, need_weights=False)[0]          # only the queries see the image
        return q + self.ffn_q(self.ln_fq(q)), t + self.ffn_t(self.ln_ft(t))


class QFormer(nn.Module):
    def __init__(self, n_layers=2, d_e=32):
        super().__init__()
        self.queries = nn.Parameter(0.02 * torch.randn(1, NQ, D))                          # learnable query embeddings
        self.tok, self.pos = nn.Embedding(len(WORDS), D), nn.Parameter(0.02 * torch.randn(1, 8, D))
        self.layers = nn.ModuleList([QFormerLayer() for _ in range(n_layers)])
        self.ln_q, self.ln_t = nn.LayerNorm(D), nn.LayerNorm(D)
        self.vision_proj, self.text_proj = nn.Linear(D, d_e), nn.Linear(D, d_e)           # ITC projections
        self.itm_head, self.lm_head = nn.Linear(D, 2), nn.Linear(D, len(WORDS))           # ITM and ITG heads
        self.temp = 0.07                                                                   # ITC temperature

    def forward(self, img, text, kind):
        """img: (B, 16, D_IMG) frozen features; text: (B, L) ids; returns query outputs Z (B, NQ, D) and text states (B, L, D)."""
        q = self.queries.expand(img.size(0), -1, -1)
        t = self.tok(text) + self.pos[:, : text.size(1)]
        mask = attention_mask(kind, text.size(1))
        for layer in self.layers:
            q, t = layer(q, t, img, mask)
        return self.ln_q(q), self.ln_t(t)


def itc_loss(qf, img, cap):
    """ITC: sim(I, T) = max_q <z_q, t_cls>, symmetric contrastive loss with in-batch negatives (uni-modal mask).
    Returned as the excess over the floor set by duplicate captions, so 0 means perfectly aligned."""
    B = img.size(0)
    Z, T = qf(img, torch.cat([torch.full((B, 1), TOK["[CLS]"]), cap], 1), "unimodal")
    z = F.normalize(qf.vision_proj(Z), dim=-1)                                            # (B, NQ, d_e)
    t = F.normalize(qf.text_proj(T[:, 0]), dim=-1)                                        # (B, d_e) from [CLS]
    sim = torch.einsum("iqd,jd->ijq", z, t).max(-1).values / qf.temp                      # (B, B): max over queries
    # only 12 captions exist, so a batch holds duplicates: every pair with an identical caption is a positive
    pos = (cap[:, None, :] == cap[None, :, :]).all(-1).float()
    target = pos / pos.sum(1, keepdim=True)
    loss = -((sim.log_softmax(1) * target).sum(1).mean() + (sim.T.log_softmax(1) * target.T).sum(1).mean()) / 2
    floor = -(target * target.clamp_min(1e-9).log()).sum(1).mean()                        # entropy of the targets = best possible loss
    return loss - floor, sim.detach().masked_fill(pos.bool(), float("-inf"))              # duplicates cannot be negatives


def itg_loss(qf, img, cap):
    """ITG: generate the caption token by token; text attends to the queries and earlier text (multimodal causal mask)."""
    text = torch.cat([torch.full((cap.size(0), 1), TOK["[DEC]"]), cap], 1)               # [DEC] a red square
    _, T = qf(img, text, "multimodal_causal")
    logits = qf.lm_head(T[:, :-1])
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), text[:, 1:].reshape(-1))


def itm_loss(qf, img, cap, sim):
    """ITM: is this pair matched? Linear head on every query output, averaged; hard negatives sampled from the ITC similarities."""
    B = img.size(0)
    w = sim.softmax(1) + 1e-6                                                             # image -> text similarities (positives masked out)
    neg_txt = torch.multinomial(w, 1).squeeze(1)                                          # a hard negative text per image
    neg_img = torch.multinomial(w.T, 1).squeeze(1)                                        # a hard negative image per text
    imgs = torch.cat([img, img, img[neg_img]]); caps = torch.cat([cap, cap[neg_txt], cap])
    text = torch.cat([torch.full((3 * B, 1), TOK["[CLS]"]), caps], 1)
    Z, _ = qf(imgs, text, "bidirectional")
    logits = qf.itm_head(Z).mean(1)                                                       # average over the NQ queries
    labels = torch.cat([torch.ones(B), torch.zeros(2 * B)]).long()
    return F.cross_entropy(logits, labels), (logits.argmax(-1) == labels).float().mean().item()


# ───────────────────────── stage 2: the frozen decoder-only LLM (Section 3.3) ─────────────────────────
class ToyLLM(nn.Module):
    def __init__(self, d=64, n_layers=2, max_len=16):
        super().__init__()
        self.emb, self.pos = nn.Embedding(len(WORDS), d), nn.Parameter(0.02 * torch.randn(1, max_len, d))
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, "gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, len(WORDS))

    def forward(self, inputs_embeds):
        L = inputs_embeds.size(1)
        h = self.blocks(inputs_embeds + self.pos[:, :L], mask=torch.triu(torch.ones(L, L, dtype=torch.bool), 1))
        return self.head(self.ln(h))


def stage2_loss(qf, fc, llm, img, cap):
    """Soft visual prompts FC(Z) prepended to "<bos> a red square <eos>"; LM loss on the caption tokens only."""
    B = img.size(0)
    text = torch.cat([torch.full((B, 1), TOK["<bos>"]), cap, torch.full((B, 1), TOK["<eos>"])], 1)
    Z, _ = qf(img, torch.full((B, 1), TOK["[CLS]"]), "unimodal")                          # queries alone (text unused)
    logits = llm(torch.cat([fc(Z), llm.emb(text)], 1))[:, NQ:-1]                          # predictions for text[1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), text[:, 1:].reshape(-1))
    acc = (logits.argmax(-1)[:, 1:3] == text[:, 2:4]).all(1).float().mean().item()        # colour and shape both right
    return loss, acc


def run_stage2(qf, llm, enc, steps=600, probe=25, tag=""):
    """Returns the caption accuracy early (after `probe` steps) and at the end: Figure 5's point is that stage-1
    representation learning makes the LLM usable with far less generative training."""
    fc = nn.Linear(D, 64)
    opt = torch.optim.Adam(list(qf.parameters()) + list(fc.parameters()), lr=3e-3)         # LLM and image encoder frozen
    accs = {}
    for step in range(1, steps + 1):
        x, cap = make_batch(32)
        loss, _ = stage2_loss(qf, fc, llm, enc(x), cap)
        opt.zero_grad(); loss.backward(); opt.step()
        if step in (probe, steps):
            with torch.no_grad():
                x, cap = make_batch(800)
                _, accs[step] = stage2_loss(qf, fc, llm, enc(x), cap)
    print(f"stage 2 {tag}: caption (colour+shape) accuracy after {probe} steps {accs[probe]:.3f}, after {steps} steps {accs[steps]:.3f}")
    return accs[probe], accs[steps]


def main():
    t0 = time.time()
    enc, qf = FrozenImageEncoder(), QFormer()
    print(f"frozen image encoder: pre-trained on colour/shape classification, final loss {enc.pretrain_and_freeze():.3f}")
    print(f"Q-Former parameters: {sum(p.numel() for p in qf.parameters()):,}  (N_q = {NQ} queries of width {D})")
    opt = torch.optim.Adam(qf.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / 600)
    for step in range(1, 601):                                                            # ── stage 1 ──
        x, cap = make_batch(32)
        img = enc(x)
        l_itc, sim = itc_loss(qf, img, cap)
        l_itg = itg_loss(qf, img, cap)
        l_itm, itm_acc = itm_loss(qf, img, cap, sim)
        loss = l_itc + l_itg + l_itm
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 150 == 0:
            print(f"stage 1 step {step:3d}  ITC {l_itc.item():.3f}  ITG {l_itg.item():.3f}  ITM {l_itm.item():.3f} (acc {itm_acc:.2f})")
    llm = ToyLLM()                                                                        # ── pretrain the "LLM" on text only ──
    opt = torch.optim.Adam(llm.parameters(), lr=2e-3)
    for step in range(300):
        _, cap = make_batch(64)
        text = torch.cat([torch.full((64, 1), TOK["<bos>"]), cap, torch.full((64, 1), TOK["<eos>"])], 1)
        # A prefix of 1..NQ tokens: mostly <pad>, and 70% of the time it also carries the colour and shape words as
        # "hints" at random positions. So the frozen LLM, like a real LLM with in-context ability, has learned to read
        # its prefix when writing the caption; the visual prompts of stage 2 must imitate such hints.
        k = int(torch.randint(1, NQ + 1, ()))
        prefix = torch.full((64, k), TOK["<pad>"])
        hinted = torch.rand(64) < 0.7
        slots = torch.rand(64, k).argsort(1)[:, :2]                                        # two distinct prefix positions
        prefix[hinted, slots[hinted, 0]] = cap[hinted, 1]; prefix[hinted, slots[hinted, 1] if k > 1 else slots[hinted, 0]] = cap[hinted, 2]
        logits = llm(llm.emb(torch.cat([prefix, text], 1)))[:, k:-1]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), text[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    for p in llm.parameters():
        p.requires_grad_(False)
    print(f"frozen text-only LLM: loss {loss.item():.3f} (pre-trained on captions with optional hint prefixes; never sees an image)")
    early_with, acc_with = run_stage2(qf, llm, enc, tag="with stage-1 Q-Former   ")      # ── stage 2 ──
    torch.manual_seed(1)
    early_without, acc_without = run_stage2(QFormer(), llm, enc, tag="without stage 1 (Fig. 5)")
    print(f"done in {time.time() - t0:.1f} s")
    assert l_itc.item() < 0.5 and itm_acc > 0.8, "stage-1 objectives did not train"
    assert acc_with > 0.85, "the frozen LLM should caption from the soft visual prompts"
    assert early_with > early_without + 0.1, "stage-1 representation learning should make stage 2 converge much faster (Figure 5)"


if __name__ == "__main__":
    main()
