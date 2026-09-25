"""Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models (Huang et al., 2025) — toy implementation.

What is implemented (section / figure numbers follow the paper):
  * a base MLLM: a decoder-only LM that reads an image as a sequence of glyph patches through a linear vision
    encoder; pretrained on captions (image -> the expression as text), direct answers, and TEXT-only step-by-step
    solutions, i.e. little exposure to multimodal chains of thought                                    (Section 3.1)
  * Vision-R1-Zero: GRPO on the base model with format + result rewards; accuracy does not take off  (Section 3.1)
  * cold start by modality bridging (Section 3.3.1, Figure 2): the MLLM describes the image (the expression), a
    text-only reasoner writes an R1-style chain with re-check steps, chains whose answer is wrong are filtered out,
    the image is re-attached -> SFT gives Vision-R1-CI
  * the overthinking diagnostic (Section 3.3.2, Figure 1A): the CI model's correct answers cluster at shorter chains
  * GRPO with the hard formatting result reward (credit only when formatted AND correct AND within the cap) and
    Progressive Thinking Suppression Training: a length cap and group size schedule (Section 3.4, Figure 3),
    compared with a fixed large cap (Table 5)
Simplifications: 3-operation arithmetic on glyph "images" (one 16-d noisy pattern per symbol) instead of geometry
diagrams; the text reasoner is a program that writes the chain with a random number of re-check steps; caps are
26 / 40 tokens with groups of 8 / 4 instead of 4K / 8K / 16K with 16 / 8 / 4.

Run:  python vision_r1.py        (CPU, about 55 s)
"""
import copy, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ───────────────────────── the world: arithmetic rendered as glyph images ─────────────────────────
MAX_VAL, K_OPS = 24, 3
PLUS, MINUS, EQ, Q, A, ANS, SEP, END, THINK, ENDTHINK, CHECK, IMG, PAD = range(MAX_VAL + 1, MAX_VAL + 14)
VOCAB = PAD + 1
GLYPH_DIM = 16
GLYPHS = torch.randn(MAX_VAL + 4, GLYPH_DIM)                   # a pattern for each number 0..24, "+", "-", "="
GLYPH = {t: i for i, t in enumerate(list(range(MAX_VAL + 1)) + [PLUS, MINUS, EQ])}


def make_problem(k=K_OPS):
    while True:
        nums = torch.randint(0, 7, (k + 1,)).tolist()
        ops = torch.randint(0, 2, (k,)).tolist()
        res, r = [], nums[0]
        for n, o in zip(nums[1:], ops):
            r = r + n if o == 0 else r - n
            res.append(r)
        if all(0 <= x <= MAX_VAL for x in res):
            return nums, ops, res


def expression(nums, ops):
    toks = [nums[0]]
    for n, o in zip(nums[1:], ops):
        toks += [PLUS if o == 0 else MINUS, n]
    return toks + [EQ]


def render(nums, ops):
    """The image: one noisy glyph patch per symbol of the expression "a + b - c + d =" (8 patches for 3 operations)."""
    e = expression(nums, ops)
    return GLYPHS[[GLYPH[t] for t in e]] + 0.5 * torch.randn(len(e), GLYPH_DIM)


def steps(nums, ops, res, checks=0):
    """A chain of thought; `checks` re-verification steps ("wait, let me check: 7 - 4 = 3") are inserted after random
    steps — the reflective style of DeepSeek-R1 chains."""
    toks, prev = [], nums[0]
    where = set(torch.randperm(len(ops))[:checks].tolist())
    for i, (n, o, r) in enumerate(zip(nums[1:], ops, res)):
        toks += [prev, PLUS if o == 0 else MINUS, n, EQ, r, SEP]
        if i in where:
            toks += [CHECK, r, MINUS if o == 0 else PLUS, n, EQ, prev, SEP]
        prev = r
    return toks


# ───────────────────────── the multimodal LM ─────────────────────────
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


class MLLM(nn.Module):
    """Decoder-only Transformer (no position embedding) whose IMG placeholder tokens are replaced by projected glyph
    patches: the toy's vision encoder is the linear layer `vis`."""
    def __init__(self, d=96, h=4, n_layers=2):
        super().__init__()
        self.tok, self.vis = nn.Embedding(VOCAB, d), nn.Linear(GLYPH_DIM, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB, bias=False)

    def forward(self, seq, img=None):
        x = self.tok(seq)
        if img is not None:
            x = x.clone(); x[seq == IMG] = self.vis(img).reshape(-1, x.shape[-1])
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


N_IMG = 2 * K_OPS + 2                                          # glyphs in a 3-operation expression


def prompt_of(nums, ops):
    """The multimodal prompt: [IMG x 8] A  (the image replaces the textual question)."""
    return [IMG] * N_IMG + [A]


def pretrain_batch(B):
    """Base-model corpus: (a) caption: image -> expression text; (b) image -> direct answer; (c) TEXT question ->
    step-by-step solution. The base model never sees an image followed by a chain."""
    seqs, imgs = [], []
    for _ in range(B):
        nums, ops, res = make_problem(torch.randint(2, 4, ()).item())
        kind = torch.multinomial(torch.tensor([0.25, 0.25, 0.5]), 1).item()   # half of the corpus is text-only solutions
        if kind == 0:
            s = [IMG] * len(expression(nums, ops)) + [Q] + expression(nums, ops) + [END]
        elif kind == 1:
            s = [IMG] * len(expression(nums, ops)) + [A, ANS, res[-1], END]
        else:
            s = [Q] + expression(nums, ops) + [A, THINK] + steps(nums, ops, res) + [ENDTHINK, ANS, res[-1], END]
        seqs.append(s)
        if kind < 2:
            imgs.append(render(nums, ops))                             # glyphs only for the sequences that show an image
    L = max(map(len, seqs))
    seq = torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])
    return seq, torch.cat(imgs)


def lm_loss(model, seq, img):
    logits = model(seq[:, :-1], img)
    return F.cross_entropy(logits.reshape(-1, VOCAB), seq[:, 1:].reshape(-1), ignore_index=PAD)


def pretrain(model, steps_, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps_)))
    for _ in range(steps_):
        loss = lm_loss(model, *pretrain_batch(32))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return loss.item()


# ───────────────────────── sampling, rewards, GRPO with a length cap (Section 3.4) ─────────────────────────
def problems(B):
    ps = [make_problem() for _ in range(B)]
    q = torch.tensor([prompt_of(n, o) for n, o, _ in ps])
    img = torch.stack([render(n, o) for n, o, _ in ps])
    return q, img, torch.tensor([r[-1] for _, _, r in ps])


@torch.no_grad()
def sample(model, q, img, cap, greedy=False):
    """Up to `cap` response tokens; a response is frozen (PAD) after END."""
    seq, done = q, torch.zeros(q.size(0), dtype=torch.bool)
    for _ in range(cap):
        logits = model(seq, img)[:, -1]
        logits[:, [IMG, PAD]] = float("-inf")                          # placeholders are never generated
        nxt = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        seq = torch.cat([seq, nxt[:, None]], 1)
        done |= nxt == END
        if done.all():
            break
    return seq[:, q.size(1):]


def judge(o, gold):
    """Per response: formatted (think tags then ANS), correct, finished (END within the cap), and its length."""
    fmt, acc, fin, length = (torch.zeros(o.size(0)) for _ in range(4))
    for i, toks in enumerate(o.tolist()):
        fin[i] = float(END in toks)
        toks = toks[: toks.index(END)] if END in toks else [t for t in toks if t != PAD]
        length[i] = len(toks)
        if ANS in toks:
            j = toks.index(ANS)
            acc[i] = float(j + 1 < len(toks) and toks[j + 1] == gold[i].item())
            fmt[i] = float(toks[:1] == [THINK] and ENDTHINK in toks and toks.index(ENDTHINK) < j)
    return fmt, acc, fin, length


def token_log_probs(model, q, img, o):
    logits = model(torch.cat([q, o], 1)[:, :-1], img)[:, q.size(1) - 1:]
    return logits.log_softmax(-1).gather(-1, o[..., None]).squeeze(-1)


def grpo(policy, ref, opt, cap, G, hard, B=12, eps=0.2, beta=0.04):
    """One GRPO iteration with G samples per problem under a response cap. hard=True is the HFRRF: r = 1 only if
    formatted AND correct AND finished within the cap; hard=False is the R1-Zero reward: format + result."""
    q, img, gold = problems(B)
    q, img, gold = q.repeat_interleave(G, 0), img.repeat_interleave(G, 0), gold.repeat_interleave(G, 0)
    o = sample(policy, q, img, cap)
    fmt, acc, fin, length = judge(o, gold)
    r = fmt * acc * fin if hard else 0.5 * fmt + acc
    rg = r.view(B, G)
    adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-4)).view(-1)
    mask = (o != PAD).float()
    logp = token_log_probs(policy, q, img, o)
    with torch.no_grad():
        logp_old, logp_ref = logp.detach(), token_log_probs(ref, q, img, o)
    ratio = (logp - logp_old).exp()
    surr = torch.min(ratio * adv[:, None], ratio.clamp(1 - eps, 1 + eps) * adv[:, None])
    kl = (logp_ref - logp).exp() - (logp_ref - logp) - 1
    loss = -((surr - beta * kl) * mask).sum(1).div(mask.sum(1).clamp(min=1)).mean()
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
    return acc.mean().item(), length.mean().item()


def rl(init, schedule, hard, lr=3e-4, tag=""):
    """schedule: list of (iterations, cap, G) stages — PTST when the cap grows and G shrinks across stages."""
    policy, ref = copy.deepcopy(init), copy.deepcopy(init).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for stage, (iters, cap, G) in enumerate(schedule, 1):
        for it in range(1, iters + 1):
            acc, length = grpo(policy, ref, opt, cap, G, hard)
            if it % iters == 0:
                print(f"  [{tag}] stage {stage} (cap {cap}, G={G}) iteration {it:3d}: sample accuracy {acc:.2f}, length {length:.1f}")
    return policy


@torch.no_grad()
def evaluate(model, cap=40, n=256):
    q, img, gold = problems(n)
    o = sample(model, q, img, cap, greedy=True)
    fmt, acc, fin, length = judge(o, gold)
    return acc.mean().item(), length.mean().item(), fmt.mean().item()


# ───────────────────────── cold start by modality bridging (Section 3.3.1) ─────────────────────────
@torch.no_grad()
def describe(model, img):
    """Step 1-2: the MLLM turns the image into text (here: reads the expression, the caption task it was trained on)."""
    seq = torch.cat([torch.full((img.size(0), N_IMG), IMG), torch.full((img.size(0), 1), Q)], 1)
    out = sample(model, seq, img, N_IMG + 1, greedy=True)
    return [t[: t.index(END)] if END in t else t for t in out.tolist()]


def text_reasoner(expr):
    """Step 3: the text-only reasoner (DeepSeek-R1's role) solves the DESCRIBED problem, writing a chain with 0-3
    re-check steps (0-2); returns (chain tokens, its answer) or None if the description is not a well-formed expression."""
    if len(expr) != N_IMG or expr[-1] != EQ or any(expr[i] > MAX_VAL for i in range(0, N_IMG - 1, 2)) or any(expr[i] not in (PLUS, MINUS) for i in range(1, N_IMG - 1, 2)):
        return None
    nums, ops = expr[0:N_IMG - 1:2], [0 if t == PLUS else 1 for t in expr[1:N_IMG - 1:2]]
    res, r = [], nums[0]
    for n, o in zip(nums[1:], ops):
        r = r + n if o == 0 else r - n
        res.append(r)
    if any(x < 0 or x > MAX_VAL for x in res):
        return None
    return [THINK] + steps(nums, ops, res, checks=torch.randint(0, 4, ()).item()) + [ENDTHINK, ANS, res[-1], END], res[-1]


def build_cold_start(model, n=2048):
    """Steps 1-4: describe, reason in text, keep the chains whose answer matches the ground truth, re-attach the image."""
    q, img, gold = problems(n)
    kept_q, kept_img, kept_chain = [], [], []
    for i, expr in enumerate(describe(model, img)):
        out = text_reasoner(expr)
        if out is not None and out[1] == gold[i].item():
            kept_q.append(q[i]); kept_img.append(img[i]); kept_chain.append(out[0])
    L = max(map(len, kept_chain))
    seqs = torch.stack([torch.cat([kq, torch.tensor(c + [PAD] * (L - len(c)))]) for kq, c in zip(kept_q, kept_chain)])
    return seqs, torch.stack(kept_img), len(kept_chain) / n


def sft(model, seqs, imgs, epochs=16, lr=1e-3):
    model = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(seqs.size(0)).split(64):
            loss = lm_loss(model, seqs[idx], imgs[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    t0 = time.time()
    base = MLLM()
    print("pretraining the base MLLM (captions, direct answers, text-only chains) ...")
    pretrain(base, 1300)
    e = evaluate(base)
    print(f"base model on image questions: accuracy {e[0]:.2f}, response length {e[1]:.1f}, uses <think> {e[2]:.2f}")
    print("Vision-R1-Zero: GRPO on the base model, format + result reward (Section 3.1)")
    zero = rl(base, [(15, 40, 8)], hard=False, tag="Zero")
    ez = evaluate(zero)
    print(f"  -> accuracy {ez[0]:.2f}, length {ez[1]:.1f}")
    print("cold start by modality bridging (Figure 2)")
    seqs, imgs, keep = build_cold_start(base)
    print(f"  {seqs.size(0)} chains kept ({keep:.0%} of the described problems had a correct text-reasoner answer)")
    ci = sft(base, seqs, imgs)
    q, img, gold = problems(512)
    o = sample(ci, q, img, 40)
    fmt, acc, fin, length = judge(o, gold)
    short, long_ = length <= length.median(), length > length.median()
    print(f"Vision-R1-CI: sampled accuracy {acc.mean():.2f}; Figure 1A — accuracy of the shorter half of the chains "
          f"{acc[short].mean():.2f} vs the longer half {acc[long_].mean():.2f} (the correct answers are the short ones)")
    print("RL from Vision-R1-CI with the hard formatting-result reward:")
    fixed = rl(ci, [(30, 40, 4)], hard=True, tag="fixed cap 40, G=4")
    ptst = rl(ci, [(18, 26, 8), (12, 40, 4)], hard=True, tag="PTST")
    ef, ep = evaluate(fixed), evaluate(ptst)
    print(f"{'model':32s} {'accuracy':>9s} {'length':>7s}")
    for name, r in (("base", e), ("Vision-R1-Zero", ez), ("Vision-R1-CI", evaluate(ci)), ("CI + GRPO, fixed cap", ef), ("CI + GRPO with PTST (Vision-R1)", ep)):
        print(f"{name:32s} {r[0]:9.2f} {r[1]:7.1f}")
    print(f"({time.time() - t0:.1f} s)")
    assert ez[0] < 0.4, "RL alone on the base model should not take off (Section 3.1)"
    assert acc[short].mean() > acc[long_].mean() + 0.1, "the CI model's correct answers should be its shorter chains (Figure 1A)"
    assert ep[0] > 0.7 and ep[0] > ef[0] + 0.05, "PTST should beat the fixed large cap (Table 5)"


if __name__ == "__main__":
    main()
