"""DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning (Zheng et al., 2025) — toy implementation.

What is implemented (section / figure numbers follow the paper):
  * interleaved multimodal chain of thought (Section 3.1): mid-generation the policy may emit a zoom-in action
    "ZOOM r c" (a box on the global image); the environment returns the crop, encoded by the SAME vision encoder,
    appended as observation tokens; generation continues to the answer
  * the agentic RL formulation (Section 3.2): the state is the interleaved text/image history; the trajectory
    reward R = R_acc + R_format + 1[R_acc > 0] * R_tool (tool bonus only if the answer is correct and a zoom happened);
    GRPO on whole multi-turn trajectories with a token-wise mask so observation tokens contribute no loss, KL = 0
  * the tool-reward ablation of Figure 4 / Table 5: conditional vs unconditional vs no tool reward
  * data curation (Section 3.3): the perception-utility filter keeps problems that are unsolvable from the global view
    but solvable from the ground-truth crop; the difficulty filter drops problems the base policy already solves
  * training dynamics (Figure 3): zoom count and accuracy over RL iterations
Simplifications: 8x8-cell images with one 4x4-pixel symbol whose identity is invisible at the 2x-downsampled global
view but readable in a full-resolution crop; a small decoder-only LM with a linear patch encoder as the VLM,
pretrained (as Qwen2.5-VL is) to caption crops and to ground the object's location; one zoom per trajectory.

Run:  python deepeyes.py        (CPU, about 30 s)
"""
import copy, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ───────────────────────── the world: a tiny symbol in a large image ─────────────────────────
# four symbols on a 4x4-pixel cell; every quadrant holds exactly 2 lit pixels, so after 2x average pooling all four
# look identical (a uniform blob) — the identity needs the full resolution
SYMBOLS = torch.tensor([
    [[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]],      # "="
    [[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]],      # "||"
    [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]],      # checker
    [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]],      # bricks
], dtype=torch.float)
CELLS, PX = 8, 4                                                   # 8x8 cells of 4x4 pixels = 32x32 image
GRID = 4                                                           # the global view has 4x4 patches, each = 2x2 cells
SYM = ["=", "||", "checker", "bricks"]
WORDS = SYM + [str(i) for i in range(GRID)] + ["ZOOM", "<a>", "<ans>", "<end>", "<img>", "<obs>", "<pad>", "<q-where>", "<q-what>"]
TOK = {w: i for i, w in enumerate(WORDS)}
ZOOM, AS, ANS, END, IMG, OBS, PAD, QWHERE, QWHAT = (TOK[w] for w in ["ZOOM", "<a>", "<ans>", "<end>", "<img>", "<obs>", "<pad>", "<q-where>", "<q-what>"])
NUM0 = len(SYM)                                                    # token of digit i is NUM0 + i
VOCAB = len(WORDS)


def make_image():
    """One symbol at a random cell of an otherwise dark image (plus noise). Returns (image, symbol, patch row, col)."""
    img = 0.1 * torch.rand(CELLS * PX, CELLS * PX)
    s, r, c = torch.randint(4, ()).item(), torch.randint(CELLS, ()).item(), torch.randint(CELLS, ()).item()
    img[r * PX:(r + 1) * PX, c * PX:(c + 1) * PX] += SYMBOLS[s]
    return img, s, r // 2, c // 2


def global_view(img):
    """2x downsampled image cut into 4x4 patches of 4x4 pixels -> (16, 16) patch vectors: the identity is gone."""
    small = F.avg_pool2d(img[None, None], 2)[0, 0]                 # 16x16
    return small.unfold(0, PX, PX).unfold(1, PX, PX).reshape(GRID * GRID, PX * PX)


def crop(img, r, c):
    """Full-resolution crop of global patch (r, c) = 2x2 cells -> 4 patch vectors of 4x4 pixels."""
    region = img[r * 2 * PX:(r + 1) * 2 * PX, c * 2 * PX:(c + 1) * 2 * PX]
    return region.unfold(0, PX, PX).unfold(1, PX, PX).reshape(4, PX * PX)


def perception_utility(img, s, r, c):
    """Data filter (Section 3.3): keep a problem only if the symbol is NOT readable from the global view (its pooled
    patch equals what any symbol would give) and IS readable from the ground-truth crop."""
    pooled = F.avg_pool2d(SYMBOLS[s][None, None], 2)[0, 0]
    ambiguous = all(torch.allclose(pooled, F.avg_pool2d(SYMBOLS[k][None, None], 2)[0, 0]) for k in range(4))
    return ambiguous and (crop(img, r, c).max() > 0.9)


# ───────────────────────── the VLM ─────────────────────────
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


class VLM(nn.Module):
    """Decoder-only LM; <img> and <obs> placeholders are replaced by patch vectors through one shared linear encoder
    (the global view and the crops go through the same encoder, as in the paper). No position embedding, so the
    caption skill learned on crops at the start of a sequence transfers to crops observed mid-trajectory."""
    def __init__(self, d=64, h=4, n_layers=2):
        super().__init__()
        self.tok, self.vis = nn.Embedding(VOCAB, d), nn.Linear(PX * PX, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB, bias=False)

    def forward(self, seq, patches):
        """patches: (B, n, 16) — one row per placeholder token of the sequence, in order."""
        x = self.tok(seq)
        ph = (seq == IMG) | (seq == OBS)
        x = x.clone(); x[ph] = self.vis(patches).reshape(-1, x.shape[-1])[: int(ph.sum())]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


def pretrain_batch(B):
    """What the base VLM knows before RL (Qwen2.5-VL's native abilities): (a) caption a crop: 4 patches -> the symbol;
    (b) grounding: global view -> "ZOOM r c" of the bright patch; (c) direct answer from the global view (a guess).
    (b) and (c) share the prompt, so the base policy sometimes grounds and sometimes guesses — but never both."""
    seqs, pats = [], []
    for _ in range(B):
        img, s, r, c = make_image()
        kind = torch.randint(3, ()).item()
        if kind == 0:
            seqs.append([OBS] * 4 + [QWHAT, ANS, s, END]); pats.append(crop(img, r, c))
        elif kind == 1:
            seqs.append([IMG] * 16 + [AS, ZOOM, NUM0 + r, NUM0 + c, END]); pats.append(global_view(img))
        else:
            seqs.append([IMG] * 16 + [AS, ANS, s, END]); pats.append(global_view(img))
    L = max(map(len, seqs))
    seq = torch.tensor([x + [PAD] * (L - len(x)) for x in seqs])
    return seq, torch.cat(pats)


def pretrain(model, steps, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        seq, pats = pretrain_batch(32)
        logits = model(seq[:, :-1], pats)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), seq[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); opt.step()


# ───────────────────────── the environment: rollouts with a zoom tool (Section 3.1-3.2) ─────────────────────────
@torch.no_grad()
def rollout(policy, imgs, greedy=False, max_tokens=12):
    """Prompt: [<img> x 16] <a>. The policy generates tokens; when it has emitted ZOOM r c, the crop's 4 patches are
    appended as <obs> tokens plus the caption scaffold <q-what> (observation and scaffolding, not actions) and it
    continues; the episode ends at <end> or after max_tokens generated tokens. Returns the token sequences (prompt +
    trajectory), the patch rows, a loss mask that is 1 only on tokens the policy produced, and who zoomed."""
    B = len(imgs)
    seqs = [[IMG] * 16 + [AS] for _ in range(B)]
    pats = [global_view(img) for img in imgs]
    masks = [[0] * 17 for _ in range(B)]
    zoomed, done = [False] * B, [False] * B
    for _ in range(max_tokens):
        L = max(map(len, seqs))
        seq = torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])
        logits = policy(seq, torch.cat(pats))
        for i in range(B):
            if done[i]:
                continue
            lg = logits[i, len(seqs[i]) - 1].clone(); lg[[IMG, OBS, PAD, QWHAT, QWHERE]] = float("-inf")
            nxt = lg.argmax().item() if greedy else torch.distributions.Categorical(logits=lg).sample().item()
            seqs[i].append(nxt); masks[i].append(1)
            if nxt == END:
                done[i] = True
            elif seqs[i][-3] == ZOOM and not zoomed[i]:                          # "ZOOM r c" complete -> observe
                r, c = seqs[i][-2] - NUM0, seqs[i][-1] - NUM0
                if 0 <= r < GRID and 0 <= c < GRID:
                    pats[i] = torch.cat([pats[i], crop(imgs[i], r, c)])
                    seqs[i] += [OBS] * 4 + [QWHAT]; masks[i] += [0] * 5
                zoomed[i] = True
        if all(done):
            break
    L = max(map(len, seqs))
    seq = torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])
    mask = torch.tensor([m + [0] * (L - len(m)) for m in masks]).float()
    return seq, torch.cat(pats), mask, torch.tensor(zoomed)


def judge(seq, golds):
    """Per trajectory: correct answer, well-formed (ANS x END at the end)."""
    acc, fmt = torch.zeros(len(golds)), torch.zeros(len(golds))
    for i, toks in enumerate(seq.tolist()):
        toks = [t for t in toks[17:] if t != PAD]
        if ANS in toks:
            j = toks.index(ANS)
            fmt[i] = float(len(toks) > j + 2 and toks[j + 1] < NUM0 and toks[j + 2] == END)
            acc[i] = float(len(toks) > j + 1 and toks[j + 1] == golds[i])
    return acc, fmt


def reward(acc, fmt, zoomed, tool_reward):
    """R = R_acc + R_format + 1[R_acc > 0] R_tool (conditional, the paper); 'unconditional': R_tool whenever a zoom
    happened; 'none': no tool term."""
    bonus = {"conditional": (acc > 0).float() * zoomed.float(), "unconditional": zoomed.float(), "none": torch.zeros_like(acc)}[tool_reward]
    return acc + 0.5 * fmt + 0.5 * bonus


def token_log_probs(model, seq, pats):
    logits = model(seq[:, :-1], pats)
    return logits.log_softmax(-1).gather(-1, seq[:, 1:, None]).squeeze(-1)


def grpo_step(policy, opt, problems, tool_reward, B=8, G=8, eps=0.2):
    """GRPO over whole trajectories: group-normalised advantages, clipped ratio (one update per batch), observation
    tokens masked out of the loss, no KL term (the paper sets the KL coefficient to 0)."""
    idx = torch.randint(len(problems), (B,))
    imgs = [problems[i][0] for i in idx for _ in range(G)]
    golds = [problems[i][1] for i in idx for _ in range(G)]
    seq, pats, mask, zoomed = rollout(policy, imgs)
    acc, fmt = judge(seq, golds)
    r = reward(acc, fmt, zoomed, tool_reward).view(B, G)
    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
    logp = token_log_probs(policy, seq, pats)
    ratio = (logp - logp.detach()).exp()
    m = mask[:, 1:]
    surr = torch.min(ratio * adv[:, None], ratio.clamp(1 - eps, 1 + eps) * adv[:, None])
    loss = -((surr * m).sum(1) / m.sum(1).clamp(min=1)).mean()
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
    return acc.mean().item(), zoomed.float().mean().item()


def train_rl(base, problems, tool_reward, iters=80, lr=3e-4):
    policy = copy.deepcopy(base)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    curve = []
    for it in range(1, iters + 1):
        acc, zoom = grpo_step(policy, opt, problems, tool_reward)
        curve.append((acc, zoom))
        if it % 20 == 0:
            print(f"  [{tool_reward:13s}] iteration {it:3d}: sample accuracy {acc:.2f}, zoom rate {zoom:.2f}")
    return policy, curve


@torch.no_grad()
def evaluate(policy, n=200):
    probs = [make_image() for _ in range(n)]
    seq, pats, mask, zoomed = rollout(policy, [p[0] for p in probs], greedy=True)
    acc, _ = judge(seq, [p[1] for p in probs])
    return acc.mean().item(), zoomed.float().mean().item()


def curate(base, n=600):
    """Section 3.3: keep problems with perception utility, then drop the ones the base policy already solves without
    zooming (difficulty filter, greedy)."""
    probs = [p for p in (make_image() for _ in range(n)) if perception_utility(*p)]
    seq, pats, mask, zoomed = rollout(base, [p[0] for p in probs], greedy=True)
    acc, _ = judge(seq, [p[1] for p in probs])
    kept = [(p[0], p[1]) for p, a, z in zip(probs, acc, zoomed) if not (a > 0 and not z)]
    return kept, len(probs) / n, len(kept) / len(probs)


def main():
    t0 = time.time()
    base = VLM()
    print("pretraining the base VLM: caption crops, ground the bright patch, guess from the global view")
    pretrain(base, 700)
    e = evaluate(base)
    print(f"base policy: accuracy {e[0]:.2f} (chance 0.25), zoom rate {e[1]:.2f}")
    problems, pu, diff = curate(base)
    print(f"data curation: {pu:.0%} pass the perception-utility filter, {diff:.0%} of those survive the difficulty filter -> {len(problems)} problems")
    results, policies = {}, {}
    for tool_reward in ("conditional", "unconditional", "none"):
        torch.manual_seed(1)
        policies[tool_reward], curve = train_rl(base, problems, tool_reward)
        results[tool_reward] = (evaluate(policies[tool_reward]), curve)
    print(f"\n{'tool reward':14s} {'final accuracy':>15s} {'zoom rate':>10s} {'sample acc, iters 1-30':>24s} {'iters 31-80':>12s}   (Figure 3 / Table 5)")
    early = {}
    for k, ((acc, zoom), curve) in results.items():
        early[k] = sum(a for a, _ in curve[:30]) / 30
        late = sum(a for a, _ in curve[30:]) / len(curve[30:])
        print(f"{k:14s} {acc:15.2f} {zoom:10.2f} {early[k]:24.2f} {late:12.2f}")
    seq, pats, mask, zoomed = rollout(policies["conditional"], [make_image()[0]], greedy=True)
    print("example trajectory:", " ".join(WORDS[t] for t in seq[0].tolist() if t not in (IMG, PAD)), f"  ({time.time() - t0:.1f} s)")
    cond, uncond, none = (results[k][0] for k in ("conditional", "unconditional", "none"))
    assert cond[0] > 0.85 and cond[1] > 0.9, "with the conditional tool reward the policy should learn to zoom and answer"
    assert early["conditional"] > early["none"] + 0.05, "without a tool reward the policy is slower to adopt the tool (early phase)"
    assert uncond[1] > 0.9 and uncond[0] <= cond[0], "an unconditional tool reward makes the policy zoom on everything without helping accuracy"


if __name__ == "__main__":
    main()
