"""RT-1: Robotics Transformer for Real-World Control at Scale (Brohan et al., 2022) — a toy PyTorch implementation.

What is implemented (section / figure numbers follow the paper):
  * instruction embedding (stand-in for the Universal Sentence Encoder)                (Fig. 3, Sec. 4.1)
  * FiLM-conditioned image encoder: feature-wise affine modulation gamma/beta from the
    instruction at every conv block, initialised to the identity ("early fusion")       (Sec. 4.1, Fig. 3)
  * TokenLearner: attention-based soft selection of S tokens out of the H*W visual tokens (Sec. 4.1)
  * decoder-only Transformer over the tokens of a history of frames, causal in time      (Sec. 4.1, Fig. 3)
  * action tokenisation: continuous dims uniformly discretised into bins (32 here, 256 in the paper) + a discrete
    mode (arm / terminate), trained with per-dimension cross-entropy (behaviour cloning) (Sec. 4.2)
  * closed-loop evaluation: success rate of the policy on seen and unseen instructions
Simplifications: a 16x16 rendered 2-D pick-and-place environment instead of real robot data; a 3-layer conv net (with
two pixel-coordinate input channels) instead of EfficientNet-B3; 4 visual tokens/frame out of 16 (paper: 8 out of 81);
history of 3 frames (paper: 6); 4 action dims (dx, dy, gripper, mode) instead of 11; a scripted teleoperator
generates the demonstrations.

Run:  python rt1.py        (CPU, about 55 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
BINS, T_HIST, GRID, MAX_STEP = 32, 3, 16, 0.18
COLORS = ["red", "blue", "green"]                       # instruction k = "pick the COLORS[k] block and place it on the target"


# ───────────────────────── toy 2-D pick-and-place environment ─────────────────────────
class PickPlaceEnv:
    """Gripper, three coloured blocks and a target on the unit square, rendered as a 5-channel 16x16 image (+2 coord channels).
    Action: (dx, dy) in [-1,1] (scaled by MAX_STEP), gripper in {-1: open, +1: close}, mode in {0: arm, 1: terminate}."""
    def __init__(self):
        yy, xx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
        self.grid = torch.stack([xx, yy], -1).float() / (GRID - 1)                    # (G, G, 2) pixel centres
        self.coords = self.grid.permute(2, 0, 1) * 2 - 1                                  # (2, G, G) coordinate channels

    def reset(self, instr):
        self.instr, self.t, self.holding, self.done, self.success = instr, 0, False, False, False
        pts = torch.rand(5, 2) * 0.8 + 0.1                                            # gripper, 3 blocks, target
        self.grip, self.blocks, self.target = pts[0], pts[1:4], pts[4]
        return self.render()

    def render(self):
        pts = torch.cat([self.grip[None], self.blocks, self.target[None]])            # (5, 2)
        d2 = ((self.grid[:, :, None] - pts) ** 2).sum(-1)                             # (G, G, 5)
        return torch.cat([torch.exp(-d2 / (2 * 0.07 ** 2)).permute(2, 0, 1), self.coords])   # (7, G, G) gaussian blobs

    def step(self, a):                                                                # a = (dx, dy, gripper, mode)
        self.t += 1
        self.grip = (self.grip + MAX_STEP * a[:2].clamp(-1, 1)).clamp(0, 1)
        k = self.instr
        if a[2] > 0 and not self.holding and (self.grip - self.blocks[k]).norm() < 0.1:
            self.holding = True
        if self.holding:
            self.blocks[k] = self.grip.clone()
            if a[2] < 0:
                self.holding = False
        if a[3] > 0.5 or self.t >= 25:                                                # terminate
            self.done, self.success = True, bool((self.blocks[k] - self.target).norm() < 0.1 and not self.holding)
        return self.render()

    def expert(self):
        """Scripted teleoperator: go to the block, close, carry to the target, open, terminate."""
        k = self.instr
        if not self.holding:
            goal, near = self.blocks[k], (self.grip - self.blocks[k]).norm() < 0.06
            return torch.tensor([*((goal - self.grip) / MAX_STEP).clamp(-1, 1), 1.0 if near else -1.0, 0.0])
        goal, near = self.target, (self.grip - self.target).norm() < 0.06
        return torch.tensor([*((goal - self.grip) / MAX_STEP).clamp(-1, 1), -1.0 if near else 1.0, 1.0 if near else 0.0])


def tokenize(a):
    """Sec. 4.2: each continuous dimension is uniformly discretised into 256 bins; the mode stays a small discrete set."""
    bins = ((a[..., :3].clamp(-1, 1) + 1) / 2 * (BINS - 1)).round().long()
    return torch.cat([bins, a[..., 3:].long()], -1)                                  # (..., 4) integer action tokens


def detokenize(tok):
    return torch.cat([tok[..., :3].float() / (BINS - 1) * 2 - 1, tok[..., 3:].float()], -1)   # bin -> bin centre


# ───────────────────────── model (Figure 3) ─────────────────────────
class FiLM(nn.Module):
    """FiLM(x) = (1 + gamma(l)) * x + beta(l) per channel, gamma/beta zero-initialised so the block starts as the identity."""
    def __init__(self, d_lang, c):
        super().__init__()
        self.to_gb = nn.Linear(d_lang, 2 * c)
        nn.init.zeros_(self.to_gb.weight); nn.init.zeros_(self.to_gb.bias)

    def forward(self, x, lang):
        gamma, beta = self.to_gb(lang).chunk(2, -1)
        return (1 + gamma)[:, :, None, None] * x + beta[:, :, None, None]


class FiLMEncoder(nn.Module):
    """Stand-in for the FiLM-conditioned EfficientNet: conv -> FiLM -> ReLU three times, output a 4x4xC map = 16 tokens."""
    def __init__(self, d_lang, c=32):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv2d(7, c, 4, stride=2, padding=1), nn.Conv2d(c, c, 4, stride=2, padding=1), nn.Conv2d(c, c, 3, padding=1)])
        self.films = nn.ModuleList([FiLM(d_lang, c) for _ in range(3)])

    def forward(self, img, lang):                                     # img (B, 7, 16, 16) -> (B, 16, c)
        x = img
        for conv, film in zip(self.convs, self.films):
            x = F.relu(film(conv(x), lang))
        return x.flatten(2).transpose(1, 2)


class TokenLearner(nn.Module):
    """TokenLearner (Sec. 4.1): S spatial attention maps alpha_i = softmax_p(MLP(x)_i); token_i = sum_p alpha_i(p) x_p."""
    def __init__(self, c, S):
        super().__init__()
        self.attn = nn.Sequential(nn.LayerNorm(c), nn.Linear(c, c), nn.GELU(), nn.Linear(c, S))

    def forward(self, x):                                             # x (B, P, c) -> (B, S, c)
        alpha = self.attn(x).softmax(dim=1)                           # (B, P, S): each of the S maps sums to 1 over positions
        return alpha.transpose(1, 2) @ x


class Block(nn.Module):
    """Pre-LN Transformer block: x + MHA(LN(x)), x + MLP(LN(x)), with a boolean attention mask (True = may attend)."""
    def __init__(self, d, h):
        super().__init__()
        self.h, self.ln1, self.ln2 = h, nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2) / math.sqrt(d // self.h)).masked_fill(~mask, float("-inf")).softmax(-1)
        x = x + self.out((att @ v).transpose(1, 2).reshape(B, L, d))
        return x + self.mlp(self.ln2(x))


class RT1(nn.Module):
    def __init__(self, n_instr, d_lang=32, c=32, S=4, d_model=64, n_layers=2, h=4):
        super().__init__()
        self.S = S
        self.lang = nn.Embedding(n_instr, d_lang)                     # stands in for the 512-d Universal Sentence Encoder
        self.encoder, self.token_learner = FiLMEncoder(d_lang, c), TokenLearner(c, S)
        self.proj = nn.Linear(c, d_model)
        self.pos = nn.Parameter(torch.randn(T_HIST * S, d_model) * 0.02)
        self.blocks = nn.ModuleList([Block(d_model, h) for _ in range(n_layers)])   # decoder-only: causal self-attention
        self.heads = nn.ModuleList([nn.Linear(d_model, BINS) for _ in range(3)] + [nn.Linear(d_model, 2)])

    def forward(self, imgs, instr):                                   # imgs (B, T, 5, G, G), instr (B,)
        B, T = imgs.shape[:2]
        lang = self.lang(instr)
        feats = self.encoder(imgs.flatten(0, 1), lang.repeat_interleave(T, 0))     # (B*T, 16, c), FiLM'ed by the instruction
        tok = self.token_learner(feats).view(B, T * self.S, -1)                     # 16 -> S tokens per frame
        x = self.proj(tok) + self.pos[: T * self.S]
        frame = torch.arange(T * self.S) // self.S
        mask = frame[:, None] >= frame[None, :]                                    # causal over frames: token may see frames <= its own
        for blk in self.blocks:
            x = blk(x, mask)
        out = x[:, -self.S :].mean(1)                                               # outputs at the last frame's tokens -> action
        return [head(out) for head in self.heads]                                  # 4 lists of logits (3 x 256 bins, 1 x 2 modes)

    @torch.no_grad()
    def act(self, imgs, instr):
        return detokenize(torch.stack([lg.argmax(-1) for lg in self(imgs, instr)], -1))


# ───────────────────────── data ─────────────────────────
def history(frames, t):
    """Frames t-2..t, repeating the first frame when the episode is shorter than the history window."""
    return torch.stack([frames[max(i, 0)] for i in range(t - T_HIST + 1, t + 1)])


def collect(env, n_episodes, instrs):
    X, I, A = [], [], []
    for ep in range(n_episodes):
        k = instrs[ep % len(instrs)]
        frames = [env.reset(k)]
        while not env.done:
            a = env.expert()
            X.append(history(frames, len(frames) - 1)); I.append(k); A.append(tokenize(a))
            frames.append(env.step(a))
    return torch.stack(X), torch.tensor(I), torch.stack(A)


@torch.no_grad()
def evaluate(model, env, instrs, n=30):
    ok = 0
    for ep in range(n):
        frames = [env.reset(instrs[ep % len(instrs)])]
        while not env.done:
            a = model.act(history(frames, len(frames) - 1)[None], torch.tensor([env.instr]))[0]
            frames.append(env.step(a))
        ok += env.success
    return ok / n


def main():
    env, t0 = PickPlaceEnv(), time.time()
    X, I, A = collect(env, 400, instrs=[0, 1, 2])
    print(f"demonstrations: {X.shape[0]} (image history, instruction, action) samples from 400 scripted episodes")
    model = RT1(n_instr=len(COLORS))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1000)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    first = None
    for step in range(1, 1001):
        idx = torch.randint(0, X.shape[0], (96,))
        logits = model(X[idx], I[idx])
        loss = sum(F.cross_entropy(lg, A[idx, d]) for d, lg in enumerate(logits))   # Sec. 4.2: cross-entropy per action dim
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        first = first or loss.item()
        if step % 250 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}")
    model.eval()
    succ = evaluate(model, env, [0, 1, 2])
    print(f"closed-loop success on 30 new episodes: {succ:.2f}   ({time.time() - t0:.1f} s)")
    gb = model.encoder.films[0].to_gb.weight.abs().mean().item()
    print(f"FiLM gamma/beta weights moved away from identity init: mean |w| = {gb:.4f}")
    assert loss.item() < 0.3 * first and succ > 0.6, "behaviour cloning did not converge as expected"


if __name__ == "__main__":
    main()
