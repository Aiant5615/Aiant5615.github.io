"""Octo: An Open-Source Generalist Robot Policy (Octo Model Team, 2024) — a toy PyTorch implementation.

What is implemented (section / figure numbers follow the paper):
  * tokenizers: a language-instruction token, goal-image patch tokens, observation-image patch tokens
    (shallow conv stem) for a window of timesteps, plus learned readout tokens per timestep            (Sec. 3.1, Fig. 2)
  * block-wise causal attention mask: observation tokens at time t attend to task tokens and
    observations at <= t; readout tokens attend to task, observations <= t and earlier readouts;
    nothing attends to a readout token, so heads can be attached after pretraining                   (Sec. 3.1)
  * diffusion action head over an action chunk: DDPM epsilon-objective
        L = E ||eps - eps_theta(sqrt(abar_k) a + sqrt(1-abar_k) eps, e, k)||^2
    and 20-step sampling x^{k-1} = alpha (x^k - gamma eps_theta) + N(0, sigma^2 I), cosine schedule     (Sec. 3.1)
  * ablation of Table 2: the diffusion head vs an MSE (mean-regression) head on a bimodal task
Simplifications: an 8x8 rendered "go around the wall" task with two equally good ways round instead of OXE data;
tiny transformer (2 layers, d=64); readout attention rules only for one action head; no proprio/wrist tokens.

Run:  python octo.py        (CPU, about 40 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
H, T_OBS, K_DIFF, GRID, STEP = 4, 2, 20, 8, 0.2               # chunk length, obs window, diffusion steps, image size, max step


# ───────────────────────── toy task: go around a wall (bimodal expert) ─────────────────────────
def render(agent, goal):
    """3-channel 8x8 image: agent blob, goal blob, wall (x = 0.5, y in [0.3, 0.7])."""
    yy, xx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
    grid = torch.stack([xx, yy], -1).float() / (GRID - 1)                              # (G, G, 2)
    blob = lambda p: torch.exp(-((grid - p) ** 2).sum(-1) / (2 * 0.08 ** 2))
    wall = ((grid[..., 0] - 0.5).abs() < 0.08) & ((grid[..., 1] - 0.5).abs() <= 0.2)
    return torch.stack([blob(agent), blob(goal), wall.float()])                         # (3, G, G)


def make_batch(B):
    """Agent starts left of the wall, goal is right of it. The expert goes over (m=+1) or under (m=-1) the wall,
    chosen at random: a two-mode action distribution. Returns obs images (B, T_OBS, 3, G, G), goal images, chunks (B, H, 2)."""
    agent = torch.rand(B, 2) * torch.tensor([0.1, 0.1]) + torch.tensor([0.15, 0.45])
    goal = torch.rand(B, 2) * torch.tensor([0.1, 0.1]) + torch.tensor([0.75, 0.45])
    m = torch.randint(0, 2, (B, 1)).float() * 2 - 1
    dx = ((goal[:, :1] - agent[:, :1]) / H) / STEP                                       # x progress per step, in action units
    dy = m * 0.75
    chunk = torch.stack([torch.cat([dx, dy], 1), torch.cat([dx, dy], 1),
                         torch.cat([dx, 0 * dy], 1), torch.cat([dx, 0 * dy], 1)], 1)   # (B, H, 2), first two steps sidestep
    chunk = (chunk + 0.03 * torch.randn_like(chunk)).clamp(-1, 1)
    prev = agent - torch.tensor([0.05, 0.0])                                             # previous frame: one small step behind
    obs = torch.stack([torch.stack([render(prev[i], goal[i]), render(agent[i], goal[i])]) for i in range(B)])
    goal_img = torch.stack([render(goal[i], goal[i]) for i in range(B)])
    return obs, goal_img, chunk, agent, goal


def collides(chunk, agent):
    """Roll the chunk out; a collision is crossing x = 0.5 while |y - 0.5| < 0.2 (the wall)."""
    pos = agent.clone(); hit = torch.zeros(agent.shape[0], dtype=torch.bool)
    for t in range(H):
        nxt = pos + STEP * chunk[:, t]
        hit |= (pos[:, 0] < 0.5) & (nxt[:, 0] >= 0.5) & ((nxt[:, 1] - 0.5).abs() < 0.2)
        pos = nxt
    return hit


# ───────────────────────── transformer with block-wise mask (Sec. 3.1) ─────────────────────────
class Block(nn.Module):
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


def block_mask(n_task, n_obs_tok, T):
    """Token groups: task (language + goal image), then per timestep t: n_obs_tok observation tokens and 1 readout.
    Returns a bool (L, L) mask, True = query row may attend to key column."""
    kind, step = [0] * n_task, [-1] * n_task                                             # 0 task, 1 obs, 2 readout
    for t in range(T):
        kind += [1] * n_obs_tok + [2]; step += [t] * (n_obs_tok + 1)
    kind, step = torch.tensor(kind), torch.tensor(step)
    is_task, is_obs, is_ro = kind == 0, kind == 1, kind == 2
    past = step[:, None] >= step[None, :]                                                # key timestep <= query timestep
    m = torch.zeros(len(kind), len(kind), dtype=torch.bool)
    m[is_task] = is_task                                                                 # task tokens see only task tokens
    m[is_obs] = (is_task | (is_obs & past[is_obs]))                                      # obs: task + obs at <= t
    m[is_ro] = (is_task | ((is_obs | is_ro) & past[is_ro]))                              # readout: task + obs/readouts at <= t
    return m, is_ro


class OctoTransformer(nn.Module):
    def __init__(self, n_lang=1, d=64, h=4, n_layers=2, patch=4):
        super().__init__()
        self.patch, self.n_patch = patch, (GRID // patch) ** 2
        self.lang = nn.Embedding(n_lang, d)                                              # stands in for the T5 encoder
        self.stem = nn.Conv2d(3, d, patch, stride=patch)                                 # shallow conv patch tokenizer
        self.readout = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        n_task = 1 + self.n_patch
        L = n_task + T_OBS * (self.n_patch + 1)
        self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        mask, is_ro = block_mask(n_task, self.n_patch, T_OBS)
        self.register_buffer("mask", mask); self.register_buffer("is_ro", is_ro)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)

    def tokens(self, img):                                                               # (N, 3, G, G) -> (N, n_patch, d)
        return self.stem(img).flatten(2).transpose(1, 2)

    def forward(self, obs, goal_img, lang):
        B = obs.shape[0]
        task = torch.cat([self.lang(lang)[:, None], self.tokens(goal_img)], 1)
        per_t = self.tokens(obs.flatten(0, 1)).view(B, T_OBS, self.n_patch, -1)
        seq = [task]
        for t in range(T_OBS):
            seq += [per_t[:, t], self.readout.expand(B, 1, -1)]
        x = torch.cat(seq, 1) + self.pos
        for blk in self.blocks:
            x = blk(x, self.mask)
        return self.ln(x[:, self.is_ro][:, -1])                                          # embedding e of the last readout token


# ───────────────────────── diffusion action head (Sec. 3.1) ─────────────────────────
def cosine_schedule(K, s=0.008):
    f = lambda t: torch.cos((t / K + s) / (1 + s) * math.pi / 2) ** 2
    t = torch.arange(K + 1).float()
    abar = f(t) / f(torch.zeros(1))                                                      # abar_0 = 1, abar_K ~ 0
    beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return abar[1:], beta                                                                # indexed by k-1 for k = 1..K


class DiffusionHead(nn.Module):
    """eps_theta(x^k, e, k): an MLP on the noisy chunk, the readout embedding and a sinusoidal step embedding."""
    def __init__(self, d, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(H * 2 + d + 16, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, H * 2))
        abar, beta = cosine_schedule(K_DIFF)
        self.register_buffer("abar", abar); self.register_buffer("beta", beta)

    def forward(self, x, e, k):                                                          # x (B, H, 2), k (B,) in 1..K
        freqs = torch.exp(-math.log(1e4) * torch.arange(8) / 8)
        k_emb = torch.cat([torch.sin(k[:, None] * freqs), torch.cos(k[:, None] * freqs)], 1)
        return self.net(torch.cat([x.flatten(1), e, k_emb], 1)).view(-1, H, 2)

    def loss(self, a, e):
        """L = E_{a, k, eps} || eps - eps_theta(sqrt(abar_k) a + sqrt(1 - abar_k) eps, e, k) ||^2."""
        k = torch.randint(1, K_DIFF + 1, (a.shape[0],))
        ab = self.abar[k - 1][:, None, None]
        eps = torch.randn_like(a)
        return F.mse_loss(self(ab.sqrt() * a + (1 - ab).sqrt() * eps, e, k), eps)

    @torch.no_grad()
    def sample(self, e):
        """x^{k-1} = alpha (x^k - gamma eps_theta(x^k, e, k)) + N(0, sigma^2 I), k = K..1, with the DDPM coefficients
        alpha = 1/sqrt(1 - beta_k), gamma = beta_k / sqrt(1 - abar_k), sigma^2 = beta_k; clipped to [-1, 1]."""
        x = torch.randn(e.shape[0], H, 2)
        for k in range(K_DIFF, 0, -1):
            beta, ab = self.beta[k - 1], self.abar[k - 1]
            eps = self(x, e, torch.full((e.shape[0],), k))
            x = (x - beta / (1 - ab).sqrt() * eps) / (1 - beta).sqrt()
            if k > 1:
                x = x + beta.sqrt() * torch.randn_like(x)
            x = x.clamp(-1, 1)
        return x


DATA = make_batch(4096)                                                                  # a fixed demonstration dataset


def train(backbone, head, steps, diffusion, tag, lr=2e-3):
    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=lr)
    first = None
    for step in range(1, steps + 1):
        idx = torch.randint(0, DATA[0].shape[0], (128,))
        obs, goal_img, chunk = DATA[0][idx], DATA[1][idx], DATA[2][idx]
        e = backbone(obs, goal_img, torch.zeros(128, dtype=torch.long))
        loss = head.loss(chunk, e) if diffusion else F.mse_loss(head(e).view(-1, H, 2), chunk)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or loss.item()
        if step % 300 == 0:
            print(f"  {tag} step {step:4d}  loss {loss.item():.4f}")
    return first, loss.item()


def main():
    t0 = time.time()
    backbone, head = OctoTransformer(), DiffusionHead(64)
    m = backbone.mask; n_task = 1 + backbone.n_patch; ro = backbone.is_ro
    assert not m[:, ro][~ro].any(), "task/observation tokens must not attend to readouts"
    assert m[ro][:, :n_task].all() and m[~ro][:, ro].sum() == 0 and m[n_task:][:, ro].sum() == T_OBS * (T_OBS + 1) // 2
    print(f"block-wise mask OK: {m.shape[0]} tokens = {n_task} task + {T_OBS} x ({backbone.n_patch} obs + 1 readout); "
          f"readouts attended by nobody, observations block-causal in time")
    print(f"parameters: backbone {sum(p.numel() for p in backbone.parameters()):,}, diffusion head {sum(p.numel() for p in head.parameters()):,}")
    print("training the diffusion head (DDPM epsilon objective)")
    first, last = train(backbone, head, 1200, True, "diffusion")
    backbone_mse, head_mse = OctoTransformer(), nn.Linear(64, H * 2)
    print("training an MSE head on the same architecture (Table 2 ablation)")
    train(backbone_mse, head_mse, 150, False, "mse")
    obs, goal_img, chunk, agent, goal = make_batch(200)
    lang = torch.zeros(200, dtype=torch.long)
    with torch.no_grad():
        a_diff = head.sample(backbone(obs, goal_img, lang))
        a_mse = head_mse(backbone_mse(obs, goal_img, lang)).view(-1, H, 2)
    up = (a_diff[:, 0, 1] > 0.3).float().mean().item(); down = (a_diff[:, 0, 1] < -0.3).float().mean().item()
    c_diff, c_mse = collides(a_diff, agent).float().mean().item(), collides(a_mse, agent).float().mean().item()
    print(f"diffusion samples: first-step dy > 0.3 in {up:.2f}, < -0.3 in {down:.2f} of 200 samples (expert: 0.5 / 0.5)")
    print(f"MSE head: mean first-step dy = {a_mse[:, 0, 1].mean().item():+.3f} (averages the two modes)")
    print(f"wall collisions: diffusion {c_diff:.2f}   MSE {c_mse:.2f}      ({time.time() - t0:.1f} s)")
    assert last < 0.5 * first and up + down > 0.85 and min(up, down) > 0.25 and c_diff < 0.25 and c_mse > 0.8


if __name__ == "__main__":
    main()
