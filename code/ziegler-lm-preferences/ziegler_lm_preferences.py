"""Fine-Tuning Language Models from Human Preferences (Ziegler et al., 2019) — a from-scratch PyTorch implementation.

What is implemented (section numbers follow the paper):
  * a language model ρ(x_{n+1} | x_1..x_n), a context x, a continuation y ∈ Σ^n, the initial policy π = ρ  (Section 2)
  * the 4-way comparison reward model: loss(r) = E[ log e^{r(x,y_b)} / Σ_i e^{r(x,y_i)} ], normalised after
    training so that r(x,y) has mean 0 and variance 1 for y ~ ρ                                          (Section 2.2)
  * the penalised reward R(x,y) = r(x,y) − β log π(y|x)/ρ(y|x) optimised with PPO                          (Section 2.2)
  * adaptive β: e_t = clip((KL(π_t, ρ) − KL_target)/KL_target, −0.2, 0.2), β_{t+1} = β_t (1 + K_β e_t)     (Section 2.2)
  * offline data collection (all labels from ρ, train r once) and online collection (relabel samples of the
    current policy and retrain r during PPO)                                                              (Section 2.3)
  * a mock reward (a fixed scorer standing in for the sentiment classifier of Section 4.1) as the synthetic human
Simplifications: ρ is a tiny tabular autoregressive LM over 10 integer tokens instead of GPT-2, the reward model is a
small MLP on token counts instead of a copy of the LM, the whole continuation is one PPO action with a per-context
value baseline, n = 8 tokens, no human labels.

Run:  python ziegler_lm_preferences.py        (CPU, about 10 s)
"""
import copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
C, n, V = 4, 8, 10                                      # number of contexts, continuation length, vocabulary |Σ|


# ───────────────────────── toy language model ρ / policy π ─────────────────────────
class TabularLM(nn.Module):
    """Autoregressive LM over integer tokens: the logits of y_t depend on the context x, the position t and the previous
    token y_{t-1} (a per-position bigram table). Stands in for GPT-2 as ρ and, once copied, as the policy π."""
    def __init__(self, scale=0.7):
        super().__init__()
        self.table = nn.Parameter(scale * torch.randn(C, n, V + 1, V))          # previous-token index V = "start"

    def log_prob(self, x, y):                                                   # log π(y|x) = Σ_t log π(y_t | x, y_<t)
        lp, prev = 0.0, torch.full_like(x, V)
        for t in range(n):
            lp = lp + self.table[x, t, prev].log_softmax(-1).gather(1, y[:, t:t + 1]).squeeze(1)
            prev = y[:, t]
        return lp

    @torch.no_grad()
    def sample(self, x):
        ys, prev = [], torch.full_like(x, V)
        for t in range(n):
            prev = torch.distributions.Categorical(logits=self.table[x, t, prev]).sample()
            ys.append(prev)
        return torch.stack(ys, 1)


# ───────────────────────── the "human": a mock reward, as in Section 4.1's mock-sentiment experiments ─────────────────────────
W_TRUE = torch.randn(C, V)                              # per-context token scores, hidden from the learner


def mock_reward(x, y):
    """r_true(x, y): mean score of the continuation's tokens under the context's hidden scoring table."""
    return W_TRUE[x].gather(1, y).mean(1)


def human_choice(x, ys):                                # ys: (B, 4, n) -> index b of the best of four
    return torch.stack([mock_reward(x, ys[:, i]) for i in range(4)], 1).argmax(1)


# ───────────────────────── reward model r(x, y) ─────────────────────────
class RewardModel(nn.Module):
    """r(x, y) from a small MLP on [one-hot(x), token counts of y / n]; gain and bias implement the normalisation
    "mean 0, variance 1 for y ~ ρ" so that the scale of r (and hence of β) is not arbitrary."""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(C + V, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.register_buffer("gain", torch.ones(()))
        self.register_buffer("bias", torch.zeros(()))

    def forward(self, x, y):
        feats = torch.cat([F.one_hot(x, C).float(), F.one_hot(y, V).float().mean(1)], -1)
        return self.gain * self.net(feats).squeeze(-1) + self.bias


def rm_loss(rm, x, ys, b):
    """-E[ log( e^{r(x,y_b)} / Σ_i e^{r(x,y_i)} ) ]: softmax over the four samples, the chosen one must be most likely."""
    scores = torch.stack([rm(x, ys[:, i]) for i in range(4)], 1)                 # (B, 4)
    return F.cross_entropy(scores, b)


def collect_labels(policy, n_queries):
    """A batch of contexts, four continuations each from the given policy, and the human's pick b."""
    x = torch.randint(0, C, (n_queries,))
    ys = torch.stack([policy.sample(x) for _ in range(4)], 1)
    return x, ys, human_choice(x, ys)


def train_rm(rm, data, rho, steps=300):
    x, ys, b = data
    opt = torch.optim.Adam(rm.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(steps):
        loss = rm_loss(rm, x, ys, b)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():                                                       # normalise r on samples of ρ
        rm.gain.fill_(1.0); rm.bias.zero_()
        xs = torch.randint(0, C, (2048,))
        r = rm(xs, rho.sample(xs))
        rm.gain.copy_(1 / r.std()); rm.bias.copy_(-r.mean() / r.std())
    return loss.item()


# ───────────────────────── PPO on the KL-penalised reward ─────────────────────────
def ppo_step(policy, value, rm, rho, opt, beta, B=256, K=4, M=64, eps=0.2):
    """Sample y ~ π_old(.|x), score with R(x,y) = r(x,y) − β log π(y|x)/ρ(y|x), then K epochs of clipped-PPO minibatches."""
    x = torch.randint(0, C, (B,))
    y = policy.sample(x)
    with torch.no_grad():
        logp_old, log_rho = policy.log_prob(x, y), rho.log_prob(x, y)
        kl = (logp_old - log_rho).mean()                                        # KL(π || ρ) estimated on the policy's own samples
        R = rm(x, y) - beta * (logp_old - log_rho)                              # penalised reward
    for _ in range(K):
        for idx in torch.randperm(B).split(M):
            ratio = (policy.log_prob(x[idx], y[idx]) - logp_old[idx]).exp()
            adv = R[idx] - value[x[idx]].detach()
            l_clip = torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
            l_vf = (value[x[idx]] - R[idx]).pow(2).mean()
            loss = -(l_clip - 0.5 * l_vf)
            opt.zero_grad(); loss.backward(); opt.step()
    return kl.item(), rm(x, y).mean().item(), mock_reward(x, y).mean().item()


def run(rho, mode, steps=150, kl_target=4.0, k_beta=0.1, n_labels=1024, lr=1e-2, beta0=0.3):
    """mode='offline': labels from ρ once. mode='online': after every 30 PPO steps, 256 fresh labels from the current
    policy are added and r is retrained on everything (Section 2.3)."""
    rm, policy, value = RewardModel(), copy.deepcopy(rho), nn.Parameter(torch.zeros(C))
    data = collect_labels(rho, n_labels)
    train_rm(rm, data, rho)
    opt = torch.optim.Adam(list(policy.parameters()) + [value], lr=lr)
    beta, log = beta0, []
    for step in range(1, steps + 1):
        if mode == "online" and step % 30 == 0:
            new = collect_labels(policy, 256)
            data = tuple(torch.cat([a, b]) for a, b in zip(data, new))
            train_rm(rm, data, rho, steps=100)
        kl, r_hat, r_true = ppo_step(policy, value, rm, rho, opt, beta)
        e = ((kl - kl_target) / kl_target)
        beta = beta * (1 + k_beta * max(-0.2, min(0.2, e)))                     # adaptive β
        log.append((kl, r_hat, r_true))
        if step % 30 == 0 or step == 1:
            print(f"  [{mode:7s}] step {step:3d}  KL(π||ρ) {kl:5.2f}  β {beta:.3f}  r(x,y) {r_hat:6.2f}  true reward {r_true:6.2f}")
    return log


def main():
    t0 = time.time()
    rho = TabularLM()
    x0 = torch.randint(0, C, (1024,))
    base = mock_reward(x0, rho.sample(x0)).mean().item()
    print(f"true reward of ρ's own samples: {base:.2f};  KL target 4.0 nats, adaptive β")
    off = run(rho, "offline")
    on = run(rho, "online")
    kls = torch.tensor([kl for kl, _, _ in off[-30:]])
    final_off, final_on = sum(r for _, _, r in off[-10:]) / 10, sum(r for _, _, r in on[-10:]) / 10
    print(f"offline: true reward {base:.2f} -> {final_off:.2f}, KL over the last 30 steps {kls.min():.2f}–{kls.max():.2f};  "
          f"online: -> {final_on:.2f}   ({time.time() - t0:.1f} s)")
    assert kls.max() < 1.5 * 4.0, "KL escaped the target region"
    assert final_off > base + 0.3, "reward did not rise"


if __name__ == "__main__":
    main()
