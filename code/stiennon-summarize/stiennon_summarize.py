"""Learning to Summarize from Human Feedback (Stiennon et al., 2020) — a from-scratch PyTorch implementation.

What is implemented (section / figure numbers follow the paper):
  * the three steps of Figure 2: collect pairwise comparisons of samples from the SFT policy, train a reward model,
    train the policy with PPO against it                                                                (Section 3.4)
  * reward-model loss(r_θ) = −E[log σ(r_θ(x, y_i) − r_θ(x, y_{1−i}))], one pass over the data, then shifted so that
    reference summaries score 0 on average; validation accuracy on held-out comparisons                (Section 3.4)
  * the per-episode policy reward R(x, y) = r_θ(x, y) − β log π^RL_φ(y|x) / π^SFT(y|x) with a FIXED β, PPO   (Section 3.4)
  * the overoptimization experiment of Figure 5: optimise against the fixed RM to increasing KL from the SFT policy
    (PPO at several β, and best-of-N with KL = log N − (N−1)/N) and measure both the RM's predicted preference and
    the labelers' actual preference over the references                                              (Section 4.3)
Simplifications: the "SFT model" is a tiny tabular autoregressive LM over 10 integer tokens, "reference summaries" are
its own samples, the labeler is a noisy Bradley-Terry oracle on a hidden true reward that penalises repeated tokens
(what the RM cannot see well far from its training distribution), the whole summary is one PPO action.

Run:  python stiennon_summarize.py        (CPU, about 15 s)
"""
import copy, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
C, n, V = 4, 8, 10                                      # posts (contexts), summary length, vocabulary


# ───────────────────────── toy LM: the supervised (SFT) policy and the RL policy ─────────────────────────
class TabularLM(nn.Module):
    """Autoregressive LM over integer tokens: logits of y_t depend on the post x, the position t and y_{t-1}."""
    def __init__(self, scale=0.7):
        super().__init__()
        self.table = nn.Parameter(scale * torch.randn(C, n, V + 1, V))          # previous-token index V = "start"

    def log_prob(self, x, y):
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


# ───────────────────────── the labelers: a noisy oracle on a hidden true reward ─────────────────────────
W_TRUE = 2.0 * torch.randn(C, V)


def true_reward(x, y, repeat_penalty=6.0):
    """Hidden quality: good tokens for this post, minus a penalty once some token occurs more than twice (degenerate,
    "gibberish" summaries). Such summaries are rare under the SFT policy, so the RM never learns about them."""
    max_count = F.one_hot(y, V).sum(1).max(1).values.float()
    return W_TRUE[x].gather(1, y).mean(1) - repeat_penalty * (max_count - 2).clamp(min=0) / n


def human_prefers(x, y0, y1, temp=0.3):
    """P(labeler picks y0) = σ((r*(y0) − r*(y1)) / temp): ~70 % agreement with the true ordering, like the paper's labelers."""
    return torch.sigmoid((true_reward(x, y0) - true_reward(x, y1)) / temp)


# ───────────────────────── reward model r_θ(x, y) ─────────────────────────
class RewardModel(nn.Module):
    """r_θ(x, y): a linear head on [one-hot(x), token counts of y / n] (the paper puts a scalar head on the SFT model)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(C + V, 1)
        self.register_buffer("shift", torch.zeros(C))                            # references score 0 on average

    def forward(self, x, y):
        feats = torch.cat([F.one_hot(x, C).float(), F.one_hot(y, V).float().mean(1)], -1)
        return self.net(feats).squeeze(-1) - self.shift[x]


def rm_loss(rm, x, y0, y1, i):
    """loss(r_θ) = −E[ log σ( r_θ(x, y_i) − r_θ(x, y_{1−i}) ) ]   with i the labeler's choice."""
    d = rm(x, y0) - rm(x, y1)
    d_chosen = torch.where(i == 0, d, -d)
    return -F.logsigmoid(d_chosen).mean()


def collect_comparisons(sft, N):
    x = torch.randint(0, C, (N,))
    y0, y1 = sft.sample(x), sft.sample(x)
    i = (torch.rand(N) > human_prefers(x, y0, y1)).long()                       # 0 if the labeler picked y0
    return x, y0, y1, i


def train_rm(rm, sft, data, val, batch=64):
    x, y0, y1, i = data
    opt = torch.optim.Adam(rm.parameters(), lr=1e-2)
    for _ in range(3):                                                            # the paper: a single epoch on 60k+ pairs
        for idx in torch.randperm(x.size(0)).split(batch):
            loss = rm_loss(rm, x[idx], y0[idx], y1[idx], i[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        xs = torch.arange(C).repeat_interleave(512)
        rm.shift.copy_(rm(xs, sft.sample(xs)).view(C, -1).mean(1))
        vx, vy0, vy1, vi = val
        acc = (((rm(vx, vy0) - rm(vx, vy1)) > 0).long() == (vi == 0).long()).float().mean().item()
    return acc


# ───────────────────────── PPO with a fixed KL coefficient β ─────────────────────────
def ppo_train(sft, rm, beta, steps=150, B=256, K=4, M=64, eps=0.2, lr=1e-2):
    policy, value = copy.deepcopy(sft), nn.Parameter(torch.zeros(C))            # policy starts from the SFT model
    opt = torch.optim.Adam(list(policy.parameters()) + [value], lr=lr)
    for _ in range(steps):
        x = torch.randint(0, C, (B,))
        y = policy.sample(x)
        with torch.no_grad():
            logp_old = policy.log_prob(x, y)
            R = rm(x, y) - beta * (logp_old - sft.log_prob(x, y))              # R(x,y) = r_θ(x,y) − β log π^RL/π^SFT
        for _ in range(K):
            for idx in torch.randperm(B).split(M):
                ratio = (policy.log_prob(x[idx], y[idx]) - logp_old[idx]).exp()
                adv = R[idx] - value[x[idx]].detach()
                l_clip = torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
                loss = -(l_clip - 0.5 * (value[x[idx]] - R[idx]).pow(2).mean())
                opt.zero_grad(); loss.backward(); opt.step()
    return policy


@torch.no_grad()
def evaluate(sft, rm, x, y):
    """KL(π || π^SFT) on the policy's samples y, the RM's predicted preference over reference summaries and the
    labelers' actual preference (both as the fraction of pairs won against a reference, i.e. Figure 5's y-axis)."""
    y_ref = sft.sample(x)
    pred = torch.sigmoid(rm(x, y) - rm(x, y_ref)).mean().item()
    actual = human_prefers(x, y, y_ref).mean().item()
    return pred, actual


def main():
    t0 = time.time()
    sft = TabularLM()
    rm = RewardModel()
    data, val = collect_comparisons(sft, 6000), collect_comparisons(sft, 1000)
    acc = train_rm(rm, sft, data, val)
    print(f"reward model: {data[0].size(0)} comparisons, held-out accuracy {acc:.2f} (labelers agree with the truth "
          f"{human_prefers(*val[:3]).sub(0.5).abs().add(0.5).mean():.2f} of the time)")
    print("Figure 5: optimise against the fixed RM to increasing KL and compare predicted vs actual preference")
    print("  method            KL(π||SFT)   RM-predicted pref   actual pref")
    x = torch.randint(0, C, (4096,))
    rows = []
    for N in (1, 4, 16, 64):                                                    # best-of-N: KL = log N − (N−1)/N
        ys = torch.stack([sft.sample(x) for _ in range(N)], 1)
        best = torch.stack([rm(x, ys[:, k]) for k in range(N)], 1).argmax(1)
        y = ys[torch.arange(x.size(0)), best]
        pred, actual = evaluate(sft, rm, x, y)
        kl = math.log(N) - (N - 1) / N
        rows.append((kl, pred, actual))
        print(f"  best-of-{N:<4d}      {kl:7.2f}       {pred:.3f}               {actual:.3f}")
    for beta in (1.0, 0.3, 0.1, 0.05, 0.02, 0.0):
        policy = ppo_train(sft, rm, beta)
        y = policy.sample(x)
        kl = (policy.log_prob(x, y) - sft.log_prob(x, y)).mean().item()
        pred, actual = evaluate(sft, rm, x, y)
        rows.append((kl, pred, actual))
        print(f"  PPO β={beta:<5.2f}       {kl:7.2f}       {pred:.3f}               {actual:.3f}")
    rows.sort()
    kls, preds, actuals = zip(*rows)
    peak = max(range(len(rows)), key=lambda k: actuals[k])
    print(f"actual preference peaks at KL {kls[peak]:.1f} ({actuals[peak]:.3f}) and falls to {actuals[-1]:.3f} at KL {kls[-1]:.1f}, "
          f"while the RM keeps predicting {preds[-1]:.3f}   ({time.time() - t0:.1f} s)")
    assert all(preds[k] <= preds[k + 1] + 0.01 for k in range(len(preds) - 1)), "RM-predicted preference should rise with KL"
    assert actuals[peak] > 0.55 and actuals[-1] < actuals[peak] - 0.1 and 0 < peak < len(rows) - 1, "no overoptimization hump"


if __name__ == "__main__":
    main()
