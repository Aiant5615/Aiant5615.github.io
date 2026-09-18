"""Deep Reinforcement Learning from Human Preferences (Christiano et al., 2017) — a from-scratch PyTorch implementation.

What is implemented (section numbers follow the paper):
  * trajectory segments σ = ((o_0,a_0), ..., (o_{k-1},a_{k-1})) and pairwise comparisons (σ^1, σ^2, μ)     (Section 2.1)
  * the three asynchronous processes of Figure 1 run as one loop: policy ↔ environment, pair selection
    and labelling, reward-predictor fitting                                                                (Section 2.2)
  * the Bradley-Terry preference model on summed predicted reward, P̂[σ^1 ≻ σ^2] = exp Σ r̂ / (exp Σ r̂ + exp Σ r̂),
    and the cross-entropy loss(r̂) over the comparison set D                                                (Section 2.2.2)
  * the tricks of Section 2.2.3: an ensemble of predictors on bootstrap samples, a 10 % assumed label-error
    rate (P̂ mixed with uniform), ℓ2 regularisation, a held-out validation split, normalisation of r̂ to zero
    mean / unit variance before it is handed to RL
  * query selection: sample candidate pairs, ask about those with the largest ensemble variance of P̂        (Section 2.2.4)
  * synthetic labels from an oracle that compares clips by true reward, as in the "synthetic" runs of 3.1.1
Simplifications: a 6×6 grid-world with a hidden goal instead of MuJoCo/Atari, a tabular softmax policy trained by
advantage actor-critic instead of TRPO/A2C on pixels, segments of k = 10 steps, no ties, everything synchronous.

Run:  python christiano_preferences.py        (CPU, about 8 s)
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
W, A, K = 6, 4, 10                                      # grid width, number of actions, segment length k
GOAL = torch.tensor([5, 2])
MOVES = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]])   # right, left, down, up


# ───────────────────────── environment with a hidden true reward ─────────────────────────
def step(pos, a):
    """pos: (N, 2) int, a: (N,) -> next pos. Walls: moves off the grid are ignored."""
    return (pos + MOVES[a]).clamp(0, W - 1)


def true_reward(pos, a):
    """r*(o, a): known to the oracle 'human' only. Closer to the goal is better, reaching it gives a bonus."""
    nxt = step(pos, a)
    dist = (nxt - GOAL).abs().sum(-1).float()
    return -dist / (2 * (W - 1)) + (dist == 0).float()


def obs(pos):
    return pos[:, 0] * W + pos[:, 1]                    # observation = cell index (one-hot inside the models)


def rollout(policy, N):
    """Each of N actors collects one segment of k steps from a random start: returns o (N,k), a (N,k), true r (N,k)."""
    pos = torch.randint(0, W, (N, 2))
    O, Ac, R = [], [], []
    for _ in range(K):
        o = obs(pos)
        a = torch.distributions.Categorical(logits=policy.logits[o]).sample()
        O.append(o); Ac.append(a); R.append(true_reward(pos, a))
        pos = step(pos, a)
    return torch.stack(O, 1), torch.stack(Ac, 1), torch.stack(R, 1), obs(pos)


# ───────────────────────── reward predictor r̂(o, a) and its ensemble ─────────────────────────
class RewardPredictor(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(W * W + A, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, o, a):                             # (...,) ints -> (...,) predicted reward
        x = torch.cat([F.one_hot(o, W * W), F.one_hot(a, A)], -1).float()
        return self.net(x).squeeze(-1)


def pref_prob(r_hat, o1, a1, o2, a2, eps=0.1):
    """P̂[σ^1 ≻ σ^2] = exp Σ_t r̂(o^1_t,a^1_t) / (exp Σ r̂^1 + exp Σ r̂^2)  (Section 2.2.2), i.e. σ(Σ r̂^1 - Σ r̂^2),
    mixed with a 10 % chance that the human answers uniformly at random (Section 2.2.3)."""
    p = torch.sigmoid(r_hat(o1, a1).sum(-1) - r_hat(o2, a2).sum(-1))
    return (1 - eps) * p + eps * 0.5


def preference_loss(r_hat, D):
    """loss(r̂) = -Σ_{(σ^1,σ^2,μ) ∈ D} [ μ(1) log P̂[σ^1 ≻ σ^2] + μ(2) log P̂[σ^2 ≻ σ^1] ]  (Section 2.2.2)."""
    o1, a1, o2, a2, mu = D
    p = pref_prob(r_hat, o1, a1, o2, a2)
    return -(mu[:, 0] * p.log() + mu[:, 1] * (1 - p).log()).mean()


class Ensemble:
    """Several predictors, each fit to its own bootstrap sample of D; the mean is used as the RL reward, the
    variance of P̂ across members drives query selection."""
    def __init__(self, n=3, weight_decay=1e-3):
        self.members = [RewardPredictor() for _ in range(n)]
        self.opts = [torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=weight_decay) for m in self.members]  # ℓ2 reg.

    def fit(self, D, steps=40):
        n = D[0].size(0)
        for m, opt in zip(self.members, self.opts):
            boot = torch.randint(0, n, (n,))                                    # bootstrap sample
            Db = tuple(x[boot] for x in D)
            for _ in range(steps):
                loss = preference_loss(m, Db)
                opt.zero_grad(); loss.backward(); opt.step()

    @torch.no_grad()
    def reward(self, o, a):
        r = torch.stack([m(o, a) for m in self.members]).mean(0)
        return (r - r.mean()) / (r.std() + 1e-8)                                 # normalised before RL (Section 2.2.2)

    @torch.no_grad()
    def disagreement(self, o1, a1, o2, a2):
        return torch.stack([pref_prob(m, o1, a1, o2, a2) for m in self.members]).var(0)


# ───────────────────────── the oracle "human" and query selection ─────────────────────────
def oracle_label(r1, r2):
    """Synthetic human of Section 3.1.1: prefers the clip with the larger true reward; μ is one-hot (uniform on a tie)."""
    s1, s2 = r1.sum(-1), r2.sum(-1)
    mu1 = torch.where(s1 > s2, 1.0, torch.where(s1 < s2, 0.0, 0.5))
    return torch.stack([mu1, 1 - mu1], -1)


def select_queries(ens, O, Ac, R, n_candidates=300, n_queries=10):
    """Sample candidate pairs of segments from the latest rollout; ask about the pairs with the largest ensemble variance."""
    N = O.size(0)
    i, j = torch.randint(0, N, (n_candidates,)), torch.randint(0, N, (n_candidates,))
    top = ens.disagreement(O[i], Ac[i], O[j], Ac[j]).topk(n_queries).indices
    i, j = i[top], j[top]
    return O[i], Ac[i], O[j], Ac[j], oracle_label(R[i], R[j])


def add(D, new):
    return new if D is None else tuple(torch.cat([x, y]) for x, y in zip(D, new))


# ───────────────────────── policy: tabular softmax, advantage actor-critic on r̂ ─────────────────────────
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(W * W, A))
        self.value = nn.Parameter(torch.zeros(W * W))


def a2c_update(policy, opt, O, Ac, r_hat, o_last, gamma=0.95):
    """One actor-critic step on the segment with predicted rewards r̂ (the RL algorithm of Figure 1)."""
    with torch.no_grad():
        ret, R = policy.value[o_last].clone(), torch.zeros_like(r_hat)
        for t in reversed(range(K)):                                            # discounted n-step returns
            ret = r_hat[:, t] + gamma * ret
            R[:, t] = ret
    logp = torch.log_softmax(policy.logits[O], -1)
    v = policy.value[O]
    adv = (R - v).detach()
    pg = -(logp.gather(-1, Ac.unsqueeze(-1)).squeeze(-1) * adv).mean()
    entropy = -(logp.exp() * logp).sum(-1).mean()
    loss = pg + 0.5 * (v - R).pow(2).mean() - 0.01 * entropy
    opt.zero_grad(); loss.backward(); opt.step()


def evaluate(ens, policy):
    """Correlation between r̂ and r* over all 144 (state, action) pairs, and the true return of the current policy."""
    o, a = torch.arange(W * W).repeat_interleave(A), torch.arange(A).repeat(W * W)
    pos = torch.stack([o // W, o % W], -1)
    r_hat, r_true = ens.reward(o, a), true_reward(pos, a)
    corr = torch.corrcoef(torch.stack([r_hat, r_true]))[0, 1].item()
    with torch.no_grad():
        _, _, R, _ = rollout(policy, 512)
    return corr, R.sum(1).mean().item()


def main():
    t0 = time.time()
    ens, policy = Ensemble(), Policy()
    opt = torch.optim.Adam(policy.parameters(), lr=0.02)
    D, D_val = None, None
    O, Ac, R, o_last = rollout(policy, 64)                                       # initial labels come from the random policy
    for _ in range(5):
        D = add(D, select_queries(ens, O, Ac, R))
    ens.fit(D)
    _, ret0 = evaluate(ens, policy)
    print(f"true return of the random policy: {ret0:.2f}")
    for it in range(1, 61):
        O, Ac, R, o_last = rollout(policy, 64)                                   # (1) policy interacts with the environment
        q = select_queries(ens, O, Ac, R)                                        # (2) pairs are chosen and labelled
        if it % 10 == 0:
            D_val = add(D_val, q)                                                #     ~10 % of comparisons held out
        else:
            D = add(D, q)
        ens.fit(D)                                                               # (3) r̂ is fit to all comparisons so far
        for _ in range(4):                                                       # RL on the predicted reward only
            O, Ac, R, o_last = rollout(policy, 64)
            a2c_update(policy, opt, O, Ac, ens.reward(O, Ac), o_last)
        if it % 12 == 0:
            corr, ret = evaluate(ens, policy)
            with torch.no_grad():
                tr, va = preference_loss(ens.members[0], D).item(), preference_loss(ens.members[0], D_val).item()
            print(f"iter {it:2d}  labels {D[0].size(0):3d}  RM loss train {tr:.3f} val {va:.3f}  "
                  f"corr(r̂, r*) {corr:.2f}  true return {ret:.2f}")
    corr, ret = evaluate(ens, policy)
    print(f"final: corr(r̂, r*) = {corr:.2f}, true return {ret0:.2f} -> {ret:.2f} with {D[0].size(0)} comparisons  "
          f"({time.time() - t0:.1f} s)")
    assert corr > 0.8 and ret > ret0 + 2, "learned reward does not track the true reward / policy did not improve"


if __name__ == "__main__":
    main()
