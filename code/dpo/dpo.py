"""Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023) — PyTorch.

What is implemented (section / equation numbers follow the paper):
  * the Bradley-Terry preference model p*(y1 ≻ y2 | x) = σ(r*(x,y1) − r*(x,y2)) used to label pairs         (eq. 1)
  * the KL-regularised RLHF objective max_π E[r(x,y)] − β KL[π(·|x) || π_ref(·|x)]                          (eq. 3)
  * its closed-form optimum π_r(y|x) = π_ref(y|x) exp(r(x,y)/β) / Z(x), computed EXACTLY by enumerating every y  (eq. 4)
  * the inverted reward r(x,y) = β log π_r(y|x)/π_ref(y|x) + β log Z(x)                                       (eq. 5)
  * the DPO loss L_DPO = −E log σ(β log π_θ(y_w|x)/π_ref(y_w|x) − β log π_θ(y_l|x)/π_ref(y_l|x))               (eq. 7)
  * the gradient's weight σ(r̂_θ(x,y_l) − r̂_θ(x,y_w)) with the implicit reward r̂_θ = β log π_θ/π_ref         (eq. 8)
  * checks: r̂_θ matches r* up to the per-prompt constant β log Z(x); π_θ converges to π_r; both reach the same
    value of the RLHF objective; r̂_θ predicts held-out preferences as well as r* does                    (Section 5)
Simplifications: π_ref is a tiny tabular autoregressive LM over 6 integer tokens with responses of length 4 (6^4 = 1296
responses per prompt, so Z(x) is a finite sum), r* is a hidden per-token score table, preference labels are sampled
from the Bradley-Terry model rather than collected from humans.

Run:  python dpo.py        (CPU, about 5 s)
"""
import copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
C, n, V, BETA = 3, 4, 6, 0.5                            # prompts, response length, vocabulary, β
ALL_Y = torch.cartesian_prod(*[torch.arange(V)] * n)    # every possible response, (V^n, n)


class TabularLM(nn.Module):
    """Autoregressive LM over integer tokens: logits of y_t depend on the prompt x, the position t and y_{t-1}."""
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

    def log_prob_all(self, x):                           # log π(y|x) for every y ∈ ALL_Y, for one prompt x: (V^n,)
        return self.log_prob(torch.full((ALL_Y.size(0),), x), ALL_Y)


# ───────────────────────── the hidden true reward and the preference data ─────────────────────────
W_TRUE = 0.8 * torch.randn(C, V)


def true_reward(x, y):                                   # r*(x, y) = Σ_t w[x, y_t]
    return W_TRUE[x].gather(1, y).sum(1)


def make_pairs(pi_ref, N):
    """(x, y_w, y_l): two samples from π_ref, the winner drawn from the Bradley-Terry model p* = σ(r*(y1) − r*(y2)) (eq. 1)."""
    x = torch.randint(0, C, (N,))
    y1, y2 = pi_ref.sample(x), pi_ref.sample(x)
    first_wins = torch.rand(N) < torch.sigmoid(true_reward(x, y1) - true_reward(x, y2))
    return x, torch.where(first_wins[:, None], y1, y2), torch.where(first_wins[:, None], y2, y1)


# ───────────────────────── DPO ─────────────────────────
def implicit_reward(pi, pi_ref, x, y):
    """r̂_θ(x, y) = β log π_θ(y|x) / π_ref(y|x)   (the reparameterisation of eq. 5 without the β log Z(x) term)."""
    return BETA * (pi.log_prob(x, y) - pi_ref.log_prob(x, y))


def dpo_loss(pi, pi_ref, x, y_w, y_l):
    """L_DPO(π_θ; π_ref) = −E[ log σ( r̂_θ(x, y_w) − r̂_θ(x, y_l) ) ]   (eq. 7). Also returns the eq. 8 weight."""
    r_w, r_l = implicit_reward(pi, pi_ref, x, y_w), implicit_reward(pi, pi_ref, x, y_l)
    return -F.logsigmoid(r_w - r_l).mean(), torch.sigmoid(r_l - r_w).mean()


# ───────────────────────── the RLHF closed form, by enumeration ─────────────────────────
@torch.no_grad()
def optimal_policy(pi_ref, x):
    """π_r(y|x) = π_ref(y|x) exp(r*(x,y)/β) / Z(x), Z(x) = Σ_y π_ref(y|x) exp(r*(x,y)/β)   (eq. 4), over all 1296 y."""
    log_ref = pi_ref.log_prob_all(x)
    r = true_reward(torch.full((ALL_Y.size(0),), x), ALL_Y)
    log_z = torch.logsumexp(log_ref + r / BETA, 0)
    return log_ref + r / BETA - log_z, log_z, r


@torch.no_grad()
def rlhf_objective(log_pi, log_ref, r):
    """E_{y~π}[r(x,y)] − β KL[π || π_ref]   (eq. 3), exact for the enumerated distribution."""
    p = log_pi.exp()
    return (p * r).sum() - BETA * (p * (log_pi - log_ref)).sum()


def main():
    t0 = time.time()
    pi_ref = TabularLM()
    pi = copy.deepcopy(pi_ref)                                                  # π_θ is initialised at π_ref
    x, y_w, y_l = make_pairs(pi_ref, 131072)
    vx, vy_w, vy_l = make_pairs(pi_ref, 2048)
    opt = torch.optim.Adam(pi.parameters(), lr=2e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 1500)
    for step in range(1, 1501):
        idx = torch.randint(0, x.size(0), (2048,))
        loss, weight = dpo_loss(pi, pi_ref, x[idx], y_w[idx], y_l[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step in (1, 300, 600, 1000, 1500):
            with torch.no_grad():
                acc = (implicit_reward(pi, pi_ref, vx, vy_w) > implicit_reward(pi, pi_ref, vx, vy_l)).float().mean()
            print(f"step {step:3d}  L_DPO {loss.item():.4f}  mean eq.8 weight σ(r̂_l − r̂_w) {weight.item():.3f}  "
                  f"held-out preference accuracy of r̂_θ {acc:.3f}")
    with torch.no_grad():
        acc_star = (true_reward(vx, vy_w) > true_reward(vx, vy_l)).float().mean().item()
        print(f"held-out accuracy of the TRUE reward r* (ceiling under Bradley-Terry noise): {acc_star:.3f}")
        print("per prompt: r̂_θ − r* should be the constant −β log Z(x); π_θ should reach the closed-form π_r (eq. 4)")
        print("  x   −β log Z(x)   mean(r̂−r*)  std(r̂−r*)  corr(r̂,r*)   KL(π_r||π_ref)  KL(π_r||π_θ)   J(π_ref)  J(π_θ)  J(π_r)")
        worst_std, worst_kl_ratio = 0.0, 0.0
        for xi in range(C):
            log_opt, log_z, r = optimal_policy(pi_ref, xi)
            log_ref, log_pi = pi_ref.log_prob_all(xi), pi.log_prob_all(xi)
            r_hat = BETA * (log_pi - log_ref)
            diff = r_hat - r
            corr = torch.corrcoef(torch.stack([r_hat, r]))[0, 1]
            p_opt = log_opt.exp()
            kl_ref, kl_pi = (p_opt * (log_opt - log_ref)).sum(), (p_opt * (log_opt - log_pi)).sum()
            J = [rlhf_objective(lp, log_ref, r).item() for lp in (log_ref, log_pi, log_opt)]
            print(f"  {xi}   {-BETA * log_z:9.3f}   {diff.mean():9.3f}   {diff.std():8.3f}   {corr:8.3f}      "
                  f"{kl_ref:8.3f}     {kl_pi:8.3f}     {J[0]:6.3f}  {J[1]:6.3f}  {J[2]:6.3f}")
            worst_std, worst_kl_ratio = max(worst_std, diff.std().item()), max(worst_kl_ratio, (kl_pi / kl_ref).item())
    print(f"({time.time() - t0:.1f} s)")
    assert acc > acc_star - 0.03, "implicit reward does not rank held-out pairs as well as the true reward"
    assert worst_std < 0.15, "r̂_θ − r* is not constant per prompt (eq. 5 violated)"
    assert worst_kl_ratio < 0.05, "π_θ did not converge to the closed-form optimal policy"


if __name__ == "__main__":
    main()
