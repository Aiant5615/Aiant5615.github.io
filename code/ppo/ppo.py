"""Proximal Policy Optimization Algorithms (Schulman et al., 2017) — a from-scratch PyTorch implementation.

What is implemented (section / equation numbers follow the paper):
  * probability ratio r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t) and the unclipped surrogate L^CPI   (Section 3, eq. 6)
  * the clipped surrogate L^CLIP = E[min(r_t Â_t, clip(r_t, 1-ε, 1+ε) Â_t)], ε = 0.2                (Section 3, eq. 7)
  * the adaptive KL-penalty alternative L^KLPEN with the "d < d_targ/1.5 → β/2, d > 1.5 d_targ → 2β" rule (Section 4, eq. 8)
  * the combined loss L^{CLIP+VF+S} = L^CLIP - c1 (V_θ(s_t) - V_t^targ)^2 + c2 S[π_θ](s_t)          (Section 5, eq. 9)
  * truncated generalized advantage estimation over a rollout of length T                             (Section 5, eqs. 11-12)
  * Algorithm 1: N actors collect T steps with π_θold, compute Â, then K epochs of minibatch Adam     (Section 5, Algorithm 1)
  * Figure 2 reproduced numerically: L^CPI, the clipped term, L^CLIP and KL along the update direction
Simplifications: a 1-D double integrator ("push a point mass to the origin") replaces MuJoCo, a diagonal Gaussian policy
with one action dimension, a tiny shared policy/value MLP, fixed-length episodes (T = episode length), no reward scaling.

Run:  python ppo.py        (CPU, about 10 s)
"""
import copy, math, time
import torch
import torch.nn as nn

torch.manual_seed(0)


# ───────────────────────── toy environment ─────────────────────────
class PointMass:
    """N parallel copies of a damped 1-D double integrator. State s = (x, v), action a = force in [-1, 1].
    Reward r_t = -0.1 (x^2 + 0.1 v^2 + 0.01 a^2): the agent must drive the mass to rest at the origin."""
    def __init__(self, N, dt=0.2):
        self.N, self.dt = N, dt

    def reset(self):
        self.x = torch.rand(self.N) * 4 - 2
        self.v = torch.rand(self.N) * 2 - 1
        return self.obs()

    def obs(self):
        return torch.stack([self.x, self.v], -1)                                   # (N, 2)

    def step(self, a):
        a = a.clamp(-1, 1)
        self.v = (0.95 * self.v + self.dt * a).clamp(-3, 3)
        self.x = (self.x + self.dt * self.v).clamp(-4, 4)
        return self.obs(), -0.1 * (self.x ** 2 + 0.1 * self.v ** 2 + 0.01 * a ** 2)


# ───────────────────────── shared policy / value network ─────────────────────────
class ActorCritic(nn.Module):
    """π_θ(a|s) = N(μ_θ(s), σ^2) and V_θ(s) from a shared trunk (the case where eq. 9 is needed)."""
    def __init__(self, obs_dim=2, hidden=64):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mu, self.v = nn.Linear(hidden, 1), nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(1))

    def dist(self, s):
        return torch.distributions.Normal(self.mu(self.trunk(s)).squeeze(-1), self.log_std.exp())

    def value(self, s):
        return self.v(self.trunk(s)).squeeze(-1)


# ───────────────────────── advantages and objectives ─────────────────────────
def gae(rewards, values, last_value, gamma=0.99, lam=0.95):
    """Â_t = δ_t + (γλ) δ_{t+1} + ... + (γλ)^{T-t+1} δ_{T-1},   δ_t = r_t + γ V(s_{t+1}) - V(s_t)     (eqs. 11-12).
    rewards, values: (T, N); last_value: V(s_T) used to bootstrap the truncated rollout."""
    T = rewards.size(0)
    adv, next_adv, next_v = torch.zeros_like(rewards), torch.zeros(rewards.size(1)), last_value
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_v - values[t]
        adv[t] = next_adv = delta + gamma * lam * next_adv
        next_v = values[t]
    return adv, adv + values                                                        # (Â_t, V_t^targ = Â_t + V(s_t))


def surrogates(ratio, adv, eps=0.2):
    """Per-timestep L^CPI_t = r_t Â_t (eq. 6), the clipped term clip(r_t,1-ε,1+ε) Â_t, and L^CLIP_t = min of the two (eq. 7)."""
    cpi = ratio * adv
    clipped = ratio.clamp(1 - eps, 1 + eps) * adv
    return cpi, clipped, torch.min(cpi, clipped)


def ppo_loss(model, batch, variant="clip", beta=None, eps=0.2, c1=0.5, c2=0.01):
    """Negative of L^{CLIP+VF+S} (eq. 9) on a minibatch, or of L^KLPEN (eq. 8) - c1 VF + c2 S when variant='klpen'."""
    s, a, logp_old, adv, v_targ, mu_old, std_old = batch
    pi = model.dist(s)
    ratio = (pi.log_prob(a) - logp_old).exp()                                       # r_t(θ), = 1 at θ_old
    if variant == "clip":
        policy_obj = surrogates(ratio, adv, eps)[2].mean()
    else:                                                                           # KL(π_θold(.|s_t) || π_θ(.|s_t)) for Gaussians
        kl = torch.distributions.kl_divergence(torch.distributions.Normal(mu_old, std_old), pi)
        policy_obj = (ratio * adv - beta * kl).mean()
    vf = (model.value(s) - v_targ).pow(2).mean()                                    # (V_θ(s_t) - V_t^targ)^2
    entropy = pi.entropy().mean()                                                   # S[π_θ](s_t)
    return -(policy_obj - c1 * vf + c2 * entropy)


# ───────────────────────── Algorithm 1 ─────────────────────────
def collect_rollout(model, env, T):
    """N actors run π_θold for T timesteps; returns tensors flattened over (T, N)."""
    S, A, LP, R, V, MU = [], [], [], [], [], []
    s = env.reset()
    with torch.no_grad():
        for _ in range(T):
            pi = model.dist(s)
            a = pi.sample()
            s_next, r = env.step(a)
            S.append(s); A.append(a); LP.append(pi.log_prob(a)); R.append(r); V.append(model.value(s)); MU.append(pi.mean)
            s = s_next
        adv, v_targ = gae(torch.stack(R), torch.stack(V), model.value(s))
    flat = lambda xs: torch.stack(xs).reshape(-1)
    std_old = model.log_std.exp().detach().expand(T * env.N)
    batch = (torch.stack(S).reshape(-1, 2), flat(A), flat(LP), adv.reshape(-1), v_targ.reshape(-1), flat(MU), std_old)
    return batch, torch.stack(R).sum(0).mean().item()


def train(variant, iters=60, N=16, T=64, K=10, M=128, lr=1e-3, d_targ=0.01):
    model, env = ActorCritic(), PointMass(N)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    beta, returns = 1.0, []
    for it in range(1, iters + 1):
        batch, ret = collect_rollout(model, env, T)                                 # run π_θold in environment for T timesteps
        returns.append(ret)
        adv = batch[3]
        batch = batch[:3] + ((adv - adv.mean()) / (adv.std() + 1e-8),) + batch[4:]   # standard advantage normalisation
        old = copy.deepcopy(model)
        for _ in range(K):                                                          # K epochs of minibatch SGD on the same data
            for idx in torch.randperm(N * T).split(M):
                loss = ppo_loss(model, tuple(x[idx] for x in batch), variant, beta)
                opt.zero_grad(); loss.backward(); opt.step()
        if variant == "klpen":                                                      # adaptive β (Section 4)
            with torch.no_grad():
                d = torch.distributions.kl_divergence(old.dist(batch[0]), model.dist(batch[0])).mean().item()
            beta = beta / 2 if d < d_targ / 1.5 else beta * 2 if d > 1.5 * d_targ else beta
        if it % 15 == 0 or it == 1:
            print(f"  [{variant:5s}] iter {it:3d}  mean episode return {ret:8.2f}  σ {model.log_std.exp().item():.2f}"
                  + (f"  β {beta:.3f}" if variant == "klpen" else ""))
    return model, old, batch, returns


def figure2(old, new, batch):
    """Figure 2: interpolate θ = θ_old + α (θ_new - θ_old) and evaluate KL, L^CPI, the clipped term and L^CLIP on the batch."""
    s, a, logp_old, adv, _, mu_old, std_old = batch
    probe, sd_old, sd_new = copy.deepcopy(new), old.state_dict(), new.state_dict()
    print("  α      KL(old||new)   L^CPI    clipped   L^CLIP")
    for alpha in (0.0, 0.5, 1.0, 1.5, 2.0):
        probe.load_state_dict({k: sd_old[k] + alpha * (sd_new[k] - sd_old[k]) for k in sd_old})
        with torch.no_grad():
            pi = probe.dist(s)
            ratio = (pi.log_prob(a) - logp_old).exp()
            cpi, clipped, lclip = (x.mean().item() for x in surrogates(ratio, adv))
            kl = torch.distributions.kl_divergence(torch.distributions.Normal(mu_old, std_old), pi).mean().item()
        print(f"  {alpha:.1f}    {kl:9.4f}    {cpi:7.4f}  {clipped:7.4f}  {lclip:7.4f}")
    return cpi, lclip


def main():
    t0 = time.time()
    print("PPO-clip (eq. 7 + eq. 9), N=16 actors, T=64, K=10 epochs, M=128, γ=0.99, λ=0.95, ε=0.2")
    model, old, batch, ret_clip = train("clip")
    print("PPO with adaptive KL penalty (eq. 8), same budget")
    _, _, _, ret_kl = train("klpen")
    print("Figure 2 on the final batch of PPO-clip (α=0 is θ_old, α=1 is the parameters after K epochs):")
    cpi, lclip = figure2(old, model, batch)
    with torch.no_grad():
        ratio = (model.dist(batch[0]).log_prob(batch[1]) - batch[2]).exp()
    frac = ((ratio - 1).abs() > 0.2).float().mean().item()
    print(f"fraction of ratios outside [1-ε, 1+ε] after the update: {frac:.2f}   ({time.time() - t0:.1f} s)")
    first, last = sum(ret_clip[:3]) / 3, sum(ret_clip[-3:]) / 3
    print(f"PPO-clip return: {first:.1f} -> {last:.1f};  KL-penalty return: {sum(ret_kl[:3]) / 3:.1f} -> {sum(ret_kl[-3:]) / 3:.1f}")
    assert last > first + 20 and last > -5, "return did not rise as expected"
    assert lclip <= cpi + 1e-6, "L^CLIP must be a lower bound on the unclipped surrogate"


if __name__ == "__main__":
    main()
