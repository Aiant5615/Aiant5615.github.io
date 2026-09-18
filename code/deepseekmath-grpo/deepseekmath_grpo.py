"""DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024) —
Group Relative Policy Optimization (GRPO) in PyTorch.

What is implemented (section / equation numbers follow the paper):
  * a group of G outputs {o_1..o_G} ~ π_θold(·|q) per question q, scored by a rule (outcome supervision)   (Section 4.1)
  * group-relative advantages Â_{i,t} = (r_i − mean(r)) / std(r), the same for every token of o_i           (Section 4.1.2, eq. 3)
  * the GRPO objective J_GRPO = E[ 1/G Σ_i 1/|o_i| Σ_t { min(ρ_{i,t} Â_{i,t}, clip(ρ_{i,t}, 1−ε, 1+ε) Â_{i,t})
    − β D_KL[π_θ || π_ref] } ] with ρ_{i,t} = π_θ(o_{i,t}|q,o_{i,<t}) / π_θold(o_{i,t}|q,o_{i,<t})        (Section 4.1, eq. 2)
  * the unbiased KL estimator D_KL = π_ref/π_θ − log(π_ref/π_θ) − 1 ("k3"), checked to be ≥ 0 per token and
    to agree with the exact per-token KL on average                                                        (eq. 4)
  * Algorithm 1: μ gradient steps per batch, and the outer "iteration" that resets π_ref ← π_θ              (Section 4.1)
  * no value model: the group mean is the baseline (Figure 4, bottom)
Simplifications: the "LLM" is a two-layer causal transformer with 32-d embeddings; the task is verifiable by a rule
(emit three digits in 0..4 whose sum equals the target given in the prompt) so there is no reward model to retrain
in the iterative loop; G = 8, ε = 0.2, β = 0.04, one prompt token.

Run:  python deepseekmath_grpo.py        (CPU, about 12 s)
"""
import copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
N_DIGITS, LEN = 5, 3                                     # output tokens are digits 0..4, three of them
TARGETS = torch.arange(LEN * (N_DIGITS - 1) + 1)         # target sums 0..12
VOCAB = N_DIGITS + len(TARGETS)                          # prompt token for target s is N_DIGITS + s


# ───────────────────────── a tiny causal transformer policy ─────────────────────────
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, d=32, h=2, layers=2, max_len=8):
        super().__init__()
        self.tok, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, N_DIGITS)          # the action space is the digits

    def forward(self, idx):                                                    # (B, L) -> logits over digits (B, L, 5)
        L = idx.size(1)
        x = self.tok(idx) + self.pos(torch.arange(L))
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)              # causal: True = not attended
        for b in self.blocks:
            x = b(x, mask)
        return self.head(self.ln(x))


def token_log_probs(model, q, o):
    """log π(o_t | q, o_<t) for every t: (B,) prompt tokens, (B, |o|) outputs -> (B, |o|)."""
    logits = model(torch.cat([q[:, None], o[:, :-1]], 1))
    return logits.log_softmax(-1).gather(-1, o[..., None]).squeeze(-1)


@torch.no_grad()
def sample(model, q, greedy=False):
    seq = q[:, None]
    for _ in range(LEN):
        logits = model(seq)[:, -1]
        nxt = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        seq = torch.cat([seq, nxt[:, None]], 1)
    return seq[:, 1:]


def reward(q, o):
    """Outcome supervision by a rule: r_i = 1 if the digits sum to the target in the prompt, else 0."""
    return (o.sum(1) == q - N_DIGITS).float()


# ───────────────────────── GRPO ─────────────────────────
def group_advantages(r, G):
    """Â_{i,t} = (r_i − mean({r_1..r_G})) / std({r_1..r_G})  (eq. 3); a group with identical rewards gets 0."""
    r = r.view(-1, G)
    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
    return adv.reshape(-1)


def kl_k3(logp, logp_ref):
    """D_KL[π_θ || π_ref] estimated per token as π_ref/π_θ − log(π_ref/π_θ) − 1  (eq. 4): nonnegative for every sample."""
    log_ratio = logp_ref - logp
    return log_ratio.exp() - log_ratio - 1


def grpo_loss(policy, q, o, logp_old, logp_ref, adv, eps=0.2, beta=0.04):
    """−J_GRPO on one batch of B questions × G outputs (eq. 2). Every output has |o| = LEN tokens, so the
    1/G Σ_i 1/|o_i| Σ_t average is the plain mean over all tokens."""
    logp = token_log_probs(policy, q, o)
    rho = (logp - logp_old).exp()                                              # ρ_{i,t}
    surrogate = torch.min(rho * adv[:, None], rho.clamp(1 - eps, 1 + eps) * adv[:, None])
    kl = kl_k3(logp, logp_ref)
    return -(surrogate - beta * kl).mean(), kl.mean().item(), ((rho - 1).abs() > eps).float().mean().item()


@torch.no_grad()
def exact_kl(policy, ref, q, o):
    """Σ_v π_θ(v|·) log π_θ(v|·)/π_ref(v|·) per token, to check the k3 estimator against."""
    inp = torch.cat([q[:, None], o[:, :-1]], 1)
    lp, lr = policy(inp).log_softmax(-1), ref(inp).log_softmax(-1)
    return (lp.exp() * (lp - lr)).sum(-1).mean().item()


@torch.no_grad()
def accuracy(policy, n=1024, greedy=False):
    q = N_DIGITS + TARGETS[torch.randint(0, len(TARGETS), (n,))]
    return reward(q, sample(policy, q, greedy)).mean().item()


def main():
    t0 = time.time()
    policy = TinyGPT()
    ref = copy.deepcopy(policy).eval()                                          # π_ref for the first iteration
    opt = torch.optim.Adam(policy.parameters(), lr=5e-4)
    G, B, mu, steps_per_iter, iters = 8, 16, 2, 150, 8
    acc0 = accuracy(policy)
    print(f"parameters: {sum(p.numel() for p in policy.parameters()):,}   accuracy before RL: {acc0:.3f}")
    for it in range(1, iters + 1):
        if it > 1:
            ref = copy.deepcopy(policy).eval()                                  # Algorithm 1: π_ref ← π_θ each iteration
        for step in range(1, steps_per_iter + 1):
            q = (N_DIGITS + TARGETS[torch.randint(0, len(TARGETS), (B,))]).repeat_interleave(G)   # B questions × G
            old = copy.deepcopy(policy).eval()                                  # π_θold: the sampling policy
            o = sample(old, q)
            with torch.no_grad():
                logp_old, logp_ref = token_log_probs(old, q, o), token_log_probs(ref, q, o)
            r = reward(q, o)
            adv = group_advantages(r, G)
            for _ in range(mu):                                                 # μ GRPO updates on the same group data
                loss, kl, clipped = grpo_loss(policy, q, o, logp_old, logp_ref, adv)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
            if step % 75 == 0:
                solved = (r.view(-1, G).mean(1) == 1).float().mean().item()
                print(f"iter {it} step {step:3d}  batch reward {r.mean():.3f}  groups all-correct {solved:.2f}  "
                      f"k3 KL {kl:.4f} (exact {exact_kl(policy, ref, q, o):.4f})  clipped {clipped:.2f}")
    acc, acc_g = accuracy(policy), accuracy(policy, greedy=True)
    with torch.no_grad():
        k3 = kl_k3(token_log_probs(policy, q, o), logp_ref)
    print(f"accuracy after RL: sampled {acc:.3f}, greedy {acc_g:.3f} (from {acc0:.3f});  min per-token k3 = {k3.min():.1e} ≥ 0   ({time.time() - t0:.1f} s)")
    assert acc > 0.8 and acc_g > 0.8 and acc > acc0 + 0.5, "GRPO did not learn the task"
    assert (k3 >= -1e-6).all(), "k3 must be nonnegative for every sample (up to float error)"


if __name__ == "__main__":
    main()
