"""Training Language Models to Follow Instructions with Human Feedback (InstructGPT, Ouyang et al., 2022) — the three-step
RLHF pipeline (Figure 2) on a toy token language model, in PyTorch.

What is implemented (section numbers follow the paper):
  * Step 0: a tiny pre-trained causal LM standing in for GPT-3, and its pre-training distribution D_pretrain
  * Step 1: supervised fine-tuning (SFT) on labeler demonstrations                                          (3.5)
  * Step 2: reward model r_theta(x, y) = the SFT model with the unembedding layer replaced by a scalar head, trained on
    K = 4 ranked completions per prompt with  loss(theta) = -1/C(K,2) E[log sigma(r(x, y_w) - r(x, y_l))],
    all C(K,2) pairs of one prompt in the same batch, a single epoch, then shifted so that demonstrations
    score 0 on average                                                                                       (3.5)
  * Step 3: PPO-ptx, maximising
    objective(phi) = E_{(x,y)~pi_RL}[ r_theta(x,y) - beta log(pi_RL(y|x) / pi_SFT(y|x)) ] + gamma E_{x~D_pretrain}[log pi_RL(x)]
    with the per-token KL penalty inside the reward, the value function initialised from the RM, PPO's clipped
    surrogate with GAE, and the pre-training-mix term; PPO with gamma = 0 is run as the "alignment tax" control (3.5)
  * the "labelers" are a hidden scoring rule r*(x, y) that writes demonstrations and ranks samples
Simplifications: 2-layer d = 32 Transformers over 16 integer tokens instead of GPT-3; prompts are 3 tokens and
completions 6; a few hundred synthetic prompts instead of 13k / 33k / 31k; the true preference r* is a rule
("reuse the prompt's tokens") rather than human judgement.

Run:  python instructgpt.py        (CPU, about 10 s)
"""
import copy, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
V, P_LEN, T_LEN = 16, 3, 6                   # vocabulary, prompt length, completion length
CTX = P_LEN + T_LEN


# ───────────────────────── a tiny causal Transformer LM ─────────────────────────
class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.attn, self.ln2 = nn.LayerNorm(d), nn.MultiheadAttention(d, h, batch_first=True), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        L = x.size(1)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), 1)
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=causal, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class LM(nn.Module):
    """Body -> hidden states; `head` = unembedding for policies, a scalar layer for the RM / value function (3.5)."""
    def __init__(self, d=32, h=2, n_layer=2, out=V):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(V, d), nn.Embedding(CTX, d)
        self.blocks, self.ln_f, self.head = nn.ModuleList([Block(d, h) for _ in range(n_layer)]), nn.LayerNorm(d), nn.Linear(d, out)

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1)))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def completion_logprobs(policy, x, y):
    """log pi(y_t | x, y_<t) for every completion token: (B, T_LEN)."""
    logits = policy(torch.cat([x, y], 1))[:, P_LEN - 1 : -1]
    return logits.log_softmax(-1).gather(-1, y[..., None]).squeeze(-1)


@torch.no_grad()
def sample(policy, x):
    y = torch.zeros(x.size(0), 0, dtype=torch.long)
    for _ in range(T_LEN):
        probs = policy(torch.cat([x, y], 1))[:, -1].softmax(-1)
        y = torch.cat([y, torch.multinomial(probs, 1)], 1)
    return y


def scalar(model, x, y):
    """r_theta(x, y) or V(x, y): read the scalar head at the last token (RM) or at every completion prefix (value)."""
    return model(torch.cat([x, y], 1))[:, P_LEN - 1 :, 0]            # (B, T_LEN + 1); [:, -1] is the RM score


# ───────────────────────── the hidden "human" (labelers) ─────────────────────────
def r_star(x, y):
    """What labelers actually prefer: completions that stay on topic, i.e. reuse the prompt's tokens (fraction in 0..1)."""
    return (y[:, :, None] == x[:, None, :]).any(-1).float().mean(1)


def demonstrations(x):
    """Labeler-written completions: 70% of the time a prompt token (never equal to the previous token), else random."""
    y, prev = [], torch.full((x.size(0),), -1)
    for _ in range(T_LEN):
        pick = x.gather(1, torch.randint(0, P_LEN, (x.size(0), 1))).squeeze(1)
        pick = torch.where(pick == prev, x.gather(1, torch.randint(0, P_LEN, (x.size(0), 1))).squeeze(1), pick)
        tok = torch.where(torch.rand(x.size(0)) < 0.7, pick, torch.randint(0, V, (x.size(0),)))
        y.append(tok); prev = tok
    return torch.stack(y, 1)


TRANS = torch.full((V, V), 0.2 / (V - 2)).scatter_(1, torch.stack([torch.randperm(V)[:2] for _ in range(V)]), 0.4)


def pretrain_batch(B):
    """D_pretrain: sequences from a hidden Markov chain, unrelated to the instruction task."""
    w, out = torch.randint(0, V, (B,)), []
    for _ in range(CTX):
        out.append(w); w = torch.multinomial(TRANS[w], 1).squeeze(1)
    return torch.stack(out, 1)


def lm_nll(policy, seq):
    logits = policy(seq[:, :-1])
    return F.cross_entropy(logits.reshape(-1, V), seq[:, 1:].reshape(-1))


prompts = lambda B: torch.randint(0, V, (B, P_LEN))


# ───────────────────────── the pipeline ─────────────────────────
def train_reward_model(sft, n_prompts=3072, K=4, epochs=2):
    """Step 2: rank K samples per prompt with r*, put all C(K,2) pairs of a prompt in one batch (3.5); the paper trains
    a single epoch on 33k prompts, the toy sees its 3k prompts twice."""
    rm = copy.deepcopy(sft); rm.head = nn.Linear(rm.head.in_features, 1)       # replace the unembedding by a scalar head
    opt = torch.optim.Adam(rm.parameters(), lr=1e-3)
    pairs = torch.combinations(torch.arange(K))                                   # the C(K,2) = 6 index pairs
    data = []
    for _ in range(n_prompts // 16):
        x = prompts(16)
        ys = torch.stack([sample(sft, x) for _ in range(K)], 1)                   # (16, K, T)
        rank = r_star(x.repeat_interleave(K, 0), ys.view(-1, T_LEN)).view(16, K).argsort(1, descending=True)
        data.append((x, ys.gather(1, rank[..., None].expand(-1, -1, T_LEN))))    # sorted best -> worst by the labeler
    for x, ys in data * epochs:
        r = scalar(rm, x.repeat_interleave(K, 0), ys.view(-1, T_LEN))[:, -1].view(16, K)
        r_w, r_l = r[:, pairs[:, 0]], r[:, pairs[:, 1]]                            # y_w preferred over y_l
        loss = -F.logsigmoid(r_w - r_l).mean()                                     # -1/C(K,2) E[log sigma(r(x,y_w) - r(x,y_l))]
        opt.zero_grad(); loss.backward(); opt.step()
    x = prompts(256)
    with torch.no_grad():                                                          # shift so demonstrations score 0 on average
        rm.head.bias -= scalar(rm, x, demonstrations(x))[:, -1].mean()
        y1, y2 = sample(sft, x), sample(sft, x)
        agree = ((scalar(rm, x, y1)[:, -1] > scalar(rm, x, y2)[:, -1]) == (r_star(x, y1) > r_star(x, y2))).float().mean()
    print(f"step 2  reward model: last pairwise loss {loss.item():.3f}, agreement with the labeler on fresh pairs {agree:.2f}")
    return rm


def gae(rewards, values, lam=0.95):
    """Generalised advantage estimation with discount 1 over the T completion tokens; V(terminal) = 0."""
    adv, last = torch.zeros_like(rewards), torch.zeros(rewards.size(0))
    for t in reversed(range(rewards.size(1))):
        delta = rewards[:, t] + (values[:, t + 1] if t + 1 < rewards.size(1) else 0) - values[:, t]
        last = delta + lam * last
        adv[:, t] = last
    return adv, adv + values[:, : rewards.size(1)]


def ppo(sft, rm, gamma_ptx, beta=0.05, iters=60, B=64, clip=0.2, tag=""):
    """Step 3 (3.5): policy pi_RL from the SFT model, value function from the RM, per-token reward
    -beta log(pi_RL / pi_SFT) plus r_theta at the last token, clipped PPO surrogate, plus gamma * log pi_RL on D_pretrain."""
    policy, value = copy.deepcopy(sft), copy.deepcopy(rm)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-4)
    for it in range(1, iters + 1):
        x = prompts(B)
        y = sample(policy, x)
        with torch.no_grad():
            logp_old, logp_ref = completion_logprobs(policy, x, y), completion_logprobs(sft, x, y)
            kl = logp_old - logp_ref                                               # per-token log(pi_RL / pi_SFT)
            rewards = -beta * kl
            rewards[:, -1] += scalar(rm, x, y)[:, -1]                              # r_theta(x, y) once, at the end
            adv, returns = gae(rewards, scalar(value, x, y))
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(2):                                                         # PPO epochs
            ratio = (completion_logprobs(policy, x, y) - logp_old).exp()
            pg_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            v_loss = 0.5 * (scalar(value, x, y)[:, :-1] - returns).pow(2).mean()
            ptx_loss = lm_nll(policy, pretrain_batch(B))                            # -gamma E_{x~D_pretrain}[log pi_RL(x)]
            loss = pg_loss + 0.5 * v_loss + gamma_ptx * ptx_loss
            opt.zero_grad(); loss.backward(); opt.step()
        if it % 15 == 0 or it == 1:
            print(f"  [{tag}] iter {it:2d}  RM reward {scalar(rm, x, y)[:, -1].mean():6.3f}  true r* {r_star(x, y).mean():.3f}"
                  f"  KL(pi_RL || pi_SFT) {kl.sum(1).mean():.3f} nats  pretrain NLL {ptx_loss.item():.3f}")
    return policy


def main():
    t0 = time.time()
    base = LM()
    opt = torch.optim.Adam(base.parameters(), lr=1e-3)
    for _ in range(300):                                                           # Step 0: pre-training on D_pretrain
        loss = lm_nll(base, pretrain_batch(64)); opt.zero_grad(); loss.backward(); opt.step()
    print(f"step 0  pre-trained LM: NLL on D_pretrain {loss.item():.3f}")
    sft = copy.deepcopy(base)
    opt = torch.optim.Adam(sft.parameters(), lr=1e-3)
    for _ in range(300):                                                           # Step 1: SFT on demonstrations
        x = prompts(64)
        loss = -completion_logprobs(sft, x, demonstrations(x)).mean(); opt.zero_grad(); loss.backward(); opt.step()
    x = prompts(512)
    r_sft, r_demo = r_star(x, sample(sft, x)).mean().item(), r_star(x, demonstrations(x)).mean().item()
    print(f"step 1  SFT: NLL of demonstrations {loss.item():.3f}, true r* of SFT samples {r_sft:.3f} (demonstrations {r_demo:.3f})")
    rm = train_reward_model(sft)
    print("step 3  PPO-ptx (gamma = 0.1) and plain PPO (gamma = 0) from the SFT model")
    rl = ppo(sft, rm, gamma_ptx=0.1, tag="PPO-ptx")
    rl_pure = ppo(sft, rm, gamma_ptx=0.0, tag="PPO    ")
    x = prompts(1000)
    with torch.no_grad():
        res = {}
        for name, pol in (("SFT", sft), ("PPO-ptx", rl), ("PPO", rl_pure)):
            y = sample(pol, x)
            res[name] = (r_star(x, y).mean().item(), (completion_logprobs(pol, x, y) - completion_logprobs(sft, x, y)).sum(1).mean().item(),
                         lm_nll(pol, pretrain_batch(1000)).item())
    print(f"{'model':8s} {'true r*':>8s} {'KL to SFT':>10s} {'NLL on D_pretrain (alignment tax)':>34s}")
    for name, (r, kl, nll) in res.items():
        print(f"{name:8s} {r:8.3f} {kl:10.2f} {nll:34.3f}")
    print(f"({time.time() - t0:.1f} s)")
    r_sft, r_ptx, r_ppo = res["SFT"][0], res["PPO-ptx"][0], res["PPO"][0]
    assert r_ppo > r_sft + 0.08 and r_ptx > r_sft + 0.03, "RL did not raise the labelers' true preference"
    assert res["PPO"][1] < 8 and res["PPO-ptx"][2] < res["PPO"][2] - 0.2, "expected a bounded KL and a smaller alignment tax with the pre-training mix"


if __name__ == "__main__":
    main()
