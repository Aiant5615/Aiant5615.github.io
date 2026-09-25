"""DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025) — toy implementation.

What is implemented (section / figure numbers follow the paper):
  * DeepSeek-R1-Zero: GRPO straight on a pretrained base model with NO supervised data and NO reward model, only
    rule-based rewards: an accuracy reward (is the final answer right?) and a format reward (is the reasoning enclosed
    in <think> ... </think> before the answer?), summed                                              (Section 2.2)
  * GRPO as in DeepSeekMath: group-normalised advantages, clipped ratio, k3 KL estimate to the reference (Section 2.2.1)
  * the template of Table 1: the prompt only asks the model to think first, nothing says how; the model discovers on
    its own that writing intermediate steps pays (Figure 1a: accuracy rises; Figure 1b: response length grows)
  * rejection sampling + SFT (stages 3 of Section 2.3): keep the correct, well-formatted samples of the RL model
  * distillation vs RL on a small model (Section 4.1): SFT of a small model on the large RL model's samples beats
    running the same RL on the small model directly
Simplifications: the "base model" is a 2-layer Transformer pretrained on synthetic multi-step arithmetic written in
several styles (with and without steps, with and without <think> tags), as in the chain-of-thought file; the small
model has width 64 instead of 96 and less pretraining; the cold start, the language-consistency reward and the final preference-RL stage are not
implemented (the toy has one language and no human preferences).

Run:  python deepseek_r1.py        (CPU, about 60 s)
"""
import copy, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ───────────────────────── the arithmetic world (same as the chain-of-thought file) ─────────────────────────
MAX_VAL = 24
PLUS, MINUS, EQ, Q, A, ANS, SEP, END, THINK, ENDTHINK, PAD = range(MAX_VAL + 1, MAX_VAL + 12)
VOCAB = PAD + 1
CTX = 96
FORMATS = ["standard", "steps", "think", "answer-then-steps"]


def make_problem(k):
    while True:
        nums = torch.randint(0, 7, (k + 1,)).tolist()
        ops = torch.randint(0, 2, (k,)).tolist()
        res, r = [], nums[0]
        for n, o in zip(nums[1:], ops):
            r = r + n if o == 0 else r - n
            res.append(r)
        if all(0 <= x <= MAX_VAL for x in res):
            return nums, ops, res


def question(nums, ops):
    toks = [Q, nums[0]]
    for n, o in zip(nums[1:], ops):
        toks += [PLUS if o == 0 else MINUS, n]
    return toks + [EQ, A]


def steps(nums, ops, res):
    toks, prev = [], nums[0]
    for n, o, r in zip(nums[1:], ops, res):
        toks += [prev, PLUS if o == 0 else MINUS, n, EQ, r, SEP]
        prev = r
    return toks


def render(nums, ops, res, fmt):
    """Pretraining styles: 'think' is the paper's template (reasoning between the tags, then the answer)."""
    q, chain, ans = question(nums, ops), steps(nums, ops, res), [ANS, res[-1]]
    body = {"standard": ans, "steps": chain + ans, "think": [THINK] + chain + [ENDTHINK] + ans,
            "answer-then-steps": ans + [SEP] + chain}[fmt]
    return q + body + [END]


def make_documents(B, n_examples=2):
    docs, longest = torch.full((B, CTX), PAD), 0
    for i in range(B):
        fmt = FORMATS[torch.randint(len(FORMATS), (1,)).item()]
        toks = []
        for _ in range(n_examples):
            toks += render(*make_problem(torch.randint(2, 5, (1,)).item()), fmt)
        docs[i, : len(toks)] = torch.tensor(toks[:CTX]); longest = max(longest, len(toks))
    return docs[:, :longest]


# ───────────────────────── the model ─────────────────────────
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


class LM(nn.Module):
    """Decoder-only Transformer without position embeddings (causal attention gives position implicitly)."""
    def __init__(self, d, h, n_layers):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, VOCAB, bias=False)

    def forward(self, seq):
        x = self.tok(seq)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


def pretrain(model, steps, batch=32, lr=3e-3, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    for step in range(1, steps + 1):
        docs = make_documents(batch)
        loss = F.cross_entropy(model(docs[:, :-1]).reshape(-1, VOCAB), docs[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % (steps // 2) == 0:
            print(f"  [{tag}] pretraining step {step:4d}  loss {loss.item():.3f}")


# ───────────────────────── prompts, sampling and the rule-based rewards (Section 2.2.2) ─────────────────────────
K_OPS, MAX_NEW = 3, 26


def prompts(B):
    """Table 1's template, zero-shot: the question, then 'A:' — the model must decide itself what to write next."""
    ps, golds = [], []
    for _ in range(B):
        nums, ops, res = make_problem(K_OPS)
        ps.append(question(nums, ops)); golds.append(res[-1])
    return torch.tensor(ps), torch.tensor(golds)


@torch.no_grad()
def sample(model, q, greedy=False):
    """Up to MAX_NEW response tokens after the prompt; a sequence is frozen (PAD) after it emits END."""
    seq, done = q, torch.zeros(q.size(0), dtype=torch.bool)
    for _ in range(MAX_NEW):
        logits = model(seq)[:, -1]
        nxt = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
        seq = torch.cat([seq, nxt[:, None]], 1)
        done |= nxt == END
        if done.all():
            break
    return seq[:, q.size(1):]


def rule_rewards(o, gold):
    """No reward model. Accuracy reward: the number after ANS equals the gold answer. Format reward: the response starts
    with <think>, closes it, and only then answers. Returns (accuracy, format) per response."""
    acc, fmt = torch.zeros(o.size(0)), torch.zeros(o.size(0))
    for i, toks in enumerate(o.tolist()):
        toks = toks[: toks.index(END)] if END in toks else toks
        if ANS in toks:
            j = toks.index(ANS)
            acc[i] = float(j + 1 < len(toks) and toks[j + 1] == gold[i].item())
            fmt[i] = float(toks[:1] == [THINK] and ENDTHINK in toks and toks.index(ENDTHINK) < j and toks.count(THINK) == 1)
    return acc, fmt


def response_length(o):
    return ((o != PAD) & (o != END)).float().sum(1)


# ───────────────────────── GRPO (DeepSeekMath, Section 2.2.1) ─────────────────────────
def token_log_probs(model, q, o):
    logits = model(torch.cat([q, o], 1)[:, :-1])[:, q.size(1) - 1:]
    return logits.log_softmax(-1).gather(-1, o[..., None]).squeeze(-1)


def grpo_step(policy, ref, opt, B=12, G=8, eps=0.2, beta=0.04, w_format=0.2):
    """One GRPO iteration: G samples per question, r = accuracy + w_format * format, group-normalised advantages,
    clipped objective, k3 KL to the frozen reference; one gradient step per batch (μ = 1, so ρ = 1 at the update)."""
    q, gold = prompts(B)
    q, gold = q.repeat_interleave(G, 0), gold.repeat_interleave(G, 0)
    o = sample(policy, q)
    acc, fmt = rule_rewards(o, gold)
    r = acc + w_format * fmt
    rg = r.view(B, G)
    adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-4)).view(-1)
    mask = (o != PAD).float()
    logp = token_log_probs(policy, q, o)
    with torch.no_grad():
        logp_old, logp_ref = logp.detach(), token_log_probs(ref, q, o)
    ratio = (logp - logp_old).exp()
    surr = torch.min(ratio * adv[:, None], ratio.clamp(1 - eps, 1 + eps) * adv[:, None])
    kl = (logp_ref - logp).exp() - (logp_ref - logp) - 1                    # k3 estimator, >= 0
    loss = -((surr - beta * kl) * mask).sum(1).div(mask.sum(1)).mean()
    opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
    return acc.mean().item(), fmt.mean().item(), response_length(o).mean().item()


def rl(base, iters, lr, tag):
    """R1-Zero: RL from the base model with the rule rewards. Prints Figure 1's two curves."""
    policy, ref = copy.deepcopy(base), copy.deepcopy(base).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for it in range(1, iters + 1):
        acc, fmt, length = grpo_step(policy, ref, opt)
        if it == 1 or it % (iters // 4) == 0:
            print(f"  [{tag}] iteration {it:3d}  accuracy {acc:.2f}  format {fmt:.2f}  response length {length:5.1f}")
    return policy


@torch.no_grad()
def evaluate(model, n=256):
    q, gold = prompts(n)
    o = sample(model, q, greedy=True)
    acc, fmt = rule_rewards(o, gold)
    return acc.mean().item(), fmt.mean().item(), response_length(o).mean().item()


def rejection_sample(model, n_prompts, G=4):
    """Stage 3: sample the RL model, keep only correct and well-formatted responses (the SFT / distillation data)."""
    q, gold = prompts(n_prompts)
    q, gold = q.repeat_interleave(G, 0), gold.repeat_interleave(G, 0)
    o = sample(model, q)
    acc, fmt = rule_rewards(o, gold)
    keep = (acc * fmt) > 0
    return torch.cat([q, o], 1)[keep], keep.float().mean().item()


def sft(model, data, epochs=8, lr=1e-3):
    """Supervised fine-tuning on the kept samples (distillation when `model` is the small model)."""
    model = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        for idx in torch.randperm(data.size(0)).split(64):
            seq = data[idx]
            loss = F.cross_entropy(model(seq[:, :-1]).reshape(-1, VOCAB), seq[:, 1:].reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def show(toks):
    names = {PLUS: "+", MINUS: "-", EQ: "=", Q: "Q:", A: "A:", ANS: "The answer is", SEP: ";", END: "", THINK: "<think>", ENDTHINK: "</think>", PAD: ""}
    return " ".join(str(t) if t <= MAX_VAL else names[t] for t in toks).strip()


def main():
    t0 = time.time()
    large, small = LM(96, 4, 2), LM(64, 4, 2)
    print("pretraining the base models on the mixed-style arithmetic corpus")
    pretrain(large, 1000, tag="large"); pretrain(small, 600, tag="small")
    e0 = evaluate(large)
    print(f"base model, zero-shot: accuracy {e0[0]:.2f}  format {e0[1]:.2f}  response length {e0[2]:.1f}")
    print("R1-Zero: GRPO on the large base model with rule rewards only (Figure 1a accuracy, 1b response length)")
    r1_zero = rl(large, 50, 3e-4, "large")
    e1 = evaluate(r1_zero)
    q, _ = prompts(1)
    print(f"  after RL: accuracy {e1[0]:.2f}  format {e1[1]:.2f}  length {e1[2]:.1f}   e.g. {show(q[0].tolist())} {show(sample(r1_zero, q, greedy=True)[0].tolist())}")
    print("Section 4.1: distillation vs RL on the small model")
    data, keep = rejection_sample(r1_zero, 768)
    print(f"  rejection sampling kept {keep:.2f} of the RL model's samples -> {data.size(0)} SFT examples")
    small_rl = rl(small, 30, 3e-4, "small")
    small_distilled = sft(small, data)
    es0, es_rl, es_d = evaluate(small), evaluate(small_rl), evaluate(small_distilled)
    print(f"  small base {es0[0]:.2f}   small + RL {es_rl[0]:.2f}   small distilled from R1-Zero {es_d[0]:.2f}   ({time.time() - t0:.1f} s)")
    assert e1[0] > e0[0] + 0.3 and e1[1] > 0.8, "RL with rule rewards should make the model think and answer correctly"
    assert e1[2] > e0[2] + 5, "the response length should grow (the model chooses to think)"
    assert es_d[0] > es_rl[0] + 0.1, "distillation should beat RL on the small model"


if __name__ == "__main__":
    main()
