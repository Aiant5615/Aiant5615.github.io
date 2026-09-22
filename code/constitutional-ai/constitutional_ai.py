"""Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022) — toy implementation.

What is implemented (section numbers follow the paper):
  * a helpful-only RLHF model: preference model on human helpfulness comparisons + PPO, which complies with
    harmful prompts because nothing penalises harm                                                     (Section 1, 3)
  * Stage 1, SL-CAI: sample responses to red-team prompts, critique them under a randomly drawn principle of the
    constitution, revise, repeat; fine-tune the pretrained model on the final revisions plus helpful data; Figure 5:
    the harmlessness score rises with the number of revisions, most of the gain in the first             (Section 3)
  * Stage 2, RL-CAI: a feedback model labels pairs of SL-CAI responses "which is less harmful?" with a principle,
    giving SOFT labels (normalised probabilities of A vs B); chain-of-thought labels are sharper and are clamped to
    40-60 %; the preference model is trained on the union of AI harmlessness labels and human helpfulness labels with
    the comparison loss on soft targets; PPO against it                                                  (Section 4)
  * the standard HH-RLHF baseline: human helpfulness + human harmlessness labels (where a refusal counts as harmless),
    which is what makes RLHF models evasive                                                             (Section 4.3)
  * evaluation on hidden true scores: helpfulness, harmfulness and evasiveness of every model (Figures 2-3, Section 4.3)
Simplifications: "responses" are 6 tokens from a 10-word vocabulary whose words are content, harmful content,
refusals, explanations or filler; the policy is a tabular LM; critique and revision are the model's edits implemented
as rules over the flagged words (each pass fixes only part of what it flags); the feedback model is a noisy judge with
access to the true harm score, sharper with chain of thought; one PPO action = one whole response.

Run:  python constitutional_ai.py        (CPU, about 30 s)
"""
import copy, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
n, V = 6, 10                                                  # response length, vocabulary
CONTENT, HARM, REFUSE, EXPLAIN, FILLER = 0, 1, 2, 3, 4
CLASS = torch.tensor([CONTENT, CONTENT, CONTENT, HARM, HARM, REFUSE, EXPLAIN, EXPLAIN, FILLER, FILLER])
BENIGN, HARMFUL = 0, 1                                        # the two kinds of prompt


def count(y, cls):
    return (CLASS[y] == cls).float().sum(1)


# ───────────────────────── hidden ground truth used by the human labelers and the evaluation ─────────────────────────
def true_helpfulness(x, y):
    """Benign prompts want content; a harmful request is 'helped' by complying (harmful content) or, less, by a
    thoughtful explanation. Refusals and filler help nobody."""
    return torch.where(x == BENIGN, count(y, CONTENT), count(y, HARM) + 0.5 * count(y, EXPLAIN))


def true_harm(x, y):
    return torch.where(x == HARMFUL, count(y, HARM), torch.zeros_like(y[:, 0], dtype=torch.float))


def true_evasiveness(x, y):
    return torch.where(x == HARMFUL, count(y, REFUSE), torch.zeros_like(y[:, 0], dtype=torch.float))


def human_choice(score0, score1, temp=0.5):
    """Crowdworkers pick response 0 with probability sigma((s0 - s1)/temp): hard labels, ~80 % agreement."""
    return (torch.rand_like(score0) > torch.sigmoid((score0 - score1) / temp)).long()     # 0 if they chose y0


# ───────────────────────── policy, preference model, PPO (the RLHF machinery of the earlier papers) ─────────────────────────
class TabularLM(nn.Module):
    """Autoregressive LM over integer tokens: logits of y_t depend on the prompt kind x, the position t and y_{t-1}."""
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(0.1 * torch.randn(2, n, V + 1, V))     # previous-token index V = "start"

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


class PreferenceModel(nn.Module):
    """r(x, y): linear in the word-class counts of y, with separate weights for each kind of prompt."""
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(2 * 5, 1)

    def forward(self, x, y):
        counts = F.one_hot(CLASS[y], 5).float().sum(1) / n
        feats = (F.one_hot(x, 2).float()[:, :, None] * counts[:, None, :]).reshape(-1, 10)
        return self.net(feats).squeeze(-1)


def train_pm(pairs, epochs=4):
    """Comparison loss with SOFT targets p = P(y0 preferred): -[p log sigma(d) + (1-p) log sigma(-d)], d = r(y0)-r(y1).
    Human labels are p in {0, 1}; the AI feedback labels of Section 4 are probabilities."""
    pm = PreferenceModel()
    x, y0, y1, p = (torch.cat(t) for t in zip(*pairs))
    opt = torch.optim.Adam(pm.parameters(), lr=3e-2)
    for _ in range(epochs):
        for idx in torch.randperm(x.size(0)).split(64):
            d = pm(x[idx], y0[idx]) - pm(x[idx], y1[idx])
            loss = -(p[idx] * F.logsigmoid(d) + (1 - p[idx]) * F.logsigmoid(-d)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return pm


def prompts(B):
    return torch.randint(0, 2, (B,))


def ppo(init, pm, beta=0.05, steps=120, B=256, K=4, M=64, eps=0.2, lr=1e-2):
    """PPO against the preference model with a KL penalty to the initial policy (as in the RLHF papers)."""
    policy, value = copy.deepcopy(init), nn.Parameter(torch.zeros(2))
    opt = torch.optim.Adam(list(policy.parameters()) + [value], lr=lr)
    for _ in range(steps):
        x = prompts(B)
        y = policy.sample(x)
        with torch.no_grad():
            logp_old = policy.log_prob(x, y)
            R = pm(x, y) - beta * (logp_old - init.log_prob(x, y))
        for _ in range(K):
            for idx in torch.randperm(B).split(M):
                ratio = (policy.log_prob(x[idx], y[idx]) - logp_old[idx]).exp()
                adv = R[idx] - value[x[idx]].detach()
                l_clip = torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
                loss = -(l_clip - 0.5 * (value[x[idx]] - R[idx]).pow(2).mean())
                opt.zero_grad(); loss.backward(); opt.step()
    return policy


def human_helpfulness_pairs(policy, N, benign_only=False):
    """Human helpfulness comparisons. The helpful-only model is trained on comparisons for every kind of request (so it
    learns to comply with harmful ones); the HH and CAI preference models take the helpfulness data from ordinary
    prompts and get their harmful-prompt signal from the harmlessness labels only (Section 4)."""
    x = torch.full((N,), BENIGN) if benign_only else prompts(N); y0, y1 = policy.sample(x), policy.sample(x)
    return x, y0, y1, 1.0 - human_choice(true_helpfulness(x, y0), true_helpfulness(x, y1)).float()


def human_harmlessness_pairs(policy, N):
    """Standard HH-RLHF harmlessness labels: crowdworkers judging harmlessness alone prefer the response that refuses
    outright over one that engages with the topic (the source of evasiveness the paper points at, Sections 1, 4.3)."""
    x = torch.full((N,), HARMFUL); y0, y1 = policy.sample(x), policy.sample(x)
    judge = lambda y: -true_harm(x, y) + 0.3 * true_evasiveness(x, y) - 0.3 * count(y, EXPLAIN)
    return x, y0, y1, 1.0 - human_choice(judge(y0), judge(y1)).float()


# ───────────────────────── Stage 1: critiques and revisions (Section 3) ─────────────────────────
CONSTITUTION = {
    "harmful":  ("Identify ways the response is harmful, unethical or dangerous.", HARM,   EXPLAIN),
    "evasive":  ("Identify ways the response is evasive or preachy instead of engaging.", REFUSE, EXPLAIN),
    "unhelpful": ("Identify parts of the response that are filler and do not help.",   FILLER, CONTENT),
}


BAD = torch.tensor([CONSTITUTION[k][1] for k in CONSTITUTION])
GOOD = torch.tensor([CONSTITUTION[k][2] for k in CONSTITUTION])
P_PRINCIPLE = torch.tensor([0.6, 0.2, 0.2])                   # most of the constitution is about harm


def critique(y, principle, p_flag=0.8):
    """The model, prompted with the principle drawn for each response, flags the offending words; like a real critique
    it misses some of them (p_flag)."""
    return (CLASS[y] == BAD[principle][:, None]) & (torch.rand(y.shape) < p_flag)


def revise(y, flags, principle):
    """The model rewrites the flagged words in the spirit of the principle (harm -> explanation, refusal ->
    explanation, filler -> content); the rest of the response is kept."""
    words = torch.stack([torch.tensor([i for i in range(V) if CLASS[i] == g][:2]) for g in GOOD])   # 2 words per class
    replacement = words[principle[:, None], torch.randint(0, 2, y.shape)]
    return torch.where(flags, replacement, y)


def critique_revision(x, y, rounds):
    """Repeat critique -> revision `rounds` times, drawing a principle per response each round; benign prompts are
    left alone."""
    for _ in range(rounds):
        principle = torch.multinomial(P_PRINCIPLE, y.size(0), replacement=True)
        y_new = revise(y, critique(y, principle), principle)
        y = torch.where((x == HARMFUL)[:, None], y_new, y)
    return y


def sft(pretrained, x, y, epochs=30):
    """Supervised fine-tuning of the pretrained LM on the revisions (harmful prompts) and helpful samples (benign)."""
    model = copy.deepcopy(pretrained)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)
    for _ in range(epochs):
        for idx in torch.randperm(x.size(0)).split(128):
            loss = -model.log_prob(x[idx], y[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


# ───────────────────────── Stage 2: AI feedback (Section 4) ─────────────────────────
def ai_feedback_pairs(policy, N, cot):
    """The feedback model reads "Which response is less harmful? (A) ... (B) ..." under a random principle and returns the
    normalised probability of (A). Without chain of thought its judgement is a blurry function of the true harm; with
    chain of thought it is sharper, and the near-0/1 labels are clamped to [0.4, 0.6] as in the paper."""
    x = torch.full((N,), HARMFUL); y0, y1 = policy.sample(x), policy.sample(x)
    score = lambda y: -true_harm(x, y) - 0.5 * true_evasiveness(x, y) + 0.5 * count(y, EXPLAIN)   # what the constitution asks for
    noise = 0.3 if cot else 1.0
    p = torch.sigmoid((score(y0) - score(y1) + noise * torch.randn(N)) / (0.5 if cot else 2.0))
    return x, y0, y1, p.clamp(0.4, 0.6) if cot else p


@torch.no_grad()
def evaluate(policy, N=4000):
    x = prompts(N); y = policy.sample(x)
    b, h = x == BENIGN, x == HARMFUL
    return true_helpfulness(x, y)[b].mean().item(), true_harm(x, y)[h].mean().item(), true_evasiveness(x, y)[h].mean().item()


def main():
    t0 = time.time()
    pretrained = TabularLM()
    # helpful-only RLHF (the starting point of the paper)
    pm_help = train_pm([human_helpfulness_pairs(pretrained, 4000)])
    helpful = ppo(pretrained, pm_help)
    # standard HH-RLHF baseline: human helpfulness + human harmlessness labels
    pm_hh = train_pm([human_helpfulness_pairs(helpful, 4000, benign_only=True), human_harmlessness_pairs(helpful, 4000)])
    hh_rlhf = ppo(helpful, pm_hh)
    # Stage 1: SL-CAI
    x_red = torch.full((2000,), HARMFUL)
    y0 = helpful.sample(x_red)
    print("Figure 5: harmlessness of the revisions (score of a PM trained on AI feedback), by number of critique-revision rounds")
    pm_ai_probe = train_pm([ai_feedback_pairs(helpful, 4000, cot=True)])
    scores = []
    for rounds in range(5):
        y_r = critique_revision(x_red, y0, rounds)
        scores.append(pm_ai_probe(x_red, y_r).mean().item())
        print(f"  {rounds} revisions: PM score {scores[-1]:+.3f}   harmful words/response {true_harm(x_red, y_r).mean():.2f}")
    revisions = critique_revision(x_red, y0, 4)
    x_benign = torch.full((2000,), BENIGN)
    sl_cai = sft(pretrained, torch.cat([x_red, x_benign]), torch.cat([revisions, helpful.sample(x_benign)]))
    # Stage 2: RL-CAI
    ai_pairs = ai_feedback_pairs(sl_cai, 4000, cot=True)
    print(f"AI feedback labels (chain of thought, clamped): mean |p - 0.5| = {(ai_pairs[3] - 0.5).abs().mean():.3f}; "
          f"without CoT: {(ai_feedback_pairs(sl_cai, 4000, cot=False)[3] - 0.5).abs().mean():.3f}")
    pm_cai = train_pm([human_helpfulness_pairs(sl_cai, 4000, benign_only=True), ai_pairs])
    rl_cai = ppo(sl_cai, pm_cai)
    print(f"\n{'model':22s} {'helpfulness (benign)':>21s} {'harmful words':>14s} {'refusals':>9s}   (harmful prompts, per 6-word response)")
    res = {}
    for name, m in [("pretrained", pretrained), ("helpful-only RLHF", helpful), ("HH RLHF (human labels)", hh_rlhf), ("SL-CAI", sl_cai), ("RL-CAI", rl_cai)]:
        res[name] = evaluate(m)
        print(f"{name:22s} {res[name][0]:21.2f} {res[name][1]:14.2f} {res[name][2]:9.2f}")
    print(f"({time.time() - t0:.1f} s)")
    assert all(scores[k] <= scores[k + 1] + 0.02 for k in range(4)) and scores[1] - scores[0] > 0.4 * (scores[4] - scores[0]), "revisions should raise harmlessness, the first one the most"
    assert res["helpful-only RLHF"][1] > 1.5, "the helpful-only model should comply with harmful requests"
    assert res["RL-CAI"][1] < 0.3 * res["helpful-only RLHF"][1] and res["RL-CAI"][1] < res["HH RLHF (human labels)"][1] + 0.2, "RL-CAI should be harmless"
    assert res["RL-CAI"][2] < 0.5 * res["HH RLHF (human labels)"][2], "RL-CAI should be less evasive than HH RLHF"
    assert res["RL-CAI"][0] > res["HH RLHF (human labels)"][0] - 0.3, "RL-CAI should stay about as helpful"


if __name__ == "__main__":
    main()
