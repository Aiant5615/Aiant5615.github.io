"""Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (Wei et al., 2022) — toy implementation.

What is implemented (section / figure numbers follow the paper):
  * a small decoder-only LM pretrained on a "corpus" of multi-step arithmetic problems written in the formats found
    in the wild: answer only, worked steps then answer (a chain of thought), answer then steps, filler then answer
  * few-shot prompting with NO training at test time: k exemplars in one format followed by the test question, greedy
    decoding, the number after "The answer is" is extracted                                    (Section 2, Figure 1)
  * standard prompting vs chain-of-thought prompting, at two model sizes: the gain from chain-of-thought appears only
    for the larger model (emergence with scale, Section 3.2, Figure 4)
  * the ablations of Figure 5: "reasoning after answer" and "variable compute only" (dots of the same length as the
    chain) do not help; only steps written BEFORE the answer do                                    (Section 3.3)
  * length generalisation (Section 5): exemplars with 3 operations, test questions with 5; with a chain of thought the
    model solves longer problems than it was shown, standard prompting collapses
Simplifications: the "web corpus" is synthetic arithmetic (a + b - c ... on small integers) and the rationale is the
chain of intermediate sums instead of English; two tiny Transformers (0.02M and 0.25M parameters, no position embedding) stand in for the
8B-540B models; "equation only" is not ablated separately because the toy chain already is an equation.

Run:  python chain_of_thought.py        (CPU, about 50 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ───────────────────────── vocabulary and the arithmetic "corpus" ─────────────────────────
MAX_VAL = 24                                         # running totals stay in 0..MAX_VAL
NUM0 = 0                                             # tokens 0..MAX_VAL are the numbers
PLUS, MINUS, EQ, Q, A, ANS, SEP, END, DOT, PAD = range(MAX_VAL + 1, MAX_VAL + 11)
VOCAB = PAD + 1
FORMATS = ["standard", "chain-of-thought", "answer-then-reasoning", "variable-compute"]
CTX = 100


def make_problem(k):
    """k operations on numbers 0..6, e.g. 3 + 4 - 2; returns (numbers, ops, intermediate results)."""
    while True:
        nums = torch.randint(0, 7, (k + 1,)).tolist()
        ops = torch.randint(0, 2, (k,)).tolist()                 # 0: +, 1: -
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
    """The chain of thought: "3 + 4 = 7 ; 7 - 2 = 5 ;" — one intermediate result per operation."""
    toks, prev = [], nums[0]
    for n, o, r in zip(nums[1:], ops, res):
        toks += [prev, PLUS if o == 0 else MINUS, n, EQ, r, SEP]
        prev = r
    return toks


def render(nums, ops, res, fmt):
    """One ⟨question, rationale, answer⟩ example in one of the four formats of Figure 5."""
    q, chain, ans = question(nums, ops), steps(nums, ops, res), [ANS, res[-1]]
    if fmt == "standard":
        body = ans
    elif fmt == "chain-of-thought":
        body = chain + ans
    elif fmt == "answer-then-reasoning":
        body = ans + [SEP] + chain
    else:                                                        # variable compute: dots, as many as the chain
        body = [DOT] * len(chain) + ans
    return q + body + [END]


def make_documents(B, n_examples=2):
    """Pretraining documents: n_examples problems with 2-4 operations, written in ONE format each (like a web page)."""
    docs = torch.full((B, CTX), PAD)
    longest = 0
    for i in range(B):
        fmt = FORMATS[torch.randint(len(FORMATS), (1,)).item()]
        toks = []
        for _ in range(n_examples):
            toks += render(*make_problem(torch.randint(2, 5, (1,)).item()), fmt)
        docs[i, : len(toks)] = torch.tensor(toks[:CTX])
        longest = max(longest, len(toks))
    return docs[:, :longest]


# ───────────────────────── the language model ─────────────────────────
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
    """Decoder-only Transformer LM. Nothing about it is changed for chain-of-thought: only the prompt differs.
    No position embedding: causal attention gives the position implicitly, and unlike learned absolute positions this
    lets the toy model continue chains beyond the lengths seen in training (Section 5)."""
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
    """Plain next-token prediction on the mixed-format corpus (the model is never told which format is 'right')."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    for step in range(1, steps + 1):
        docs = make_documents(batch)
        logits = model(docs[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), docs[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % (steps // 4) == 0:
            print(f"  [{tag}] step {step:4d}  loss {loss.item():.3f}")


# ───────────────────────── few-shot prompting and evaluation ─────────────────────────
@torch.no_grad()
def answer(model, prompts, max_new=40):
    """Greedy decoding until END (prompts of one format and size have equal length, so they decode as one batch);
    the prediction is the number after ANS ("The answer is ..."), else -1."""
    seq = torch.tensor(prompts)
    n0 = seq.shape[1]
    for _ in range(max_new):
        if seq.shape[1] >= CTX:
            break
        nxt = model(seq)[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat([seq, nxt], 1)
        if (seq[:, n0:] == END).any(1).all():
            break
    preds = []
    for out in seq[:, n0:].tolist():
        out = out[: out.index(END)] if END in out else out
        i = out.index(ANS) if ANS in out else -1
        preds.append(out[i + 1] if 0 <= i and i + 1 < len(out) and out[i + 1] <= MAX_VAL else -1)
    return preds


@torch.no_grad()
def accuracy(model, fmt, k_shot=1, k_exemplar=3, k_test=3, n=100):
    """Few-shot prompt: k_shot exemplars with k_exemplar operations in format fmt, then a k_test-operation question."""
    model.eval()
    prompts, golds = [], []
    for _ in range(n):
        prompt = []
        for _ in range(k_shot):
            prompt += render(*make_problem(k_exemplar), fmt)
        nums, ops, res = make_problem(k_test)
        prompts.append(prompt + question(nums, ops)); golds.append(res[-1])
    preds = answer(model, prompts)
    return sum(p == g for p, g in zip(preds, golds)) / n


def main():
    t0 = time.time()
    torch.manual_seed(0)
    models = {"small (1 layer, d=32)": LM(32, 2, 1), "large (2 layers, d=96)": LM(96, 4, 2)}
    for name, m in models.items():
        print(f"pretraining {name}: {sum(p.numel() for p in m.parameters()):,} parameters")
        pretrain(m, 500 if "small" in name else 1200, tag=name.split()[0])
    print("\nFigure 4 / Figure 5: accuracy on 3-step problems by prompt format (1 exemplar, greedy decoding)")
    print(f"{'model':26s}" + "".join(f"{f:>24s}" for f in FORMATS))
    acc = {}
    for name, m in models.items():
        acc[name] = {f: accuracy(m, f) for f in FORMATS}
        print(f"{name:26s}" + "".join(f"{acc[name][f]:24.2f}" for f in FORMATS))
    large, small = models["large (2 layers, d=96)"], models["small (1 layer, d=32)"]
    print("\nSection 5, length generalisation: one 3-operation exemplar, test questions with 5 operations (never seen in training)")
    lg = {f: accuracy(large, f, k_shot=1, k_test=5) for f in ("standard", "chain-of-thought")}
    print(f"  large model: standard {lg['standard']:.2f}   chain-of-thought {lg['chain-of-thought']:.2f}")
    nums, ops, res = make_problem(3)
    prompt = render(*make_problem(3), "chain-of-thought") + question(nums, ops)
    seq = torch.tensor(prompt)[None]
    with torch.no_grad():
        for _ in range(30):
            nxt = large(seq)[0, -1].argmax(); seq = torch.cat([seq, nxt.view(1, 1)], 1)
            if nxt.item() == END:
                break
    names = {PLUS: "+", MINUS: "-", EQ: "=", Q: "Q:", A: "A:", ANS: "The answer is", SEP: ";", END: "", DOT: "."}
    show = lambda toks: " ".join(str(t) if t <= MAX_VAL else names[t] for t in toks)
    print(f"  example  {show(question(nums, ops))}  ->  {show(seq[0, len(prompt):].tolist())}   ({time.time() - t0:.1f} s)")
    gl, gs = acc["large (2 layers, d=96)"], acc["small (1 layer, d=32)"]
    assert gl["chain-of-thought"] > gl["standard"] + 0.3, "chain-of-thought should help the large model a lot"
    assert gs["chain-of-thought"] - gs["standard"] < gl["chain-of-thought"] - gl["standard"] - 0.2, "the gain should be much smaller for the small model (emergence)"
    assert gl["answer-then-reasoning"] < gl["chain-of-thought"] - 0.3 and gl["variable-compute"] < gl["chain-of-thought"] - 0.3, "ablations should not help"
    assert lg["chain-of-thought"] > lg["standard"] + 0.15, "chain-of-thought should generalise to longer problems better than standard prompting"


if __name__ == "__main__":
    main()
