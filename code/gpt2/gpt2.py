"""Language Models are Unsupervised Multitask Learners (GPT-2, Radford et al., 2019) — a from-scratch PyTorch implementation.

What is implemented (section numbers follow the paper):
  * the framing p(output | input, task): the task is written inside the text, the model is trained only as a
    language model p(x) = prod_i p(s_i | s_1..s_{i-1}) and evaluated zero-shot by prompting             (1, 2)
  * byte-level input: the 256 byte values are the vocabulary, so any string is encodable                (2.2)
  * the GPT decoder with the paper's modifications: LayerNorm moved to the input of each sub-block (pre-norm),
    an extra LayerNorm after the final block, residual-path weights scaled by 1/sqrt(N) at initialisation
    (N = number of residual layers), learned positions, tied input/output embeddings                     (2.3)
  * zero-shot task evaluation: condition on "task: input =" and read the greedy continuation up to "\\n"     (3)
  * top-k sampling (k = 40), the decoding used for the paper's generated samples                           (3, Table 13)
Simplifications: 2 blocks of d_model = 64 and a 64-byte context instead of 48 x 1600 / 1024; a synthetic "WebText"
whose lines demonstrate three string tasks (reverse, sort, upper) on 6 letters instead of 40 GB of Reddit links; raw
bytes instead of byte-level BPE (no merges); no dropout; held-out strings never appear in the training text.

Run:  python gpt2.py        (CPU, about 30 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
VOCAB, CTX = 256, 64                       # byte-level vocabulary (2.2), context size


# ───────────────────────── model (2.3) ─────────────────────────
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, h):
        super().__init__()
        self.h, self.d_k = h, d_model // h
        self.c_attn, self.c_proj = nn.Linear(d_model, 3 * d_model), nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = (t.transpose(1, 2) for t in self.c_attn(x).view(B, L, 3, self.h, self.d_k).unbind(2))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), float("-inf"))
        return self.c_proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(B, L, -1))


class Block(nn.Module):
    """Pre-norm block: x + Attn(LN(x)), then x + MLP(LN(x)).  GPT-1 applied LN after the residual sum instead."""
    def __init__(self, d_model, h):
        super().__init__()
        self.ln_1, self.attn = nn.LayerNorm(d_model), CausalSelfAttention(d_model, h)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT2(nn.Module):
    def __init__(self, vocab=VOCAB, ctx=CTX, d_model=64, h=4, n_layer=2):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab, d_model), nn.Embedding(ctx, d_model)
        self.blocks = nn.ModuleList([Block(d_model, h) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)                  # the additional layer norm after the final block
        self.apply(lambda m: isinstance(m, (nn.Linear, nn.Embedding)) and nn.init.normal_(m.weight, std=0.02))
        N = 2 * n_layer                                    # residual layers: one attention + one MLP path per block
        for block in self.blocks:                          # scale residual-path output weights by 1 / sqrt(N)
            nn.init.normal_(block.attn.c_proj.weight, std=0.02 / math.sqrt(N))
            nn.init.normal_(block.mlp[2].weight, std=0.02 / math.sqrt(N))

    def forward(self, idx):
        x = self.wte(idx) + self.wpe(torch.arange(idx.size(1)))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x) @ self.wte.weight.T           # tied output embedding

    @torch.no_grad()
    def generate(self, prompt, max_new=12, top_k=None):
        """Greedy (top_k=None) or top-k sampling; stops at the newline byte, the end of a "document" line."""
        idx = torch.tensor([list(prompt)])
        for _ in range(max_new):
            logits = self(idx[:, -CTX:])[0, -1]
            if top_k is not None:
                thresh = logits.topk(top_k).values[-1]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
                nxt = torch.multinomial(logits.softmax(-1), 1)
            else:
                nxt = logits.argmax().view(1)
            idx = torch.cat([idx, nxt.view(1, 1)], 1)
            if nxt.item() == ord("\n"):
                break
        return bytes(idx[0, len(prompt):].tolist())


# ───────────────────────── synthetic WebText: documents that happen to demonstrate tasks (Section 2) ─────────────────────────
LETTERS, TASKS = "abcdef", ("reverse", "sort", "upper")
SOLVE = {"reverse": lambda s: s[::-1], "sort": lambda s: "".join(sorted(s)), "upper": lambda s: s.upper()}


def random_word():
    n = torch.randint(3, 6, (1,)).item()
    return "".join(LETTERS[i] for i in torch.randint(0, len(LETTERS), (n,)).tolist())


HELD_OUT = set()
while len(HELD_OUT) < 60:
    HELD_OUT.add(random_word())


def make_webtext(n_lines):
    """Lines like "sort: fca = acf\\n": the task, its input and its output, stated in natural text (Section 2)."""
    lines = []
    while len(lines) < n_lines:
        w = random_word()
        if w in HELD_OUT:
            continue
        task = TASKS[torch.randint(0, 3, (1,)).item()]
        lines.append(f"{task}: {w} = {SOLVE[task](w)}\n")
    return torch.tensor(list("".join(lines).encode()))     # one byte stream


def zero_shot_accuracy(model, task):
    """Prompt "task: w =" and compare the greedy continuation with " answer\\n"; no fine-tuning, no examples (Section 3)."""
    hits = 0
    for w in sorted(HELD_OUT):
        out = model.generate(f"{task}: {w} =".encode()).decode(errors="replace")
        hits += out == f" {SOLVE[task](w)}\n"
    return hits / len(HELD_OUT)


def main():
    t0 = time.time()
    data = make_webtext(6000)
    model = GPT2()
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}   training bytes: {data.numel():,}")
    with torch.no_grad():                                  # what the 1/sqrt(N) init buys: a residual stream of ~unit scale
        x = model.wte(data[None, :CTX]) + model.wpe(torch.arange(CTX))
        for block in model.blocks:
            x = block(x)
        print(f"residual-stream std after {len(model.blocks)} blocks at init: {x.std().item():.3f} (embeddings: 0.028)")
    opt, steps = torch.optim.Adam(model.parameters(), lr=2e-3), 1500
    for step in range(1, steps + 1):
        for g in opt.param_groups:                         # cosine decay of the learning rate (as in GPT-3's recipe)
            g["lr"] = 1e-3 * (1 + math.cos(math.pi * step / steps))
        starts = torch.randint(0, data.numel() - CTX - 1, (32,))
        batch = torch.stack([data[s : s + CTX + 1] for s in starts])
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), batch[:, 1:].reshape(-1))   # plain next-byte prediction
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1:
            first = loss.item()
        if step % 300 == 0:
            print(f"step {step:4d}  LM loss {loss.item():.3f} nats/byte")
    model.eval()
    accs = {task: zero_shot_accuracy(model, task) for task in TASKS}
    print("zero-shot accuracy on 60 held-out words:", {k: round(v, 2) for k, v in accs.items()}, f"({time.time() - t0:.1f} s)")
    w = sorted(HELD_OUT)[0]
    print(f'greedy:  "sort: {w} =" ->', repr(model.generate(f"sort: {w} =".encode()).decode()))
    print('top-k (k = 40) samples after "reverse: ":', [model.generate(b"reverse: ", 16, top_k=40).decode() for _ in range(3)])
    assert loss.item() < 0.5 * first and min(accs.values()) > 0.75, "zero-shot task ability did not emerge as expected"


if __name__ == "__main__":
    main()
