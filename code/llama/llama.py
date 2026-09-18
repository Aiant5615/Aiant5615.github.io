"""LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023) — a from-scratch PyTorch LLaMA block.

What is implemented (section numbers follow the paper):
  * pre-normalisation with RMSNorm: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g, applied to the INPUT of each sub-layer  (2.2)
  * SwiGLU feed-forward: FFN(x) = (Swish_1(x W_1) o x V) W_2 with hidden width 2/3 * 4d                                  (2.2)
  * rotary positional embeddings on q and k in every layer, no absolute position embeddings                              (2.2)
  * causal multi-head self-attention without biases, tied to a tiny decoder-only LM                                     (2.2)
  * AdamW with beta_2 = 0.95, weight decay 0.1, gradient clipping 1.0, warm-up + cosine decay to 10% of the peak lr     (2.3)
Simplifications: 2 layers, d = 64, a synthetic "word" corpus of integer tokens instead of 1.4T public tokens;
  the xformers memory-efficient attention / checkpointing of Section 2.4 are not needed at this size.

Run:  python llama.py        (CPU, about 15 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ───────────────────────── the three architecture changes (Section 2.2) ─────────────────────────
class RMSNorm(nn.Module):
    """RMSNorm(x) = x / sqrt(1/d sum_i x_i^2 + eps) * g  (Zhang & Sennrich, 2019); no mean-centering, no bias."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.g = eps, nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.g


class SwiGLU(nn.Module):
    """FFN_SwiGLU(x) = (Swish_1(x W_1) o x V) W_2,  Swish_1(z) = z sigma(z)  (Shazeer, 2020; PaLM).
    Hidden width 2/3 * 4d keeps the parameter count of a 2-matrix 4d MLP despite the third matrix V."""
    def __init__(self, d, multiple_of=8):
        super().__init__()
        hidden = int(2 * (4 * d) / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)   # LLaMA rounds up to a multiple (256 in the release)
        self.w1, self.v, self.w2 = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False), nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.v(x))


def rope_cos_sin(L, d_k, base=10000.0):
    """theta_i = base^(-2i/d_k); returns cos, sin of m*theta_i, each (L, d_k) with every angle repeated for its pair."""
    theta = base ** (-torch.arange(0, d_k, 2).float() / d_k)
    ang = torch.arange(L).float()[:, None] * theta[None, :]
    return ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)


def apply_rope(x, cos, sin):
    """R_m x = x * cos + rotate_half(x) * sin  (Su et al., 2021, eq. 34), x: (B, h, L, d_k)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), -1).flatten(-2)
    return x * cos + rot * sin


class Attention(nn.Module):
    """Causal multi-head attention, no biases, rotary embeddings on q and k."""
    def __init__(self, d, h):
        super().__init__()
        self.h, self.d_k = h, d // h
        self.wq, self.wk, self.wv, self.wo = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x):
        B, L, _ = x.shape
        split = lambda t: t.view(B, L, self.h, self.d_k).transpose(1, 2)
        q, k, v = split(self.wq(x)), split(self.wk(x)), split(self.wv(x))
        cos, sin = rope_cos_sin(L, self.d_k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~torch.tril(torch.ones(L, L, dtype=torch.bool)), float("-inf"))
        return self.wo((scores.softmax(-1) @ v).transpose(1, 2).reshape(B, L, -1))


class LlamaBlock(nn.Module):
    """x = x + Attn(RMSNorm(x));  x = x + SwiGLU(RMSNorm(x))   (pre-norm, Section 2.2)."""
    def __init__(self, d, h):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(d), RMSNorm(d)
        self.attn, self.ffn = Attention(d, h), SwiGLU(d)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class Llama(nn.Module):
    def __init__(self, vocab, d=64, h=4, n_layers=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([LlamaBlock(d, h) for _ in range(n_layers)])
        self.norm, self.output = RMSNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, idx):
        x = self.tok(idx)                                   # no positional embedding is added: positions enter through RoPE only
        for layer in self.layers:
            x = layer(x)
        return self.output(self.norm(x))

    @torch.no_grad()
    def generate(self, idx, n):
        for _ in range(n):
            idx = torch.cat([idx, self(idx)[:, -1].argmax(-1, keepdim=True)], 1)
        return idx


# ───────────────────────── toy corpus and the LLaMA training recipe (Section 2.3) ─────────────────────────
class ToyCorpus:
    """A 'language' of 16 words over a vocabulary of 32 tokens. Each word is a fixed sequence of 2-4 tokens whose FIRST token
    is unique to the word, so inside a word the next token is deterministic; word order follows a random bigram Markov chain,
    so at word boundaries the next token is stochastic (loss floor = the chain's entropy). One long token stream is generated."""
    def __init__(self, n_words=16, vocab=32, n_stream_words=40000):
        self.vocab = vocab
        self.words = [torch.cat([torch.tensor([w]), torch.randint(0, vocab, (int(torch.randint(1, 4, ())),))]) for w in range(n_words)]
        self.trans = torch.softmax(torch.randn(n_words, n_words) * 2, -1)       # bigram transition matrix over words
        self.entropy = -(self.trans * self.trans.log()).sum(-1).mean().item()  # nats per word boundary
        cdf, u, w, toks, inside = self.trans.cumsum(-1), torch.rand(n_stream_words), 0, [], []
        for i in range(n_stream_words):
            toks += self.words[w].tolist(); inside += [0] + [1] * (len(self.words[w]) - 1)
            w = min(int(torch.searchsorted(cdf[w], u[i])), n_words - 1)
        self.stream, self.inside = torch.tensor(toks), torch.tensor(inside)    # inside[t] = 1: within-word (deterministic) token

    def batch(self, B, L):
        start = torch.randint(0, len(self.stream) - L - 1, (B,))
        idx = start[:, None] + torch.arange(L + 1)
        x, m = self.stream[idx], self.inside[idx]
        return x[:, :-1], x[:, 1:], m[:, 1:]


def lr_at(step, total, peak, warmup):
    """Linear warm-up then cosine decay to 10% of the peak (Section 2.3)."""
    if step < warmup:
        return peak * step / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.9 * peak * 0.5 * (1 + math.cos(math.pi * prog))


def main():
    t0 = time.time()
    corpus, B, L, steps = ToyCorpus(), 32, 32, 1000
    model = Llama(corpus.vocab)
    ffn = model.layers[0].ffn
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}; SwiGLU hidden width {ffn.w1.out_features} "
          f"(= 2/3 * 4d for d=64, 3 matrices ~ the params of one 4d two-matrix MLP: {3 * 64 * ffn.w1.out_features} vs {2 * 64 * 256})")
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]      # weight decay on matrices only, not on norms/gains
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=3e-3, betas=(0.9, 0.95))                    # beta_2 = 0.95, wd = 0.1 (Section 2.3)
    first = None
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, peak=3e-3, warmup=100)             # 2000 warm-up steps in the paper; 100 here
        x, y, _ = corpus.batch(B, L)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)             # gradient clipping 1.0
        opt.step()
        first = first or loss.item()
        if step % 300 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}  lr {opt.param_groups[0]['lr']:.2e}")
    model.eval()
    with torch.no_grad():
        x, y, inside = corpus.batch(128, L)
        logits = model(x)
        nll = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view_as(y)
        acc_inside = (logits.argmax(-1)[inside == 1] == y[inside == 1]).float().mean().item()
        loss_inside, loss_boundary = nll[inside == 1].mean().item(), nll[inside == 0].mean().item()
    print(f"held-out loss: within-word tokens {loss_inside:.3f} (deterministic, floor 0), word boundaries {loss_boundary:.3f} "
          f"(floor = bigram entropy {corpus.entropy:.3f});  within-word accuracy {acc_inside:.3f}   ({time.time() - t0:.1f} s)")
    prompt = x[:1, :6]
    print("greedy continuation of", prompt[0].tolist(), "->", model.generate(prompt, 12)[0, 6:].tolist())
    assert acc_inside > 0.98 and loss_inside < 0.1, "the LM should learn the deterministic within-word structure"
    assert loss_boundary < corpus.entropy + 0.3 and nll.mean().item() < 0.5 * first, "boundary loss should approach the entropy floor"


if __name__ == "__main__":
    main()
