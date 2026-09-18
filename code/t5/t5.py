"""Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5, Raffel et al., 2020) — a from-scratch PyTorch implementation.

What is implemented (section numbers follow the paper):
  * encoder-decoder Transformer with pre-norm and a bias-free (RMS) layer norm                            (2.1)
  * relative position biases: attention logits q_i k_j^T / sqrt(d_k) + b_bucket(j - i) with 32 log-spaced
    buckets up to distance 128, one scalar per head, the table shared across the layers of a stack
    (bidirectional buckets in the encoder, one-sided in the decoder), no position embeddings on tokens     (2.1)
  * span-corruption pre-training: 15% of the tokens are corrupted with mean span length 3; every corrupted
    span is replaced by a single sentinel <X>, <Y>, ... in the input, the target is the dropped spans
    delimited by the same sentinels plus a final sentinel                                        (3.1.4, Figure 2, 3.3)
  * text-to-text fine-tuning: task prefixes, both tasks mixed in one batch, teacher-forced cross-entropy,
    greedy decoding, and a classification task whose label is emitted as a token                   (2.4, 3.5.2)
Simplifications: 2 + 2 blocks of d_model = 64 instead of 12 + 12 of 768; a toy vocabulary of integer tokens whose
documents come from a hidden Markov chain instead of C4 + SentencePiece; the downstream tasks are "reverse: s" -> s
reversed and "mode: s" -> the most frequent symbol of s; no dropout; Adam instead of AdaFactor, no beam search.

Run:  python t5.py        (CPU, about 25 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
PAD, EOS, SENT, N_SENT = 0, 1, 2, 10               # sentinels <X>, <Y>, ... are ids SENT .. SENT + N_SENT - 1
REVERSE, MODE, WORD0, V_TEXT = 12, 13, 14, 16      # two task-prefix tokens, then 16 "word" tokens
VOCAB = WORD0 + V_TEXT


# ───────────────────────── model (2.1) ─────────────────────────
class T5LayerNorm(nn.Module):
    """LayerNorm without mean subtraction or additive bias: x / sqrt(mean(x^2) + eps) * w."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def relative_position_bucket(relative_position, bidirectional, num_buckets=32, max_distance=128):
    """Map the offset j - i to one of 32 buckets: exact for small |offset|, logarithmically spaced up to 128, then one
    shared bucket.  In the encoder half the buckets are for negative offsets; in the decoder only j <= i occur."""
    ret, n = 0, -relative_position
    if bidirectional:
        num_buckets //= 2
        ret = (n < 0).long() * num_buckets
        n = n.abs()
    else:
        n = n.clamp(min=0)
    max_exact = num_buckets // 2
    is_small = n < max_exact
    large = max_exact + (torch.log(n.float().clamp(min=1) / max_exact) / math.log(max_distance / max_exact)
                         * (num_buckets - max_exact)).long()
    return ret + torch.where(is_small, n, large.clamp(max=num_buckets - 1))


class RelativeBias(nn.Module):
    """b_bucket(j - i), a learned scalar per (bucket, head); one instance per stack, shared by all of its layers."""
    def __init__(self, h, bidirectional):
        super().__init__()
        self.table, self.bidirectional = nn.Embedding(32, h), bidirectional

    def forward(self, Lq, Lk):
        rel = torch.arange(Lk)[None, :] - torch.arange(Lq)[:, None]          # memory position - context position
        return self.table(relative_position_bucket(rel, self.bidirectional)).permute(2, 0, 1)[None]   # (1, h, Lq, Lk)


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.d_k = h, d // h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x, mem, mask, bias=None):                # mask: (B, 1, Lq, Lk) True = attend
        B, Lq, Lk = x.size(0), x.size(1), mem.size(1)
        split = lambda t, lin: lin(t).view(B, -1, self.h, self.d_k).transpose(1, 2)
        scores = split(x, self.q) @ split(mem, self.k).transpose(-2, -1) / math.sqrt(self.d_k)
        if bias is not None:
            scores = scores + bias                               # softmax(q k^T / sqrt(d_k) + b_bucket(j - i))
        attn = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        return self.o((attn @ split(mem, self.v)).transpose(1, 2).reshape(B, Lq, -1))


class Layer(nn.Module):
    """Pre-norm sub-blocks: x + SelfAttn(LN(x)); [x + CrossAttn(LN(x), memory)]; x + FFN(LN(x))."""
    def __init__(self, d, h, cross):
        super().__init__()
        self.ln1, self.self_attn = T5LayerNorm(d), Attention(d, h)
        self.ln2, self.cross_attn = (T5LayerNorm(d), Attention(d, h)) if cross else (None, None)
        self.ln3, self.ffn = T5LayerNorm(d), nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.ReLU(), nn.Linear(4 * d, d, bias=False))

    def forward(self, x, mask, bias, mem=None, mem_mask=None):
        x = x + self.self_attn(self.ln1(x), self.ln1(x), mask, bias)
        if mem is not None:
            x = x + self.cross_attn(self.ln2(x), mem, mem_mask)   # no position bias in encoder-decoder attention
        return x + self.ffn(self.ln3(x))


class T5(nn.Module):
    def __init__(self, vocab=VOCAB, d=64, h=4, n_layers=2):
        super().__init__()
        self.d, self.emb = d, nn.Embedding(vocab, d)                # shared by encoder, decoder and the output layer
        self.enc = nn.ModuleList([Layer(d, h, cross=False) for _ in range(n_layers)])
        self.dec = nn.ModuleList([Layer(d, h, cross=True) for _ in range(n_layers)])
        self.enc_bias, self.dec_bias = RelativeBias(h, bidirectional=True), RelativeBias(h, bidirectional=False)
        self.enc_ln, self.dec_ln = T5LayerNorm(d), T5LayerNorm(d)

    def encode(self, src):
        mask = (src != PAD)[:, None, None, :]
        x, bias = self.emb(src), self.enc_bias(src.size(1), src.size(1))
        for layer in self.enc:
            x = layer(x, mask, bias)
        return self.enc_ln(x), mask

    def decode(self, tgt_in, mem, mem_mask):
        L = tgt_in.size(1)
        mask = torch.tril(torch.ones(L, L, dtype=torch.bool))[None, None]
        x, bias = self.emb(tgt_in), self.dec_bias(L, L)
        for layer in self.dec:
            x = layer(x, mask, bias, mem, mem_mask)
        return self.dec_ln(x) @ self.emb.weight.T * self.d ** -0.5    # tied output, rescaled as in T5

    def forward(self, src, tgt):
        """Teacher forcing: the decoder input is the target shifted right, starting with PAD (T5's start token)."""
        tgt_in = torch.cat([torch.full_like(tgt[:, :1], PAD), tgt[:, :-1]], 1)
        return self.decode(tgt_in, *self.encode(src))

    @torch.no_grad()
    def greedy_decode(self, src, max_len):
        mem, mem_mask = self.encode(src)
        ys = torch.full((src.size(0), 1), PAD)
        for _ in range(max_len):
            ys = torch.cat([ys, self.decode(ys, mem, mem_mask)[:, -1].argmax(-1, keepdim=True)], 1)
        return ys[:, 1:]


# ───────────────────────── data ─────────────────────────
TRANS = torch.full((V_TEXT, V_TEXT), 0.2 / (V_TEXT - 2)).scatter_(1, torch.stack([torch.randperm(V_TEXT)[:2] for _ in range(V_TEXT)]), 0.4)


def sample_words(B, L):
    """Documents from a hidden Markov chain (each word has two likely successors) — the unlabeled corpus."""
    w, out = torch.randint(0, V_TEXT, (B,)), []
    for _ in range(L):
        out.append(w + WORD0)
        w = torch.multinomial(TRANS[w], 1).squeeze(1)
    return torch.stack(out, 1)


def random_spans_noise_mask(L, noise_density=0.15, mean_span=3.0):
    """T5's span sampler (3.3.4): choose round(0.15 L) noise tokens grouped into round(noise / 3) spans, split the
    remaining tokens into the same number of spans, and interleave non-noise / noise / non-noise / ..."""
    n_noise = max(1, round(L * noise_density))
    n_spans = max(1, round(n_noise / mean_span))
    segment = lambda n, k: torch.diff(torch.cat([torch.tensor([0]), torch.randperm(n - 1)[: k - 1].sort().values + 1, torch.tensor([n])]))
    noise_len, clean_len = segment(n_noise, n_spans), segment(L - n_noise, n_spans)
    mask = []
    for c, n in zip(clean_len.tolist(), noise_len.tolist()):
        mask += [False] * c + [True] * n
    return mask


def span_corrupt(seq):
    """Figure 2: input  "Thank you <X> me to your party <Y> week."   target  "<X> for inviting <Y> last <Z>"."""
    inp, tgt, s, prev = [], [], 0, False
    for tok, noisy in zip(seq, random_spans_noise_mask(len(seq))):
        if noisy:
            if not prev:
                inp.append(SENT + s); tgt.append(SENT + s); s += 1
            tgt.append(tok)
        else:
            inp.append(tok)
        prev = noisy
    return inp, tgt + [SENT + s, EOS]                                  # final sentinel, then end of sequence


def pad(seqs):
    out = torch.full((len(seqs), max(map(len, seqs))), PAD)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


def make_pretrain_batch(B, L=24):
    pairs = [span_corrupt(doc) for doc in sample_words(B, L).tolist()]
    return pad([p[0] for p in pairs]), pad([p[1] for p in pairs])


def make_task_batch(B):
    """Text-to-text (2.4): "reverse: s" -> s reversed;  "mode: s" -> the most frequent symbol (classification as text).
    Both tasks are mixed in one batch, as in the paper's multi-task mixing."""
    src, tgt = [], []
    for i in range(B):
        n = torch.randint(4, 9, (1,)).item()
        s = torch.randint(WORD0, VOCAB, (n,)).tolist()
        if i % 2 == 0:
            src.append([REVERSE] + s); tgt.append(s[::-1] + [EOS])
        else:
            m = s[0]                                                    # plant a unique mode: s[0] fills > half of s
            s = [m] * (n // 2 + 1) + s[n // 2 + 1:]
            s = [s[j] for j in torch.randperm(n).tolist()]
            src.append([MODE] + s); tgt.append([m, EOS])
    return pad(src), pad(tgt)


def seq2seq_loss(model, src, tgt):
    logits = model(src, tgt)
    return F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1), ignore_index=PAD)


def accuracy(model, src, tgt):
    pred = model.greedy_decode(src, tgt.size(1))
    return ((pred == tgt) | (tgt == PAD)).all(1).float().mean().item()


def main():
    t0 = time.time()
    model = T5()
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    src, tgt = make_pretrain_batch(1)
    print("span corruption example:  input", src[0].tolist(), " target", tgt[0].tolist(), f"(sentinels are {SENT}..{SENT + N_SENT - 1})")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(1, 501):                                          # unsupervised pre-training (3.1.4)
        loss = seq2seq_loss(model, *make_pretrain_batch(32))
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1:
            first = loss.item()
        if step % 125 == 0:
            print(f"pre-train step {step:3d}  span-corruption loss {loss.item():.3f}")
    assert loss.item() < 0.7 * first, "pre-training loss did not fall"
    for step in range(1, 601):                                          # text-to-text fine-tuning on the mixture (2.4)
        loss = seq2seq_loss(model, *make_task_batch(32))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 150 == 0:
            print(f"fine-tune step {step:3d}  loss {loss.item():.3f}")
    model.eval()
    src, tgt = make_task_batch(400)
    acc_rev, acc_mode = accuracy(model, src[0::2], tgt[0::2]), accuracy(model, src[1::2], tgt[1::2])
    print(f"exact-match accuracy on 200 held-out examples each:  reverse {acc_rev:.2f}   mode {acc_mode:.2f}   ({time.time() - t0:.1f} s)")
    pred = model.greedy_decode(src[:1], tgt.size(1))[0]
    print("example:  reverse:", src[0, 1:][src[0, 1:] != PAD].tolist(), "->", pred[: (tgt[0] != PAD).sum() - 1].tolist())
    assert acc_rev > 0.8 and acc_mode > 0.8, "text-to-text fine-tuning did not converge as expected"


if __name__ == "__main__":
    main()
