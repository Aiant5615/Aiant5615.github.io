"""Attention Is All You Need (Vaswani et al., 2017) — a from-scratch PyTorch implementation.

What is implemented (section numbers follow the paper):
  * scaled dot-product attention and multi-head attention        (3.2.1, 3.2.2)
  * position-wise feed-forward network                            (3.3)
  * sinusoidal positional encoding                                (3.5)
  * encoder / decoder stacks with residual + LayerNorm (post-LN)  (3.1)
  * causal (look-ahead) mask in the decoder, padding masks
  * Noam learning-rate schedule (warm-up then inverse sqrt)       (5.3)
  * label smoothing eps = 0.1                                     (5.4)
Simplifications: tiny model, a synthetic "reverse the sequence" task instead of WMT, greedy decoding.

Run:  python transformer.py        (CPU, about 35 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ───────────────────────── building blocks ─────────────────────────
def scaled_dot_product_attention(q, k, v, mask=None):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V.   q,k,v: (B, h, L, d_k); mask: broadcastable bool, True = keep."""
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)             # (B, h, Lq, Lk)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = scores.softmax(dim=-1)
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    """MultiHead(Q,K,V) = Concat(head_1..head_h) W^O,  head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)."""
    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k = h, d_model // h
        self.w_q, self.w_k, self.w_v, self.w_o = (nn.Linear(d_model, d_model) for _ in range(4))

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        split = lambda x, lin: lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)   # (B, h, L, d_k)
        out, self.attn = scaled_dot_product_attention(split(q, self.w_q), split(k, self.w_k), split(v, self.w_v), mask)
        return self.w_o(out.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k))


class FeedForward(nn.Module):
    """FFN(x) = max(0, x W_1 + b_1) W_2 + b_2, applied to every position separately."""
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class PositionalEncoding(nn.Module):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d)).  Added to the scaled embeddings."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):                     # x: (B, L, d_model)
        return x + self.pe[: x.size(1)]


class SublayerConnection(nn.Module):
    """Post-LN residual block from the paper: LayerNorm(x + Sublayer(x)), with dropout on the sublayer output."""
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm, self.drop = nn.LayerNorm(d_model), nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.drop(sublayer(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.self_attn, self.ff = MultiHeadAttention(d_model, h), FeedForward(d_model, d_ff, dropout)
        self.sub = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.sub[0](x, lambda x: self.self_attn(x, x, x, src_mask))
        return self.sub[1](x, self.ff)


class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, dropout):
        super().__init__()
        self.self_attn, self.cross_attn = MultiHeadAttention(d_model, h), MultiHeadAttention(d_model, h)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.sub = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sub[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))            # masked self-attention
        x = self.sub[1](x, lambda x: self.cross_attn(x, memory, memory, src_mask))  # encoder-decoder attention
        return self.sub[2](x, self.ff)


class Transformer(nn.Module):
    def __init__(self, vocab, d_model=64, h=4, d_ff=128, n_layers=2, dropout=0.1, pad=0):
        super().__init__()
        self.pad, self.d_model = pad, d_model
        self.src_emb = nn.Embedding(vocab, d_model, padding_idx=pad)
        self.tgt_emb = nn.Embedding(vocab, d_model, padding_idx=pad)
        self.pos = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, h, d_ff, dropout) for _ in range(n_layers)])
        self.proj = nn.Linear(d_model, vocab)
        self.proj.weight = self.tgt_emb.weight        # weight tying between embedding and pre-softmax (3.4)

    @staticmethod
    def causal_mask(L):
        return torch.tril(torch.ones(L, L, dtype=torch.bool))          # (L, L): position i may attend to j <= i

    def encode(self, src):
        src_mask = (src != self.pad)[:, None, None, :]                  # (B, 1, 1, Ls)
        x = self.pos(self.src_emb(src) * math.sqrt(self.d_model))       # embeddings scaled by sqrt(d_model) (3.4)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt, memory, src_mask):
        L = tgt.size(1)
        tgt_mask = (tgt != self.pad)[:, None, None, :] & self.causal_mask(L)[None, None]
        y = self.pos(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            y = layer(y, memory, src_mask, tgt_mask)
        return self.proj(y)

    def forward(self, src, tgt_in):
        memory, src_mask = self.encode(src)
        return self.decode(tgt_in, memory, src_mask)

    @torch.no_grad()
    def greedy_decode(self, src, bos, eos, max_len):
        memory, src_mask = self.encode(src)
        ys = torch.full((src.size(0), 1), bos, dtype=torch.long)
        for _ in range(max_len):
            nxt = self.decode(ys, memory, src_mask)[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
        return ys[:, 1:]


# ───────────────────────── training utilities ─────────────────────────
def noam_lr(step, d_model, warmup=400):
    """lrate = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)   (eq. 3)."""
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def label_smoothed_nll(logits, target, eps, pad):
    """Cross-entropy against the smoothed distribution (1-eps) on the gold token, eps spread over the rest; pads ignored."""
    logp = logits.log_softmax(-1)
    V = logits.size(-1)
    nll = -logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    smooth = -logp.mean(-1)
    loss = (1 - eps) * nll + eps * smooth
    keep = target != pad
    return (loss * keep).sum() / keep.sum()


def make_batch(B, vocab, min_len=4, max_len=10, pad=0, bos=1, eos=2):
    """Toy task: output the input sequence reversed. Tokens 3..vocab-1, variable length, padded."""
    L = torch.randint(min_len, max_len + 1, (B,))
    src = torch.full((B, max_len), pad); tgt = torch.full((B, max_len + 2), pad)
    for i, l in enumerate(L.tolist()):
        s = torch.randint(3, vocab, (l,))
        src[i, :l] = s
        tgt[i, 0], tgt[i, 1 : l + 1], tgt[i, l + 1] = bos, s.flip(0), eos
    return src, tgt


def main():
    vocab, pad, bos, eos = 20, 0, 1, 2
    model = Transformer(vocab, pad=pad, dropout=0.0)   # the paper uses 0.1; the toy task converges faster without it
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)   # lr set by the schedule
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    t0, first_loss = time.time(), None
    for step in range(1, 1501):
        for g in opt.param_groups:
            g["lr"] = noam_lr(step, model.d_model)
        src, tgt = make_batch(64, vocab)
        logits = model(src, tgt[:, :-1])
        loss = label_smoothed_nll(logits, tgt[:, 1:], eps=0.1, pad=pad)
        opt.zero_grad(); loss.backward(); opt.step()
        first_loss = first_loss or loss.item()
        if step % 250 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}  lr {opt.param_groups[0]['lr']:.2e}")
    model.eval()
    src, tgt = make_batch(200, vocab)
    pred = model.greedy_decode(src, bos, eos, max_len=src.size(1) + 1)
    gold = tgt[:, 1:]
    keep = gold != pad
    acc = ((pred[:, : gold.size(1)] == gold) | ~keep).all(1).float().mean().item()
    print(f"exact-match accuracy on 200 held-out sequences: {acc:.2f}   ({time.time() - t0:.1f} s)")
    print("example:", src[0][src[0] != pad].tolist(), "->", pred[0][: (gold[0] != pad).sum()].tolist())
    assert loss.item() < first_loss * 0.5 and acc > 0.9, "training did not converge as expected"


if __name__ == "__main__":
    main()
