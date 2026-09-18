"""Effective Approaches to Attention-based Neural Machine Translation (Luong, Pham & Manning, 2015) — PyTorch.

What is implemented (section numbers follow the paper):
  * global attention: a_t(s) = align(h_t, h̄_s) = softmax_s(score(h_t, h̄_s)) over all source states,
    computed from the *current* top-layer decoder state h_t                                        (Section 3.1, eq. 7)
  * the three content-based score functions                                                        (Section 3.1, eq. 8)
      dot: h_t^T h̄_s     general: h_t^T W_a h̄_s     concat: v_a^T tanh(W_a [h_t; h̄_s])
  * context c_t = sum_s a_t(s) h̄_s, attentional hidden state h̃_t = tanh(W_c [c_t; h_t]),
    p(y_t | y_<t, x) = softmax(W_s h̃_t)                                                            (Section 3, eqs. 5–6)
  * local-p attention: p_t = S sigmoid(v_p^T tanh(W_p h_t)), window [p_t - D, p_t + D], weights multiplied by the
    Gaussian exp(-(s - p_t)^2 / (2 sigma^2)) with sigma = D/2                                        (Section 3.2, eqs. 9–10)
  * input feeding: the next decoder input is [E y_t; h̃_t]                                          (Section 3.3)
  * a comparison of the four attention variants at equal training budget (Table 1 / Section 4)
Simplifications: 2-layer LSTMs of 48 units, a toy task (translate through a dictionary, sentence reversed in blocks)
instead of WMT'14 En–De, Adam instead of SGD, greedy decoding, D = 2 for the local window (paper: 10), no dropout.

Run:  python luong_attention.py        (CPU, about 25 s)
"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class Attention(nn.Module):
    """score(h_t, h̄_s) for one of {dot, general, concat}; optional local-p prediction of p_t (eqs. 9–10)."""
    def __init__(self, d, score="dot", local=False, D=2):
        super().__init__()
        self.score_fn, self.local, self.D = score, local, D
        if score == "general":
            self.W_a = nn.Linear(d, d, bias=False)
        elif score == "concat":
            self.W_a, self.v_a = nn.Linear(2 * d, d, bias=False), nn.Linear(d, 1, bias=False)
        if local:
            self.W_p, self.v_p = nn.Linear(d, d, bias=False), nn.Linear(d, 1, bias=False)

    def score(self, h_t, h_bar):                                         # h_t: (B, d), h_bar: (B, S, d) -> (B, S)
        if self.score_fn == "dot":
            return torch.bmm(h_bar, h_t.unsqueeze(-1)).squeeze(-1)                                     # h_t^T h̄_s
        if self.score_fn == "general":
            return torch.bmm(self.W_a(h_bar), h_t.unsqueeze(-1)).squeeze(-1)                           # h_t^T W_a h̄_s
        both = torch.cat([h_t.unsqueeze(1).expand_as(h_bar), h_bar], -1)
        return self.v_a(torch.tanh(self.W_a(both))).squeeze(-1)                                        # v_a^T tanh(W_a [h_t; h̄_s])

    def forward(self, h_t, h_bar, mask):
        e = self.score(h_t, h_bar).masked_fill(~mask, float("-inf"))
        if self.local:
            S = mask.sum(1, keepdim=True).float()                                                      # source length
            p_t = S * torch.sigmoid(self.v_p(torch.tanh(self.W_p(h_t))))                               # eq. 9: p_t in [0, S]
            s = torch.arange(h_bar.size(1)).float().unsqueeze(0)
            in_window = (s - p_t).abs() <= self.D                                                      # [p_t - D, p_t + D]
            e = e.masked_fill(~in_window & mask, float("-inf"))
            a = F.softmax(e, -1) * torch.exp(-((s - p_t) ** 2) / (2 * (self.D / 2) ** 2))              # eq. 10, sigma = D/2
            a = torch.nan_to_num(a)                                                                    # (all-masked rows)
        else:
            a = F.softmax(e, -1)                                                                       # eq. 7: global align
        c_t = torch.bmm(a.unsqueeze(1), h_bar).squeeze(1)                                              # c_t = sum_s a_t(s) h̄_s
        return c_t, a


class LuongNMT(nn.Module):
    def __init__(self, vocab, score, local=False, input_feeding=True, d_emb=24, d=48, n_layers=2, pad=0):
        super().__init__()
        self.pad, self.d, self.input_feeding = pad, d, input_feeding
        self.E_src, self.E_tgt = nn.Embedding(vocab, d_emb, padding_idx=pad), nn.Embedding(vocab, d_emb, padding_idx=pad)
        self.encoder = nn.LSTM(d_emb, d, n_layers, batch_first=True)               # stacked LSTM encoder (Figure 1)
        dec_in = d_emb + (d if input_feeding else 0)                                # input feeding: [E y_t; h̃_t]  (Section 3.3)
        self.decoder = nn.ModuleList([nn.LSTMCell(dec_in if l == 0 else d, d) for l in range(n_layers)])
        self.attn = Attention(d, score, local)
        self.W_c = nn.Linear(2 * d, d, bias=False)                                  # h̃_t = tanh(W_c [c_t; h_t])   eq. 5
        self.W_s = nn.Linear(d, vocab)                                              # p(y_t | ...) = softmax(W_s h̃_t)   eq. 6

    def encode(self, src):
        lengths = (src != self.pad).sum(1)
        packed = nn.utils.rnn.pack_padded_sequence(self.E_src(src), lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, c) = self.encoder(packed)
        h_bar, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=src.size(1))
        return h_bar, [(h[l], c[l]) for l in range(h.size(0))], src != self.pad     # encoder states initialise the decoder

    def step(self, y_tok, h_tilde, states, h_bar, mask):
        x = self.E_tgt(y_tok)
        if self.input_feeding:
            x = torch.cat([x, h_tilde], -1)
        new_states = []
        for cell, st in zip(self.decoder, states):
            x, c = cell(x, st); new_states.append((x, c))
        h_t = x                                                                     # top-layer state, *current* step
        c_t, a_t = self.attn(h_t, h_bar, mask)                                      # h_t -> a_t -> c_t -> h̃_t
        h_tilde = torch.tanh(self.W_c(torch.cat([c_t, h_t], -1)))
        return self.W_s(h_tilde), h_tilde, new_states, a_t

    def forward(self, src, tgt_in):
        h_bar, states, mask = self.encode(src)
        h_tilde, logits = torch.zeros(src.size(0), self.d), []
        for t in range(tgt_in.size(1)):
            lg, h_tilde, states, _ = self.step(tgt_in[:, t], h_tilde, states, h_bar, mask)
            logits.append(lg)
        return torch.stack(logits, 1)

    @torch.no_grad()
    def greedy_decode(self, src, bos, max_len):
        h_bar, states, mask = self.encode(src)
        h_tilde, y, out, attn = torch.zeros(src.size(0), self.d), torch.full((src.size(0),), bos, dtype=torch.long), [], []
        for _ in range(max_len):
            lg, h_tilde, states, a_t = self.step(y, h_tilde, states, h_bar, mask)
            y = lg.argmax(-1); out.append(y); attn.append(a_t)
        return torch.stack(out, 1), torch.stack(attn, 1)


def make_batch(B, vocab, dictionary, min_len=4, max_len=8, pad=0, bos=1, eos=2):
    """Target = dictionary[source] with every block of two tokens swapped (a mostly monotonic alignment with local
    reordering, the situation local attention is designed for)."""
    L = torch.randint(min_len, max_len + 1, (B,))
    src, tgt = torch.full((B, max_len), pad), torch.full((B, max_len + 2), pad)
    for i, l in enumerate(L.tolist()):
        s = torch.randint(3, vocab, (l,))
        y = dictionary[s].clone(); y[: l - l % 2] = y[: l - l % 2].view(-1, 2).flip(1).reshape(-1)
        src[i, :l] = s; tgt[i, 0], tgt[i, 1 : l + 1], tgt[i, l + 1] = bos, y, eos
    return src, tgt


def run_variant(name, vocab, dictionary, steps, pad=0, bos=1, **kw):
    torch.manual_seed(1)
    model = LuongNMT(vocab, pad=pad, **kw)
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    losses = []
    for step in range(1, steps + 1):
        src, tgt = make_batch(64, vocab, dictionary)
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab), tgt[:, 1:].reshape(-1), ignore_index=pad)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        losses.append(loss.item())
    model.eval()
    src, tgt = make_batch(200, vocab, dictionary)
    pred, attn = model.greedy_decode(src, bos, max_len=tgt.size(1) - 1)
    gold = tgt[:, 1:]; keep = gold != pad
    acc = ((pred == gold) | ~keep).all(1).float().mean().item()
    print(f"  {name:28s} loss@100 {sum(losses[90:110]) / 20:.3f}  final loss {sum(losses[-20:]) / 20:.3f}  exact match {acc:.2f}")
    return acc, sum(losses[-20:]) / 20, (src, pred, attn)


def main():
    t0 = time.time()
    vocab, pad, steps = 20, 0, 350
    dictionary = torch.cat([torch.tensor([0, 1, 2]), torch.randperm(vocab - 3) + 3])
    print(f"{steps} training steps per variant, 2-layer LSTM encoder/decoder, 48 units")
    results = {}
    results["global, dot, no input feeding"] = run_variant("global dot (no input feed)", vocab, dictionary, steps, score="dot", input_feeding=False)
    for score in ("dot", "general", "concat"):
        results[f"global, {score}"] = run_variant(f"global {score} + input feed", vocab, dictionary, steps, score=score)
    results["local-p, general"] = run_variant("local-p general + input feed", vocab, dictionary, steps, score="general", local=True)
    print(f"({time.time() - t0:.1f} s)")

    src, pred, attn = results["local-p, general"][2]
    b = int(((src != pad).sum(1) == 6).nonzero()[0]); l = 6
    print(f"local-p alignment a_t(s) for source {src[b, :l].tolist()} -> {pred[b, : l + 1].tolist()} (rows t, cols s; window D = 2, Gaussian-damped):")
    for t in range(l + 1):
        print("   " + " ".join(f"{a:.2f}" for a in attn[b, t, :l].tolist()))
    accs = {k: v[0] for k, v in results.items()}
    print("ranking by final loss:", sorted(results, key=lambda k: results[k][1]))
    assert all(a > 0.85 for a in accs.values()), f"an attention variant failed to learn the toy task: {accs}"


if __name__ == "__main__":
    main()
