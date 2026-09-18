"""Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation (Cho et al., 2014).

What is implemented (section numbers follow the paper):
  * the gated hidden unit, written out from the equations                                        (Section 2.3, eqs. 5–8)
      r_j = sigma([W_r x]_j + [U_r h_{t-1}]_j)                           reset gate
      z_j = sigma([W_z x]_j + [U_z h_{t-1}]_j)                           update gate
      h~_j = tanh([W x]_j + [U (r ⊙ h_{t-1})]_j)                         candidate
      h_j  = z_j h_j^{t-1} + (1 - z_j) h~_j                              interpolation
  * encoder RNN h_t = f(h_{t-1}, x_t) whose final state is the summary c = h_T                   (Section 2.2, eq. 1–2)
  * decoder RNN h_t = f(h_{t-1}, y_{t-1}, c) with c entering every step, and
    P(y_t | y_{<t}, c) = g(h_t, y_{t-1}, c) with a maxout layer before the softmax               (Section 2.2, eq. 3–4; 3)
  * joint training by maximising 1/N sum_n log p(y_n | x_n)                                     (Section 2.2, eq. 4)
  * both uses of the trained model: scoring a candidate pair log p(y | x) (the SMT feature of Section 3)
    and generating y by greedy decoding
Simplifications: 96 hidden units and a toy "translation" (each token mapped through a dictionary, adjacent pairs
swapped) instead of WMT'14 phrase pairs; Adam instead of Adadelta; greedy decoding instead of beam search.

Run:  python rnn_encoder_decoder_gru.py        (CPU, about 20 s)
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ───────────────────────── the gated recurrent unit (Section 2.3) ─────────────────────────
class GRUCell(nn.Module):
    """h_t = f(h_{t-1}, x): x is whatever the step conditions on (the input for the encoder, [E y_{t-1}; c] for the decoder)."""
    def __init__(self, d_in, d_h):
        super().__init__()
        self.W_r, self.U_r = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h, bias=False)     # reset gate
        self.W_z, self.U_z = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h, bias=False)     # update gate
        self.W, self.U = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h, bias=False)         # candidate

    def forward(self, x, h_prev):
        r = torch.sigmoid(self.W_r(x) + self.U_r(h_prev))                                # eq. 5
        z = torch.sigmoid(self.W_z(x) + self.U_z(h_prev))                                # eq. 6
        h_tilde = torch.tanh(self.W(x) + self.U(r * h_prev))                             # eq. 8
        return z * h_prev + (1 - z) * h_tilde                                            # eq. 7


class RNNEncoderDecoder(nn.Module):
    def __init__(self, vocab, d_emb=32, d_h=96, maxout_pieces=2, pad=0):
        super().__init__()
        self.pad, self.d_h = pad, d_h
        self.E_src, self.E_tgt = nn.Embedding(vocab, d_emb, padding_idx=pad), nn.Embedding(vocab, d_emb, padding_idx=pad)
        self.enc = GRUCell(d_emb, d_h)                     # encoder: h_t = f(h_{t-1}, x_t)
        self.dec = GRUCell(d_emb + d_h, d_h)               # decoder: h_t = f(h_{t-1}, y_{t-1}, c)
        # g(h_t, y_{t-1}, c): maxout hidden layer (Section 3 / Appendix) then softmax over the target vocabulary
        self.k = maxout_pieces
        self.out_hidden = nn.Linear(d_h + d_emb + d_h, d_h * maxout_pieces)
        self.out = nn.Linear(d_h, vocab)

    def encode(self, src):                                  # src: (B, T); pads at the end are skipped via masking
        B, T = src.shape
        h = torch.zeros(B, self.d_h)
        x = self.E_src(src)
        for t in range(T):
            h_new = self.enc(x[:, t], h)
            h = torch.where((src[:, t] != self.pad)[:, None], h_new, h)   # keep h_T of the real sequence as the summary
        return h                                             # c = h_<T>   (eq. 2)

    def g(self, h, y_prev, c):
        """P(y_t | ...) = softmax(W_o maxout(...)), maxout over k pieces: s_i = max_j s'_{i,j}."""
        s = self.out_hidden(torch.cat([h, y_prev, c], -1)).view(h.size(0), self.d_h, self.k).max(-1).values
        return self.out(s)

    def decode_step(self, y_prev_tok, h, c):
        y_prev = self.E_tgt(y_prev_tok)
        h = self.dec(torch.cat([y_prev, c], -1), h)          # the summary c is an input at *every* step (eq. 3)
        return self.g(h, y_prev, c), h

    def log_prob(self, src, tgt_in, tgt_out):
        """log p(y | x) = sum_t log P(y_t | y_{<t}, c) per pair; pads contribute 0."""
        c = self.encode(src)
        h = torch.tanh(c)                                    # decoder initial state h_0 = tanh(V c) simplified to tanh(c)
        lp = torch.zeros(src.size(0))
        for t in range(tgt_in.size(1)):
            logits, h = self.decode_step(tgt_in[:, t], h, c)
            step_lp = F.log_softmax(logits, -1).gather(1, tgt_out[:, t : t + 1]).squeeze(1)
            lp = lp + step_lp * (tgt_out[:, t] != self.pad)
        return lp

    @torch.no_grad()
    def greedy_decode(self, src, bos, max_len):
        c = self.encode(src)
        h, y = torch.tanh(c), torch.full((src.size(0),), bos, dtype=torch.long)
        out = []
        for _ in range(max_len):
            logits, h = self.decode_step(y, h, c)
            y = logits.argmax(-1); out.append(y)
        return torch.stack(out, 1)


# ───────────────────────── toy phrase pairs ─────────────────────────
def make_batch(B, vocab, dictionary, min_len=3, max_len=7, pad=0, bos=1, eos=2):
    """Source phrase of 3..7 tokens in 3..vocab-1. 'Translation' = map every token through a fixed dictionary and swap
    each adjacent pair (a little reordering, like adjective–noun order between languages)."""
    L = torch.randint(min_len, max_len + 1, (B,))
    src, tgt = torch.full((B, max_len), pad), torch.full((B, max_len + 2), pad)
    for i, l in enumerate(L.tolist()):
        s = torch.randint(3, vocab, (l,))
        y = dictionary[s].clone()
        y[: l - l % 2] = y[: l - l % 2].view(-1, 2).flip(1).reshape(-1)
        src[i, :l] = s
        tgt[i, 0], tgt[i, 1 : l + 1], tgt[i, l + 1] = bos, y, eos
    return src, tgt


def main():
    t0 = time.time()
    vocab, pad, bos, eos = 20, 0, 1, 2
    dictionary = torch.cat([torch.tensor([0, 1, 2]), torch.randperm(vocab - 3) + 3])      # source word -> target word
    model = RNNEncoderDecoder(vocab, pad=pad)
    opt = torch.optim.Adam(model.parameters(), lr=4e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 if s < 1000 else 0.2)   # lower the rate for the second half
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    first = None
    for step in range(1, 2001):
        src, tgt = make_batch(64, vocab, dictionary)
        n_tok = (tgt[:, 1:] != pad).sum()
        loss = -model.log_prob(src, tgt[:, :-1], tgt[:, 1:]).sum() / n_tok                # maximise 1/N sum log p(y|x)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        first = first or loss.item()
        if step % 400 == 0:
            print(f"step {step:4d}  nll/token {loss.item():.3f}")

    model.eval()
    src, tgt = make_batch(200, vocab, dictionary)
    pred = model.greedy_decode(src, bos, max_len=tgt.size(1) - 1)
    gold = tgt[:, 1:]; keep = gold != pad
    acc = ((pred == gold) | ~keep).all(1).float().mean().item()
    print(f"exact-match accuracy of greedy decoding on 200 held-out phrases: {acc:.2f}")
    print("example:", src[0][src[0] != pad].tolist(), "->", pred[0][: keep[0].sum()].tolist(), " gold", gold[0][keep[0]].tolist())

    # Use as a scoring feature (Section 3): log p(y | x) of the true pair vs. a shuffled (wrong) target
    with torch.no_grad():
        lp_true = model.log_prob(src, tgt[:, :-1], tgt[:, 1:])
        wrong = tgt[torch.randperm(200)]
        lp_wrong = model.log_prob(src, wrong[:, :-1], wrong[:, 1:])
    frac = (lp_true > lp_wrong).float().mean().item()
    print(f"score log p(y|x): true pair {lp_true.mean():.2f} vs mismatched pair {lp_wrong.mean():.2f}; "
          f"true pair scores higher in {100 * frac:.0f}% of cases   ({time.time() - t0:.1f} s)")
    assert loss.item() < 0.3 * first and acc > 0.9 and frac > 0.95, "encoder–decoder did not learn the toy translation"


if __name__ == "__main__":
    main()
