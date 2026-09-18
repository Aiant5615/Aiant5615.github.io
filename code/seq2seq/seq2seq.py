"""Sequence to Sequence Learning with Neural Networks (Sutskever, Vinyals & Le, 2014) — from-scratch PyTorch.

What is implemented (section numbers follow the paper):
  * the LSTM cell written out from the Graves-style equations used in the paper                  (Section 2)
      i_t = sigma(W_xi x_t + W_hi h_{t-1} + b_i),  f_t = sigma(W_xf x_t + W_hf h_{t-1} + b_f),  o_t = sigma(W_xo x_t + W_ho h_{t-1} + b_o)
      c_t = f_t ⊙ c_{t-1} + i_t ⊙ tanh(W_xc x_t + W_hc h_{t-1} + b_c),   h_t = o_t ⊙ tanh(c_t)
  * two different multi-layer LSTMs: the encoder reads the source into v = (h_T, c_T) of every layer, the decoder
    defines p(y_1..y_T' | x) = prod_t p(y_t | v, y_1..y_{t-1}) with <EOS> terminating both sequences   (Section 2, eq. 1)
  * training objective 1/|S| sum log p(T | S), uniform init in [-0.08, 0.08], gradient norm clipping
    g <- 5 g / s when s > 5, learning-rate halving late in training                                (Section 3.4)
  * the source-reversal trick, compared against the un-reversed model at equal budget                (Section 3.3)
  * left-to-right beam search decoding with a small beam                                            (Section 3.2)
Simplifications: 2 layers x 128 cells instead of 4 x 1000, a toy "translate through a dictionary" task instead of WMT'14,
Adam instead of the paper's SGD with lr 0.7 (which needs far more than our 1200 steps to move at all), one sequence per
beam-search call, no length-sorted batching. The reversal effect is measured as the mean training loss (speed of
convergence), which is where the paper's "minimal time lag" argument predicts it.

Run:  python seq2seq.py        (CPU, about 25 s)
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# ───────────────────────── the LSTM (Section 2) ─────────────────────────
class LSTMCell(nn.Module):
    """One layer, one time step. The four gates share one matmul: [i; f; o; g] = W_x x_t + W_h h_{t-1} + b."""
    def __init__(self, d_in, d_h):
        super().__init__()
        self.W_x, self.W_h = nn.Linear(d_in, 4 * d_h), nn.Linear(d_h, 4 * d_h, bias=False)

    def forward(self, x, state):
        h_prev, c_prev = state
        i, f, o, g = (self.W_x(x) + self.W_h(h_prev)).chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)         # input, forget, output gates
        c = f * c_prev + i * torch.tanh(g)                                     # c_t = f ⊙ c_{t-1} + i ⊙ tanh(...)
        h = o * torch.tanh(c)                                                  # h_t = o ⊙ tanh(c_t)
        return h, c


class StackedLSTM(nn.Module):
    """n_layers cells; layer l takes h of layer l-1 as input. Returns the top-layer h and the list of all (h, c)."""
    def __init__(self, d_in, d_h, n_layers):
        super().__init__()
        self.cells = nn.ModuleList([LSTMCell(d_in if l == 0 else d_h, d_h) for l in range(n_layers)])

    def forward(self, x, states):
        new_states = []
        for cell, st in zip(self.cells, states):
            x, c = cell(x, st)
            new_states.append((x, c))
        return x, new_states

    def zero_state(self, B, d_h):
        return [(torch.zeros(B, d_h), torch.zeros(B, d_h)) for _ in self.cells]


class Seq2Seq(nn.Module):
    def __init__(self, vocab, d_emb=32, d_h=128, n_layers=2, pad=0):
        super().__init__()
        self.pad, self.d_h = pad, d_h
        self.E_src, self.E_tgt = nn.Embedding(vocab, d_emb, padding_idx=pad), nn.Embedding(vocab, d_emb, padding_idx=pad)
        self.encoder, self.decoder = StackedLSTM(d_emb, d_h, n_layers), StackedLSTM(d_emb, d_h, n_layers)   # two LSTMs
        self.out = nn.Linear(d_h, vocab)
        for p in self.parameters():
            nn.init.uniform_(p, -0.08, 0.08)                                  # Section 3.4

    def encode(self, src):
        """Read x_1..x_T (already reversed if the trick is on). v = final (h, c) of every layer; pads leave the state unchanged."""
        B, T = src.shape
        states, x = self.encoder.zero_state(B, self.d_h), self.E_src(src)
        for t in range(T):
            _, new = self.encoder(x[:, t], states)
            m = (src[:, t] != self.pad)[:, None]
            states = [(torch.where(m, hn, h), torch.where(m, cn, c)) for (hn, cn), (h, c) in zip(new, states)]
        return states

    def forward(self, src, tgt_in):
        """Teacher-forced logits for every target position: p(y_t | v, y_1..y_{t-1})."""
        states, y = self.encode(src), self.E_tgt(tgt_in)
        logits = []
        for t in range(tgt_in.size(1)):
            h, states = self.decoder(y[:, t], states)
            logits.append(self.out(h))
        return torch.stack(logits, 1)

    @torch.no_grad()
    def beam_search(self, src, bos, eos, beam, max_len):
        """Section 3.2: keep the B most likely partial hypotheses (sum of log-probs); when <EOS> is appended the
        hypothesis is complete and is removed from the beam. Returns the best complete hypothesis (without <EOS>)."""
        states = self.encode(src.unsqueeze(0))
        beams, done = [(0.0, [bos], states)], []
        for _ in range(max_len):
            cand = []
            for score, toks, st in beams:
                h, st2 = self.decoder(self.E_tgt(torch.tensor([toks[-1]])), st)
                logp = F.log_softmax(self.out(h), -1)[0]
                for lp, tok in zip(*logp.topk(beam)):
                    cand.append((score + lp.item(), toks + [tok.item()], st2))
            cand.sort(key=lambda c: -c[0])
            beams = []
            for c in cand[:beam]:
                (done if c[1][-1] == eos else beams).append(c)
            if not beams:
                break
        best = max(done or beams, key=lambda c: c[0])
        return [t for t in best[1][1:] if t != eos]


# ───────────────────────── toy task and training ─────────────────────────
def make_batch(B, vocab, dictionary, min_len=3, max_len=6, pad=0, bos=1, eos=2, reverse=False):
    """Target = source translated token-by-token through a fixed dictionary, then <EOS>. With reverse=True the encoder
    sees x_T..x_1, so the first target word is next to its source word (the 'minimal time lag' argument of Section 3.3)."""
    L = torch.randint(min_len, max_len + 1, (B,))
    src, tgt = torch.full((B, max_len), pad), torch.full((B, max_len + 2), pad)
    for i, l in enumerate(L.tolist()):
        s = torch.randint(3, vocab, (l,))
        src[i, :l] = s.flip(0) if reverse else s
        tgt[i, 0], tgt[i, 1 : l + 1], tgt[i, l + 1] = bos, dictionary[s], eos
    return src, tgt


def train(model, dictionary, vocab, steps, reverse, lr=4e-3, clip=5.0, tag=""):
    """1/|S| sum log p(T | S); gradient rescaled to norm 5 when larger (g <- 5g/s), lr halved twice late in training."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    curve = []
    for step in range(1, steps + 1):
        if step in (int(0.6 * steps), int(0.8 * steps)):
            for g in opt.param_groups:
                g["lr"] /= 2
        src, tgt = make_batch(64, vocab, dictionary, reverse=reverse)
        logits = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1), ignore_index=model.pad)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)                     # g <- 5 g / s if s > 5
        opt.step()
        curve.append(loss.item())
        if step % (steps // 4) == 0:
            print(f"  {tag} step {step:4d}  nll/token {sum(curve[-50:]) / 50:.3f}")
    return sum(curve) / len(curve), sum(curve[-50:]) / 50


@torch.no_grad()
def exact_match(model, dictionary, vocab, reverse, n=100, beam=2, pad=0, bos=1, eos=2):
    src, tgt = make_batch(n, vocab, dictionary, reverse=reverse)
    ok = 0
    for i in range(n):
        gold = tgt[i, 1:][(tgt[i, 1:] != pad) & (tgt[i, 1:] != eos)].tolist()
        ok += model.beam_search(src[i][src[i] != pad], bos, eos, beam, max_len=len(gold) + 2) == gold
    return ok / n, src, tgt


def main():
    t0 = time.time()
    vocab, pad, bos, eos, steps = 24, 0, 1, 2, 1200
    dictionary = torch.cat([torch.tensor([0, 1, 2]), torch.randperm(vocab - 3) + 3])
    results = {}
    for reverse in (False, True):
        torch.manual_seed(1)
        model = Seq2Seq(vocab, pad=pad)
        tag = "reversed  " if reverse else "in order  "
        if not reverse:
            print(f"parameters: {sum(p.numel() for p in model.parameters()):,}   ({steps} SGD steps per model)")
        mean, final = train(model, dictionary, vocab, steps, reverse, tag=tag)
        acc, src, tgt = exact_match(model, dictionary, vocab, reverse)
        results[reverse] = (mean, final, acc, model)
        print(f"{tag} mean nll/token over training {mean:.3f}, final {final:.3f}, beam-2 exact match {acc:.2f}   ({time.time() - t0:.1f} s)")
    model = results[True][3]
    s = src[0][src[0] != pad]
    print("example (reversed input shown as fed):", s.tolist(), "->", model.beam_search(s, bos, eos, 3, 12),
          " gold", tgt[0, 1:][(tgt[0, 1:] > 2)].tolist())
    assert results[True][0] < results[False][0], "reversing the source did not speed up training"
    assert results[True][2] > 0.9, "reversed seq2seq did not learn the toy translation"


if __name__ == "__main__":
    main()
