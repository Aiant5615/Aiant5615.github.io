"""Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau, Cho & Bengio, 2015) — PyTorch.

What is implemented (section numbers follow the paper):
  * bidirectional RNN encoder with annotations h_j = [->h_j ; <-h_j]                              (Section 3.2, eq. 7)
  * alignment model e_ij = a(s_{i-1}, h_j) = v_a^T tanh(W_a s_{i-1} + U_a h_j), with U_a h_j precomputed  (Appendix A.1.2, eq. 6)
  * alpha_ij = softmax_j(e_ij), context c_i = sum_j alpha_ij h_j (the expected annotation)         (Section 3.1, eqs. 5–6)
  * decoder s_i = f(s_{i-1}, y_{i-1}, c_i) (a GRU whose gates take c_i through matrices C, C_z, C_r),
    p(y_i | y_<i, x) = g(y_{i-1}, s_i, c_i), s_0 = tanh(W_s <-h_1)                                  (Section 3.1, eq. 4; Appendix A.2)
  * the alignment matrix alpha (Figure 3) printed for a held-out sentence
Simplifications: 48-unit GRUs and 16-dim embeddings instead of 1000/620; a toy task ("translate every token through a
dictionary and emit the sentence backwards", so the gold alignment is the anti-diagonal); g is a tanh layer instead of
maxout; Adam instead of Adadelta; greedy decoding instead of beam search.

Run:  python bahdanau_attention.py        (CPU, about 12 s)
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class GRUCell(nn.Module):
    """Gated unit of Appendix A.1.1 / Cho et al. 2014:  z = sigma(W_z x + U_z h),  r = sigma(W_r x + U_r h),
    h~ = tanh(W x + U (r ⊙ h)),  h' = (1 - z) ⊙ h + z ⊙ h~.   The decoder's extra term C c_i is included by giving it
    x = [E y_{i-1}; c_i], which makes W_z x = W_z E y_{i-1} + C_z c_i with W_z = [W_z | C_z] (Appendix A.2.2, eq. 8)."""
    def __init__(self, d_in, d_h):
        super().__init__()
        self.W_z, self.U_z = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h, bias=False)
        self.W_r, self.U_r = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h, bias=False)
        self.W, self.U = nn.Linear(d_in, d_h), nn.Linear(d_h, d_h, bias=False)

    def forward(self, x, h):
        z = torch.sigmoid(self.W_z(x) + self.U_z(h))
        r = torch.sigmoid(self.W_r(x) + self.U_r(h))
        h_tilde = torch.tanh(self.W(x) + self.U(r * h))
        return (1 - z) * h + z * h_tilde


class Encoder(nn.Module):
    """Bidirectional RNN (Section 3.2): forward states ->h_j and backward states <-h_j, concatenated into h_j (eq. 7)."""
    def __init__(self, vocab, d_emb, n, pad):
        super().__init__()
        self.E, self.fwd, self.bwd, self.n, self.pad = nn.Embedding(vocab, d_emb, padding_idx=pad), GRUCell(d_emb, n), GRUCell(d_emb, n), n, pad

    def run(self, cell, x, mask, order):
        h, out = torch.zeros(x.size(0), self.n), [None] * x.size(1)
        for j in order:
            h = torch.where(mask[:, j : j + 1], cell(x[:, j], h), h)          # pads do not update the state
            out[j] = h
        return torch.stack(out, 1)

    def forward(self, src):
        x, mask = self.E(src), src != self.pad
        T = src.size(1)
        fwd = self.run(self.fwd, x, mask, range(T))
        bwd = self.run(self.bwd, x, mask, range(T - 1, -1, -1))
        return torch.cat([fwd, bwd], -1), bwd[:, 0], mask                     # h_j (B, T, 2n), <-h_1 for s_0, mask


class AdditiveAttention(nn.Module):
    """e_ij = v_a^T tanh(W_a s_{i-1} + U_a h_j);  alpha_ij = exp(e_ij) / sum_k exp(e_ik);  c_i = sum_j alpha_ij h_j."""
    def __init__(self, n, n_prime):
        super().__init__()
        self.W_a, self.U_a, self.v_a = nn.Linear(n, n_prime, bias=False), nn.Linear(2 * n, n_prime, bias=False), nn.Linear(n_prime, 1, bias=False)

    def precompute(self, h):                                                   # U_a h_j does not depend on i: once per sentence
        return self.U_a(h)

    def forward(self, s_prev, h, Uh, mask):
        e = self.v_a(torch.tanh(self.W_a(s_prev).unsqueeze(1) + Uh)).squeeze(-1)   # (B, T_x)   eq. 6
        alpha = F.softmax(e.masked_fill(~mask, float("-inf")), dim=-1)              # eq. 5, pads excluded
        c = (alpha.unsqueeze(-1) * h).sum(1)                                        # eq. 5: c_i = sum_j alpha_ij h_j
        return c, alpha


class RNNsearch(nn.Module):
    def __init__(self, vocab, d_emb=16, n=48, n_prime=48, pad=0):
        super().__init__()
        self.pad, self.n = pad, n
        self.encoder = Encoder(vocab, d_emb, n, pad)
        self.E = nn.Embedding(vocab, d_emb, padding_idx=pad)
        self.attn = AdditiveAttention(n, n_prime)
        self.W_s = nn.Linear(n, n)                                             # s_0 = tanh(W_s <-h_1)   (Appendix A.2.2)
        self.dec = GRUCell(d_emb + 2 * n, n)                                   # s_i = f(s_{i-1}, y_{i-1}, c_i)   eq. 4
        self.g_hidden = nn.Linear(n + d_emb + 2 * n, n)                        # g(y_{i-1}, s_i, c_i): tanh layer (maxout in paper)
        self.g_out = nn.Linear(n, vocab)

    def step(self, y_prev_tok, s, h, Uh, mask):
        y_prev = self.E(y_prev_tok)
        c, alpha = self.attn(s, h, Uh, mask)                                   # scores use s_{i-1}, the *previous* state
        s = self.dec(torch.cat([y_prev, c], -1), s)                            # s_i
        logits = self.g_out(torch.tanh(self.g_hidden(torch.cat([s, y_prev, c], -1))))
        return logits, s, alpha

    def forward(self, src, tgt_in):
        h, h_back_1, mask = self.encoder(src)
        Uh, s = self.attn.precompute(h), torch.tanh(self.W_s(h_back_1))
        logits, alphas = [], []
        for i in range(tgt_in.size(1)):
            lg, s, alpha = self.step(tgt_in[:, i], s, h, Uh, mask)
            logits.append(lg); alphas.append(alpha)
        return torch.stack(logits, 1), torch.stack(alphas, 1)                  # (B, T_y, V), (B, T_y, T_x)

    @torch.no_grad()
    def greedy_decode(self, src, bos, max_len):
        h, h_back_1, mask = self.encoder(src)
        Uh, s = self.attn.precompute(h), torch.tanh(self.W_s(h_back_1))
        y, out, alphas = torch.full((src.size(0),), bos, dtype=torch.long), [], []
        for _ in range(max_len):
            lg, s, alpha = self.step(y, s, h, Uh, mask)
            y = lg.argmax(-1); out.append(y); alphas.append(alpha)
        return torch.stack(out, 1), torch.stack(alphas, 1)


def make_batch(B, vocab, dictionary, min_len=3, max_len=8, pad=0, bos=1, eos=2):
    """Target = dictionary[source] in reverse order, then <EOS>: the correct alignment of target word i is source word T-i."""
    L = torch.randint(min_len, max_len + 1, (B,))
    src, tgt = torch.full((B, max_len), pad), torch.full((B, max_len + 2), pad)
    for i, l in enumerate(L.tolist()):
        s = torch.randint(3, vocab, (l,))
        src[i, :l] = s
        tgt[i, 0], tgt[i, 1 : l + 1], tgt[i, l + 1] = bos, dictionary[s].flip(0), eos
    return src, tgt, L


def main():
    t0 = time.time()
    vocab, pad, bos, eos = 20, 0, 1, 2
    dictionary = torch.cat([torch.tensor([0, 1, 2]), torch.randperm(vocab - 3) + 3])
    model = RNNsearch(vocab, pad=pad)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    first = None
    for step in range(1, 901):
        src, tgt, _ = make_batch(64, vocab, dictionary)
        logits, _ = model(src, tgt[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab), tgt[:, 1:].reshape(-1), ignore_index=pad)
        opt.zero_grad(); loss.backward(); opt.step()
        first = first or loss.item()
        if step % 300 == 0:
            print(f"step {step:4d}  nll/token {loss.item():.3f}")

    model.eval()
    src, tgt, L = make_batch(200, vocab, dictionary)
    pred, alphas = model.greedy_decode(src, bos, max_len=tgt.size(1) - 1)
    gold = tgt[:, 1:]; keep = gold != pad
    acc = ((pred == gold) | ~keep).all(1).float().mean().item()
    # alignment quality: for target position i (0-based) the source word is L-1-i; count argmax_j alpha_ij hits
    hits, total = 0, 0
    for b in range(200):
        for i in range(L[b].item()):
            hits += alphas[b, i].argmax().item() == L[b].item() - 1 - i; total += 1
    print(f"exact-match accuracy {acc:.2f};  argmax of alpha_ij lands on the true source word in {100 * hits / total:.0f}% of steps   ({time.time() - t0:.1f} s)")

    b = int((L == 6).nonzero()[0])                                            # print a 6-word example, Figure 3 style
    l = L[b].item()
    print(f"alignment matrix alpha_ij for source {src[b, :l].tolist()} -> output {pred[b, : l + 1].tolist()} (rows = target i, cols = source j):")
    for i in range(l + 1):
        print("   " + " ".join(f"{a:.2f}" for a in alphas[b, i, :l].tolist()) + ("   <EOS>" if i == l else ""))
    assert loss.item() < 0.2 * first and acc > 0.9 and hits / total > 0.9, "attention model did not learn to align and translate"


if __name__ == "__main__":
    main()
