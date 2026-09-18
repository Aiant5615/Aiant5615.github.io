"""FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Dao et al., 2022) — PyTorch.

What is implemented (section numbers follow the paper):
  * standard attention (Algorithm 0): S = Q K^T, P = softmax(S), O = P V with the full N x N matrices     (2.2 / 3.1)
  * the online-softmax decomposition  m(x) = max(m(x1), m(x2)),  l(x) = e^{m1-m} l1 + e^{m2-m} l2         (3.1, eq. "softmax of a concatenation")
  * FlashAttention forward (Algorithm 1): outer loop over K,V blocks, inner loop over Q blocks, running
    row-max m_i and row-sum l_i, block sizes B_c = ceil(M / 4d), B_r = min(B_c, d), the rescaling
    O_i <- diag(l_new)^-1 (diag(l_i) e^{m_i - m_new} O_i + e^{m~_ij - m_new} P~_ij V_j)                   (3.1, Algorithm 1)
  * the logsumexp L = m + log l saved for the backward pass, and the backward with recomputation of
    S_ij, P_ij from Q_i, K_j, L_i instead of storing P (Algorithm 4 in Appendix B)                       (3.1 "Recomputation")
  * causal masking fused into the kernel and skipping of fully-masked blocks (block-sparse idea)          (3.3, Appendix B.2)
  * checks: FlashAttention == standard attention (fwd and bwd) to 1e-5, and the peak size of the
    largest intermediate: N x N for standard vs B_r x B_c for FlashAttention                            (Theorem 2, Section 3.2)
Simplifications: plain PyTorch loops on CPU (the point of the paper is a fused CUDA kernel; here the
  "SRAM" is just Python locals), no dropout, one head at a time.

Run:  python flashattention.py        (CPU, about 1 s)
"""
import math, time
import torch

torch.manual_seed(0)


def standard_attention(Q, K, V, causal=False):
    """Algorithm 0: materialises S and P (N x N) in "HBM"."""
    N, d = Q.shape
    S = Q @ K.T / math.sqrt(d)                                          # write S (N x N)
    if causal:
        S = S.masked_fill(~torch.tril(torch.ones(N, N, dtype=torch.bool)), float("-inf"))
    P = S.softmax(-1)                                                   # write P (N x N)
    return P @ V


def flash_attention_forward(Q, K, V, M=65536, causal=False):
    """Algorithm 1. Q, K, V: (N, d) in HBM; M = on-chip SRAM size in floats. Returns O and the logsumexp L (Algorithm 2 keeps L)."""
    N, d = Q.shape
    tau = 1 / math.sqrt(d)                                               # softmax scaling (Algorithm 2)
    B_c, B_r = math.ceil(M / (4 * d)), min(math.ceil(M / (4 * d)), d)   # line 1: block sizes from the SRAM size
    O = torch.zeros(N, d)                                                # line 2: O = 0, l = 0, m = -inf in HBM
    l = torch.zeros(N)
    m = torch.full((N,), float("-inf"))
    peak_block = 0
    for j in range(0, N, B_c):                                           # line 5: outer loop over K_j, V_j blocks
        K_j, V_j = K[j: j + B_c], V[j: j + B_c]                          # line 6: load K_j, V_j into SRAM
        for i in range(0, N, B_r):                                       # line 7: inner loop over Q_i blocks
            if causal and j > i + B_r - 1:
                continue                                                 # block entirely above the diagonal: skip (block-sparse, 3.3)
            Q_i, O_i, l_i, m_i = Q[i: i + B_r], O[i: i + B_r], l[i: i + B_r], m[i: i + B_r]   # line 8: load into SRAM
            S_ij = tau * Q_i @ K_j.T                                     # line 9:  S_ij = Q_i K_j^T   (B_r x B_c, on chip)
            if causal:
                rows, cols = torch.arange(i, i + S_ij.size(0))[:, None], torch.arange(j, j + S_ij.size(1))[None, :]
                S_ij = S_ij.masked_fill(cols > rows, float("-inf"))      # masking fused into the kernel
            peak_block = max(peak_block, S_ij.numel())
            m_ij = S_ij.max(-1).values                                   # line 10: m~_ij = rowmax(S_ij)
            P_ij = torch.exp(S_ij - m_ij[:, None])                       #          P~_ij = exp(S_ij - m~_ij)
            l_ij = P_ij.sum(-1)                                          #          l~_ij = rowsum(P~_ij)
            m_new = torch.maximum(m_i, m_ij)                             # line 11: m_new = max(m_i, m~_ij)
            l_new = torch.exp(m_i - m_new) * l_i + torch.exp(m_ij - m_new) * l_ij   # l_new = e^{m_i-m_new} l_i + e^{m~_ij-m_new} l~_ij
            O[i: i + B_r] = (torch.exp(m_i - m_new)[:, None] * l_i[:, None] * O_i    # line 12: rescale the old partial output
                             + torch.exp(m_ij - m_new)[:, None] * (P_ij @ V_j)) / l_new[:, None]     # + new block, / l_new
            l[i: i + B_r], m[i: i + B_r] = l_new, m_new                  # line 13: write l_i, m_i back to HBM
    return O, m + torch.log(l), (B_r, B_c, peak_block)                  # L_i = m_i + log l_i (logsumexp, for the backward)


def flash_attention_backward(Q, K, V, O, dO, L, M=65536, causal=False):
    """Algorithm 4 (Appendix B): recompute S_ij, P_ij block-wise from Q_i, K_j and the saved L_i; never store P."""
    N, d = Q.shape
    tau = 1 / math.sqrt(d)
    B_c, B_r = math.ceil(M / (4 * d)), min(math.ceil(M / (4 * d)), d)
    dQ, dK, dV = torch.zeros_like(Q), torch.zeros_like(K), torch.zeros_like(V)
    D = (dO * O).sum(-1)                                                 # D_i = rowsum(dO_i o O_i)
    for j in range(0, N, B_c):
        K_j, V_j = K[j: j + B_c], V[j: j + B_c]
        for i in range(0, N, B_r):
            if causal and j > i + B_r - 1:
                continue
            Q_i, dO_i, L_i, D_i = Q[i: i + B_r], dO[i: i + B_r], L[i: i + B_r], D[i: i + B_r]
            S_ij = tau * Q_i @ K_j.T                                     # recompute S_ij
            if causal:
                rows, cols = torch.arange(i, i + S_ij.size(0))[:, None], torch.arange(j, j + S_ij.size(1))[None, :]
                S_ij = S_ij.masked_fill(cols > rows, float("-inf"))
            P_ij = torch.exp(S_ij - L_i[:, None])                        # P_ij = exp(S_ij - L_i): exact softmax rows, from L
            dV[j: j + B_c] += P_ij.T @ dO_i                              # dV_j += P_ij^T dO_i
            dP_ij = dO_i @ V_j.T                                         # dP_ij = dO_i V_j^T
            dS_ij = P_ij * (dP_ij - D_i[:, None])                        # dS_ij = P_ij o (dP_ij - D_i)
            dQ[i: i + B_r] += tau * dS_ij @ K_j                          # dQ_i += tau dS_ij K_j
            dK[j: j + B_c] += tau * dS_ij.T @ Q_i                        # dK_j += tau dS_ij^T Q_i
    return dQ, dK, dV


def online_softmax_identity_check(N=64):
    """The block decomposition of softmax from Section 3.1 on a random row split in two."""
    x = torch.randn(N) * 5
    x1, x2 = x[: N // 2], x[N // 2:]
    m1, m2 = x1.max(), x2.max()
    f1, f2 = torch.exp(x1 - m1), torch.exp(x2 - m2)
    l1, l2 = f1.sum(), f2.sum()
    m = torch.maximum(m1, m2)
    l = torch.exp(m1 - m) * l1 + torch.exp(m2 - m) * l2
    sm = torch.cat([torch.exp(m1 - m) * f1, torch.exp(m2 - m) * f2]) / l
    assert torch.allclose(sm, x.softmax(0), atol=1e-6)
    return (sm - x.softmax(0)).abs().max().item()


def main():
    t0 = time.time()
    print(f"online-softmax identity, max error: {online_softmax_identity_check():.1e}")
    N, d, M = 1024, 64, 65536                                            # M floats of "SRAM" (256 KiB) -> B_c = 256, B_r = 64
    Q, K, V = (torch.randn(N, d, requires_grad=True) for _ in range(3))
    for causal in (False, True):
        O_ref = standard_attention(Q, K, V, causal)
        O, L, (B_r, B_c, peak_block) = flash_attention_forward(Q.detach(), K.detach(), V.detach(), M, causal)
        err_fwd = (O - O_ref).abs().max().item()
        # backward: autograd through the standard implementation vs Algorithm 4 with recomputation
        dO = torch.randn_like(O)
        dQ_ref, dK_ref, dV_ref = torch.autograd.grad(O_ref, (Q, K, V), dO)
        dQ, dK, dV = flash_attention_backward(Q.detach(), K.detach(), V.detach(), O, dO, L, M, causal)
        err_bwd = max((dQ - dQ_ref).abs().max().item(), (dK - dK_ref).abs().max().item(), (dV - dV_ref).abs().max().item())
        # the logsumexp L is the exact row statistic: exp(S - L) reproduces P
        S = Q.detach() @ K.detach().T / math.sqrt(d)
        if causal:
            S = S.masked_fill(~torch.tril(torch.ones(N, N, dtype=torch.bool)), float("-inf"))
        err_L = (torch.exp(S - L[:, None]) - S.softmax(-1)).abs().max().item()
        print(f"N={N} d={d} causal={causal!s:5}  blocks B_r={B_r} B_c={B_c}  max|O - O_ref| {err_fwd:.1e}  "
              f"max|grad diff| {err_bwd:.1e}  |exp(S-L) - P| {err_L:.1e}")
        assert err_fwd < 1e-5 and err_bwd < 1e-4 and err_L < 1e-5, "FlashAttention must equal standard attention exactly"
    bytes_std, bytes_flash = N * N * 4, peak_block * 4
    print(f"largest intermediate: standard S/P = {N}x{N} = {bytes_std / 2 ** 20:.1f} MiB (quadratic in N); "
          f"FlashAttention block = {B_r}x{B_c} = {bytes_flash / 1024:.0f} KiB, plus O(N) for m, l, L  "
          f"({bytes_std / bytes_flash:.0f}x smaller)   ({time.time() - t0:.1f} s)")
    # HBM-access count of Theorem 2: Theta(N^2 d^2 / M) for FlashAttention vs Theta(N d + N^2) for standard attention
    hbm_flash = 2 * N * d + (N / B_c) * 3 * N * d                        # K,V read once; Q read, O read+written once per K,V block
    hbm_std = 4 * N * d + 4 * N * N                                      # read Q,K,V, write O; write S, read S, write P, read P
    print(f"HBM accesses (floats): standard ~{hbm_std:.2e},  FlashAttention ~{hbm_flash:.2e}  (Theorem 2: Theta(N^2 d^2 / M) = {N * N * d * d / M:.2e})")
    assert bytes_flash * 10 < bytes_std and hbm_flash < hbm_std


if __name__ == "__main__":
    main()
