"""Training Compute-Optimal Large Language Models (Chinchilla, Hoffmann et al., 2022) — Approach 3 in PyTorch.

What is implemented (section numbers follow the paper):
  * the parametric loss  L_hat(N, D) = E + A / N^alpha + B / D^beta                                (3.3, eq. 2)
  * the fit: minimise the Huber loss (delta = 1e-3) between log L_hat and log L over all runs, with
    L-BFGS from a grid of initialisations, log L_hat written as a logsumexp of (a - alpha log N, b - beta log D, e)  (3.3, eq. 3 / Appendix D)
  * the compute constraint  FLOPs(N, D) ~= 6 N D                                                   (Section 1 / Appendix F)
  * the closed-form optimum  N_opt = G (C/6)^a,  D_opt = G^-1 (C/6)^b,  G = (alpha A / (beta B))^(1/(alpha+beta)),
    a = beta / (alpha + beta),  b = alpha / (alpha + beta)                                          (3.3, eq. 4 / Appendix D.1)
  * a numerical IsoFLOP check of the closed form, and the Gopher-budget prediction (5.76e23 FLOPs)   (3.3, Table 3)
Simplifications: the ~400 training runs are synthetic: losses are generated from the paper's own fitted constants
  (E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28) plus multiplicative noise, so the fit is a recovery test;
  Approaches 1 and 2 (envelope, IsoFLOP parabolas) are not fitted.

Run:  python chinchilla.py        (CPU, about 3 s)
"""
import math, time
import torch

torch.manual_seed(0)

TRUE = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)     # Section 3.3, the paper's fitted values


def loss_law(N, D, E, A, B, alpha, beta):
    """L_hat(N, D) = E + A / N^alpha + B / D^beta   (eq. 2)."""
    return E + A / N ** alpha + B / D ** beta


def synthetic_runs(n_models=15, n_tokens=15, noise=0.005):
    """Fake the paper's grid of runs (Section 3): model sizes 10M..100B, tokens 100M..1T, loss from eq. 2 times log-normal
    noise (0.5%). The paper's real grid is 70M-16B params and 5B-500B tokens; a wider grid makes the exponents identifiable."""
    N = torch.logspace(7, 11, n_models)
    D = torch.logspace(8, 12, n_tokens)
    N, D = torch.meshgrid(N, D, indexing="ij")
    N, D = N.flatten(), D.flatten()
    L = loss_law(N, D, **TRUE) * torch.exp(noise * torch.randn_like(N))
    return N, D, L


def fit_approach3(N, D, L, delta=1e-3):
    """Approach 3 (eq. 3): min over (a, b, e, alpha, beta) of sum_runs Huber_delta( LSE(a - alpha log N, b - beta log D, e) - log L ),
    where A = e^a, B = e^b, E = e^e. L-BFGS from a small grid of initialisations, keeping the best (Appendix D.2)."""
    logN, logD, logL = N.log(), D.log(), L.log()
    best = None
    for a0 in (0.0, 5.0, 10.0):
        for alpha0 in (0.2, 0.5, 0.8):
            for beta0 in (0.2, 0.5, 0.8):
                theta = torch.tensor([a0, a0, 0.0, alpha0, beta0], requires_grad=True)  # (a, b, e, alpha, beta)
                opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

                def objective():
                    a, b, e, alpha, beta = theta
                    pred = torch.logsumexp(torch.stack([a - alpha * logN, b - beta * logD, e.expand_as(logN)]), 0)
                    return torch.nn.functional.huber_loss(pred, logL, delta=delta, reduction="sum")

                def closure():
                    opt.zero_grad(); l = objective(); l.backward(); return l
                opt.step(closure)
                val = objective().item()
                if best is None or val < best[0]:
                    best = (val, theta.detach().clone())
    a, b, e, alpha, beta = best[1].tolist()
    return dict(E=math.exp(e), A=math.exp(a), B=math.exp(b), alpha=alpha, beta=beta), best[0]


def compute_optimal(C, E, A, B, alpha, beta):
    """Minimise eq. 2 subject to 6 N D = C:  N_opt = G (C/6)^a,  D_opt = G^-1 (C/6)^b  (eq. 4)."""
    a, b = beta / (alpha + beta), alpha / (alpha + beta)
    G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
    return G * (C / 6) ** a, (C / 6) ** b / G, a, b


def isoflop_argmin(C, params, n=2000):
    """Numerical check: scan N along the IsoFLOP curve D = C / (6N) and take the loss minimiser (the spirit of Figure 3/4)."""
    N = torch.logspace(6, 14, n)
    L = loss_law(N, C / (6 * N), **params)
    i = L.argmin()
    return N[i].item(), (C / (6 * N[i])).item()


def main():
    t0 = time.time()
    N, D, L = synthetic_runs()
    print(f"{len(L)} synthetic runs: N in [{N.min():.1e}, {N.max():.1e}], D in [{D.min():.1e}, {D.max():.1e}], "
          f"loss in [{L.min():.2f}, {L.max():.2f}]")

    fit, huber = fit_approach3(N, D, L)
    print("Approach 3 fit of L(N, D) = E + A/N^alpha + B/D^beta   (true values in brackets)")
    for k in ("E", "A", "B", "alpha", "beta"):
        print(f"   {k:5s} = {fit[k]:8.3f}   [{TRUE[k]}]")
    print(f"   final Huber objective {huber:.4f}")

    print("compute-optimal allocation under C = 6 N D (eq. 4):")
    _, _, a, b = compute_optimal(1e20, **fit)
    _, _, a_true, b_true = compute_optimal(1e20, **TRUE)
    print(f"   N_opt ~ C^a with a = {a:.3f} (true {a_true:.3f}),  D_opt ~ C^b with b = {b:.3f} (true {b_true:.3f});"
          f"  paper: a = 0.46, b = 0.54  ->  N and D scale ~equally")
    print(f"   {'C (FLOPs)':>10s}  {'N_opt':>9s}  {'D_opt':>9s}  {'D/N':>6s}   numerical IsoFLOP argmin")
    for C in (1e18, 1e20, 1e22, 5.76e23):
        n_opt, d_opt, _, _ = compute_optimal(C, **fit)
        n_num, d_num = isoflop_argmin(C, fit)
        print(f"   {C:10.2e}  {n_opt:9.2e}  {d_opt:9.2e}  {d_opt / n_opt:6.1f}   N {n_num:.2e}  D {d_num:.2e}")
        assert abs(math.log(n_num / n_opt)) < 0.05, "closed-form optimum should match the IsoFLOP scan"
    n_g, d_g, _, _ = compute_optimal(5.76e23, **fit)
    print(f"Gopher's budget 5.76e23 FLOPs: optimal N = {n_g / 1e9:.0f}B parameters on {d_g / 1e12:.1f}T tokens "
          f"(Gopher itself: 280B on 0.3T; Chinchilla: 70B on 1.4T)   ({time.time() - t0:.1f} s)")

    assert abs(fit["alpha"] - TRUE["alpha"]) < 0.05 and abs(fit["beta"] - TRUE["beta"]) < 0.05, "exponents not recovered"
    assert abs(fit["E"] - TRUE["E"]) < 0.1, "irreducible loss not recovered"
    assert abs(a - 0.5) < 0.1 and abs(b - 0.5) < 0.1 and abs(a + b - 1) < 1e-9, "N and D should scale roughly equally"
    assert n_g < 280e9 / 2 and d_g > 1e12, "the Gopher budget calls for a much smaller model on many more tokens"


if __name__ == "__main__":
    main()
