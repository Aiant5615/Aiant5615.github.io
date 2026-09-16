# Paper implementations

One small, self-contained PyTorch file per reviewed paper, written from the paper's equations. Each file trains or runs
the method on a synthetic task in well under a minute on a CPU, prints what it is doing, and asserts that it worked.
The reviews at https://aiant5615.github.io/papers/ show these files inline ("Implementation" section).

- Python 3.10+ with `torch` and `numpy` only; no downloads, no GPU, fixed seeds.
- Run one: `python code/<slug>/<slug>.py`. Run all: `bash code/run_all.sh`.
- Every file starts with a docstring listing what is implemented (with the paper's section or equation numbers) and what
  was simplified.
