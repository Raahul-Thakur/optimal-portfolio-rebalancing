# A Variational Control Framework for Optimal Portfolio Rebalancing Under Transaction Costs

A reproducible study of portfolio rebalancing as a finite-horizon stochastic
control problem. We solve the optimal rebalancing policy by dynamic programming,
show that proportional transaction costs induce a **no-trade region**, and ask
whether a small neural network can imitate the optimal control as a fast
approximation.

The contribution is **educational and computational**, not a new financial
model: we reproduce classical optimal-control results, visualize the structure
of the optimal policy, and benchmark machine learning as a cheap surrogate for
optimal portfolio control.

📄 **The paper:** [`paper/main.pdf`](paper/main.pdf) (18 pages) — built from
[`paper/main.tex`](paper/main.tex).

---

## Key results

| Result | Value |
|---|---|
| Frictionless Merton fraction `w*` | **0.300** |
| Optimal no-trade band at `t=0` (1% cost) | **[0.235, 0.365]**, width 0.130 |
| Band widening with cost | 0.04 (0.05% cost) → 0.33 (3% cost) |
| DP solve time (52 steps, 201 grid) | **~0.02 s** |
| ML imitation network | 2×64 MLP, **4,417 params**, val MSE `2.1e-4` |
| DP vs ML (paired, common random numbers) | DP wins, `t ≈ 20` (highly significant) |
| ML over-trading | turnover **0.214** vs DP **3e-5**, despite tracking error only `5e-3` |
| Repair (dead-zone, δ=0.05) | closes **100%** of the CE gap; `t` vs DP falls `20.1 → -0.4` |
| Repair (cost-aware loss) | in-band residual **0.0096 → 0.0026**; closes 54% alone |
| Robustness | fails at **10/10 seeds** and **7/7 architectures** (337–67k params); largest net has the *worst* residual |

The headline finding: a naive supervised regressor learns the policy *shape*
accurately (low MSE) yet **fails to reproduce the no-trade region**, because the
"do nothing" set has measure zero for a smooth function. The resulting tiny
action errors trigger continuous spurious trading, eroding performance through
transaction costs.

---

## Repository layout

```
.
├── notebooks/                  # narrative walkthroughs (run top-to-bottom)
│   ├── 01_theory_validation.ipynb
│   ├── 02_dynamic_programming.ipynb
│   ├── 03_ml_imitation.ipynb
│   └── 04_evaluation.ipynb
├── src/                        # the reusable engine
│   ├── config.py               # all hyperparameters (single source of truth)
│   ├── dynamics.py             # GBM returns + Gauss-Hermite quadrature
│   ├── transaction_costs.py    # proportional-cost wealth bookkeeping
│   ├── dp_solver.py            # backward-induction DP, no-trade region extraction
│   ├── ml_policy.py            # imitation-learning MLP (PyTorch)
│   ├── simulation.py           # Monte-Carlo strategy evaluation + paired tests
│   ├── visualization.py        # all paper figures
│   └── remedies.py             # dead-zone wrapper + cost-aware training
├── tests/                      # pytest suite (see "Tests" below)
│   ├── test_dynamics.py
│   ├── test_transaction_costs.py
│   ├── test_dp_solver.py
│   ├── test_ml_policy.py
│   ├── test_simulation.py
│   ├── test_remedies.py        # the dead-zone and cost-aware repairs
│   ├── test_robustness.py      # the seed/architecture sweep claims
│   ├── test_reproducibility.py # byte-for-byte rerun check
│   └── test_paper_consistency.py  # the manuscript vs the generated results
├── scripts/
│   ├── run_all.py              # reproduce every figure/table/checkpoint
│   ├── run_robustness.py       # seed + architecture sweep (~10 min)
│   └── build_paper.py          # check the LaTeX sources and build main.pdf
├── results/
│   ├── figures/                # fig1..fig10 PNGs
│   ├── tables/                 # CSVs + metrics.json
│   └── checkpoints/            # dp_solution.npz, ml_policy.pt, ml_cost_aware.pt
├── paper/
│   ├── main.pdf                # the typeset paper
│   ├── main.tex                # LaTeX source (authoritative)
│   ├── paper_preview.html      # browser-readable preview (MathJax + figures)
│   └── refs.bib                # bibliography
├── requirements.txt            # minimum supported versions
├── requirements-lock.txt       # the exact environment the results came from
├── pytest.ini                  # test config; registers the `slow` marker
├── CITATION.cff                # how to cite this work
├── LICENSE                     # MIT
└── README.md
```

---

## Quick start

```bash
pip install -r requirements.txt
```

Reproduce the solver, the imitation study and the remedies in about 40 seconds:

```bash
python scripts/run_all.py
```

Faster smoke test (fewer paths and epochs):

```bash
python scripts/run_all.py --quick
```

The robustness study — 10 retraining seeds and 7 architectures, establishing
that the imitation failure is not an artifact of one fit — is separate because
it takes about seven minutes:

```bash
python scripts/run_robustness.py
```

Or step through the narrative notebooks:

```bash
jupyter lab notebooks/
```

Individual modules are runnable for self-checks, e.g.:

```bash
python -m src.dp_solver
```

---

## The model in one paragraph

An investor allocates wealth between one risky asset (discrete-time geometric
Brownian motion) and one risk-free asset, rebalancing at `n_steps` dates and
paying a proportional cost `tc` on traded value, to maximize expected terminal
CRRA utility `U(W) = W^{1-γ}/(1-γ)`. Because CRRA utility is homogeneous and
costs are proportional, the value function factorizes as
`V_t(W, x) = W^{1-γ} g_t(x)`, collapsing the control problem to a
one-dimensional function of the risky fraction `x`. Backward induction over a
discretized `(state, action)` grid — with the return expectation evaluated by
Gauss-Hermite quadrature — yields the optimal policy `y*(t, x)` and its
no-trade region. Full derivation in [`paper/main.tex`](paper/main.tex), §3.

---

## Data

**There is no dataset.** This is a pure simulation study: every return path is
drawn from the model's own geometric Brownian motion under `Config.seed`, and
no historical prices, external files or estimation steps enter the pipeline at
any point. The parameters in [`src/config.py`](src/config.py) (`mu=0.08`,
`sigma=0.20`, `r=0.02`, `gamma=5.0`, `tc=0.01`) are **stylized and
illustrative, not calibrated** — round numbers of the right order for a broad
equity index over one year, chosen so the Merton target sits in the interior of
`[0, 1]` and the no-trade band is legible.

So every result is a statement about *this model*, not an empirical claim about
any market. The transportable content is the qualitative structure — the
existence of the no-trade region, the direction of its comparative statics, the
failure mode of action-space imitation. The point estimates are not. See
§4 "Data provenance" and §8(vii) in the paper.

---

## Reproducibility

All randomness derives from `Config.seed` (default `12345`), and every economic
and numerical parameter lives in [`src/config.py`](src/config.py). Re-running
`scripts/run_all.py` regenerates the committed figures, tables and checkpoints
**byte for byte** — verified by `tests/test_reproducibility.py`, which reruns
the whole pipeline into a scratch directory and diffs the MD5 of every artifact.

Two details make that guarantee hold, and are worth knowing if you extend the code:

- **Figures carry no timestamp.** Matplotlib writes a PNG `Date` chunk by
  default, which alone would make identical plots differ between runs;
  `src/visualization.py` suppresses it.
- **Timings live outside the metrics.** Wall-clock measurements are
  machine-dependent, so they go to `results/tables/timings.json` (gitignored)
  rather than into `metrics.json`.

The environment the committed results were produced in — Python 3.10.11, NumPy
1.26.4, SciPy 1.15.3, pandas 2.2.3, Matplotlib 3.10.3, PyTorch 2.7.1 (CPU; the
default Windows wheel reports `+cu118`, but no GPU is used) — is pinned in
[`requirements-lock.txt`](requirements-lock.txt).

Install with `python -m pip`, **not** a bare `pip`: on the machine these results
were produced, `pip` on `PATH` belonged to Python 3.11 while `python` was 3.10,
so a bare `pip install` would have populated the wrong interpreter.

```bash
python -m pip install -r requirements-lock.txt
```

**Scope of the guarantee.** Byte-for-byte identity is verified on a single
machine and a single pinned environment. Across platforms and library versions,
the reported *values* reproduce, but exact bytes need not: PNG encoding depends
on the Matplotlib version, and floating-point reduction order in PyTorch can
depend on the CPU thread count. The notebooks in `notebooks/` are also outside
the guarantee — they are committed with their outputs as narrative documents
and are not regenerated or diffed by `run_all.py` or the test suite.

---

## Tests

```bash
pytest
```

The suite checks the paper's claims, not just that the code runs:

| File | What it pins down |
|---|---|
| `test_dynamics.py` | Gauss-Hermite quadrature against the analytic lognormal moments |
| `test_transaction_costs.py` | the wealth-update fixed point; no-trade is *exactly* costless |
| `test_dp_solver.py` | band brackets `w*`, widens with `tc` at a fitted log-log slope ≈ 1/3, narrows with `γ`, collapses at `tc=0`, stable under grid/quadrature refinement |
| `test_ml_policy.py` | the network is off the `y=x` diagonal at *every* point inside the band — the paper's mechanism |
| `test_simulation.py` | wealth bookkeeping, certainty equivalents, and that pairing beats unpaired comparison |
| `test_remedies.py` | the dead-zone restores exact inaction; the cost-aware loss shrinks the residual; both close the CE gap |
| `test_robustness.py` | every seed and every architecture over-trades; the repair works for all of them |
| `test_reproducibility.py` | a fresh full run is byte-identical to the committed results |
| `test_paper_consistency.py` | every figure, table cell and quoted statistic in `main.tex` matches `metrics.json` |

The reproducibility test runs the full pipeline and takes about half a minute. Skip it with:

```bash
pytest -m "not slow"
```

---

## Building the paper

[`paper/main.pdf`](paper/main.pdf) is committed, so you only need this if you edit the source:

```bash
python scripts/build_paper.py
```

This checks the sources (balanced environments, every figure present, every
citation and reference resolving, no orphaned figures) and then runs the full
pdflatex → bibtex → pdflatex → pdflatex cycle. Use `--check` to validate without
building. It needs a TeX distribution on `PATH`; without one, upload the
repository to [Overleaf](https://overleaf.com) and compile `main.tex` there, or
open [`paper/paper_preview.html`](paper/paper_preview.html) in a browser.

---

## License

MIT — see [`LICENSE`](LICENSE). If you use this work, see
[`CITATION.cff`](CITATION.cff).
