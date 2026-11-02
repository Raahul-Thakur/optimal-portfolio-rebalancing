"""
Generate the four narrative notebooks from code, then (optionally) execute them
so that figures and outputs are embedded.

Run from the project root:

    python notebooks/_build_notebooks.py            # build only
    python notebooks/_build_notebooks.py --execute  # build and run all cells

The notebooks import the project ``src`` package and use light-weight settings
(fewer paths/epochs than scripts/run_all.py) so they execute quickly while
telling the same story.  The canonical, paper-quality artifacts come from
scripts/run_all.py and live in results/.
"""

from __future__ import annotations

import argparse
import os
import sys

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# A preamble every notebook runs first: make src importable regardless of the
# kernel's working directory.
PREAMBLE = (
    "import os, sys\n"
    "ROOT = os.path.abspath(os.path.join(os.getcwd(), '..')) "
    "if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd()\n"
    "if ROOT not in sys.path:\n"
    "    sys.path.insert(0, ROOT)\n"
    "%matplotlib inline\n"
    "import numpy as np, pandas as pd, matplotlib.pyplot as plt\n"
)


def md(text):
    return new_markdown_cell(text)


def code(src):
    return new_code_cell(src)


# --------------------------------------------------------------------------- #
def nb_theory():
    cells = [
        md("# 01 · Theory Validation\n"
           "\n"
           "**A Variational Control Framework for Optimal Portfolio Rebalancing "
           "Under Transaction Costs**\n"
           "\n"
           "This notebook validates the building blocks of the model before we "
           "solve any control problem:\n"
           "\n"
           "1. the discrete-time geometric-Brownian-motion return model and its "
           "Gauss–Hermite quadrature;\n"
           "2. the CRRA utility and the frictionless **Merton fraction** "
           "$w^\\* = (\\mu-r)/(\\gamma\\sigma^2)$;\n"
           "3. the homogeneity property that lets us reduce the state space."),
        code(PREAMBLE),
        code("from src.config import Config\n"
             "from src import dynamics\n"
             "cfg = Config()\n"
             "print('Annualized parameters:')\n"
             "print(f'  mu={cfg.mu}, sigma={cfg.sigma}, r={cfg.r}, gamma={cfg.gamma}, tc={cfg.tc}')\n"
             "print(f'  horizon T={cfg.T}y over n_steps={cfg.n_steps} (dt={cfg.dt:.5f})')\n"
             "print(f'  Merton fraction w* = {cfg.merton_fraction:.4f}')"),
        md("## 1. Return model and quadrature\n"
           "\n"
           "Over one step the risky gross return is "
           "$R=\\exp\\big((\\mu-\\tfrac12\\sigma^2)\\,dt+\\sigma\\sqrt{dt}\\,Z\\big)$, "
           "$Z\\sim N(0,1)$, and the risk-free gross return is $R_f=e^{r\\,dt}$.\n"
           "\n"
           "We evaluate conditional expectations with Gauss–Hermite quadrature. "
           "The quadrature moments should match the closed-form lognormal "
           "moments $E[R]=e^{\\mu dt}$ and $E[R^2]=e^{2\\mu dt+\\sigma^2 dt}$."),
        code("R, p = dynamics.gross_return_quadrature(cfg)\n"
             "Rf = dynamics.risk_free_gross(cfg)\n"
             "ER_quad, ER2_quad = np.sum(p*R), np.sum(p*R**2)\n"
             "ER_exact = np.exp(cfg.mu*cfg.dt)\n"
             "ER2_exact = np.exp(2*cfg.mu*cfg.dt + cfg.sigma**2*cfg.dt)\n"
             "print(f'Risk-free gross R_f = {Rf:.6f}')\n"
             "print(f'E[R]:  quadrature={ER_quad:.8f}  exact={ER_exact:.8f}  "
             "err={abs(ER_quad-ER_exact):.2e}')\n"
             "print(f'E[R^2]:quadrature={ER2_quad:.8f}  exact={ER2_exact:.8f}  "
             "err={abs(ER2_quad-ER2_exact):.2e}')\n"
             "print(f'weights sum = {p.sum():.8f}, # nodes = {cfg.n_quad}')"),
        code("# Visualize the one-step gross-return distribution sampled by the quadrature.\n"
             "fig, ax = plt.subplots(figsize=(7,3.6))\n"
             "ax.stem(R, p)\n"
             "ax.axvline(Rf, color='k', ls='--', label=f'risk-free $R_f$={Rf:.4f}')\n"
             "ax.axvline(ER_quad, color='C1', ls=':', label=f'$E[R]$={ER_quad:.4f}')\n"
             "ax.set_xlabel('one-step gross return $R$'); ax.set_ylabel('quadrature weight')\n"
             "ax.set_title('Gauss–Hermite nodes for the risky return'); ax.legend(); plt.show()"),
        md("## 2. CRRA utility and the Merton fraction\n"
           "\n"
           "Preferences are $U(W)=W^{1-\\gamma}/(1-\\gamma)$. In the frictionless, "
           "continuous-time limit the optimal constant risky weight is the "
           "**Merton fraction** $w^\\*=(\\mu-r)/(\\gamma\\sigma^2)$, which will be "
           "the centre of the no-trade region."),
        code("import numpy as np\n"
             "for g in [2,3,5,8,12]:\n"
             "    w = (cfg.mu-cfg.r)/(g*cfg.sigma**2)\n"
             "    print(f'gamma={g:2d}  ->  Merton fraction w* = {w:.4f}')"),
        md("## 3. Homogeneity ⇒ dimensionality reduction\n"
           "\n"
           "Because $U$ is homogeneous of degree $1-\\gamma$ and transaction costs "
           "are proportional, scaling wealth by $\\kappa$ scales every dollar "
           "trade and the next wealth by $\\kappa$ while leaving all *fractions* "
           "unchanged. Hence\n"
           "$$V_t(\\kappa W, x)=\\kappa^{1-\\gamma}V_t(W,x)\\ \\Rightarrow\\ "
           "V_t(W,x)=W^{1-\\gamma}g_t(x).$$\n"
           "The whole problem collapses to a 1-D function $g_t(x)$ of the risky "
           "fraction. We verify the multiplier identity that makes this work."),
        code("from src.transaction_costs import wealth_multiplier, trade_fraction\n"
             "tc = cfg.tc\n"
             "print('Check: post-trade fraction is recovered exactly for any (x,y).')\n"
             "for (x,y) in [(0.30,0.30),(0.10,0.50),(0.80,0.30)]:\n"
             "    m = wealth_multiplier(x,y,tc); d = trade_fraction(x,y,tc)\n"
             "    rec = (x+d)/m\n"
             "    tag = 'no trade' if abs(y-x)<1e-12 else ('buy' if d>0 else 'sell')\n"
             "    print(f'  x={x:.2f} y={y:.2f} [{tag:8s}]  m={m:.5f}  delta={d:+.5f}  recovered_y={rec:.5f}')"),
        md("**Takeaways.** The quadrature reproduces the lognormal moments to "
           "machine precision; the Merton fraction is $0.30$ for our base case "
           "($\\gamma=5$); and the proportional-cost bookkeeping is internally "
           "consistent. We can now solve the control problem in one dimension — "
           "see **02_dynamic_programming**."),
    ]
    nb = new_notebook(cells=cells)
    return nb


# --------------------------------------------------------------------------- #
def nb_dp():
    cells = [
        md("# 02 · Dynamic Programming\n"
           "\n"
           "We solve the finite-horizon Bellman recursion by backward induction "
           "and extract the optimal policy and its **no-trade region**.\n"
           "\n"
           "$$g_t(x)=\\max_y\\ m(x,y)^{1-\\gamma}\\,"
           "\\mathbb{E}_R\\!\\big[\\phi(y,R)^{1-\\gamma}\\,g_{t+1}(x'(y,R))\\big],"
           "\\qquad g_N(x)=\\tfrac{1}{1-\\gamma},$$\n"
           "with $\\phi(y,R)=yR+(1-y)R_f$ and $x'=yR/\\phi$."),
        code(PREAMBLE),
        code("from src.config import Config\n"
             "from src.dp_solver import solve\n"
             "from src import visualization as viz\n"
             "cfg = Config()\n"
             "import time; t0=time.time()\n"
             "sol = solve(cfg)\n"
             "print(f'Solved {cfg.n_steps}-step DP on a {cfg.n_x}-point grid in {time.time()-t0:.3f}s')\n"
             "lo,hi = sol.no_trade_band(0)\n"
             "print(f'No-trade band at t=0: [{lo:.4f}, {hi:.4f}] (width {hi-lo:.4f}); "
             "Merton w*={cfg.merton_fraction:.4f}')"),
        md("## No-trade region over the horizon\n"
           "Inside the shaded band it is optimal **not to trade**; outside it the "
           "investor trades just to the nearest boundary. Note the **horizon "
           "effect**: the band flares open near $t=T$ because there is too little "
           "time left to benefit from re-optimizing the allocation."),
        code("viz.plot_no_trade_region(sol); plt.show()"),
        md("## The optimal policy $y^\\*(x)$\n"
           "The policy is *bang-bang to the boundary*: flat at the lower edge for "
           "under-allocated states, the identity $y=x$ inside the band, and flat "
           "at the upper edge for over-allocated states."),
        code("viz.plot_policy_slices(sol); plt.show()"),
        code("viz.plot_value_function(sol); plt.show()"),
        md("## Comparative statics\n"
           "**Band vs cost.** The no-trade region widens with the transaction-cost "
           "rate (the classical $\\propto tc^{1/3}$ small-cost behaviour)."),
        code("_, (tc_levels, lo, hi) = viz.plot_band_vs_cost(cfg); plt.show()\n"
             "pd.DataFrame({'tc':tc_levels,'lower':lo,'upper':hi,'width':hi-lo})"),
        md("**Band vs risk aversion.** Higher $\\gamma$ lowers the target weight "
           "and tightens the band."),
        code("from dataclasses import replace\n"
             "rows=[]\n"
             "for g in [2,3,5,8,12]:\n"
             "    s = solve(replace(cfg, gamma=float(g)))\n"
             "    l,h = s.no_trade_band(0)\n"
             "    rows.append({'gamma':g,'w*':replace(cfg,gamma=float(g)).merton_fraction,\n"
             "                 'lower':l,'upper':h,'width':h-l})\n"
             "pd.DataFrame(rows)"),
        md("**Takeaway.** Proportional costs convert the single Merton point into "
           "a band of inaction whose width scales with cost and shrinks with risk "
           "aversion — reproducing the classical Davis–Norman / Constantinides "
           "picture. Next we ask whether a neural network can imitate this policy "
           "(**03_ml_imitation**)."),
    ]
    return new_notebook(cells=cells)


# --------------------------------------------------------------------------- #
def nb_ml():
    cells = [
        md("# 03 · Machine-Learning Imitation\n"
           "\n"
           "We treat the DP policy $y^\\*(t,x)$ as a teacher and train a small MLP "
           "(student) by supervised imitation:\n"
           "$$\\text{input } (\\tau, x)\\ \\longrightarrow\\ \\text{target } "
           "y^\\*(t,x),\\qquad \\tau=t/N.$$\n"
           "The student is a 2×64 tanh MLP with a sigmoid output (so $y\\in[0,1]$), "
           "trained with Adam on MSE."),
        code(PREAMBLE),
        code("from src.config import Config\n"
             "from src.dp_solver import solve\n"
             "from src.ml_policy import build_dataset, train_policy, policy_action\n"
             "from src import visualization as viz\n"
             "# Lighter epoch budget for a quick notebook run.\n"
             "cfg = Config(ml_epochs=150)\n"
             "sol = solve(cfg)\n"
             "X, Y = build_dataset(sol)\n"
             "print(f'Dataset: X={X.shape} (columns [tau, x]), Y={Y.shape}')\n"
             "print(f'tau range [{X[:,0].min():.3f},{X[:,0].max():.3f}], "
             "x range [{X[:,1].min():.3f},{X[:,1].max():.3f}]')"),
        code("model, hist = train_policy(sol, cfg, verbose=True)\n"
             "n_params = sum(p.numel() for p in model.parameters())\n"
             "print(f'\\nParameters: {n_params};  final val MSE = {hist.final_val:.3e}')"),
        code("viz.plot_training_history(hist); plt.show()"),
        md("## Learned vs optimal policy\n"
           "The network captures the overall shape but **smooths the kinks** of "
           "the no-trade region: inside the band, where the optimum is exactly "
           "$y=x$, the smooth approximation lies slightly off the diagonal."),
        code("viz.plot_dp_vs_ml(sol, model, cfg); plt.show()"),
        md("## Why smoothness is costly: the no-trade region is hard to imitate\n"
           "A regressor cannot represent the exact identity $y=x$ on a positive-"
           "measure set without infinitely sharp kinks. The residual "
           "$|y_{ML}-x|$ inside the band is tiny in MSE terms yet **non-zero "
           "everywhere**, so the imitation policy keeps trading where the optimum "
           "would hold. We quantify the residual inside the $t=0$ band:"),
        code("lo,hi = sol.no_trade_band(0)\n"
             "xs = np.linspace(lo, hi, 50)\n"
             "y_ml = policy_action(model, np.zeros_like(xs), xs)\n"
             "resid = np.abs(y_ml - xs)\n"
             "print(f'Inside the t=0 no-trade band [{lo:.3f},{hi:.3f}]:')\n"
             "print(f'  mean |y_ML - x| = {resid.mean():.4f}  (max {resid.max():.4f})')\n"
             "print('Each non-zero residual is a trade the optimal policy would NOT make.')\n"
             "print('This is the mechanism behind the ML turnover seen in 04_evaluation.')"),
        md("**Takeaway.** Low supervised error does **not** imply good control: "
           "imitation in *action space* ignores that small errors inside the "
           "no-trade region are economically amplified into turnover and cost. "
           "We measure the consequences in **04_evaluation**."),
    ]
    return new_notebook(cells=cells)


# --------------------------------------------------------------------------- #
def nb_eval():
    cells = [
        md("# 04 · Evaluation\n"
           "\n"
           "We compare four strategies on a **common** set of simulated return "
           "paths (common random numbers):\n"
           "\n"
           "* **DP optimal** — the interpolated dynamic-programming policy;\n"
           "* **ML imitation** — the trained network;\n"
           "* **Periodic** — rebalance to $w^\\*$ every few steps;\n"
           "* **Buy & hold** — trade to $w^\\*$ once, then never.\n"
           "\n"
           "Metrics: terminal wealth, expected utility / certainty equivalent, "
           "turnover, transaction cost, and tracking error vs the optimal policy."),
        code(PREAMBLE),
        code("from src.config import Config\n"
             "from src.dp_solver import solve\n"
             "from src.ml_policy import train_policy\n"
             "from src.simulation import run_all_strategies, results_table, paired_comparison\n"
             "from src import visualization as viz\n"
             "# Lighter settings for a quick notebook run.\n"
             "cfg = Config(n_paths=6000, ml_epochs=150)\n"
             "sol = solve(cfg)\n"
             "model, _ = train_policy(sol, cfg)\n"
             "results = run_all_strategies(cfg, sol, model)\n"
             "summary = results_table(results, cfg.gamma)\n"
             "df = pd.DataFrame(summary)[['strategy','certainty_equivalent','ce_ci_low',\n"
             "    'ce_ci_high','mean_terminal_wealth','mean_turnover','mean_total_cost','tracking_error']]\n"
             "df.round(5)"),
        md("## Terminal-wealth distributions"),
        code("viz.plot_wealth_distributions(results); plt.show()"),
        md("## Performance bars (certainty equivalent with 95% CI, turnover, cost)"),
        code("viz.plot_performance_bars(summary); plt.show()"),
        md("## Is DP *significantly* better? Paired test with common random numbers\n"
           "Because every strategy sees the same returns, the per-path utility "
           "differences have far smaller variance than the individual estimates, "
           "giving a powerful paired test."),
        code("pd.DataFrame(paired_comparison(results, 'DP optimal')).round(6)"),
        md("**Reading the table.** DP beats ML and Periodic with large "
           "$t$-statistics. DP vs Buy & hold is *not* significant when we start "
           "at the target — economically correct, since the wide no-trade band "
           "makes optimal behaviour ≈ \"do nothing\" here.\n"
           "\n"
           "## Starting away from the target\n"
           "When the initial allocation is far from $w^\\*$, rebalancing matters "
           "and DP dominates *all* baselines by trading in most cost-efficiently."),
        code("from src.simulation import (simulate_strategy, make_dp_action, make_ml_action,\n"
             "    make_periodic_action, make_buy_hold_action)\n"
             "from src.dynamics import sample_gross_returns\n"
             "rng = np.random.default_rng(cfg.seed+7)\n"
             "returns = sample_gross_returns(cfg, (cfg.n_paths, cfg.n_steps), rng)\n"
             "x0 = 0.80\n"
             "acts = {'DP optimal':(make_dp_action(sol),None),\n"
             "        'ML imitation':(make_ml_action(model,cfg),sol),\n"
             "        'Periodic':(make_periodic_action(cfg),sol),\n"
             "        'Buy & hold':(make_buy_hold_action(cfg),sol)}\n"
             "off = {n: simulate_strategy(cfg,a,returns,n,dp_for_tracking=t,initial_fraction=x0)\n"
             "       for n,(a,t) in acts.items()}\n"
             "pd.DataFrame(results_table(off, cfg.gamma))[['strategy','certainty_equivalent',\n"
             "    'mean_turnover','mean_total_cost','tracking_error']].round(5)"),
        code("pd.DataFrame(paired_comparison(off, 'DP optimal')).round(6)"),
        md("**Conclusions.**\n"
           "1. The DP policy is statistically the best strategy (or tied with "
           "buy & hold when starting on target).\n"
           "2. The ML imitation, despite a tiny action-space error, **over-trades** "
           "(highest turnover and cost) because it cannot reproduce the no-trade "
           "region — it is the *worst* performer by certainty equivalent.\n"
           "3. Periodic rebalancing wastes cost on unnecessary trades; buy & hold "
           "is surprisingly strong over a one-year horizon when started on target.\n"
           "\n"
           "The practical lesson: imitating optimal *control* requires respecting "
           "the inaction region, not just minimizing action-space MSE. See the "
           "paper's Discussion for cost-aware and dead-zone remedies."),
    ]
    return new_notebook(cells=cells)


# --------------------------------------------------------------------------- #
BUILDERS = {
    "01_theory_validation.ipynb": nb_theory,
    "02_dynamic_programming.ipynb": nb_dp,
    "03_ml_imitation.ipynb": nb_ml,
    "04_evaluation.ipynb": nb_eval,
}


def main(execute: bool):
    for fname, builder in BUILDERS.items():
        nb = builder()
        path = os.path.join(HERE, fname)
        if execute:
            from nbclient import NotebookClient
            client = NotebookClient(nb, timeout=600, kernel_name="python3",
                                    resources={"metadata": {"path": HERE}})
            print(f"Executing {fname} ...")
            client.execute()
        nbformat.write(nb, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true",
                    help="run all cells and embed outputs")
    args = ap.parse_args()
    main(execute=args.execute)
