"""
Plotting utilities for the rebalancing project.

Every figure used in the paper is produced by a function here, each returning a
Matplotlib Figure and (when given an ``outpath``) saving a tight-bbox PNG at
publication resolution.  The module is import-safe in headless environments
(uses the Agg backend) and keeps a single, consistent visual style.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

# NOTE: we deliberately do NOT force a Matplotlib backend here so that the
# functions render inline inside notebooks (``%matplotlib inline``).  Headless
# batch scripts (e.g. scripts/run_all.py) select the non-interactive "Agg"
# backend themselves *before* importing this module.

from .config import Config
from .dp_solver import DPSolution


# Consistent, color-blind-friendly palette and style.
_PALETTE = {
    "DP optimal": "#0072B2",
    "ML imitation": "#D55E00",
    "Periodic": "#009E73",
    "Buy & hold": "#CC79A7",
    "band": "#0072B2",
    "merton": "#000000",
}
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


# PNG writers embed a creation timestamp (and the Matplotlib version) by
# default, which makes otherwise identical figures differ byte-for-byte between
# runs.  Suppressing those chunks is what lets ``scripts/run_all.py`` regenerate
# bit-identical figures from the same seed -- see tests/test_reproducibility.py.
_DETERMINISTIC_PNG_METADATA = {"Software": None, "Date": None}


def _save(fig, outpath: Optional[str]):
    if outpath:
        fig.savefig(outpath, bbox_inches="tight",
                    metadata=_DETERMINISTIC_PNG_METADATA)
    return fig


# --------------------------------------------------------------------------- #
# 1. No-trade region over time
# --------------------------------------------------------------------------- #
def plot_no_trade_region(sol: DPSolution, outpath: Optional[str] = None):
    """Shaded no-trade band [lower(t), upper(t)] across the horizon."""
    cfg = sol.cfg
    t = np.arange(cfg.n_steps)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(t, sol.no_trade_lower, sol.no_trade_upper,
                    color=_PALETTE["band"], alpha=0.25, label="No-trade region")
    ax.plot(t, sol.no_trade_lower, color=_PALETTE["band"], lw=1.5)
    ax.plot(t, sol.no_trade_upper, color=_PALETTE["band"], lw=1.5)
    ax.axhline(cfg.merton_fraction, color=_PALETTE["merton"], ls="--", lw=1.2,
               label=f"Merton target $w^*={cfg.merton_fraction:.3f}$")
    ax.set_xlabel("Rebalancing date $t$")
    ax.set_ylabel("Risky allocation fraction")
    ax.set_title("Optimal no-trade region over the investment horizon")
    ax.set_ylim(cfg.x_min, cfg.x_max)
    ax.legend(loc="upper left")
    return _save(fig, outpath)


# --------------------------------------------------------------------------- #
# 2. Policy function y*(x) at selected dates
# --------------------------------------------------------------------------- #
def plot_policy_slices(sol: DPSolution, dates=None, outpath: Optional[str] = None):
    """Optimal post-trade fraction y*(x) versus pre-trade fraction x."""
    cfg = sol.cfg
    if dates is None:
        dates = [0, cfg.n_steps // 2, cfg.n_steps - 1]
    fig, ax = plt.subplots(figsize=(6.4, 6))
    x = sol.x_grid
    ax.plot(x, x, color="0.6", ls=":", lw=1.2, label="No-trade line $y=x$")
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(dates)))
    for c, t in zip(cmap, dates):
        ax.plot(x, sol.policy[t], color=c, lw=1.8,
                label=f"$t={t}$ ($\\tau={t/cfg.n_steps:.2f}$)")
    ax.axvline(cfg.merton_fraction, color=_PALETTE["merton"], ls="--", lw=1.0,
               alpha=0.7)
    ax.set_xlabel("Pre-trade fraction $x$")
    ax.set_ylabel("Optimal post-trade fraction $y^*(x)$")
    ax.set_title("Optimal rebalancing policy (no-trade region = flat-on-diagonal)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    return _save(fig, outpath)


# --------------------------------------------------------------------------- #
# 3. DP vs ML policy comparison
# --------------------------------------------------------------------------- #
def plot_dp_vs_ml(sol: DPSolution, ml_model, cfg: Config,
                  dates=None, outpath: Optional[str] = None):
    """Overlay the optimal DP action and the learned ML action at given dates."""
    from .ml_policy import policy_action
    if dates is None:
        dates = [0, cfg.n_steps // 2, cfg.n_steps - 1]
    x = sol.x_grid
    fig, axes = plt.subplots(1, len(dates), figsize=(4.6 * len(dates), 4.2),
                             sharey=True)
    if len(dates) == 1:
        axes = [axes]
    for ax, t in zip(axes, dates):
        tau = t / cfg.n_steps
        y_ml = policy_action(ml_model, np.full_like(x, tau), x)
        ax.plot(x, x, color="0.6", ls=":", lw=1.0, label="$y=x$")
        ax.plot(x, sol.policy[t], color=_PALETTE["DP optimal"], lw=1.8,
                label="DP optimal")
        ax.plot(x, y_ml, color=_PALETTE["ML imitation"], lw=1.8, ls="--",
                label="ML imitation")
        ax.set_title(f"$t={t}$ ($\\tau={tau:.2f}$)")
        ax.set_xlabel("Pre-trade fraction $x$")
    axes[0].set_ylabel("Post-trade fraction $y$")
    axes[0].legend(loc="upper left")
    fig.suptitle("Learned policy vs optimal policy", y=1.02)
    return _save(fig, outpath)


# --------------------------------------------------------------------------- #
# 4. ML training history
# --------------------------------------------------------------------------- #
def plot_training_history(history, outpath: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    epochs = np.arange(1, len(history.train_loss) + 1)
    ax.plot(epochs, history.train_loss, color=_PALETTE["DP optimal"],
            label="Train MSE")
    ax.plot(epochs, history.val_loss, color=_PALETTE["ML imitation"],
            label="Validation MSE")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (log scale)")
    ax.set_title("Imitation-learning training curve")
    ax.legend()
    return _save(fig, outpath)


# --------------------------------------------------------------------------- #
# 5. Terminal-wealth distributions
# --------------------------------------------------------------------------- #
def plot_wealth_distributions(results: Dict, outpath: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(
        min(r.terminal_wealth.min() for r in results.values()),
        np.percentile(np.concatenate([r.terminal_wealth for r in results.values()]),
                      99.5),
        80)
    for name, res in results.items():
        ax.hist(res.terminal_wealth, bins=bins, histtype="step", lw=1.8,
                color=_PALETTE.get(name, None), label=name, density=True)
    ax.set_xlabel("Terminal wealth $W_T$ (initial wealth $=1$)")
    ax.set_ylabel("Density")
    ax.set_title("Terminal-wealth distributions across strategies")
    ax.legend()
    return _save(fig, outpath)


# --------------------------------------------------------------------------- #
# 6. Performance bars (certainty equivalent, turnover, cost)
# --------------------------------------------------------------------------- #
def plot_performance_bars(summary_rows, outpath: Optional[str] = None):
    names = [r["strategy"] for r in summary_rows]
    colors = [_PALETTE.get(n, "0.5") for n in names]
    metrics = [
        ("certainty_equivalent", "Certainty equivalent"),
        ("mean_turnover", "Mean turnover"),
        ("mean_total_cost", "Mean transaction cost"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, title) in zip(axes, metrics):
        vals = [r[key] for r in summary_rows]
        yerr = None
        if key == "certainty_equivalent" and "ce_ci_low" in summary_rows[0]:
            lo = np.array([r["ce_ci_low"] for r in summary_rows])
            hi = np.array([r["ce_ci_high"] for r in summary_rows])
            yerr = np.vstack([np.array(vals) - lo, hi - np.array(vals)])
        ax.bar(names, vals, color=colors, yerr=yerr, capsize=3,
               error_kw={"elinewidth": 1, "ecolor": "0.3"})
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        if key == "certainty_equivalent":
            ax.set_ylim(min(vals) * 0.999, max(vals) * 1.001)
    fig.suptitle("Strategy performance comparison", y=1.03)
    return _save(fig, outpath)


# --------------------------------------------------------------------------- #
# 7. No-trade band width vs transaction-cost level
# --------------------------------------------------------------------------- #
def plot_band_vs_cost(cfg: Config, tc_levels=None, t: int = 0,
                      outpath: Optional[str] = None):
    """
    No-trade band boundaries at date ``t`` as the transaction-cost rate varies.

    Reproduces the classical comparative static: the band widens with cost
    (roughly proportional to tc^{1/3} for small tc).
    """
    from dataclasses import replace
    from .dp_solver import solve
    if tc_levels is None:
        tc_levels = np.array([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03])
    lowers, uppers = [], []
    for tc in tc_levels:
        sol = solve(replace(cfg, tc=float(tc)))
        lo, hi = sol.no_trade_band(t)
        lowers.append(lo); uppers.append(hi)
    lowers = np.array(lowers); uppers = np.array(uppers)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.fill_between(tc_levels * 100, lowers, uppers, color=_PALETTE["band"],
                    alpha=0.25, label="No-trade region")
    ax.plot(tc_levels * 100, lowers, "o-", color=_PALETTE["band"], lw=1.5, ms=4)
    ax.plot(tc_levels * 100, uppers, "o-", color=_PALETTE["band"], lw=1.5, ms=4)
    ax.axhline(cfg.merton_fraction, color=_PALETTE["merton"], ls="--", lw=1.2,
               label=f"Merton target $w^*={cfg.merton_fraction:.3f}$")
    ax.set_xlabel("Transaction-cost rate (%)")
    ax.set_ylabel("Risky allocation fraction")
    ax.set_title(f"No-trade region widens with transaction costs (t={t})")
    ax.legend(loc="upper left")
    return _save(fig, outpath), (tc_levels, lowers, uppers)


# --------------------------------------------------------------------------- #
# 9. Do the remedies work?
# --------------------------------------------------------------------------- #
def plot_remedies(summary_rows, deltas, outpath: Optional[str] = None):
    """
    Certainty equivalent and turnover of the dead-zone wrapper as the dead-zone
    half-width varies, against the DP optimum and the two ML endpoints.

    The story the figure has to tell is whether restoring an inaction region
    recovers the performance that naive imitation gives away.
    """
    by_name = {r["strategy"]: r for r in summary_rows}
    dz = [by_name[f"ML + dead-zone {d:g}"] for d in deltas]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    refs = [("DP optimal", _PALETTE["DP optimal"], "-"),
            ("ML imitation", _PALETTE["ML imitation"], "--"),
            ("ML cost-aware", _PALETTE["Periodic"], ":")]

    for ax, key, label in ((axes[0], "certainty_equivalent", "Certainty equivalent"),
                           (axes[1], "mean_turnover", "Mean turnover")):
        ax.plot(deltas, [r[key] for r in dz], "o-",
                color=_PALETTE["band"], lw=1.8, label="ML + dead-zone")
        for name, color, ls in refs:
            if name in by_name:
                ax.axhline(by_name[name][key], color=color, ls=ls, lw=1.4,
                           label=name)
        ax.set_xscale("log")
        # Label the deltas actually swept: the default log ticks show only
        # 10^-2, which hides every value the reader needs to read off.
        ax.set_xticks(list(deltas))
        ax.set_xticklabels([f"{d:g}" for d in deltas])
        ax.minorticks_off()
        ax.set_xlabel(r"Dead-zone half-width $\delta$")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[1].set_yscale("log")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Restoring the inaction region repairs the learned policy")
    fig.tight_layout()
    return _save(fig, outpath)


# --------------------------------------------------------------------------- #
# 10. Robustness of the failure across seeds and architectures
# --------------------------------------------------------------------------- #
def plot_robustness(seeds_df, arch_df, outpath: Optional[str] = None):
    """
    Is the imitation failure universal, or an artifact of one fit?

    Left: turnover across retraining seeds, against the optimal policy's.
    Right: in-band residual against network capacity.  Both are plotted on log
    scales because the quantity of interest spans orders of magnitude.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: side-by-side bars, not overlaid -- overlaying makes it impossible to
    # tell an occluded bar from a short one.
    ax = axes[0]
    idx = np.arange(len(seeds_df))
    w = 0.4
    ax.bar(idx - w / 2, seeds_df["turnover"], w,
           color=_PALETTE["ML imitation"], label="ML imitation")
    ax.bar(idx + w / 2, seeds_df["turnover_repaired"], w,
           color=_PALETTE["Periodic"], label="after dead-zone repair")
    ax.axhline(float(seeds_df["turnover_dp"].iloc[0]),
               color=_PALETTE["DP optimal"], ls="-", lw=1.6, label="DP optimal")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(int(s)) for s in seeds_df["seed"]],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Mean turnover (log scale)")
    ax.set_title(f"Every seed over-trades ({len(seeds_df)} retrainings)")
    ax.legend(fontsize=9, loc="lower right")

    # Right: scatter by depth.  A connecting line would imply a single trend
    # through what are really two families (2-layer widths, and 64-wide depths),
    # and would double back on itself in parameter order.
    ax = axes[1]
    markers = {2: "o", 3: "s", 4: "^"}
    for depth, grp in arch_df.groupby("depth"):
        grp = grp.sort_values("n_params")
        ax.scatter(grp["n_params"], grp["in_band_residual"],
                   marker=markers.get(int(depth), "D"), s=55,
                   color=_PALETTE["ML imitation"],
                   label=f"{int(depth)} hidden layers")
    # Alternate the label offset: the 3- and 4-layer points sit close together
    # in both axes and their labels would otherwise overlap.
    ordered = arch_df.sort_values("n_params").reset_index(drop=True)
    for i, r in ordered.iterrows():
        ax.annotate(r["architecture"], (r["n_params"], r["in_band_residual"]),
                    textcoords="offset points",
                    xytext=(0, 9 if i % 2 == 0 else -16), fontsize=7,
                    ha="center")
    ax.axhline(0.0, color=_PALETTE["DP optimal"], lw=1.6,
               label="DP optimal (exactly zero)")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel(r"In-band mean $|\hat y - x|$")
    ax.set_ylim(0.0, float(arch_df["in_band_residual"].max()) * 1.35)
    ax.set_title("More capacity does not restore inaction")
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("The failure is structural, not an artifact of one fit")
    fig.tight_layout()
    return _save(fig, outpath)


# --------------------------------------------------------------------------- #
# 8. Value function slices
# --------------------------------------------------------------------------- #
def plot_value_function(sol: DPSolution, dates=None, outpath: Optional[str] = None):
    cfg = sol.cfg
    if dates is None:
        dates = [0, cfg.n_steps // 2, cfg.n_steps - 1]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.8, len(dates)))
    for c, t in zip(cmap, dates):
        ax.plot(sol.x_grid, sol.value[t], color=c, lw=1.8,
                label=f"$t={t}$")
    ax.axvline(cfg.merton_fraction, color=_PALETTE["merton"], ls="--", lw=1.0)
    ax.set_xlabel("Pre-trade fraction $x$")
    ax.set_ylabel(r"Reduced value $g_t(x)$")
    ax.set_title("Reduced value function across the horizon")
    ax.legend()
    return _save(fig, outpath)


if __name__ == "__main__":
    from .dp_solver import solve
    cfg = Config()
    sol = solve(cfg)
    plot_no_trade_region(sol, "results/figures/_smoke_no_trade.png")
    print("Saved smoke-test figure.")
