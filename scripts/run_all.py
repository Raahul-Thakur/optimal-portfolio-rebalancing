"""
End-to-end experiment driver.

Running this script reproduces every numerical artifact used in the paper:

  * solves the dynamic program and saves the value function / policy;
  * trains the ML imitation policy and saves the checkpoint;
  * runs the Monte-Carlo strategy comparison (on-target and off-target starts);
  * computes comparative-static tables (band width vs cost and vs risk aversion);
  * writes all figures to results/figures/ and all tables to results/tables/;
  * dumps a machine-readable metrics summary to results/tables/metrics.json.

``metrics.json`` holds only deterministic, seed-derived quantities, so two runs
of this script from the same seed produce a byte-identical file.  Wall-clock
timings -- machine-dependent by nature -- are written separately to
``results/tables/timings.json`` and are excluded from the reproducibility check
in ``tests/test_reproducibility.py``.

Usage
-----
    python scripts/run_all.py            # full run with config defaults
    python scripts/run_all.py --quick    # smaller/faster run for smoke testing

All randomness is seeded from Config.seed for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")          # headless: select non-interactive backend before viz import
import numpy as np
import pandas as pd

# Make the project root importable when run as a script.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import Config
from src.dp_solver import solve
from src.ml_policy import train_policy, save_checkpoint, policy_action
from src.remedies import (train_cost_aware_policy, make_dead_zone_action,
                          in_band_residual)
from src.simulation import (run_all_strategies, results_table, paired_comparison,
                            simulate_strategy, make_dp_action, make_ml_action,
                            make_periodic_action, make_buy_hold_action)
from src.dynamics import sample_gross_returns
from src import visualization as viz

DEFAULT_OUTDIR = os.path.join(ROOT, "results")

# Dead-zone half-widths swept in the remedy study.  The interesting range
# straddles the network's in-band residual: below it the wrapper does
# nothing, far above it the wrapper starts suppressing correct trades too.
DEADZONE_DELTAS = (0.002, 0.005, 0.01, 0.02, 0.05)


def _ensure_dirs(outdir=None):
    """Create (and return) the figures/tables/checkpoints directories.

    ``outdir`` defaults to the repository's ``results/``.  Redirecting it
    lets a caller -- notably tests/test_reproducibility.py -- regenerate
    every artifact into a scratch directory and diff it against the
    committed one without touching the committed one.
    """
    root = outdir or DEFAULT_OUTDIR
    dirs = tuple(os.path.join(root, name)
                 for name in ("figures", "tables", "checkpoints"))
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs


def _band_vs_gamma(cfg: Config, gammas, t: int = 0):
    """No-trade band at date t as relative risk aversion varies."""
    rows = []
    for g in gammas:
        sol = solve(replace(cfg, gamma=float(g)))
        lo, hi = sol.no_trade_band(t)
        rows.append({"gamma": float(g),
                     "merton_fraction": replace(cfg, gamma=float(g)).merton_fraction,
                     "band_lower": lo, "band_upper": hi, "band_width": hi - lo})
    return pd.DataFrame(rows)


def main(quick: bool = False, outdir: str = None):
    # perf_counter, not time(): the DP solve is a few milliseconds and
    # time() has ~16 ms resolution on Windows, which would report it as 0.
    t_start = time.perf_counter()
    FIG, TAB, CKPT = _ensure_dirs(outdir)

    if quick:
        cfg = Config(n_paths=3000, ml_epochs=80)
        print(">> QUICK MODE (reduced paths/epochs)")
    else:
        cfg = Config()

    metrics = {"config": cfg.to_dict()}
    # Wall-clock measurements are kept out of ``metrics`` so that file stays
    # byte-stable across runs; they are written to timings.json instead.
    timings = {}
    print(f"Merton fraction w* = {cfg.merton_fraction:.4f},  dt = {cfg.dt:.5f}")

    # --------------------------------------------------------------- #
    # Phase 2: Dynamic programming
    # --------------------------------------------------------------- #
    print("\n[1/7] Solving dynamic program ...")
    t0 = time.perf_counter()
    sol = solve(cfg, verbose=False)
    dp_secs = time.perf_counter() - t0
    np.savez_compressed(os.path.join(CKPT, "dp_solution.npz"),
                        x_grid=sol.x_grid, value=sol.value, policy=sol.policy,
                        no_trade_lower=sol.no_trade_lower,
                        no_trade_upper=sol.no_trade_upper)
    lo0, hi0 = sol.no_trade_band(0)
    metrics["dp"] = {
        "band_t0_lower": lo0, "band_t0_upper": hi0, "band_t0_width": hi0 - lo0,
        "merton_fraction": cfg.merton_fraction,
    }
    timings["dp_solve_seconds"] = dp_secs
    print(f"      solved in {dp_secs:.2f}s; no-trade band at t=0 = "
          f"[{lo0:.4f}, {hi0:.4f}] (width {hi0-lo0:.4f})")

    viz.plot_no_trade_region(sol, os.path.join(FIG, "fig1_no_trade_region.png"))
    viz.plot_policy_slices(sol, outpath=os.path.join(FIG, "fig2_policy_slices.png"))
    viz.plot_value_function(sol, outpath=os.path.join(FIG, "fig3_value_function.png"))

    # --------------------------------------------------------------- #
    # Phase 2b: Comparative statics
    # --------------------------------------------------------------- #
    print("\n[2/7] Comparative statics (band vs cost, band vs risk aversion) ...")
    _, (tc_levels, lowers, uppers) = viz.plot_band_vs_cost(
        cfg, outpath=os.path.join(FIG, "fig4_band_vs_cost.png"))
    band_cost_df = pd.DataFrame({
        "tc": tc_levels, "band_lower": lowers, "band_upper": uppers,
        "band_width": uppers - lowers})
    band_cost_df.to_csv(os.path.join(TAB, "band_vs_cost.csv"), index=False)

    gamma_df = _band_vs_gamma(cfg, gammas=[2, 3, 5, 8, 12])
    gamma_df.to_csv(os.path.join(TAB, "band_vs_gamma.csv"), index=False)
    metrics["band_vs_cost"] = band_cost_df.to_dict(orient="list")
    metrics["band_vs_gamma"] = gamma_df.to_dict(orient="list")

    # --------------------------------------------------------------- #
    # Phase 3: ML imitation
    # --------------------------------------------------------------- #
    print("\n[3/7] Training ML imitation policy ...")
    t0 = time.perf_counter()
    model, history = train_policy(sol, cfg, verbose=True)
    ml_secs = time.perf_counter() - t0
    save_checkpoint(model, cfg, os.path.join(CKPT, "ml_policy.pt"))
    viz.plot_training_history(history,
                              os.path.join(FIG, "fig5_training_history.png"))
    viz.plot_dp_vs_ml(sol, model, cfg,
                      outpath=os.path.join(FIG, "fig6_dp_vs_ml_policy.png"))
    metrics["ml"] = {
        "final_train_mse": history.final_train,
        "final_val_mse": history.final_val,
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    timings["ml_train_seconds"] = ml_secs
    print(f"      trained in {ml_secs:.2f}s; final val MSE = "
          f"{history.final_val:.3e}; params = {metrics['ml']['n_params']}")

    # --------------------------------------------------------------- #
    # Phase 4: Strategy evaluation (on-target start)
    # --------------------------------------------------------------- #
    print("\n[4/7] Monte-Carlo evaluation (start at Merton target) ...")
    results = run_all_strategies(cfg, sol, model)
    summary = results_table(results, cfg.gamma)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(TAB, "performance_on_target.csv"), index=False)
    paired = paired_comparison(results, reference="DP optimal")
    pd.DataFrame(paired).to_csv(
        os.path.join(TAB, "paired_comparison_on_target.csv"), index=False)
    viz.plot_wealth_distributions(
        results, os.path.join(FIG, "fig7_wealth_distributions.png"))
    viz.plot_performance_bars(summary,
                              os.path.join(FIG, "fig8_performance_bars.png"))
    metrics["performance_on_target"] = summary
    metrics["paired_on_target"] = paired

    # --------------------------------------------------------------- #
    # Phase 4b: Strategy evaluation (off-target start)
    # --------------------------------------------------------------- #
    print("\n[5/7] Monte-Carlo evaluation (start off-target x0=0.80) ...")
    off_results = _evaluate_off_target(cfg, sol, model, x0=0.80)
    off = results_table(off_results, cfg.gamma)
    off_df = pd.DataFrame(off)
    off_df.to_csv(os.path.join(TAB, "performance_off_target.csv"), index=False)
    off_paired = paired_comparison(off_results, reference="DP optimal")
    pd.DataFrame(off_paired).to_csv(
        os.path.join(TAB, "paired_comparison_off_target.csv"), index=False)
    metrics["performance_off_target"] = off
    metrics["paired_off_target"] = off_paired

    # --------------------------------------------------------------- #
    # Phase 5: Do the proposed remedies actually work?
    # --------------------------------------------------------------- #
    print("\n[6/7] Testing the remedies (dead-zone, cost-aware loss) ...")
    t0 = time.perf_counter()
    aware_model, aware_history = train_cost_aware_policy(sol, cfg)
    timings["cost_aware_train_seconds"] = time.perf_counter() - t0
    save_checkpoint(aware_model, cfg, os.path.join(CKPT, "ml_cost_aware.pt"))

    metrics["remedy_diagnostics"] = {
        "deltas": list(DEADZONE_DELTAS),
        "in_band_residual_plain": in_band_residual(model, sol),
        "in_band_residual_cost_aware": in_band_residual(aware_model, sol),
        "cost_aware_final_val_mse": aware_history.final_val,
        "cost_aware_lambda": cfg.cost_aware_lambda,
    }
    print(f"      in-band mean |y-x|:  plain "
          f"{metrics['remedy_diagnostics']['in_band_residual_plain']:.5f}"
          f"  ->  cost-aware "
          f"{metrics['remedy_diagnostics']['in_band_residual_cost_aware']:.5f}")

    for tag, x0, seed_off in (("on_target", None, 1), ("off_target", 0.80, 7)):
        rem = _evaluate_remedies(cfg, sol, model, aware_model, x0, seed_off)
        rows = results_table(rem, cfg.gamma)
        pd.DataFrame(rows).to_csv(
            os.path.join(TAB, f"remedies_{tag}.csv"), index=False)
        rem_paired = paired_comparison(rem, reference="DP optimal")
        pd.DataFrame(rem_paired).to_csv(
            os.path.join(TAB, f"remedies_paired_{tag}.csv"), index=False)
        metrics[f"remedies_{tag}"] = rows
        metrics[f"remedies_paired_{tag}"] = rem_paired
        if tag == "on_target":
            viz.plot_remedies(rows, DEADZONE_DELTAS,
                              os.path.join(FIG, "fig9_remedies.png"))
            print("      " + "-" * 62)
            _print_table(pd.DataFrame(rows))

    # --------------------------------------------------------------- #
    # Write metrics + console summary
    # --------------------------------------------------------------- #
    print("\n[7/7] Writing metrics summary ...")
    with open(os.path.join(TAB, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    timings["total_seconds"] = time.perf_counter() - t_start
    with open(os.path.join(TAB, "timings.json"), "w") as f:
        json.dump(timings, f, indent=2, default=float)

    print("\n================ PERFORMANCE (on-target start) ================")
    _print_table(summary_df)
    print("  Paired utility advantage of DP optimal (common random numbers):")
    _print_paired(paired)
    print("\n================ PERFORMANCE (off-target start, x0=0.80) ======")
    _print_table(off_df)
    print("  Paired utility advantage of DP optimal (common random numbers):")
    _print_paired(off_paired)

    print(f"\nDone in {time.perf_counter() - t_start:.1f}s. "
          f"Figures -> results/figures/, tables -> results/tables/, "
          f"checkpoints -> results/checkpoints/")
    return metrics


def _evaluate_remedies(cfg: Config, sol, model, aware_model,
                       x0, seed_offset: int):
    """Evaluate both remedies against the baselines on shared paths.

    Uses exactly the return matrix of the corresponding headline experiment
    (same seed offset), so the remedy rows are directly comparable to the
    performance tables rather than to a fresh draw.
    """
    rng = np.random.default_rng(cfg.seed + seed_offset)
    returns = sample_gross_returns(cfg, (cfg.n_paths, cfg.n_steps), rng)

    actions = [("DP optimal", make_dp_action(sol), None),
               ("ML imitation", make_ml_action(model, cfg), sol)]
    for d in DEADZONE_DELTAS:
        actions.append((f"ML + dead-zone {d:g}",
                        make_dead_zone_action(model, cfg, d), sol))
    actions.append(("ML cost-aware", make_ml_action(aware_model, cfg), sol))
    actions.append((f"ML cost-aware + dead-zone {cfg.deadzone_delta:g}",
                    make_dead_zone_action(aware_model, cfg,
                                          cfg.deadzone_delta), sol))

    out = {}
    for name, act, track in actions:
        out[name] = simulate_strategy(cfg, act, returns, name,
                                      dp_for_tracking=track,
                                      initial_fraction=x0)
    return out


def _evaluate_off_target(cfg: Config, sol, model, x0: float):
    """Run all strategies on a shared return matrix starting away from target.

    Returns a dict of StrategyResult keyed by name (so callers can build both
    the summary table and the paired comparison)."""
    rng = np.random.default_rng(cfg.seed + 7)
    returns = sample_gross_returns(cfg, (cfg.n_paths, cfg.n_steps), rng)
    actions = {
        "DP optimal": (make_dp_action(sol), None),
        "ML imitation": (make_ml_action(model, cfg), sol),
        "Periodic": (make_periodic_action(cfg), sol),
        "Buy & hold": (make_buy_hold_action(cfg), sol),
    }
    out = {}
    for name, (act, track) in actions.items():
        out[name] = simulate_strategy(cfg, act, returns, name,
                                      dp_for_tracking=track,
                                      initial_fraction=x0)
    return out


def _print_table(df: pd.DataFrame):
    cols = ["strategy", "certainty_equivalent", "mean_terminal_wealth",
            "std_terminal_wealth", "mean_turnover", "mean_total_cost",
            "tracking_error"]
    cols = [c for c in cols if c in df.columns]
    with pd.option_context("display.float_format", lambda v: f"{v:.5f}",
                           "display.width", 160):
        print(df[cols].to_string(index=False))


def _print_paired(paired):
    for row in paired:
        sig = "yes" if row["reference_better_95pct"] else "no "
        print(f"    DP vs {row['vs']:<13s}: "
              f"adv={row['mean_utility_advantage']:+.3e}  "
              f"SE={row['se']:.1e}  t={row['t_stat']:+.1f}  "
              f"DP better@95%={sig}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smaller/faster run for smoke testing")
    ap.add_argument("--outdir", default=None,
                    help="write figures/tables/checkpoints under this "
                         "directory instead of results/")
    args = ap.parse_args()
    main(quick=args.quick, outdir=args.outdir)
