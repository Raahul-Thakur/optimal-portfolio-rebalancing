"""
Is the imitation failure a property of the problem, or of one lucky bad fit?

The headline experiment trains a single network, from a single seed, at a single
architecture. That is enough to describe a failure but not enough to establish
one: a referee is entitled to ask whether the result survives retraining. This
script answers that by re-running the imitation experiment across

  * ``N_SEEDS`` independent training seeds at the published architecture, and
  * a sweep of architectures (width and depth) at a fixed seed,

recording for each fit the in-band residual, the realized turnover, the
certainty equivalent, and the certainty equivalent after the dead-zone repair.

The DP solution is seed-independent, so it is solved once and shared; only the
student is retrained. Runtime is roughly ``(N_SEEDS + len(ARCHITECTURES))``
trainings, about seven minutes on a laptop CPU -- which is why this lives outside
``run_all.py`` rather than inside it.

Usage
-----
    python scripts/run_robustness.py
    python scripts/run_robustness.py --quick      # fewer seeds, fewer epochs
    python scripts/run_robustness.py --outdir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import Config
from src.dp_solver import solve
from src.dynamics import sample_gross_returns
from src.ml_policy import train_policy
from src.remedies import in_band_residual, make_dead_zone_action
from src.simulation import simulate_strategy, make_dp_action, make_ml_action
from src import visualization as viz

N_SEEDS = 10
ARCHITECTURES = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256),
                 (64, 64, 64), (64, 64, 64, 64)]
REPAIR_DELTA = 0.05


def _evaluate(cfg, sol, model, returns):
    """One trained student: its diagnostic, its economics, and its repair."""
    naive = simulate_strategy(cfg, make_ml_action(model, cfg), returns, "ml")
    repaired = simulate_strategy(
        cfg, make_dead_zone_action(model, cfg, REPAIR_DELTA), returns, "dz")
    dp = simulate_strategy(cfg, make_dp_action(sol), returns, "dp")
    dp_ce = dp.summary(cfg.gamma)["certainty_equivalent"]

    naive_ce = naive.summary(cfg.gamma)["certainty_equivalent"]
    repaired_ce = repaired.summary(cfg.gamma)["certainty_equivalent"]
    gap = dp_ce - naive_ce
    return {
        "in_band_residual": in_band_residual(model, sol),
        "turnover": float(naive.turnover.mean()),
        "turnover_repaired": float(repaired.turnover.mean()),
        "turnover_dp": float(dp.turnover.mean()),
        "ce": naive_ce,
        "ce_repaired": repaired_ce,
        "ce_dp": dp_ce,
        "gap": gap,
        "gap_closed_frac": (repaired_ce - naive_ce) / gap if gap > 0 else np.nan,
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }


def main(quick: bool = False, outdir: str = None) -> dict:
    t_start = time.perf_counter()
    root = outdir or os.path.join(ROOT, "results")
    fig_dir = os.path.join(root, "figures")
    tab_dir = os.path.join(root, "tables")
    for d in (fig_dir, tab_dir):
        os.makedirs(d, exist_ok=True)

    base = Config(n_paths=4000, ml_epochs=80) if quick else Config()
    n_seeds = 3 if quick else N_SEEDS
    architectures = ARCHITECTURES[:3] if quick else ARCHITECTURES

    sol = solve(base)                       # seed-independent: solve once
    rng = np.random.default_rng(base.seed + 1)
    returns = sample_gross_returns(base, (base.n_paths, base.n_steps), rng)
    print(f"DP solved; evaluating on {base.n_paths} shared paths.\n")

    # ---------------------------------------------------------------- seeds
    print(f"[1/2] Retraining at the published architecture, {n_seeds} seeds ...")
    seed_rows = []
    for i in range(n_seeds):
        cfg = replace(base, seed=base.seed + 1000 * i)
        model, hist = train_policy(sol, cfg)
        row = {"seed": cfg.seed, "final_val_mse": hist.final_val}
        row.update(_evaluate(base, sol, model, returns))
        seed_rows.append(row)
        print(f"      seed {cfg.seed:>6}: val MSE={hist.final_val:.2e}  "
              f"residual={row['in_band_residual']:.5f}  "
              f"turnover={row['turnover']:.4f}  "
              f"gap closed by repair={row['gap_closed_frac']:.1%}")
    seeds_df = pd.DataFrame(seed_rows)
    seeds_df.to_csv(os.path.join(tab_dir, "robustness_seeds.csv"), index=False)

    # -------------------------------------------------------- architectures
    print(f"\n[2/2] Sweeping {len(architectures)} architectures at the base seed ...")
    arch_rows = []
    for hidden in architectures:
        cfg = replace(base, ml_hidden=tuple(hidden))
        model, hist = train_policy(sol, cfg)
        row = {"architecture": "x".join(str(h) for h in hidden),
                "depth": len(hidden), "width": hidden[0],
                "final_val_mse": hist.final_val}
        row.update(_evaluate(base, sol, model, returns))
        arch_rows.append(row)
        print(f"      {row['architecture']:>14} ({row['n_params']:>6} params): "
              f"val MSE={hist.final_val:.2e}  "
              f"residual={row['in_band_residual']:.5f}  "
              f"turnover={row['turnover']:.4f}")
    arch_df = pd.DataFrame(arch_rows)
    arch_df.to_csv(os.path.join(tab_dir, "robustness_architectures.csv"), index=False)

    viz.plot_robustness(seeds_df, arch_df,
                        os.path.join(fig_dir, "fig10_robustness.png"))

    summary = {
        "n_seeds": int(n_seeds),
        "repair_delta": REPAIR_DELTA,
        "seeds": {
            "residual_min": float(seeds_df.in_band_residual.min()),
            "residual_max": float(seeds_df.in_band_residual.max()),
            "residual_mean": float(seeds_df.in_band_residual.mean()),
            "turnover_min": float(seeds_df.turnover.min()),
            "turnover_max": float(seeds_df.turnover.max()),
            "turnover_mean": float(seeds_df.turnover.mean()),
            "turnover_std": float(seeds_df.turnover.std(ddof=1)),
            "ce_min": float(seeds_df.ce.min()),
            "ce_max": float(seeds_df.ce.max()),
            "ce_dp": float(seeds_df.ce_dp.iloc[0]),
            "n_underperforming_dp": int((seeds_df.ce < seeds_df.ce_dp).sum()),
            "gap_closed_min": float(seeds_df.gap_closed_frac.min()),
            "gap_closed_mean": float(seeds_df.gap_closed_frac.mean()),
        },
        "architectures": {
            "residual_min": float(arch_df.in_band_residual.min()),
            "residual_max": float(arch_df.in_band_residual.max()),
            "turnover_min": float(arch_df.turnover.min()),
            "turnover_max": float(arch_df.turnover.max()),
            "n_underperforming_dp": int((arch_df.ce < arch_df.ce_dp).sum()),
            "n_architectures": int(len(arch_df)),
            "params_min": int(arch_df.n_params.min()),
            "params_max": int(arch_df.n_params.max()),
        },
    }
    with open(os.path.join(tab_dir, "robustness_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    s = summary["seeds"]
    a = summary["architectures"]
    print("\n" + "=" * 66)
    print(f"Across {n_seeds} seeds: in-band residual "
          f"[{s['residual_min']:.5f}, {s['residual_max']:.5f}], "
          f"turnover {s['turnover_mean']:.4f} +/- {s['turnover_std']:.4f}")
    print(f"  networks underperforming the DP optimum: "
          f"{s['n_underperforming_dp']}/{n_seeds}")
    print(f"  dead-zone repair closes >= {s['gap_closed_min']:.1%} of the gap "
          f"in every case (mean {s['gap_closed_mean']:.1%})")
    print(f"Across {a['n_architectures']} architectures "
          f"({a['params_min']}-{a['params_max']} params): "
          f"residual [{a['residual_min']:.5f}, {a['residual_max']:.5f}], "
          f"underperforming: {a['n_underperforming_dp']}/{a['n_architectures']}")
    print(f"\nDone in {time.perf_counter() - t_start:.1f}s.")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fewer seeds, fewer epochs, fewer architectures")
    ap.add_argument("--outdir", default=None,
                    help="write tables/figures under this directory")
    args = ap.parse_args()
    main(quick=args.quick, outdir=args.outdir)
