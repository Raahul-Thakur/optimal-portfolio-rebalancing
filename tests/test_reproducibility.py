"""
End-to-end reproducibility.

The claim the repository makes is strong: re-running ``scripts/run_all.py`` from
the same seed regenerates every committed artifact *byte for byte*.  This module
tests exactly that, by running the full pipeline into a scratch directory and
diffing it against ``results/``.

Two categories are deliberately exempt:

* ``results/tables/timings.json`` -- wall-clock measurements, machine-dependent
  by construction, which is why they are not in ``metrics.json``;
* PNG creation timestamps, which ``src.visualization`` suppresses so that the
  figures themselves are stable (see ``_DETERMINISTIC_PNG_METADATA``).

The full run takes roughly half a minute.  Skip it with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

EXPECTED_ARTIFACTS = [
    "figures/fig1_no_trade_region.png",
    "figures/fig2_policy_slices.png",
    "figures/fig3_value_function.png",
    "figures/fig4_band_vs_cost.png",
    "figures/fig5_training_history.png",
    "figures/fig6_dp_vs_ml_policy.png",
    "figures/fig7_wealth_distributions.png",
    "figures/fig8_performance_bars.png",
    "figures/fig9_remedies.png",
    "tables/band_vs_cost.csv",
    "tables/band_vs_gamma.csv",
    "tables/metrics.json",
    "tables/performance_on_target.csv",
    "tables/performance_off_target.csv",
    "tables/paired_comparison_on_target.csv",
    "tables/paired_comparison_off_target.csv",
    "tables/remedies_on_target.csv",
    "tables/remedies_off_target.csv",
    "tables/remedies_paired_on_target.csv",
    "tables/remedies_paired_off_target.csv",
    "checkpoints/dp_solution.npz",
    "checkpoints/ml_policy.pt",
    "checkpoints/ml_cost_aware.pt",
]

# Produced by scripts/run_robustness.py, not run_all.py.  That sweep retrains
# the network seventeen times, so it is checked for existence and internal
# consistency (tests/test_robustness.py) rather than rerun for byte-identity.
ROBUSTNESS_ARTIFACTS = [
    "figures/fig10_robustness.png",
    "tables/robustness_seeds.csv",
    "tables/robustness_architectures.csv",
    "tables/robustness_summary.json",
]


def _md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory, project_root):
    """Run the full pipeline into a scratch directory."""
    outdir = tmp_path_factory.mktemp("rerun")
    proc = subprocess.run(
        [sys.executable, os.path.join("scripts", "run_all.py"),
         "--outdir", str(outdir)],
        cwd=project_root, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return str(outdir)


# --------------------------------------------------------------------------- #
# The committed artifacts exist and are well-formed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rel", EXPECTED_ARTIFACTS + ROBUSTNESS_ARTIFACTS)
def test_committed_artifact_exists(results_dir, rel):
    path = os.path.join(results_dir, rel)
    assert os.path.isfile(path), f"missing committed artifact: {rel}"
    assert os.path.getsize(path) > 0


def test_committed_dp_checkpoint_is_loadable(results_dir):
    data = np.load(os.path.join(results_dir, "checkpoints", "dp_solution.npz"))
    assert set(data.files) == {"x_grid", "value", "policy",
                               "no_trade_lower", "no_trade_upper"}
    assert data["policy"].shape == (52, 201)
    assert np.all(np.isfinite(data["value"]))


def test_metrics_json_carries_no_wall_clock_timings(results_dir):
    """Timings live in timings.json; keeping them out is what makes this stable."""
    metrics = json.load(open(os.path.join(results_dir, "tables", "metrics.json")))
    blob = json.dumps(metrics)
    assert "solve_seconds" not in blob
    assert "train_seconds" not in blob


# --------------------------------------------------------------------------- #
# A fresh run reproduces them exactly
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_rerun_produces_every_artifact(regenerated):
    for rel in EXPECTED_ARTIFACTS:
        assert os.path.isfile(os.path.join(regenerated, rel)), rel


@pytest.mark.slow
@pytest.mark.parametrize("rel", EXPECTED_ARTIFACTS)
def test_rerun_is_byte_identical(regenerated, results_dir, rel):
    """The headline reproducibility guarantee, artifact by artifact."""
    fresh = os.path.join(regenerated, rel)
    committed = os.path.join(results_dir, rel)
    assert _md5(fresh) == _md5(committed), (
        f"{rel} changed between runs from the same seed")


@pytest.mark.slow
def test_timings_are_written_but_excluded_from_the_guarantee(regenerated):
    timings = json.load(open(os.path.join(regenerated, "tables", "timings.json")))
    assert {"dp_solve_seconds", "ml_train_seconds", "total_seconds"} <= set(timings)
    assert all(v > 0 for v in timings.values())


# --------------------------------------------------------------------------- #
# Determinism of the pieces, without a full run
# --------------------------------------------------------------------------- #
def test_figures_carry_no_creation_timestamp(results_dir):
    """
    Matplotlib stamps a PNG ``tEXt`` Date chunk by default, which alone would
    make identical figures differ between runs.  We suppress it.
    """
    for rel in EXPECTED_ARTIFACTS + ROBUSTNESS_ARTIFACTS:
        if not rel.endswith(".png"):
            continue
        blob = open(os.path.join(results_dir, rel), "rb").read()
        assert b"Date" not in blob[:2048], f"{rel} still embeds a timestamp"


def test_two_solves_agree_bit_for_bit(cfg):
    from src.dp_solver import solve
    a, b = solve(cfg), solve(cfg)
    assert np.array_equal(a.value, b.value)
    assert np.array_equal(a.policy, b.policy)
    assert np.array_equal(a.no_trade_lower, b.no_trade_lower)


def test_committed_dp_checkpoint_matches_a_fresh_solve(cfg, results_dir):
    """The stored solution is the one today's code produces."""
    from src.dp_solver import solve
    stored = np.load(os.path.join(results_dir, "checkpoints", "dp_solution.npz"))
    fresh = solve(cfg)
    assert np.array_equal(stored["policy"], fresh.policy)
    assert np.array_equal(stored["value"], fresh.value)
    assert np.array_equal(stored["x_grid"], fresh.x_grid)
