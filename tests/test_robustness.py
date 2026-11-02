"""
Tests for the robustness study.

These guard the paper's strongest empirical claim: that the imitation failure is
a property of the control problem rather than of one unlucky fit. The committed
sweep results are checked for internal consistency and for the claims the
manuscript draws from them; the sweep itself is far too slow to rerun here (it
retrains the network seventeen times), so it is exercised in ``--quick`` form
under the ``slow`` marker.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def seeds(results_dir):
    return pd.read_csv(os.path.join(results_dir, "tables",
                                    "robustness_seeds.csv"))


@pytest.fixture(scope="module")
def arch(results_dir):
    return pd.read_csv(os.path.join(results_dir, "tables",
                                    "robustness_architectures.csv"))


@pytest.fixture(scope="module")
def summary(results_dir):
    return json.load(open(os.path.join(results_dir, "tables",
                                       "robustness_summary.json")))


# --------------------------------------------------------------------------- #
# The sweep is well-formed
# --------------------------------------------------------------------------- #
def test_every_seed_is_distinct_and_the_sweep_is_the_advertised_size(seeds):
    assert len(seeds) == 10
    assert seeds["seed"].nunique() == len(seeds)


def test_architecture_sweep_spans_two_orders_of_capacity(arch):
    assert len(arch) == 7
    assert arch["n_params"].max() > 100 * arch["n_params"].min()
    assert set(arch["depth"]) >= {2, 3, 4}


def test_published_configuration_appears_in_both_sweeps(seeds, arch):
    """The headline run must be one of the points, not a separate claim."""
    assert (seeds["seed"] == 12345).sum() == 1
    assert (arch["architecture"] == "64x64").sum() == 1
    assert int(arch.loc[arch["architecture"] == "64x64", "n_params"].iloc[0]) == 4417


def test_all_fits_share_the_same_dp_baseline(seeds, arch):
    """The teacher is seed-independent; a drifting baseline would be a bug."""
    everything = pd.concat([seeds["ce_dp"], arch["ce_dp"]])
    assert everything.nunique() == 1


# --------------------------------------------------------------------------- #
# The claims the paper makes from it
# --------------------------------------------------------------------------- #
def test_every_seed_overtrades_relative_to_the_optimum(seeds):
    """The universality claim: no retraining escapes the failure."""
    assert (seeds["turnover"] > 100 * seeds["turnover_dp"]).all()


def test_every_seed_underperforms_the_optimum(seeds):
    assert (seeds["ce"] < seeds["ce_dp"]).all()
    assert (seeds["gap"] > 0).all()


def test_no_architecture_escapes_the_failure(arch):
    """Capacity is not the issue -- the largest network fails too."""
    assert (arch["ce"] < arch["ce_dp"]).all()
    assert (arch["in_band_residual"] > 0).all()


def test_more_capacity_does_not_restore_inaction(arch):
    """
    The residual does not vanish with size.  A smooth function of any width
    still cannot be exactly the identity on a positive-measure set.
    """
    biggest = arch.loc[arch["n_params"].idxmax()]
    smallest = arch.loc[arch["n_params"].idxmin()]
    assert biggest["in_band_residual"] > 0
    # Orders of magnitude more parameters buy less than an order of magnitude.
    assert biggest["in_band_residual"] > 0.1 * smallest["in_band_residual"]


def test_the_repair_works_for_every_single_fit(seeds):
    """The constructive claim, across all retrainings rather than one."""
    assert (seeds["gap_closed_frac"] > 0.9).all()
    assert (seeds["turnover_repaired"] < 0.05 * seeds["turnover"]).all()


def test_validation_mse_is_uniformly_small(seeds):
    """Every fit is a *good* fit in action space -- that is the paradox."""
    assert (seeds["final_val_mse"] < 1e-2).all()


def test_low_validation_error_does_not_imply_low_turnover(seeds):
    """
    The paper's central lesson, restated as a fact about the sweep: ranking the
    fits by validation MSE tells you almost nothing about their economics.
    """
    assert len(seeds) >= 5
    best_mse = seeds.loc[seeds["final_val_mse"].idxmin()]
    assert best_mse["turnover"] > 100 * best_mse["turnover_dp"]


# --------------------------------------------------------------------------- #
# The summary agrees with the rows it summarizes
# --------------------------------------------------------------------------- #
def test_summary_matches_the_underlying_tables(summary, seeds, arch):
    s, a = summary["seeds"], summary["architectures"]
    assert summary["n_seeds"] == len(seeds)
    assert s["turnover_mean"] == pytest.approx(seeds["turnover"].mean())
    assert s["residual_max"] == pytest.approx(seeds["in_band_residual"].max())
    assert s["n_underperforming_dp"] == len(seeds)
    assert a["n_architectures"] == len(arch)
    assert a["n_underperforming_dp"] == len(arch)


# --------------------------------------------------------------------------- #
# The script still runs
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_robustness_script_runs_end_to_end(tmp_path, project_root):
    proc = subprocess.run(
        [sys.executable, os.path.join("scripts", "run_robustness.py"),
         "--quick", "--outdir", str(tmp_path)],
        cwd=project_root, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    for rel in ("tables/robustness_seeds.csv",
                "tables/robustness_architectures.csv",
                "tables/robustness_summary.json",
                "figures/fig10_robustness.png"):
        assert os.path.isfile(os.path.join(tmp_path, rel)), rel
