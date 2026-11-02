"""
Tests for the two repairs of the imitation failure.

The paper's constructive claim is that the failure is *fixable once diagnosed*:
restoring an inaction region, either at inference time (dead-zone) or through
the training objective (excess-turnover penalty), recovers the performance that
naive imitation gives away. These tests pin down both the mechanism and the
economic consequence.

The economic tests train for more epochs than the rest of the suite
(``remedy_cfg``). With the 40-epoch budget used elsewhere the student is a poor
enough fit that its in-band residual swamps any reasonable dead-zone, and the
test would be measuring undertraining rather than the remedy.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.dynamics import sample_gross_returns
from src.ml_policy import train_policy, policy_action
from src.remedies import (dead_zone, make_dead_zone_action, in_band_residual,
                          train_cost_aware_policy)
from src.simulation import simulate_strategy, make_dp_action, make_ml_action

# A dead-zone comfortably wider than the student's in-band residual.  Below the
# residual the wrapper is a no-op; the interesting regime is above it.
WIDE_DELTA = 0.05


@pytest.fixture(scope="module")
def remedy_cfg(small_cfg):
    return replace(small_cfg, ml_epochs=200)


@pytest.fixture(scope="module")
def plain(small_sol, remedy_cfg):
    model, _ = train_policy(small_sol, remedy_cfg)
    return model


@pytest.fixture(scope="module")
def aware(small_sol, remedy_cfg):
    model, _ = train_cost_aware_policy(small_sol, remedy_cfg)
    return model


@pytest.fixture(scope="module")
def returns(remedy_cfg):
    rng = np.random.default_rng(remedy_cfg.seed + 1)
    return sample_gross_returns(
        remedy_cfg, (remedy_cfg.n_paths, remedy_cfg.n_steps), rng)


# --------------------------------------------------------------------------- #
# The dead-zone transform itself
# --------------------------------------------------------------------------- #
def test_dead_zone_snaps_small_trades_to_exact_inaction():
    x = np.array([0.30, 0.30, 0.30, 0.30])
    y_hat = np.array([0.3005, 0.295, 0.34, 0.20])
    out = dead_zone(y_hat, x, delta=0.01)
    # The first two lie inside the dead-zone: snapped to x *exactly*, since only
    # an exact match is costless in the simulator.
    assert out[0] == x[0] and out[1] == x[1]
    # The last two are genuine trades and must pass through untouched.
    assert out[2] == y_hat[2] and out[3] == y_hat[3]


def test_dead_zone_of_zero_width_is_the_identity():
    x = np.linspace(0, 1, 25)
    y_hat = x + 1e-6
    assert np.array_equal(dead_zone(y_hat, x, 0.0), y_hat)


def test_dead_zone_boundary_is_inclusive():
    """A trade exactly delta away is held. Uses dyadic values so |y-x| is exact."""
    assert dead_zone(np.array([0.75]), np.array([0.5]), delta=0.25)[0] == 0.5


@pytest.mark.parametrize("delta", [0.002, 0.01, 0.05])
def test_dead_zone_never_increases_the_trade_size(delta):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 500)
    y_hat = np.clip(x + rng.normal(0, 0.05, 500), 0, 1)
    out = dead_zone(y_hat, x, delta)
    assert np.all(np.abs(out - x) <= np.abs(y_hat - x) + 1e-15)


def test_dead_zone_restores_exact_inaction_inside_the_band(plain, small_sol):
    """The mechanism: with delta above the residual, in-band steps are costless."""
    lo, hi = small_sol.no_trade_band(0)
    inside = small_sol.x_grid[(small_sol.x_grid >= lo) & (small_sol.x_grid <= hi)]
    y_hat = policy_action(plain, 0.0, inside)

    residual = float(np.max(np.abs(y_hat - inside)))
    assert residual > 0                       # the failure the paper documents
    snapped = dead_zone(y_hat, inside, delta=residual + 1e-6)
    assert np.array_equal(snapped, inside)    # ... and its repair


# --------------------------------------------------------------------------- #
# The cost-aware objective
# --------------------------------------------------------------------------- #
def test_cost_aware_training_shrinks_the_in_band_residual(plain, aware, small_sol):
    """The training-time fix attacks the same quantity the wrapper snaps away."""
    assert in_band_residual(aware, small_sol) < in_band_residual(plain, small_sol)


def test_cost_aware_policy_is_still_a_valid_controller(aware):
    y = policy_action(aware, 0.25, np.linspace(0, 1, 41))
    assert np.all((y >= 0.0) & (y <= 1.0))
    assert np.all(np.isfinite(y))


def test_cost_aware_training_is_seed_deterministic(small_sol, remedy_cfg):
    a, _ = train_cost_aware_policy(small_sol, remedy_cfg)
    b, _ = train_cost_aware_policy(small_sol, remedy_cfg)
    xs = np.linspace(0, 1, 51)
    assert np.allclose(policy_action(a, 0.0, xs), policy_action(b, 0.0, xs))


def test_zero_penalty_reproduces_plain_imitation(small_sol, remedy_cfg, plain):
    """lambda = 0 must collapse to the ordinary MSE objective."""
    same, _ = train_cost_aware_policy(small_sol, remedy_cfg, lam=0.0)
    xs = np.linspace(0, 1, 51)
    assert np.allclose(policy_action(same, 0.0, xs),
                       policy_action(plain, 0.0, xs), atol=1e-6)


def test_teacher_is_the_exact_identity_strictly_inside_the_band(small_sol):
    """
    Sanity check on the diagnostic, and on a subtlety of the band extraction.

    ``_extract_no_trade_region`` admits grid points within 1.5 spacings of the
    diagonal, so the *reported* band overshoots the exact inaction set by up to
    one grid cell at each end. Strictly inside, the DP policy is exactly y = x
    -- which is the property the network provably cannot match.
    """
    lo, hi = small_sol.no_trade_band(0)
    inside = small_sol.x_grid[(small_sol.x_grid >= lo) & (small_sol.x_grid <= hi)]
    assert inside.size > 4
    interior = inside[1:-1]
    assert np.max(np.abs(small_sol.policy_at(0, interior) - interior)) < 1e-12


# --------------------------------------------------------------------------- #
# The economic payoff -- the claim the paper actually makes
# --------------------------------------------------------------------------- #
def _ce(cfg, action, returns, name):
    return simulate_strategy(cfg, action, returns,
                             name).summary(cfg.gamma)["certainty_equivalent"]


def test_dead_zone_cuts_turnover_dramatically(remedy_cfg, plain, returns):
    naive = simulate_strategy(remedy_cfg, make_ml_action(plain, remedy_cfg),
                              returns, "ml")
    fixed = simulate_strategy(
        remedy_cfg, make_dead_zone_action(plain, remedy_cfg, WIDE_DELTA),
        returns, "dz")
    assert fixed.turnover.mean() < 0.1 * naive.turnover.mean()
    assert fixed.total_cost.mean() < 0.1 * naive.total_cost.mean()


def test_dead_zone_improves_the_certainty_equivalent(remedy_cfg, plain, returns):
    """Lower turnover is only interesting if it buys economic performance."""
    naive = _ce(remedy_cfg, make_ml_action(plain, remedy_cfg), returns, "ml")
    fixed = _ce(remedy_cfg,
                make_dead_zone_action(plain, remedy_cfg, WIDE_DELTA), returns, "dz")
    assert fixed > naive


def test_dead_zone_closes_most_of_the_gap_to_the_optimum(remedy_cfg, small_sol,
                                                         plain, returns):
    """The headline constructive result, as an inequality."""
    dp = _ce(remedy_cfg, make_dp_action(small_sol), returns, "dp")
    naive = _ce(remedy_cfg, make_ml_action(plain, remedy_cfg), returns, "ml")
    fixed = _ce(remedy_cfg,
                make_dead_zone_action(plain, remedy_cfg, WIDE_DELTA), returns, "dz")

    gap_before = dp - naive
    gap_after = dp - fixed
    assert gap_before > 0
    assert gap_after < 0.25 * gap_before, (
        f"dead-zone closed only {1 - gap_after / gap_before:.0%} of the gap")


def test_wider_dead_zones_monotonically_reduce_turnover(remedy_cfg, plain, returns):
    turnovers = [
        simulate_strategy(remedy_cfg,
                          make_dead_zone_action(plain, remedy_cfg, d),
                          returns, f"dz{d}").turnover.mean()
        for d in (0.002, 0.005, 0.01, 0.02, 0.05)
    ]
    assert all(b <= a for a, b in zip(turnovers, turnovers[1:])), turnovers


def test_an_overwide_dead_zone_eventually_suppresses_correct_trades(
        remedy_cfg, plain, returns):
    """
    The wrapper is not free: pushed far enough it also blocks the large, correct
    trades made from outside the band, so delta is a genuine trade-off rather
    than a free win.
    """
    modest = simulate_strategy(
        remedy_cfg, make_dead_zone_action(plain, remedy_cfg, 0.01),
        returns, "dz-small", initial_fraction=0.80)
    huge = simulate_strategy(
        remedy_cfg, make_dead_zone_action(plain, remedy_cfg, 0.60),
        returns, "dz-huge", initial_fraction=0.80)
    # Starting far off target, a 0.60 dead-zone refuses to correct at all.
    assert huge.turnover.mean() < modest.turnover.mean()
    assert huge.summary(remedy_cfg.gamma)["certainty_equivalent"] < \
        modest.summary(remedy_cfg.gamma)["certainty_equivalent"]
