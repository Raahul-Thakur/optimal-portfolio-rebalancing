"""
Tests for the Monte-Carlo evaluation engine.

The simulator is where the paper's economic verdict is produced, so the checks
here cover both the bookkeeping (wealth accounting, turnover, cost) and the
statistics (certainty equivalents, common random numbers, paired tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from dataclasses import replace

from src.dynamics import sample_gross_returns
from src.simulation import (crra_utility, certainty_equivalent, simulate_strategy,
                            make_dp_action, make_periodic_action,
                            make_buy_hold_action, paired_comparison,
                            results_table, StrategyResult)


@pytest.fixture(scope="module")
def returns(small_cfg):
    """A shared return matrix -- the common random numbers of the experiment."""
    rng = np.random.default_rng(small_cfg.seed + 1)
    return sample_gross_returns(small_cfg, (small_cfg.n_paths, small_cfg.n_steps), rng)


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def test_certainty_equivalent_inverts_the_utility_function():
    for gamma in (2.0, 5.0, 8.0):
        for w in (0.5, 1.0, 1.37, 4.0):
            u = float(crra_utility(np.array([w]), gamma)[0])
            assert certainty_equivalent(u, gamma) == pytest.approx(w, rel=1e-12)


def test_certainty_equivalent_is_monotone_in_expected_utility():
    gamma = 5.0
    us = crra_utility(np.array([0.8, 1.0, 1.2, 1.5]), gamma)
    ces = [certainty_equivalent(float(u), gamma) for u in us]
    assert np.all(np.diff(ces) > 0)


def test_certainty_equivalent_penalises_risk():
    """A risky payoff is worth less than its mean to a risk-averse investor."""
    gamma = 5.0
    payoff = np.array([0.5, 1.5])
    ce = certainty_equivalent(float(np.mean(crra_utility(payoff, gamma))), gamma)
    assert ce < payoff.mean()


# --------------------------------------------------------------------------- #
# Wealth bookkeeping
# --------------------------------------------------------------------------- #
def test_buy_and_hold_never_trades_after_the_first_step(small_cfg, returns):
    res = simulate_strategy(small_cfg, make_buy_hold_action(small_cfg), returns,
                            "bh", initial_fraction=0.8)
    # Exactly one trade of size |0.30 - 0.80| = 0.50 per path.
    assert np.allclose(res.turnover, 0.5)
    assert np.allclose(res.total_cost, small_cfg.tc * 0.5)


def test_starting_on_target_makes_buy_and_hold_costless(small_cfg, returns):
    res = simulate_strategy(small_cfg, make_buy_hold_action(small_cfg), returns,
                            "bh", initial_fraction=small_cfg.merton_fraction)
    assert np.allclose(res.turnover, 0.0, atol=1e-12)
    assert np.allclose(res.total_cost, 0.0, atol=1e-12)


def test_periodic_rebalancing_trades_more_than_the_optimal_policy(
        small_cfg, small_sol, returns):
    """Turnover ordering is the paper's cost story: DP holds, periodic churns."""
    dp = simulate_strategy(small_cfg, make_dp_action(small_sol), returns, "dp")
    per = simulate_strategy(small_cfg, make_periodic_action(small_cfg), returns, "per")
    assert per.turnover.mean() > dp.turnover.mean()
    assert per.total_cost.mean() > dp.total_cost.mean()


def test_cost_is_exactly_the_cost_rate_times_turnover(small_cfg, returns):
    """Bookkeeping identity, checked on a strategy with a known single trade."""
    res = simulate_strategy(small_cfg, make_buy_hold_action(small_cfg), returns,
                            "bh", initial_fraction=0.8)
    assert np.allclose(res.total_cost, small_cfg.tc * res.turnover)


def test_zero_cost_removes_all_cost_but_not_turnover(small_cfg, returns):
    free = replace(small_cfg, tc=0.0)
    res = simulate_strategy(free, make_periodic_action(free), returns, "per")
    assert res.turnover.mean() > 0
    assert np.allclose(res.total_cost, 0.0)


def test_wealth_and_metrics_are_well_behaved(small_cfg, small_sol, returns):
    res = simulate_strategy(small_cfg, make_dp_action(small_sol), returns, "dp")
    assert res.terminal_wealth.shape == (small_cfg.n_paths,)
    assert np.all(res.terminal_wealth > 0)
    assert np.all(np.isfinite(res.realized_utility))
    assert np.all(res.turnover >= 0) and np.all(res.total_cost >= 0)


def test_the_optimal_policy_barely_trades_from_the_target(small_cfg, small_sol,
                                                          returns):
    """Started on target, the optimal action is essentially always 'hold'."""
    res = simulate_strategy(small_cfg, make_dp_action(small_sol), returns, "dp")
    assert res.turnover.mean() < 1e-2


def test_simulation_is_deterministic_given_the_returns(small_cfg, small_sol,
                                                       returns):
    a = simulate_strategy(small_cfg, make_dp_action(small_sol), returns, "dp")
    b = simulate_strategy(small_cfg, make_dp_action(small_sol), returns, "dp")
    assert np.array_equal(a.terminal_wealth, b.terminal_wealth)


# --------------------------------------------------------------------------- #
# Tracking error
# --------------------------------------------------------------------------- #
def test_dp_has_no_tracking_error_against_itself(small_cfg, small_sol, returns):
    res = simulate_strategy(small_cfg, make_dp_action(small_sol), returns, "dp",
                            dp_for_tracking=small_sol)
    assert res.tracking_error == pytest.approx(0.0, abs=1e-12)


def test_tracking_error_is_reported_for_other_strategies(small_cfg, small_sol,
                                                         returns):
    res = simulate_strategy(small_cfg, make_periodic_action(small_cfg), returns,
                            "per", dp_for_tracking=small_sol)
    assert res.tracking_error is not None and res.tracking_error > 0


def test_tracking_error_is_none_when_not_requested(small_cfg, small_sol, returns):
    res = simulate_strategy(small_cfg, make_dp_action(small_sol), returns, "dp")
    assert res.tracking_error is None


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #
def _make_results(small_cfg, small_sol, returns):
    return {
        "DP optimal": simulate_strategy(small_cfg, make_dp_action(small_sol),
                                        returns, "DP optimal"),
        "Periodic": simulate_strategy(small_cfg, make_periodic_action(small_cfg),
                                      returns, "Periodic",
                                      dp_for_tracking=small_sol),
        "Buy & hold": simulate_strategy(small_cfg, make_buy_hold_action(small_cfg),
                                        returns, "Buy & hold",
                                        dp_for_tracking=small_sol),
    }


def test_optimal_policy_beats_periodic_rebalancing(small_cfg, small_sol, returns):
    """The headline ordering, on a paired test with common random numbers."""
    rows = paired_comparison(_make_results(small_cfg, small_sol, returns),
                             reference="DP optimal")
    per = next(r for r in rows if r["vs"] == "Periodic")
    assert per["mean_utility_advantage"] > 0
    assert per["reference_better_95pct"] is True


def test_paired_comparison_excludes_the_reference_and_is_well_formed(
        small_cfg, small_sol, returns):
    results = _make_results(small_cfg, small_sol, returns)
    rows = paired_comparison(results, reference="DP optimal")
    assert {r["vs"] for r in rows} == {"Periodic", "Buy & hold"}
    for r in rows:
        assert r["se"] > 0
        assert r["ci_low"] < r["mean_utility_advantage"] < r["ci_high"]
        assert r["t_stat"] == pytest.approx(
            r["mean_utility_advantage"] / r["se"], rel=1e-9)


def test_pairing_is_more_precise_than_unpaired_comparison(small_cfg, small_sol,
                                                          returns):
    """
    The justification for common random numbers: the paired standard error is
    far below the standard error of the difference of independent estimates.
    """
    results = _make_results(small_cfg, small_sol, returns)
    paired_se = next(r for r in paired_comparison(results)
                     if r["vs"] == "Periodic")["se"]
    summaries = {r["strategy"]: r for r in results_table(results, small_cfg.gamma)}
    unpaired_se = np.hypot(summaries["DP optimal"]["expected_utility_se"],
                           summaries["Periodic"]["expected_utility_se"])
    assert paired_se < 0.5 * unpaired_se


def test_summary_table_is_complete(small_cfg, small_sol, returns):
    rows = results_table(_make_results(small_cfg, small_sol, returns),
                         small_cfg.gamma)
    assert len(rows) == 3
    for row in rows:
        assert row["ce_ci_low"] < row["certainty_equivalent"] < row["ce_ci_high"]
        assert row["mean_turnover"] >= 0
