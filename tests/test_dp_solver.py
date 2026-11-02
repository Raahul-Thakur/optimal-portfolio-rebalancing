"""
Tests for the dynamic-programming solver.

These are the tests that guard the paper's economic claims: that the solver
recovers a genuine no-trade region centred on the Merton fraction, that the
region widens with the transaction-cost rate at roughly the classical
``tc^{1/3}`` rate, narrows with risk aversion, and flares open near the horizon.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.config import Config
from src.dp_solver import solve


def band_width(sol, t: int = 0) -> float:
    """Width of the no-trade region at date ``t``."""
    lo, hi = sol.no_trade_band(t)
    return hi - lo


# --------------------------------------------------------------------------- #
# Structure of the solution object
# --------------------------------------------------------------------------- #
def test_solution_shapes(small_sol, small_cfg):
    assert small_sol.x_grid.shape == (small_cfg.n_x,)
    assert small_sol.value.shape == (small_cfg.n_steps + 1, small_cfg.n_x)
    assert small_sol.policy.shape == (small_cfg.n_steps, small_cfg.n_x)
    assert small_sol.no_trade_lower.shape == (small_cfg.n_steps,)
    assert small_sol.no_trade_upper.shape == (small_cfg.n_steps,)


def test_terminal_condition(small_sol, small_cfg):
    """g_N(x) = 1 / (1 - gamma), constant in x."""
    expected = 1.0 / (1.0 - small_cfg.gamma)
    assert np.allclose(small_sol.value[small_cfg.n_steps], expected)


def test_everything_is_finite(small_sol):
    assert np.all(np.isfinite(small_sol.value))
    assert np.all(np.isfinite(small_sol.policy))


def test_policy_respects_the_action_bounds(small_sol, small_cfg):
    assert small_sol.policy.min() >= small_cfg.x_min - 1e-12
    assert small_sol.policy.max() <= small_cfg.x_max + 1e-12


def test_value_is_negative_for_gamma_above_one(small_sol):
    """With gamma > 1, U(W) = W^{1-gamma}/(1-gamma) < 0, and so is g_t."""
    assert np.all(small_sol.value < 0)


def test_log_utility_is_rejected_rather_than_silently_wrong():
    with pytest.raises(ValueError):
        solve(Config(gamma=1.0, n_steps=4, n_x=51))


def test_solver_is_deterministic(small_cfg):
    a, b = solve(small_cfg), solve(small_cfg)
    assert np.array_equal(a.policy, b.policy)
    assert np.array_equal(a.value, b.value)


# --------------------------------------------------------------------------- #
# The no-trade region
# --------------------------------------------------------------------------- #
def test_no_trade_band_is_nonempty_and_ordered(full_sol, cfg):
    lo, hi = full_sol.no_trade_band(0)
    assert lo < hi
    assert hi - lo > 2 * (cfg.x_max - cfg.x_min) / (cfg.n_x - 1)


def test_no_trade_band_brackets_the_merton_fraction(full_sol, cfg):
    """The frictionless optimum must lie inside the band it is the centre of."""
    lo, hi = full_sol.no_trade_band(0)
    assert lo < cfg.merton_fraction < hi


def test_band_is_roughly_centred_on_the_merton_fraction(full_sol, cfg):
    lo, hi = full_sol.no_trade_band(0)
    midpoint = 0.5 * (lo + hi)
    assert abs(midpoint - cfg.merton_fraction) < 0.15 * (hi - lo)


def test_policy_is_the_identity_inside_the_band(full_sol, cfg):
    """Inside the no-trade region the optimal action is exactly y = x."""
    lo, hi = full_sol.no_trade_band(0)
    inside = (full_sol.x_grid >= lo) & (full_sol.x_grid <= hi)
    assert inside.sum() > 5
    residual = np.abs(full_sol.policy[0][inside] - full_sol.x_grid[inside])
    assert residual.max() <= (cfg.x_max - cfg.x_min) / (cfg.n_x - 1) * 1.5


def test_policy_trades_to_the_boundary_from_outside(full_sol, cfg):
    """Outside the band the investor trades only as far as the nearest edge."""
    lo, hi = full_sol.no_trade_band(0)
    grid, pol = full_sol.x_grid, full_sol.policy[0]
    tol = 2.0 * (cfg.x_max - cfg.x_min) / (cfg.n_x - 1)

    below = grid < lo - tol
    above = grid > hi + tol
    assert below.sum() > 0 and above.sum() > 0
    # From below: buy up to (approximately) the lower edge, never past the upper.
    assert np.all(pol[below] >= grid[below] - tol)
    assert np.all(pol[below] <= hi + tol)
    # From above: sell down toward the upper edge, never below the lower.
    assert np.all(pol[above] <= grid[above] + tol)
    assert np.all(pol[above] >= lo - tol)


def test_band_flares_open_near_the_horizon(full_sol, cfg):
    """Horizon effect: with little time left, rebalancing cannot pay for itself."""
    assert band_width(full_sol, cfg.n_steps - 1) > band_width(full_sol, 0)


# --------------------------------------------------------------------------- #
# Comparative statics -- the paper's Q1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tc", [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03])
def test_band_still_brackets_merton_at_every_cost_level(cfg, tc):
    lo, hi = solve(replace(cfg, tc=tc)).no_trade_band(0)
    assert lo <= cfg.merton_fraction <= hi


def test_band_widens_monotonically_with_transaction_cost(cfg):
    tcs = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03]
    widths = [band_width(solve(replace(cfg, tc=t))) for t in tcs]
    assert np.all(np.diff(widths) > 0), widths


def test_band_width_follows_the_cube_root_scaling(cfg):
    """
    Small-cost asymptotics predict band width ~ tc^{1/3}.  We regress
    log(width) on log(tc) over the small-cost range and check the slope.
    """
    tcs = np.array([0.0005, 0.001, 0.002, 0.005])
    widths = np.array([band_width(solve(replace(cfg, tc=float(t)))) for t in tcs])
    slope = np.polyfit(np.log(tcs), np.log(widths), 1)[0]
    assert 0.22 < slope < 0.45, f"log-log slope {slope:.3f} is far from 1/3"


def test_band_narrows_with_risk_aversion(cfg):
    gammas = [2.0, 3.0, 5.0, 8.0, 12.0]
    widths = [band_width(solve(replace(cfg, gamma=g))) for g in gammas]
    assert np.all(np.diff(widths) < 0), widths


def test_zero_cost_collapses_the_band_to_the_merton_point(cfg):
    """
    With no friction the no-trade region degenerates: the investor rebalances to
    a single point, which must be the Merton fraction.
    """
    sol = solve(replace(cfg, tc=0.0))
    lo, hi = sol.no_trade_band(0)
    spacing = (cfg.x_max - cfg.x_min) / (cfg.n_x - 1)
    assert hi - lo <= 4 * spacing
    assert abs(0.5 * (lo + hi) - cfg.merton_fraction) < 0.02
    # And the policy maps essentially every state to that same target.
    assert np.allclose(sol.policy[0], cfg.merton_fraction, atol=0.02)


# --------------------------------------------------------------------------- #
# Numerical convergence
# --------------------------------------------------------------------------- #
def test_band_is_stable_under_grid_refinement(cfg):
    """Halving the grid spacing must not move the band by more than the spacing."""
    coarse = solve(replace(cfg, n_x=101)).no_trade_band(0)
    fine = solve(replace(cfg, n_x=401)).no_trade_band(0)
    assert abs(coarse[0] - fine[0]) < 0.02
    assert abs(coarse[1] - fine[1]) < 0.02


def test_band_is_stable_under_quadrature_refinement(cfg):
    coarse = solve(replace(cfg, n_quad=11)).no_trade_band(0)
    fine = solve(replace(cfg, n_quad=41)).no_trade_band(0)
    assert coarse == pytest.approx(fine, abs=0.01)


def test_policy_interpolation_matches_the_grid(full_sol):
    """policy_at must reproduce the tabulated policy on grid points."""
    assert np.allclose(full_sol.policy_at(0, full_sol.x_grid), full_sol.policy[0])
