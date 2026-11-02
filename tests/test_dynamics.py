"""
Tests for the market-dynamics layer.

The Gauss-Hermite quadrature is the only place where the DP replaces an integral
with a finite sum, so its accuracy bounds the accuracy of everything downstream.
We check it against the analytic lognormal moments.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import Config
from src.dynamics import (gross_return_quadrature, risk_free_gross,
                          sample_gross_returns, expected_gross_return)


def test_quadrature_weights_are_a_probability_measure(cfg):
    _, probs = gross_return_quadrature(cfg)
    assert np.all(probs > 0)
    assert np.isclose(probs.sum(), 1.0, atol=1e-12)


def test_quadrature_matches_analytic_first_moment(cfg):
    """E[R] = exp(mu * dt) for the one-step lognormal return."""
    returns, probs = gross_return_quadrature(cfg)
    assert np.isclose(float(np.sum(probs * returns)),
                      expected_gross_return(cfg), rtol=1e-12)


def test_quadrature_matches_analytic_second_moment(cfg):
    """E[R^2] = exp(2 mu dt + sigma^2 dt)."""
    returns, probs = gross_return_quadrature(cfg)
    analytic = np.exp(2 * cfg.mu * cfg.dt + cfg.sigma ** 2 * cfg.dt)
    assert np.isclose(float(np.sum(probs * returns ** 2)), analytic, rtol=1e-12)


@pytest.mark.parametrize("n_quad", [7, 11, 21, 31])
def test_quadrature_accuracy_is_stable_in_node_count(n_quad):
    """Even a coarse node count nails the moments: the integrand is smooth."""
    cfg = Config(n_quad=n_quad)
    returns, probs = gross_return_quadrature(cfg)
    assert np.isclose(float(np.sum(probs * returns)),
                      expected_gross_return(cfg), rtol=1e-10)


def test_risk_free_gross_is_exp_r_dt(cfg):
    assert np.isclose(risk_free_gross(cfg), np.exp(cfg.r * cfg.dt), rtol=1e-15)


def test_sampler_is_seed_deterministic(cfg):
    a = sample_gross_returns(cfg, (500, cfg.n_steps), np.random.default_rng(7))
    b = sample_gross_returns(cfg, (500, cfg.n_steps), np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_sampler_mean_converges_to_analytic(cfg):
    """Monte-Carlo mean of R agrees with the quadrature/analytic value."""
    rng = np.random.default_rng(cfg.seed)
    draws = sample_gross_returns(cfg, 400_000, rng)
    se = draws.std(ddof=1) / np.sqrt(draws.size)
    assert abs(draws.mean() - expected_gross_return(cfg)) < 5 * se


def test_returns_are_strictly_positive(cfg):
    """Lognormal returns can never wipe out the risky account."""
    rng = np.random.default_rng(1)
    assert np.all(sample_gross_returns(cfg, (200, cfg.n_steps), rng) > 0)
