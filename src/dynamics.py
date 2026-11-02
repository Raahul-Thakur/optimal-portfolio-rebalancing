"""
Market dynamics for the single-risky-asset model.

The risky asset follows a discrete-time geometric Brownian motion.  Over one
rebalancing step of length ``dt`` its gross return is

    R = exp( (mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z ),   Z ~ N(0, 1).

The risk-free asset grows deterministically by

    R_f = exp(r * dt).

This module provides

  * Gauss-Hermite quadrature nodes/weights for the gross-return distribution,
    used by the dynamic-programming solver to evaluate conditional
    expectations exactly (to quadrature order); and
  * a Monte-Carlo sampler for gross returns, used by the simulation engine.
"""

from __future__ import annotations

import numpy as np

from .config import Config


def risk_free_gross(cfg: Config) -> float:
    """Risk-free gross return over a single step, R_f = exp(r * dt)."""
    return float(np.exp(cfg.r * cfg.dt))


def gross_return_quadrature(cfg: Config):
    """
    Gauss-Hermite quadrature for E[f(R)] under the one-step lognormal return.

    We approximate, for Z ~ N(0, 1),

        E[f(R)] = (1/sqrt(pi)) * sum_i w_i * f( R(sqrt(2) * z_i) ),

    where (z_i, w_i) are physicists' Hermite nodes/weights and

        R(x) = exp( (mu - 0.5 sigma^2) dt + sigma sqrt(dt) * x ).

    Returns
    -------
    returns : (n_quad,) ndarray
        Gross-return values at the quadrature nodes.
    probs : (n_quad,) ndarray
        Associated probability weights (sum to 1).
    """
    nodes, weights = np.polynomial.hermite.hermgauss(cfg.n_quad)
    probs = weights / np.sqrt(np.pi)
    drift = (cfg.mu - 0.5 * cfg.sigma ** 2) * cfg.dt
    diffusion = cfg.sigma * np.sqrt(cfg.dt)
    returns = np.exp(drift + diffusion * np.sqrt(2.0) * nodes)
    return returns, probs


def sample_gross_returns(cfg: Config, shape, rng: np.random.Generator) -> np.ndarray:
    """
    Draw i.i.d. risky-asset gross returns of the requested shape.

    Parameters
    ----------
    cfg : Config
    shape : int or tuple of ints
    rng : numpy.random.Generator

    Returns
    -------
    ndarray of gross returns with the given shape.
    """
    drift = (cfg.mu - 0.5 * cfg.sigma ** 2) * cfg.dt
    diffusion = cfg.sigma * np.sqrt(cfg.dt)
    z = rng.standard_normal(shape)
    return np.exp(drift + diffusion * z)


def expected_gross_return(cfg: Config) -> float:
    """Analytic E[R] = exp(mu * dt) for the lognormal one-step return."""
    return float(np.exp(cfg.mu * cfg.dt))


if __name__ == "__main__":
    cfg = Config()
    r, p = gross_return_quadrature(cfg)
    print(f"Risk-free gross R_f       = {risk_free_gross(cfg):.6f}")
    print(f"Quadrature E[R]           = {np.sum(p * r):.6f}")
    print(f"Analytic   E[R]=exp(mu dt)= {expected_gross_return(cfg):.6f}")
    print(f"Quadrature E[R^2]         = {np.sum(p * r**2):.6f}")
    analytic_r2 = np.exp(2 * cfg.mu * cfg.dt + cfg.sigma**2 * cfg.dt)
    print(f"Analytic   E[R^2]         = {analytic_r2:.6f}")
    print(f"probs sum                 = {p.sum():.6f}")
