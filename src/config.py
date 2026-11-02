"""
Central configuration for the variational portfolio rebalancing project.

All economic, numerical, and machine-learning hyperparameters live here so that
every module, notebook, and script reads from a single source of truth.  The
defaults reproduce a textbook single-risky-asset Merton problem with proportional
transaction costs, chosen so that the frictionless optimum (the Merton fraction)
sits comfortably in the interior of the [0, 1] allocation interval, producing a
clean no-trade band.

Usage
-----
>>> from src.config import Config
>>> cfg = Config()                 # defaults
>>> cfg = Config(gamma=3.0)        # override a single field
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Any


@dataclass(frozen=True)
class Config:
    # -- Market dynamics (annualized) ------------------------------------
    mu: float = 0.08          # risky-asset drift (expected log-ish return rate)
    sigma: float = 0.20       # risky-asset volatility
    r: float = 0.02           # risk-free rate (continuously compounded)

    # -- Investor preferences --------------------------------------------
    gamma: float = 5.0        # CRRA relative risk aversion (gamma != 1)

    # -- Transaction costs -----------------------------------------------
    tc: float = 0.01          # proportional transaction-cost rate (1%)

    # -- Time horizon -----------------------------------------------------
    T: float = 1.0            # investment horizon in years
    n_steps: int = 52         # number of rebalancing dates (weekly)

    # -- State / action discretization -----------------------------------
    n_x: int = 201            # grid points for risky-allocation fraction x in [x_min, x_max]
    x_min: float = 0.0        # minimum allocation (no shorting)
    x_max: float = 1.0        # maximum allocation (no leverage)

    # -- Quadrature -------------------------------------------------------
    n_quad: int = 21          # Gauss-Hermite nodes for the return expectation

    # -- Machine-learning policy -----------------------------------------
    ml_hidden: tuple = (64, 64)   # MLP hidden-layer widths
    ml_epochs: int = 400
    ml_batch_size: int = 256
    ml_lr: float = 1e-3
    ml_weight_decay: float = 1e-5
    ml_val_frac: float = 0.15

    # -- Remedies for the imitation failure (see src/remedies.py) ---------
    deadzone_delta: float = 0.01      # snap trades smaller than this to "hold"
    cost_aware_lambda: float = 1.0    # weight on the excess-turnover penalty

    # -- Monte-Carlo evaluation ------------------------------------------
    n_paths: int = 20000      # simulated paths for strategy comparison
    periodic_every: int = 4   # rebalance every k steps for the periodic baseline

    # -- Reproducibility --------------------------------------------------
    seed: int = 12345

    @property
    def dt(self) -> float:
        """Length of a single time step in years."""
        return self.T / self.n_steps

    @property
    def merton_fraction(self) -> float:
        """
        Frictionless Merton optimal risky fraction
            w* = (mu - r) / (gamma * sigma^2).
        This is the target allocation in the absence of transaction costs and the
        natural center of the no-trade region.
        """
        return (self.mu - self.r) / (self.gamma * self.sigma ** 2)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dt"] = self.dt
        d["merton_fraction"] = self.merton_fraction
        return d


# A module-level default instance for convenience.
DEFAULT = Config()


if __name__ == "__main__":
    import json

    cfg = Config()
    print(json.dumps(cfg.to_dict(), indent=2, default=str))
    print(f"\nMerton fraction w* = {cfg.merton_fraction:.4f}")
    print(f"dt = {cfg.dt:.6f} years")
