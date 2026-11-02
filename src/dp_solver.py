"""
Finite-horizon dynamic-programming solver for optimal rebalancing.

Problem
-------
An investor holds wealth W split between one risky and one risk-free asset, with
pre-trade risky fraction ``x``.  At each of ``n_steps`` dates she chooses a
post-trade fraction ``y``, paying proportional transaction cost ``tc`` on the
traded amount, then the market evolves one step.  She maximizes expected
terminal CRRA utility U(W_T) = W_T^{1-gamma} / (1 - gamma).

Dimensionality reduction (CRRA homogeneity)
-------------------------------------------
Because U is homogeneous of degree (1 - gamma) in wealth and transaction costs
are proportional, the value function factorizes as

    V_t(W, x) = W^{1-gamma} * g_t(x),

so the entire control problem collapses to a one-dimensional function g_t(x) of
the pre-trade risky fraction.  Setting W = 1, the Bellman recursion is

    g_t(x) = max_y  m(x, y)^{1-gamma} * E_R[ phi(y, R)^{1-gamma} * g_{t+1}(x'(y, R)) ],

where
    m(x, y)        post-trade wealth multiplier (see transaction_costs.py),
    phi(y, R)      = y * R + (1 - y) * R_f      (one-step gross portfolio return),
    x'(y, R)       = y * R / phi(y, R)          (next pre-trade fraction),

and the terminal condition is g_N(x) = 1 / (1 - gamma) for all x.

The maximizing ``y`` defines the optimal policy.  When the optimum equals the
pre-trade fraction (y* = x) the investor does not trade; the set of such x is the
**no-trade region**.

Numerics
--------
* x and y are discretized on the same grid in [x_min, x_max].
* The return expectation E_R uses Gauss-Hermite quadrature (dynamics.py).
* g_{t+1} is evaluated at the continuous next-states x'(y, R) by linear
  interpolation on the grid.
* Everything is vectorized over (x, y) so each backward step is a handful of
  array operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .dynamics import gross_return_quadrature, risk_free_gross
from .transaction_costs import wealth_multiplier_matrix


@dataclass
class DPSolution:
    """Container for the solved dynamic program."""
    cfg: Config
    x_grid: np.ndarray          # (n_x,) allocation grid (also the action grid)
    value: np.ndarray           # (n_steps + 1, n_x) reduced value function g_t(x)
    policy: np.ndarray          # (n_steps, n_x) optimal post-trade fraction y*(t, x)
    no_trade_lower: np.ndarray  # (n_steps,) lower no-trade boundary per date
    no_trade_upper: np.ndarray  # (n_steps,) upper no-trade boundary per date

    def policy_at(self, t: int, x) -> np.ndarray:
        """Interpolated optimal post-trade fraction at date t for fraction(s) x."""
        return np.interp(np.asarray(x, dtype=float),
                         self.x_grid, self.policy[t])

    def no_trade_band(self, t: int):
        """(lower, upper) no-trade boundaries at date t."""
        return float(self.no_trade_lower[t]), float(self.no_trade_upper[t])


def _utility_constant(gamma: float) -> float:
    """Terminal reduced value g_N(x) = 1 / (1 - gamma) (constant in x)."""
    if np.isclose(gamma, 1.0):
        raise ValueError("gamma == 1 (log utility) is not supported by this solver; "
                         "use gamma != 1.")
    return 1.0 / (1.0 - gamma)


def solve(cfg: Config, verbose: bool = False) -> DPSolution:
    """
    Solve the dynamic program by backward induction.

    Returns
    -------
    DPSolution
    """
    gamma = cfg.gamma
    one_minus_g = 1.0 - gamma

    x_grid = np.linspace(cfg.x_min, cfg.x_max, cfg.n_x)
    y_grid = x_grid                      # actions share the state grid
    n_x = cfg.n_x

    # Quadrature for the risky gross return and the deterministic risk-free leg.
    R, probs = gross_return_quadrature(cfg)        # (K,), (K,)
    Rf = risk_free_gross(cfg)

    # Pre-compute quantities that depend only on the action y and the return R.
    #   phi[j, k] = one-step gross portfolio return for post-trade fraction y_j
    #   xnext[j, k] = next pre-trade fraction
    y_col = y_grid[:, None]                          # (n_y, 1)
    phi = y_col * R[None, :] + (1.0 - y_col) * Rf    # (n_y, K)
    xnext = (y_col * R[None, :]) / phi               # (n_y, K)
    phi_pow = phi ** one_minus_g                     # (n_y, K)
    weighted = phi_pow * probs[None, :]              # (n_y, K), folds in quadrature weights

    # Wealth-multiplier matrix m(x, y) raised to (1 - gamma): shape (n_x, n_y).
    m_mat = wealth_multiplier_matrix(x_grid, y_grid, cfg.tc)
    m_pow = m_mat ** one_minus_g                     # (n_x, n_y)

    # Storage.
    value = np.empty((cfg.n_steps + 1, n_x))
    policy = np.empty((cfg.n_steps, n_x))
    value[cfg.n_steps, :] = _utility_constant(gamma)

    # Backward induction.
    for t in range(cfg.n_steps - 1, -1, -1):
        g_next = value[t + 1]                        # (n_x,)
        # Interpolate g_{t+1} at all next-states x'(y, R): shape (n_y, K).
        g_interp = np.interp(xnext.ravel(), x_grid, g_next).reshape(xnext.shape)
        # Inner conditional expectation E_R[phi^{1-g} g_{t+1}(x')] per action y.
        inner = np.sum(weighted * g_interp, axis=1)  # (n_y,)
        # Full objective J(x, y) = m(x,y)^{1-g} * inner(y): shape (n_x, n_y).
        objective = m_pow * inner[None, :]
        # Optimize over the action y for each pre-trade state x.
        #
        # For gamma > 1 the reduced value is negative and we maximize (toward 0);
        # for gamma < 1 it is positive.  np.argmax/np.max maximize J directly,
        # which is correct in both regimes because the recursion is written in
        # value (not loss) form.
        best_idx = np.argmax(objective, axis=1)      # (n_x,)
        value[t, :] = objective[np.arange(n_x), best_idx]
        policy[t, :] = y_grid[best_idx]
        if verbose and (t % max(1, cfg.n_steps // 10) == 0):
            print(f"  step {t:3d}: g(mean)={value[t].mean():.4e}")

    lower, upper = _extract_no_trade_region(x_grid, policy)
    return DPSolution(cfg=cfg, x_grid=x_grid, value=value, policy=policy,
                      no_trade_lower=lower, no_trade_upper=upper)


def _extract_no_trade_region(x_grid: np.ndarray, policy: np.ndarray):
    """
    Identify, for each date, the band of pre-trade fractions where the optimal
    action is "do not trade" (y* == x to within one grid spacing).

    Returns (lower, upper) arrays of shape (n_steps,).  When no no-trade region
    is detected at a date (e.g. the final step), both boundaries fall back to the
    Merton-like single point given by the policy's fixed point.
    """
    n_steps, n_x = policy.shape
    tol = (x_grid[-1] - x_grid[0]) / (n_x - 1) * 1.5   # ~1.5 grid spacings
    lower = np.empty(n_steps)
    upper = np.empty(n_steps)
    for t in range(n_steps):
        no_trade = np.abs(policy[t] - x_grid) <= tol
        if no_trade.any():
            idx = np.where(no_trade)[0]
            lower[t] = x_grid[idx[0]]
            upper[t] = x_grid[idx[-1]]
        else:
            # Fixed point of the policy map as a degenerate band.
            fp = _policy_fixed_point(x_grid, policy[t])
            lower[t] = upper[t] = fp
    return lower, upper


def _policy_fixed_point(x_grid: np.ndarray, policy_row: np.ndarray) -> float:
    """Crossing of y*(x) with the 45-degree line, by linear interpolation."""
    diff = policy_row - x_grid
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        return float(x_grid[np.argmin(np.abs(diff))])
    i = sign_change[0]
    # Linear interpolation of the zero crossing of diff between i and i+1.
    x0, x1 = x_grid[i], x_grid[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    return float(x0 - d0 * (x1 - x0) / (d1 - d0))


if __name__ == "__main__":
    cfg = Config()
    sol = solve(cfg, verbose=True)
    t0 = 0
    lo, hi = sol.no_trade_band(t0)
    print(f"\nMerton fraction        : {cfg.merton_fraction:.4f}")
    print(f"No-trade band at t=0   : [{lo:.4f}, {hi:.4f}]  (width {hi - lo:.4f})")
    lo, hi = sol.no_trade_band(cfg.n_steps - 1)
    print(f"No-trade band at t=N-1 : [{lo:.4f}, {hi:.4f}]  (width {hi - lo:.4f})")
