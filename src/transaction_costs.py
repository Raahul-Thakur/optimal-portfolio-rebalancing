"""
Proportional transaction-cost model and the induced wealth bookkeeping.

We normalize pre-trade wealth to 1 (legitimate by the CRRA homogeneity exploited
throughout the project) and describe a rebalancing trade purely in terms of the
*pre-trade* risky fraction ``x`` and the *post-trade* risky fraction ``y``.

Trade accounting
----------------
Pre-trade: risky dollars = x, cash dollars = 1 - x, total wealth = 1.
We trade ``delta`` dollars into the risky asset (delta < 0 means selling).  The
proportional cost ``tc * |delta|`` is paid out of the cash account, so the
post-trade total wealth is

    m = 1 - tc * |delta|.

Because the post-trade risky fraction must satisfy ``y = (x + delta) / m`` the
trade size is implicit in ``(x, y)``.  Solving the fixed point gives a clean
closed form (see derivation in the paper, Section 4.3):

    m(x, y) = (1 + tc * x) / (1 + tc * y)      if  y >= x   (buying risky),
              (1 - tc * x) / (1 - tc * y)      if  y <  x   (selling risky),

with the convenient property that ``delta = (y - x) / (1 +/- tc * y)`` so that
``sign(delta) = sign(y - x)`` and ``m = 1`` exactly when ``y = x`` (no trade).

This module exposes the wealth-multiplier ``m`` both as a scalar helper and as a
fully vectorized ``(n_x, n_y)`` matrix for the DP solver, plus a dollar-cost
helper used by the Monte-Carlo simulation engine.
"""

from __future__ import annotations

import numpy as np


def wealth_multiplier(x: float, y: float, tc: float) -> float:
    """
    Post-trade wealth multiplier m for moving pre-trade fraction x -> post-trade y.

    Returns m in (0, 1]; m == 1 iff no trade (y == x) or tc == 0.
    """
    if y >= x:
        return (1.0 + tc * x) / (1.0 + tc * y)
    return (1.0 - tc * x) / (1.0 - tc * y)


def wealth_multiplier_matrix(x_grid: np.ndarray, y_grid: np.ndarray, tc: float) -> np.ndarray:
    """
    Vectorized wealth multiplier over a grid of pre-trade and post-trade fractions.

    Parameters
    ----------
    x_grid : (n_x,) ndarray of pre-trade fractions.
    y_grid : (n_y,) ndarray of candidate post-trade fractions.
    tc : float, proportional cost rate.

    Returns
    -------
    m : (n_x, n_y) ndarray with m[i, j] = wealth_multiplier(x_grid[i], y_grid[j], tc).
    """
    x = x_grid[:, None]          # (n_x, 1)
    y = y_grid[None, :]          # (1, n_y)
    buy = (1.0 + tc * x) / (1.0 + tc * y)
    sell = (1.0 - tc * x) / (1.0 - tc * y)
    return np.where(y >= x, buy, sell)


def trade_fraction(x: float, y: float, tc: float) -> float:
    """
    Dollar amount traded into the risky asset (per unit pre-trade wealth).

    delta = y * m - x, where m = wealth_multiplier(x, y, tc).
    Positive means buying risky, negative means selling.
    """
    m = wealth_multiplier(x, y, tc)
    return y * m - x


def dollar_cost(pre_risky_value: float, post_risky_value: float, tc: float) -> float:
    """
    Absolute transaction cost (in dollars) for moving the risky holding from
    ``pre_risky_value`` to ``post_risky_value`` at proportional rate ``tc``.

    Used by the path-level simulation engine, which tracks dollar accounts
    directly rather than normalized fractions.
    """
    return tc * abs(post_risky_value - pre_risky_value)


if __name__ == "__main__":
    tc = 0.01
    for (x, y) in [(0.3, 0.3), (0.1, 0.5), (0.6, 0.2)]:
        m = wealth_multiplier(x, y, tc)
        d = trade_fraction(x, y, tc)
        # Consistency check: post-trade risky fraction must equal y.
        recovered_y = (x + d) / m
        print(f"x={x:.2f} y={y:.2f}  m={m:.6f}  delta={d:+.6f}  "
              f"cost={tc*abs(d):.6f}  recovered_y={recovered_y:.6f}")
