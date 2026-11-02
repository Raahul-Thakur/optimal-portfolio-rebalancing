"""
Tests for the proportional-cost wealth-update identity.

The closed form

    m(x, y) = (1 + tc x) / (1 + tc y)   if y >= x,
              (1 - tc x) / (1 - tc y)   if y <  x,

is what makes the Bellman recursion exact and vectorizable, so it is worth
pinning down.  The defining property is the *fixed point*: after paying the cost
out of cash, the realized post-trade risky fraction must be exactly ``y``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.transaction_costs import (wealth_multiplier, wealth_multiplier_matrix,
                                   trade_fraction, dollar_cost)

TCS = [0.0, 0.0005, 0.01, 0.03]
PAIRS = [(0.3, 0.3), (0.0, 0.0), (1.0, 1.0), (0.1, 0.5), (0.6, 0.2),
         (0.0, 0.9), (0.95, 0.05), (0.5, 0.5001)]


@pytest.mark.parametrize("tc", TCS)
@pytest.mark.parametrize("x,y", PAIRS)
def test_fixed_point_identity(x, y, tc):
    """The whole point of the identity: (x + delta) / m == y, exactly."""
    m = wealth_multiplier(x, y, tc)
    delta = trade_fraction(x, y, tc)
    assert np.isclose((x + delta) / m, y, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("tc", TCS)
@pytest.mark.parametrize("x,y", PAIRS)
def test_cost_is_exactly_the_wealth_lost(x, y, tc):
    """Post-trade wealth is 1 minus the proportional cost on traded dollars."""
    m = wealth_multiplier(x, y, tc)
    delta = trade_fraction(x, y, tc)
    assert np.isclose(m, 1.0 - tc * abs(delta), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("tc", TCS)
@pytest.mark.parametrize("x", [0.0, 0.2, 0.5, 0.87, 1.0])
def test_no_trade_is_exactly_free(x, tc):
    """m == 1 and delta == 0 when y == x -- the no-trade region must be costless."""
    assert wealth_multiplier(x, x, tc) == pytest.approx(1.0, abs=1e-15)
    assert trade_fraction(x, x, tc) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("x,y", PAIRS)
def test_zero_cost_is_frictionless(x, y):
    assert wealth_multiplier(x, y, 0.0) == pytest.approx(1.0)
    assert trade_fraction(x, y, 0.0) == pytest.approx(y - x)


@pytest.mark.parametrize("tc", [0.0005, 0.01, 0.03])
@pytest.mark.parametrize("x,y", PAIRS)
def test_multiplier_is_in_unit_interval(x, y, tc):
    m = wealth_multiplier(x, y, tc)
    assert 0.0 < m <= 1.0 + 1e-15


@pytest.mark.parametrize("tc", [0.0, 0.01, 0.03])
def test_trade_direction_matches_sign(tc):
    assert trade_fraction(0.2, 0.7, tc) > 0      # buying risky
    assert trade_fraction(0.7, 0.2, tc) < 0      # selling risky


@pytest.mark.parametrize("tc", TCS)
def test_matrix_form_agrees_with_scalar_form(tc):
    """The vectorized (n_x, n_y) matrix used by the DP must match the scalar."""
    xs = np.linspace(0.0, 1.0, 23)
    ys = np.linspace(0.0, 1.0, 17)
    mat = wealth_multiplier_matrix(xs, ys, tc)
    assert mat.shape == (23, 17)
    expected = np.array([[wealth_multiplier(float(x), float(y), tc) for y in ys]
                         for x in xs])
    assert np.allclose(mat, expected, rtol=1e-14, atol=1e-15)


@pytest.mark.parametrize("tc", [0.0005, 0.01, 0.03])
def test_larger_trades_cost_more(tc):
    """Cost is monotone in the distance travelled, on both sides of x."""
    x = 0.4
    buys = [1.0 - wealth_multiplier(x, y, tc) for y in [0.5, 0.6, 0.7, 0.8]]
    sells = [1.0 - wealth_multiplier(x, y, tc) for y in [0.3, 0.2, 0.1, 0.0]]
    assert np.all(np.diff(buys) > 0)
    assert np.all(np.diff(sells) > 0)


def test_dollar_cost_is_proportional_and_symmetric():
    assert dollar_cost(100.0, 130.0, 0.01) == pytest.approx(0.30)
    assert dollar_cost(130.0, 100.0, 0.01) == pytest.approx(0.30)
    assert dollar_cost(100.0, 100.0, 0.01) == pytest.approx(0.0)
