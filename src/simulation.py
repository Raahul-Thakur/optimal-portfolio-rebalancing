"""
Monte-Carlo simulation and evaluation of rebalancing strategies.

We simulate many wealth paths under a *common* set of random risky-asset returns
(so strategies are compared on identical market scenarios) and record, for each
strategy, the terminal wealth, realized utility, turnover, cumulative
transaction costs, and the tracking error of its actions relative to the optimal
DP control.

Strategies
----------
* ``dp``        : the optimal dynamic-programming policy (interpolated).
* ``ml``        : the trained neural-network imitation policy.
* ``periodic``  : rebalance to the Merton target every ``periodic_every`` steps,
                  otherwise hold.
* ``buy_hold``  : trade once at t=0 to the Merton target, then never again.

Each strategy is described by a callable ``action(t, x) -> y`` returning the
desired post-trade fraction; ``None`` means "do not trade this step".  Wealth
bookkeeping tracks dollar accounts directly and charges proportional costs on the
traded dollar amount.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from .config import Config
from .dynamics import sample_gross_returns, risk_free_gross
from .dp_solver import DPSolution
from .transaction_costs import dollar_cost


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class StrategyResult:
    name: str
    terminal_wealth: np.ndarray          # (n_paths,)
    turnover: np.ndarray                 # (n_paths,) sum of |traded dollars| / wealth
    total_cost: np.ndarray               # (n_paths,) cumulative cost / initial wealth
    tracking_error: Optional[float]      # RMS action gap vs DP (None for DP itself)
    realized_utility: np.ndarray         # (n_paths,) U(W_T)

    def summary(self, gamma: float) -> Dict[str, float]:
        u = self.realized_utility
        n = u.shape[0]
        eu = float(np.mean(u))
        eu_se = float(np.std(u, ddof=1) / np.sqrt(n))
        ce = certainty_equivalent(eu, gamma)
        # 95% CI on the certainty equivalent, propagated through the (monotone)
        # CE-of-expected-utility map.
        ce_lo = certainty_equivalent(eu - 1.96 * eu_se, gamma)
        ce_hi = certainty_equivalent(eu + 1.96 * eu_se, gamma)
        return {
            "strategy": self.name,
            "mean_terminal_wealth": float(np.mean(self.terminal_wealth)),
            "median_terminal_wealth": float(np.median(self.terminal_wealth)),
            "std_terminal_wealth": float(np.std(self.terminal_wealth)),
            "expected_utility": eu,
            "expected_utility_se": eu_se,
            "certainty_equivalent": ce,
            "ce_ci_low": float(min(ce_lo, ce_hi)),
            "ce_ci_high": float(max(ce_lo, ce_hi)),
            "mean_turnover": float(np.mean(self.turnover)),
            "mean_total_cost": float(np.mean(self.total_cost)),
            "tracking_error": (float(self.tracking_error)
                               if self.tracking_error is not None else np.nan),
        }


def crra_utility(wealth: np.ndarray, gamma: float) -> np.ndarray:
    """U(W) = W^{1-gamma} / (1 - gamma)."""
    return wealth ** (1.0 - gamma) / (1.0 - gamma)


def certainty_equivalent(expected_utility: float, gamma: float) -> float:
    """Wealth level whose utility equals the expected utility."""
    return ((1.0 - gamma) * expected_utility) ** (1.0 / (1.0 - gamma))


# --------------------------------------------------------------------------- #
# Strategy action functions
# --------------------------------------------------------------------------- #
def make_dp_action(sol: DPSolution) -> Callable:
    """Action callable for the optimal DP policy."""
    def action(t: int, x: np.ndarray) -> np.ndarray:
        return sol.policy_at(t, x)
    return action


def make_ml_action(model, cfg: Config) -> Callable:
    """Action callable for the trained ML policy."""
    from .ml_policy import policy_action

    def action(t: int, x: np.ndarray) -> np.ndarray:
        tau = t / cfg.n_steps
        return policy_action(model, np.full_like(x, tau), x)
    return action


def make_periodic_action(cfg: Config) -> Callable:
    """Rebalance to the Merton target every ``periodic_every`` steps."""
    target = cfg.merton_fraction
    every = cfg.periodic_every

    def action(t: int, x: np.ndarray):
        if t % every == 0:
            return np.full_like(x, target)
        return None     # hold
    return action


def make_buy_hold_action(cfg: Config) -> Callable:
    """Trade to the Merton target once at t=0, then never trade."""
    target = cfg.merton_fraction

    def action(t: int, x: np.ndarray):
        if t == 0:
            return np.full_like(x, target)
        return None
    return action


# --------------------------------------------------------------------------- #
# Core simulator
# --------------------------------------------------------------------------- #
def simulate_strategy(cfg: Config,
                      action: Callable,
                      returns: np.ndarray,
                      name: str,
                      dp_for_tracking: Optional[DPSolution] = None,
                      initial_fraction: Optional[float] = None) -> StrategyResult:
    """
    Simulate one strategy across all paths on a shared return matrix.

    Parameters
    ----------
    cfg : Config
    action : callable (t, x_array) -> y_array or None
        Desired post-trade fraction given the current date and pre-trade
        fractions across paths.  Return ``None`` to hold (no trade).
    returns : (n_paths, n_steps) ndarray of risky gross returns (shared).
    name : str strategy label.
    dp_for_tracking : DPSolution or None
        If provided, accumulate the RMS gap between this strategy's chosen action
        and the optimal DP action (a tracking-error metric).
    initial_fraction : float or None
        Pre-trade risky fraction at t=0 before any trading.  Defaults to the
        Merton fraction.

    Returns
    -------
    StrategyResult
    """
    n_paths, n_steps = returns.shape
    Rf = risk_free_gross(cfg)
    gamma = cfg.gamma
    x0 = cfg.merton_fraction if initial_fraction is None else initial_fraction

    # Dollar accounts, initial wealth normalized to 1 per path.
    wealth = np.ones(n_paths)
    risky = np.full(n_paths, x0)         # risky dollars
    cash = wealth - risky                # cash dollars

    turnover = np.zeros(n_paths)
    total_cost = np.zeros(n_paths)
    track_sq = 0.0
    track_count = 0

    for t in range(n_steps):
        wealth = risky + cash
        # Guard against degenerate non-positive wealth (shouldn't occur here).
        frac = np.divide(risky, wealth, out=np.full_like(risky, x0),
                         where=wealth > 0)

        y = action(t, frac)
        # Tracking error vs the optimal DP action, evaluated at EVERY step.
        # The strategy's post-decision fraction is its chosen y if it trades, or
        # the current fraction if it holds.  Comparing this to the DP-optimal
        # action from the same state measures policy disagreement uniformly,
        # including steps where the strategy (wrongly or rightly) holds.
        if dp_for_tracking is not None:
            post_frac = frac if y is None else np.clip(y, cfg.x_min, cfg.x_max)
            y_star = dp_for_tracking.policy_at(t, frac)
            track_sq += float(np.sum((post_frac - y_star) ** 2))
            track_count += n_paths
        if y is not None:
            y = np.clip(y, cfg.x_min, cfg.x_max)
            # Desired post-trade risky dollars, accounting for cost paid from cash.
            # Solve m = 1 - tc*|y*m*W - risky| with W = wealth; use closed form via
            # the wealth multiplier on normalized fractions, then rescale by wealth.
            new_risky_target = y * wealth
            traded = new_risky_target - risky
            # Cost is proportional to traded dollars and reduces wealth/cash.
            cost = cfg.tc * np.abs(traded)
            # Pay cost from cash; recompute post-trade positions.
            risky = new_risky_target
            cash = wealth - risky - cost
            turnover += np.abs(traded) / np.where(wealth > 0, wealth, 1.0)
            total_cost += cost

        # Market evolves one step.
        risky = risky * returns[:, t]
        cash = cash * Rf

    wealth = risky + cash
    util = crra_utility(wealth, gamma)
    tracking_error = (np.sqrt(track_sq / track_count)
                      if track_count > 0 else None)
    return StrategyResult(name=name, terminal_wealth=wealth, turnover=turnover,
                          total_cost=total_cost, tracking_error=tracking_error,
                          realized_utility=util)


def run_all_strategies(cfg: Config, sol: DPSolution, ml_model,
                       rng: Optional[np.random.Generator] = None
                       ) -> Dict[str, StrategyResult]:
    """
    Simulate all four strategies on a common return matrix and return their
    results keyed by strategy name.
    """
    rng = rng or np.random.default_rng(cfg.seed + 1)
    returns = sample_gross_returns(cfg, (cfg.n_paths, cfg.n_steps), rng)

    strategies = {
        "DP optimal":      make_dp_action(sol),
        "ML imitation":    make_ml_action(ml_model, cfg),
        "Periodic":        make_periodic_action(cfg),
        "Buy & hold":      make_buy_hold_action(cfg),
    }
    results = {}
    for name, action in strategies.items():
        track = None if name == "DP optimal" else sol
        results[name] = simulate_strategy(cfg, action, returns, name,
                                          dp_for_tracking=track)
    return results


def results_table(results: Dict[str, StrategyResult], gamma: float):
    """Assemble a list of per-strategy summary dicts (ready for a DataFrame)."""
    return [results[name].summary(gamma) for name in results]


def paired_comparison(results: Dict[str, StrategyResult],
                      reference: str = "DP optimal"):
    """
    Paired utility comparison of the reference strategy against each other.

    Because all strategies are simulated on a *common* return matrix (common
    random numbers), the per-path utility differences u_ref - u_other have far
    smaller variance than the individual utility estimates, giving a powerful
    test of whether the reference strategy truly dominates.

    Returns a list of dicts with the mean utility advantage of ``reference``,
    its standard error, t-statistic, and 95% confidence interval.
    """
    ref_u = results[reference].realized_utility
    rows = []
    for name, res in results.items():
        if name == reference:
            continue
        diff = ref_u - res.realized_utility            # >0 means reference better
        n = diff.shape[0]
        mean = float(np.mean(diff))
        se = float(np.std(diff, ddof=1) / np.sqrt(n))
        rows.append({
            "reference": reference,
            "vs": name,
            "mean_utility_advantage": mean,
            "se": se,
            "t_stat": mean / se if se > 0 else np.nan,
            "ci_low": mean - 1.96 * se,
            "ci_high": mean + 1.96 * se,
            "reference_better_95pct": bool(mean - 1.96 * se > 0),
        })
    return rows


if __name__ == "__main__":
    from .dp_solver import solve
    from .ml_policy import train_policy
    import pprint

    cfg = Config(n_paths=4000, ml_epochs=60)
    sol = solve(cfg)
    model, _ = train_policy(sol, cfg)
    res = run_all_strategies(cfg, sol, model)
    for row in results_table(res, cfg.gamma):
        pprint.pprint(row)
