"""
Two repairs for the failure of naive imitation.

The diagnosis in the paper is that a smooth regressor cannot represent the
optimal control's *inaction* region: inside the no-trade band the optimum is
exactly ``y = x``, and a network that is merely close to the diagonal trades on
every step. Both remedies here attack that specific defect, from opposite ends.

1. ``make_dead_zone_action`` -- an inference-time wrapper. It leaves the trained
   network untouched and simply snaps small predicted trades to "hold":

       y = x                whenever |y_hat - x| <= delta,
       y = y_hat            otherwise.

   This restores an inaction region of half-width ``delta`` by construction. It
   costs nothing to train and one comparison to evaluate.

2. ``train_cost_aware_policy`` -- a training-time fix. It adds a penalty on
   trading *more than the teacher does*:

       L = MSE(y_hat, y*) + lambda * mean( relu( |y_hat - x| - |y* - x| ) ).

   Note the asymmetry. Penalizing |y_hat - x| outright would also discourage the
   large, correct trades the optimal policy makes from outside the band, biasing
   the student toward inaction everywhere. Penalizing only the *excess* over the
   teacher's own trade size leaves correct trading unpunished and charges the
   student solely for the spurious turnover it invents inside the band -- which
   is exactly the quantity the transaction-cost model bills for.

Both return objects the existing simulation engine consumes unchanged, so they
are evaluated on the same paths, against the same baselines, as everything else.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .dp_solver import DPSolution
from .ml_policy import PolicyMLP, build_dataset, policy_action, TrainingHistory


# --------------------------------------------------------------------------- #
# Remedy 1: a dead-zone wrapper around an already-trained policy
# --------------------------------------------------------------------------- #
def dead_zone(y_hat: np.ndarray, x: np.ndarray, delta: float) -> np.ndarray:
    """
    Snap trades smaller than ``delta`` to exact inaction.

    Returning ``x`` itself (rather than something near it) matters: the
    simulator charges cost on ``|y * W - risky|``, so only an exact match
    produces a genuinely costless step.
    """
    return np.where(np.abs(y_hat - x) <= delta, x, y_hat)


def make_dead_zone_action(model: PolicyMLP, cfg: Config, delta: float) -> Callable:
    """Action callable for the dead-zone-wrapped ML policy."""
    def action(t: int, x: np.ndarray) -> np.ndarray:
        tau = t / cfg.n_steps
        y_hat = policy_action(model, np.full_like(x, tau), x)
        return dead_zone(y_hat, x, delta)
    return action


def in_band_residual(model: PolicyMLP, sol: DPSolution, t: int = 0) -> float:
    """
    Mean |y_hat - x| over the grid points inside the no-trade band at date ``t``.

    This is the paper's diagnostic: it is small but strictly positive for a
    smooth network, and every unit of it becomes turnover.
    """
    lo, hi = sol.no_trade_band(t)
    inside = sol.x_grid[(sol.x_grid >= lo) & (sol.x_grid <= hi)]
    if inside.size == 0:
        return float("nan")
    tau = t / sol.cfg.n_steps
    return float(np.mean(np.abs(policy_action(model, tau, inside) - inside)))


# --------------------------------------------------------------------------- #
# Remedy 2: a cost-aware training objective
# --------------------------------------------------------------------------- #
def train_cost_aware_policy(sol: DPSolution,
                            cfg: Optional[Config] = None,
                            lam: Optional[float] = None,
                            verbose: bool = False):
    """
    Train the imitation MLP with an excess-turnover penalty.

    Identical to ``ml_policy.train_policy`` in architecture, data, optimizer and
    seeding, so any difference in behaviour is attributable to the loss alone.

    Returns
    -------
    model : PolicyMLP (eval mode)
    history : TrainingHistory   (val_loss is plain MSE, for comparability)
    """
    cfg = cfg or sol.cfg
    lam = cfg.cost_aware_lambda if lam is None else lam

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    X, Y = build_dataset(sol)          # X = [tau, x], Y = y*
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = int(cfg.ml_val_frac * n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Xtr = torch.from_numpy(X[train_idx]); Ytr = torch.from_numpy(Y[train_idx])
    Xva = torch.from_numpy(X[val_idx]);   Yva = torch.from_numpy(Y[val_idx])

    model = PolicyMLP(hidden=cfg.ml_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ml_lr,
                           weight_decay=cfg.ml_weight_decay)
    mse = nn.MSELoss()

    n_train = Xtr.shape[0]
    history = TrainingHistory(train_loss=[], val_loss=[])

    for epoch in range(cfg.ml_epochs):
        model.train()
        epoch_perm = torch.from_numpy(rng.permutation(n_train))
        running = 0.0
        for start in range(0, n_train, cfg.ml_batch_size):
            idx = epoch_perm[start:start + cfg.ml_batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            pre_trade = xb[:, 1:2]                     # the x column

            opt.zero_grad()
            pred = model(xb)
            fit = mse(pred, yb)
            # Charge only for trading beyond what the teacher would trade.
            excess = torch.relu((pred - pre_trade).abs()
                                - (yb - pre_trade).abs()).mean()
            loss = fit + lam * excess
            loss.backward()
            opt.step()
            running += loss.item() * xb.shape[0]
        train_loss = running / n_train

        model.eval()
        with torch.no_grad():
            val_loss = mse(model(Xva), Yva).item()     # plain MSE: comparable
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        if verbose and (epoch % max(1, cfg.ml_epochs // 10) == 0):
            print(f"  epoch {epoch:4d}: train={train_loss:.3e}  "
                  f"val MSE={val_loss:.3e}")

    model.eval()
    return model, history


if __name__ == "__main__":
    from .dp_solver import solve
    from .ml_policy import train_policy

    cfg = Config(ml_epochs=120)
    sol = solve(cfg)

    plain, _ = train_policy(sol, cfg)
    aware, _ = train_cost_aware_policy(sol, cfg)

    print(f"in-band mean |y-x|, plain MSE   : {in_band_residual(plain, sol):.5f}")
    print(f"in-band mean |y-x|, cost-aware  : {in_band_residual(aware, sol):.5f}")
    lo, hi = sol.no_trade_band(0)
    xs = np.linspace(lo, hi, 9)
    print(f"dead-zone (delta=0.01) inside the band snaps to hold: "
          f"{np.allclose(dead_zone(policy_action(plain, 0.0, xs), xs, 0.01), xs)}")
