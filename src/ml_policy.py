"""
Lightweight machine-learning approximation of the optimal rebalancing policy.

We treat the dynamic-programming policy y*(t, x) as a teacher and train a small
multilayer perceptron (the student) by supervised imitation learning:

    input   : (tau, x)   with tau = t / n_steps in [0, 1] the normalized time-to-go
              and x the pre-trade risky fraction,
    target  : y*(t, x)   the optimal post-trade fraction from the DP solver.

The student is a 2-hidden-layer MLP with tanh activations and a sigmoid output
(so predictions stay in [0, 1], matching the no-shorting / no-leverage state
space).  Training uses Adam with mean-squared error.

The point of the exercise is computational: once trained, the network maps a
state to an action in microseconds with no grid lookup or interpolation, and we
ask how closely this cheap approximation tracks the optimal control and its
economic performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .dp_solver import DPSolution


# --------------------------------------------------------------------------- #
# Dataset generation
# --------------------------------------------------------------------------- #
def build_dataset(sol: DPSolution):
    """
    Flatten the DP policy table into a supervised (features, targets) dataset.

    Features are (tau, x) for every (date, grid-point) pair; targets are the
    corresponding optimal post-trade fractions.

    Returns
    -------
    X : (N, 2) float32 ndarray  -- columns [tau, x]
    Y : (N, 1) float32 ndarray  -- optimal y*
    """
    cfg = sol.cfg
    n_steps, n_x = sol.policy.shape
    taus = np.arange(n_steps) / cfg.n_steps          # normalized time index
    tau_grid, x_grid = np.meshgrid(taus, sol.x_grid, indexing="ij")  # (n_steps, n_x)
    X = np.column_stack([tau_grid.ravel(), x_grid.ravel()]).astype(np.float32)
    Y = sol.policy.ravel().astype(np.float32)[:, None]
    return X, Y


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class PolicyMLP(nn.Module):
    """Small MLP mapping (tau, x) -> y in [0, 1] via a sigmoid output."""

    def __init__(self, hidden=(64, 64)):
        super().__init__()
        layers = []
        in_dim = 2
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.Tanh()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class TrainingHistory:
    train_loss: list
    val_loss: list

    @property
    def final_train(self) -> float:
        return self.train_loss[-1]

    @property
    def final_val(self) -> float:
        return self.val_loss[-1]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_policy(sol: DPSolution, cfg: Optional[Config] = None,
                 verbose: bool = False):
    """
    Train the imitation MLP on the DP policy.

    Returns
    -------
    model : PolicyMLP (in eval mode)
    history : TrainingHistory
    """
    cfg = cfg or sol.cfg
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    X, Y = build_dataset(sol)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_val = int(cfg.ml_val_frac * n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    Xtr = torch.from_numpy(X[train_idx]); Ytr = torch.from_numpy(Y[train_idx])
    Xva = torch.from_numpy(X[val_idx]);   Yva = torch.from_numpy(Y[val_idx])

    model = PolicyMLP(hidden=cfg.ml_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ml_lr,
                           weight_decay=cfg.ml_weight_decay)
    loss_fn = nn.MSELoss()

    n_train = Xtr.shape[0]
    history = TrainingHistory(train_loss=[], val_loss=[])

    for epoch in range(cfg.ml_epochs):
        model.train()
        epoch_perm = torch.from_numpy(rng.permutation(n_train))
        running = 0.0
        for start in range(0, n_train, cfg.ml_batch_size):
            idx = epoch_perm[start:start + cfg.ml_batch_size]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.shape[0]
        train_loss = running / n_train

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), Yva).item()
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        if verbose and (epoch % max(1, cfg.ml_epochs // 10) == 0):
            print(f"  epoch {epoch:4d}: train MSE={train_loss:.3e}  "
                  f"val MSE={val_loss:.3e}")

    model.eval()
    return model, history


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def policy_action(model: PolicyMLP, tau, x) -> np.ndarray:
    """
    Evaluate the trained policy at normalized time ``tau`` and fraction(s) ``x``.

    Both arguments broadcast; returns an ndarray of post-trade fractions.
    """
    tau = np.asarray(tau, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    tau, x = np.broadcast_arrays(tau, x)
    feats = np.column_stack([tau.ravel(), x.ravel()]).astype(np.float32)
    out = model(torch.from_numpy(feats)).numpy().ravel()
    return out.reshape(x.shape)


def save_checkpoint(model: PolicyMLP, cfg: Config, path: str) -> None:
    torch.save({"state_dict": model.state_dict(), "hidden": cfg.ml_hidden}, path)


def load_checkpoint(path: str) -> PolicyMLP:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = PolicyMLP(hidden=tuple(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    from .dp_solver import solve

    cfg = Config(ml_epochs=80)
    sol = solve(cfg)
    model, hist = train_policy(sol, cfg, verbose=True)
    print(f"\nFinal train MSE = {hist.final_train:.3e}, "
          f"val MSE = {hist.final_val:.3e}")
    # Spot-check against the optimal band center at t = 0.
    y_pred = policy_action(model, 0.0, cfg.merton_fraction)
    print(f"ML action at (tau=0, x=w*={cfg.merton_fraction:.3f}) = {float(y_pred):.4f}")
