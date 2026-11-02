"""
Tests for the imitation-learning policy.

Two things matter here.  First, the mechanics: the dataset, the architecture,
and the checkpoint round-trip.  Second -- and this is the paper's central claim
-- that a well-fit network still *fails to reproduce the no-trade region*: its
action inside the band is close to, but never exactly, the identity, which is
what generates the spurious turnover documented in the results section.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.ml_policy import (build_dataset, PolicyMLP, train_policy, policy_action,
                           save_checkpoint, load_checkpoint)


@pytest.fixture(scope="module")
def trained(small_sol, small_cfg):
    model, history = train_policy(small_sol, small_cfg)
    return model, history


# --------------------------------------------------------------------------- #
# Dataset and architecture
# --------------------------------------------------------------------------- #
def test_dataset_covers_the_whole_policy_table(small_sol, small_cfg):
    X, Y = build_dataset(small_sol)
    assert X.shape == (small_cfg.n_steps * small_cfg.n_x, 2)
    assert Y.shape == (small_cfg.n_steps * small_cfg.n_x, 1)
    assert X.dtype == np.float32 and Y.dtype == np.float32
    # tau in [0, 1), x spans the allocation grid, targets are valid actions.
    assert 0.0 <= X[:, 0].min() and X[:, 0].max() < 1.0
    assert X[:, 1].min() == pytest.approx(small_cfg.x_min)
    assert X[:, 1].max() == pytest.approx(small_cfg.x_max)
    assert Y.min() >= small_cfg.x_min and Y.max() <= small_cfg.x_max


def test_dataset_targets_match_the_dp_policy(small_sol):
    X, Y = build_dataset(small_sol)
    n_x = small_sol.x_grid.size
    for t in (0, 3):
        rows = slice(t * n_x, (t + 1) * n_x)
        assert np.allclose(Y[rows, 0], small_sol.policy[t])


def test_published_architecture_has_the_reported_parameter_count():
    """The paper quotes a 4,417-parameter network: 2x64 tanh MLP, sigmoid out."""
    model = PolicyMLP(hidden=(64, 64))
    assert sum(p.numel() for p in model.parameters()) == 4417


def test_output_is_confined_to_the_action_space():
    """The sigmoid head guarantees admissible actions -- no shorting, no leverage."""
    model = PolicyMLP(hidden=(16, 16))
    feats = torch.rand(500, 2) * 3.0 - 1.0        # deliberately out of range
    out = model(feats).detach().numpy()
    assert out.min() > 0.0 and out.max() < 1.0


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def test_training_reduces_loss(trained, small_cfg):
    _, history = trained
    assert len(history.train_loss) == small_cfg.ml_epochs
    assert history.final_train < history.train_loss[0]
    assert history.final_val < history.val_loss[0]


def test_training_reaches_a_small_validation_error(trained):
    """Fitting the policy in MSE is easy -- that is precisely the paper's point."""
    _, history = trained
    assert history.final_val < 1e-2


def test_training_is_seed_deterministic(small_sol, small_cfg):
    a, _ = train_policy(small_sol, small_cfg)
    b, _ = train_policy(small_sol, small_cfg)
    xs = np.linspace(0, 1, 51)
    assert np.allclose(policy_action(a, 0.0, xs), policy_action(b, 0.0, xs))


def test_model_is_left_in_eval_mode(trained):
    model, _ = trained
    assert not model.training


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def test_policy_action_broadcasts_and_stays_admissible(trained):
    model, _ = trained
    xs = np.linspace(0.0, 1.0, 37)
    y = policy_action(model, 0.25, xs)
    assert y.shape == xs.shape
    assert np.all((y >= 0.0) & (y <= 1.0))


def test_policy_action_tracks_the_dp_policy(trained, small_sol):
    model, _ = trained
    xs = small_sol.x_grid
    gap = policy_action(model, 0.0, xs) - small_sol.policy_at(0, xs)
    assert np.sqrt(np.mean(gap ** 2)) < 0.1


def test_learned_policy_does_not_reproduce_the_no_trade_identity(trained, small_sol):
    """
    The paper's mechanism, stated as a test: inside the band the optimal action
    is exactly y = x, but the smooth network is off the diagonal *everywhere*.
    Each such residual is a trade the optimal policy would not make.
    """
    model, _ = trained
    lo, hi = small_sol.no_trade_band(0)
    inside = small_sol.x_grid[(small_sol.x_grid >= lo) & (small_sol.x_grid <= hi)]
    assert inside.size > 3

    residual = np.abs(policy_action(model, 0.0, inside) - inside)
    assert np.all(residual > 0), "a smooth regressor cannot hit y == x exactly"
    # Small in action space, yet strictly nonzero -- the whole failure mode.
    assert residual.mean() < 0.1


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def test_checkpoint_round_trip_preserves_the_policy(trained, small_cfg, tmp_path):
    model, _ = trained
    path = str(tmp_path / "policy.pt")
    save_checkpoint(model, small_cfg, path)
    reloaded = load_checkpoint(path)
    xs = np.linspace(0, 1, 101)
    assert np.array_equal(policy_action(model, 0.5, xs),
                          policy_action(reloaded, 0.5, xs))
    assert not reloaded.training


def test_published_checkpoint_loads_and_matches_the_reported_size(results_dir):
    """The committed checkpoint must still load with the current code."""
    import os
    path = os.path.join(results_dir, "checkpoints", "ml_policy.pt")
    model = load_checkpoint(path)
    assert sum(p.numel() for p in model.parameters()) == 4417
    y = policy_action(model, 0.0, np.linspace(0, 1, 11))
    assert np.all((y >= 0.0) & (y <= 1.0))
