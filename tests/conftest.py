"""
Shared pytest fixtures.

The heavy objects in this project (the DP solution, the trained network) are
deterministic functions of ``Config``, so they are built once per session and
reused across tests.  Tests that need a *fast* solve use ``small_cfg``, which
shrinks the grid and horizon without changing any economics.
"""

from __future__ import annotations

import os
import sys

import pytest

# Make the project root importable when pytest is run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import Config          # noqa: E402
from src.dp_solver import solve        # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The published configuration -- the one the paper's numbers come from."""
    return Config()


@pytest.fixture(scope="session")
def small_cfg() -> Config:
    """A coarser, faster configuration for tests that only need structure."""
    return Config(n_steps=12, n_x=101, n_quad=15, n_paths=2000, ml_epochs=40)


@pytest.fixture(scope="session")
def small_sol(small_cfg):
    return solve(small_cfg)


@pytest.fixture(scope="session")
def full_sol(cfg):
    """Full-resolution DP solution (~0.1 s, cheap enough to solve once)."""
    return solve(cfg)


@pytest.fixture(scope="session")
def results_dir() -> str:
    return os.path.join(ROOT, "results")


@pytest.fixture(scope="session")
def project_root() -> str:
    return ROOT
