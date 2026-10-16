"""Shared fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random generator for deterministic tests."""
    return np.random.default_rng(42)
