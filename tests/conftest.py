"""Shared test fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_longitudinal(
    *,
    seed: int,
    n_subj: int,
    n_visit: int,
    n_feat: int,
    batches: tuple[str, ...],
    batch_add: dict[str, float] | None,
    batch_mult: dict[str, float] | None,
    subj_re_sd: float = 1.0,
    noise_sd: float = 1.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    L = n_subj * n_visit
    subid = np.repeat(np.arange(n_subj), n_visit)
    time = np.tile(np.arange(n_visit), n_subj).astype(float)
    age = np.repeat(rng.integers(20, 60, n_subj), n_visit).astype(float)
    dx = np.repeat(rng.integers(0, 2, n_subj), n_visit)
    batch = rng.choice(list(batches), size=L)
    subj_re = rng.normal(0, subj_re_sd, n_subj)
    subj_signal = np.array([subj_re[s] for s in subid])

    if batch_add is None:
        batch_add = {b: 0.0 for b in batches}
    if batch_mult is None:
        batch_mult = {b: 1.0 for b in batches}
    bmul_vec = np.array([batch_mult[b] for b in batch])
    badd_vec = np.array([batch_add[b] for b in batch])

    df = pd.DataFrame(
        {"subid": subid, "time": time, "age": age, "diagnosis": dx, "batch": batch}
    )
    for v in range(n_feat):
        noise = rng.normal(0, noise_sd, L)
        signal = (
            0.01 * age - 0.1 * time + 0.5 * dx - 0.2 * dx * time + subj_signal
        )
        df[f"feat{v + 1}"] = signal + bmul_vec * noise + badd_vec
    return df


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Small longitudinal dataset with known additive + multiplicative batch effects.

    30 subjects × 4 visits = 120 rows, 3 features, 3 batches.
    """
    return _make_longitudinal(
        seed=1,
        n_subj=30,
        n_visit=4,
        n_feat=3,
        batches=("A", "B", "C"),
        batch_add={"A": 0.0, "B": 2.0, "C": -1.5},
        batch_mult={"A": 1.0, "B": 0.5, "C": 1.5},
    )


@pytest.fixture
def clean_data() -> pd.DataFrame:
    """Longitudinal dataset with *no* batch effects — baseline for sanity checks."""
    return _make_longitudinal(
        seed=2,
        n_subj=30,
        n_visit=4,
        n_feat=3,
        batches=("A", "B", "C"),
        batch_add=None,
        batch_mult=None,
    )


@pytest.fixture
def single_batch_data() -> pd.DataFrame:
    """Data with only one batch level (should error on long_combat)."""
    return _make_longitudinal(
        seed=3,
        n_subj=10,
        n_visit=3,
        n_feat=2,
        batches=("A",),
        batch_add=None,
        batch_mult=None,
    )
