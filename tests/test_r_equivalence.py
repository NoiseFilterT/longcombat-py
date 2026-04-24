"""Numerical-equivalence tests against R's longCombat outputs.

Skipped when the CSV fixtures in tests/fixtures/ are not present. To
generate them, see tests/fixtures/README.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from longcombat import long_combat


FIXTURE_DIR = Path(__file__).parent / "fixtures"
REQUIRED = [
    "r_input.csv",
    "r_data_combat.csv",
    "r_gammahat.csv",
    "r_delta2hat.csv",
    "r_gammastarhat.csv",
    "r_delta2starhat.csv",
]

pytestmark = pytest.mark.skipif(
    not all((FIXTURE_DIR / f).exists() for f in REQUIRED),
    reason=(
        "R-equivalence fixtures not present. Run tests/fixtures/regenerate.R"
        " (requires R + lme4 + longCombat). See tests/fixtures/README.md."
    ),
)


# Tolerances: see DIFFERENCES_FROM_R.md — lme4 (bobyqa) vs statsmodels
# (lbfgs) use different optimizers and parameterizations, but on the
# seed-1 fixture we observe agreement of ~1e-5 relative for the
# intermediate quantities and ~5e-5 absolute on harmonized values (where
# a few values near zero inflate relative error to ~1e-3). Tolerances
# below leave generous headroom so innocuous numerical changes in
# statsmodels don't flake the test.
TOL_DATA_COMBAT = dict(rtol=5e-3, atol=5e-4)
TOL_GAMMAHAT = dict(rtol=2e-4, atol=5e-5)
TOL_DELTA2HAT = dict(rtol=2e-4, atol=5e-5)
TOL_EB = dict(rtol=2e-4, atol=5e-5)


@pytest.fixture
def r_input() -> pd.DataFrame:
    return pd.read_csv(FIXTURE_DIR / "r_input.csv")


@pytest.fixture
def r_outputs() -> dict[str, pd.DataFrame]:
    return {
        "data_combat": pd.read_csv(FIXTURE_DIR / "r_data_combat.csv"),
        "gammahat": pd.read_csv(FIXTURE_DIR / "r_gammahat.csv", index_col=0),
        "delta2hat": pd.read_csv(FIXTURE_DIR / "r_delta2hat.csv", index_col=0),
        "gammastarhat": pd.read_csv(FIXTURE_DIR / "r_gammastarhat.csv", index_col=0),
        "delta2starhat": pd.read_csv(FIXTURE_DIR / "r_delta2starhat.csv", index_col=0),
    }


def _run_python(r_input: pd.DataFrame):
    feature_names = [c for c in r_input.columns if c.startswith("feat")]
    return long_combat(
        data=r_input,
        batch_col="batch",
        id_col="subid",
        time_col="time",
        features=feature_names,
        formula="age + diagnosis*time",
        ranef="(1|subid)",
        niter=30,
        method="REML",
        verbose=False,
    )


class TestREquivalence:
    def test_data_combat(self, r_input, r_outputs):
        py = _run_python(r_input)
        feature_names = [c for c in r_input.columns if c.startswith("feat")]
        combat_cols = [f"{f}.combat" for f in feature_names]

        r_vals = r_outputs["data_combat"][combat_cols].to_numpy()
        py_vals = py.data_combat[combat_cols].to_numpy()
        np.testing.assert_allclose(py_vals, r_vals, **TOL_DATA_COMBAT)

    def test_gammahat(self, r_input, r_outputs):
        py = _run_python(r_input)
        feature_names = [c for c in r_input.columns if c.startswith("feat")]
        r_vals = r_outputs["gammahat"].loc[["A", "B", "C"], feature_names].to_numpy()
        py_vals = py.gammahat.loc[["A", "B", "C"], feature_names].to_numpy()
        np.testing.assert_allclose(py_vals, r_vals, **TOL_GAMMAHAT)

    def test_delta2hat(self, r_input, r_outputs):
        py = _run_python(r_input)
        feature_names = [c for c in r_input.columns if c.startswith("feat")]
        r_vals = r_outputs["delta2hat"].loc[["A", "B", "C"], feature_names].to_numpy()
        py_vals = py.delta2hat.loc[["A", "B", "C"], feature_names].to_numpy()
        np.testing.assert_allclose(py_vals, r_vals, **TOL_DELTA2HAT)

    def test_gammastarhat(self, r_input, r_outputs):
        py = _run_python(r_input)
        feature_names = [c for c in r_input.columns if c.startswith("feat")]
        r_vals = r_outputs["gammastarhat"].loc[["A", "B", "C"], feature_names].to_numpy()
        py_vals = py.gammastarhat.loc[["A", "B", "C"], feature_names].to_numpy()
        np.testing.assert_allclose(py_vals, r_vals, **TOL_EB)

    def test_delta2starhat(self, r_input, r_outputs):
        py = _run_python(r_input)
        feature_names = [c for c in r_input.columns if c.startswith("feat")]
        r_vals = r_outputs["delta2starhat"].loc[["A", "B", "C"], feature_names].to_numpy()
        py_vals = py.delta2starhat.loc[["A", "B", "C"], feature_names].to_numpy()
        np.testing.assert_allclose(py_vals, r_vals, **TOL_EB)
