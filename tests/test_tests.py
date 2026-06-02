"""Tests for add_test and mult_test."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from longcombat import add_test, mult_test


COMMON = dict(
    batch_col="batch",
    id_col="subid",
    formula="age + diagnosis*time",
    ranef="(1|subid)",
    verbose=False,
)


class TestAddTest:
    def test_returns_dataframe_with_expected_cols(self, synthetic_data):
        out = add_test(synthetic_data, features=["feat1", "feat2"], **COMMON)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["feature", "chi2", "df", "p_value"]

    def test_sorted_by_chi2_desc(self, synthetic_data):
        out = add_test(
            synthetic_data, features=["feat1", "feat2", "feat3"], **COMMON
        )
        chi2 = out["chi2"].to_numpy()
        assert (chi2[:-1] >= chi2[1:]).all()

    def test_chi2_nonnegative(self, synthetic_data):
        out = add_test(
            synthetic_data, features=["feat1", "feat2", "feat3"], **COMMON
        )
        assert (out["chi2"] >= 0).all()

    def test_detects_additive_effects(self, synthetic_data, clean_data):
        """With true additive batch effects, p-values should be much smaller
        than on matched data with no batch effects."""
        dirty = add_test(synthetic_data, features=["feat1"], **COMMON)
        clean = add_test(clean_data, features=["feat1"], **COMMON)
        assert dirty.loc[0, "p_value"] < clean.loc[0, "p_value"]
        assert dirty.loc[0, "p_value"] < 0.01

    def test_method_kr_raises(self, synthetic_data):
        with pytest.raises(NotImplementedError, match="Kenward-Roger"):
            add_test(
                synthetic_data, features=["feat1"], method="kr", **COMMON
            )

    def test_invalid_method_raises(self, synthetic_data):
        with pytest.raises(ValueError, match="method must be"):
            add_test(
                synthetic_data, features=["feat1"], method="foo", **COMMON
            )


class TestMultTest:
    def test_returns_dataframe_with_expected_cols(self, synthetic_data):
        out = mult_test(synthetic_data, features=["feat1", "feat2"], **COMMON)
        assert list(out.columns) == ["feature", "chi2", "df", "p_value"]

    def test_sorted_by_chi2_desc(self, synthetic_data):
        out = mult_test(
            synthetic_data, features=["feat1", "feat2", "feat3"], **COMMON
        )
        chi2 = out["chi2"].to_numpy()
        assert (chi2[:-1] >= chi2[1:]).all()

    def test_detects_multiplicative_effects(self, synthetic_data, clean_data):
        dirty = mult_test(synthetic_data, features=["feat1"], **COMMON)
        clean = mult_test(clean_data, features=["feat1"], **COMMON)
        # Dirty data has batch_mult = {A:1, B:0.5, C:1.5} — strong multiplicative effect
        assert dirty.loc[0, "p_value"] < clean.loc[0, "p_value"]


class TestMissingDataValidation:
    """NaN in a feature column must raise a clear ValueError, not an opaque
    IndexError from statsmodels' row-dropping (matches long_combat)."""

    def test_add_test_missing_data_raises(self, synthetic_data):
        df = synthetic_data.copy()
        df.loc[df.index[0], "feat1"] = np.nan
        with pytest.raises(ValueError, match="Missing data in variables"):
            add_test(df, features=["feat1", "feat2"], **COMMON)

    def test_mult_test_missing_data_raises(self, synthetic_data):
        df = synthetic_data.copy()
        df.loc[df.index[0], "feat1"] = np.nan
        with pytest.raises(ValueError, match="Missing data in variables"):
            mult_test(df, features=["feat1", "feat2"], **COMMON)

    def test_batch_col_with_dot_in_name(self, synthetic_data):
        df = synthetic_data.rename(columns={"batch": "scanner.site"})
        common = {**COMMON, "batch_col": "scanner.site"}
        out = add_test(df, features=["feat1", "feat2"], **common)
        assert list(out.columns) == ["feature", "chi2", "df", "p_value"]
