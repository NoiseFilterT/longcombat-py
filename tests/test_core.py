"""Tests for long_combat."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from longcombat import LongCombatResult, long_combat


BASE_KWARGS = dict(
    batch_col="batch",
    id_col="subid",
    time_col="time",
    formula="age + diagnosis*time",
    ranef="(1|subid)",
    verbose=False,
)


class TestBasicHarmonization:
    def test_returns_result_dataclass(self, synthetic_data):
        result = long_combat(
            synthetic_data, features=["feat1", "feat2", "feat3"], **BASE_KWARGS
        )
        assert isinstance(result, LongCombatResult)

    def test_data_combat_shape_and_columns(self, synthetic_data):
        features = ["feat1", "feat2", "feat3"]
        result = long_combat(synthetic_data, features=features, **BASE_KWARGS)
        assert result.data_combat.shape == (len(synthetic_data), 3 + len(features))
        assert list(result.data_combat.columns) == [
            "subid", "time", "batch",
            "feat1.combat", "feat2.combat", "feat3.combat",
        ]

    def test_gammahat_shape(self, synthetic_data):
        result = long_combat(
            synthetic_data, features=["feat1", "feat2", "feat3"], **BASE_KWARGS
        )
        assert result.gammahat.shape == (3, 3)
        assert list(result.gammahat.index) == ["A", "B", "C"]
        assert list(result.gammahat.columns) == ["feat1", "feat2", "feat3"]

    def test_harmonization_reduces_batch_mean_spread(self, synthetic_data):
        """Per-batch means of harmonized data should be closer to a common mean
        than per-batch means of the raw data."""
        features = ["feat1", "feat2", "feat3"]
        result = long_combat(synthetic_data, features=features, **BASE_KWARGS)

        pre_spread = []
        post_spread = []
        for feat in features:
            raw_means = synthetic_data.groupby("batch", observed=True)[feat].mean().to_numpy()
            harmonized_means = (
                result.data_combat.groupby("batch", observed=True)[f"{feat}.combat"]
                .mean()
                .to_numpy()
            )
            pre_spread.append(float(raw_means.std()))
            post_spread.append(float(harmonized_means.std()))
        assert all(p < r for p, r in zip(post_spread, pre_spread, strict=True)), (
            f"post={post_spread} not all < pre={pre_spread}"
        )

    def test_harmonization_reduces_batch_variance_spread(self, synthetic_data):
        features = ["feat1", "feat2", "feat3"]
        result = long_combat(synthetic_data, features=features, **BASE_KWARGS)

        pre_spread = []
        post_spread = []
        for feat in features:
            raw_vars = synthetic_data.groupby("batch", observed=True)[feat].var().to_numpy()
            harm_vars = (
                result.data_combat.groupby("batch", observed=True)[f"{feat}.combat"]
                .var()
                .to_numpy()
            )
            pre_spread.append(float(raw_vars.std()))
            post_spread.append(float(harm_vars.std()))
        assert all(p < r for p, r in zip(post_spread, pre_spread, strict=True))

    def test_id_and_time_preserved(self, synthetic_data):
        features = ["feat1", "feat2"]
        result = long_combat(synthetic_data, features=features, **BASE_KWARGS)
        pd.testing.assert_series_equal(
            result.data_combat["subid"].reset_index(drop=True),
            synthetic_data["subid"].reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            result.data_combat["time"].reset_index(drop=True),
            synthetic_data["time"].reset_index(drop=True),
            check_names=False,
        )


class TestEbToggle:
    def test_eb_false_uses_raw_hyperparameters(self, synthetic_data):
        features = ["feat1", "feat2"]
        result = long_combat(
            synthetic_data, features=features, eb=False, **BASE_KWARGS
        )
        np.testing.assert_allclose(
            result.gammastarhat.values, result.gammahat.values, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            result.delta2starhat.values, result.delta2hat.values, rtol=0, atol=0
        )

    def test_eb_true_differs_from_raw(self, synthetic_data):
        features = ["feat1", "feat2"]
        result = long_combat(
            synthetic_data, features=features, eb=True, **BASE_KWARGS
        )
        # EB shrinkage should meaningfully move gammastar off gammahat
        # (unless data is pathological — the fixture is not).
        assert not np.allclose(
            result.gammastarhat.values, result.gammahat.values, atol=1e-6
        )


class TestMeanOnly:
    def test_delta2starhat_all_ones(self, synthetic_data):
        features = ["feat1", "feat2"]
        result = long_combat(
            synthetic_data, features=features, mean_only=True, **BASE_KWARGS
        )
        np.testing.assert_array_equal(
            result.delta2starhat.values, np.ones_like(result.delta2starhat.values)
        )

    def test_single_feature_mean_only_raises(self, synthetic_data):
        # mean_only also needs the cross-feature hyperprior tau2bar; a single
        # feature must raise, not silently emit all-NaN output.
        with pytest.raises(ValueError, match="requires at least 2 features"):
            long_combat(
                synthetic_data, features=["feat1"], mean_only=True, **BASE_KWARGS
            )

    def test_gammastarhat_has_shrinkage(self, synthetic_data):
        """Closed-form γ* when δ²=1: (n τ² γ̂ + γ̄)/(n τ² + 1)."""
        features = ["feat1", "feat2", "feat3"]
        result = long_combat(
            synthetic_data, features=features, mean_only=True, **BASE_KWARGS
        )
        # Reproduce the closed form manually from gammahat and confirm match.
        ni = (
            synthetic_data.groupby("batch", observed=True).size().sort_index().to_numpy(
                dtype=float
            )
        )
        gammahat = result.gammahat.values  # (m, V)
        gammabar = gammahat.mean(axis=1)
        tau2bar = gammahat.var(axis=1, ddof=1)
        expected = (
            (ni[:, None] * tau2bar[:, None] * gammahat + gammabar[:, None])
            / (ni[:, None] * tau2bar[:, None] + 1.0)
        )
        np.testing.assert_allclose(result.gammastarhat.values, expected, rtol=1e-12)


class TestMethodArgument:
    def test_msr_differs_from_reml(self, synthetic_data):
        features = ["feat1", "feat2"]
        r_reml = long_combat(synthetic_data, features=features, method="REML", **BASE_KWARGS)
        r_msr = long_combat(synthetic_data, features=features, method="MSR", **BASE_KWARGS)
        # MSR residuals-based estimator is systematically different from REML.
        assert not np.allclose(r_reml.data_combat.iloc[:, 3:].values,
                               r_msr.data_combat.iloc[:, 3:].values)

    def test_rejects_unknown_method(self, synthetic_data):
        with pytest.raises(ValueError, match="method must be"):
            long_combat(
                synthetic_data, features=["feat1"], method="FOO", **BASE_KWARGS
            )


class TestEdgeCases:
    def test_missing_data_raises(self, synthetic_data):
        df = synthetic_data.copy()
        df.loc[0, "feat1"] = np.nan
        with pytest.raises(ValueError, match="Missing data in variables"):
            long_combat(df, features=["feat1", "feat2"], **BASE_KWARGS)

    def test_single_batch_raises(self, single_batch_data):
        with pytest.raises(ValueError, match="at least 2 batches"):
            long_combat(single_batch_data, features=["feat1", "feat2"], **BASE_KWARGS)

    def test_single_observation_batch_raises(self, synthetic_data):
        df = synthetic_data.copy()
        df.loc[df.index[0], "batch"] = "SOLO"
        with pytest.raises(ValueError, match="fewer than 2 observations"):
            long_combat(df, features=["feat1", "feat2"], **BASE_KWARGS)

    def test_single_feature_requires_eb_false(self, synthetic_data):
        # EB needs cross-feature moments; single-feature harmonization is
        # only defined with eb=False.
        with pytest.raises(ValueError, match="requires at least 2 features"):
            long_combat(synthetic_data, features=["feat1"], **BASE_KWARGS)

        result = long_combat(
            synthetic_data, features=["feat1"], eb=False, **BASE_KWARGS
        )
        assert result.data_combat.shape[1] == 3 + 1
        assert result.gammahat.shape == (3, 1)
        # No NaNs in the harmonized output.
        assert not result.data_combat["feat1.combat"].isna().any()

    def test_integer_feature_indices(self, synthetic_data):
        # columns 5,6,7 are feat1, feat2, feat3
        col_ix = [
            synthetic_data.columns.get_loc(c) for c in ("feat1", "feat2", "feat3")
        ]
        result = long_combat(synthetic_data, features=col_ix, **BASE_KWARGS)
        assert list(result.gammahat.columns) == ["feat1", "feat2", "feat3"]

    def test_missing_feature_column_raises(self, synthetic_data):
        with pytest.raises(KeyError, match="feature column"):
            long_combat(synthetic_data, features=["does_not_exist"], **BASE_KWARGS)

    def test_batch_col_with_dot_in_name(self, synthetic_data):
        # Column names with dots/spaces are legal in R and common in
        # neuroimaging tables; they must not break the Patsy formula.
        df = synthetic_data.rename(columns={"batch": "scanner.site"})
        kwargs = {**BASE_KWARGS, "batch_col": "scanner.site"}
        result = long_combat(df, features=["feat1", "feat2"], **kwargs)
        assert not result.data_combat.iloc[:, 3:].isna().any().any()
        assert list(result.gammahat.index) == ["A", "B", "C"]

    def test_features_accepts_numpy_array_and_index(self, synthetic_data):
        # The signature is typed Sequence; ndarray / Index / range should work,
        # not just list / tuple.
        r1 = long_combat(
            synthetic_data, features=np.array(["feat1", "feat2"]), **BASE_KWARGS
        )
        r2 = long_combat(
            synthetic_data, features=synthetic_data.columns[5:7], **BASE_KWARGS
        )
        assert list(r1.gammahat.columns) == ["feat1", "feat2"]
        assert list(r2.gammahat.columns) == ["feat1", "feat2"]

    def test_no_spurious_convergence_warning(self, synthetic_data):
        # A well-behaved random-intercept fit must not emit the new
        # non-convergence warning.
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            long_combat(synthetic_data, features=["feat1", "feat2"], **BASE_KWARGS)
        assert not any("did not converge" in str(x.message) for x in w)

    def test_random_slope_ranef(self, synthetic_data):
        # Should fit without error.
        result = long_combat(
            synthetic_data,
            features=["feat1", "feat2"],
            batch_col="batch",
            id_col="subid",
            time_col="time",
            formula="age + diagnosis*time",
            ranef="(1 + time|subid)",
            verbose=False,
        )
        assert result.data_combat.shape == (len(synthetic_data), 5)

    def test_dict_like_access(self, synthetic_data):
        result = long_combat(synthetic_data, features=["feat1", "feat2"], **BASE_KWARGS)
        assert result["data_combat"] is result.data_combat
        assert "gammahat" in result.keys()


class TestSumConstraint:
    def test_gammahat_row_weighted_sum_zero(self, synthetic_data):
        """The sum-to-L constraint Σ n_i γ_i = 0 should hold on the
        *recovered* (adjusted) batch effects. The standardized residuals'
        column means (gammahat rows), weighted by batch size, should be ~0
        per feature — that's the identity this constraint imposes."""
        features = ["feat1", "feat2", "feat3"]
        result = long_combat(synthetic_data, features=features, **BASE_KWARGS)
        ni = (
            synthetic_data.groupby("batch", observed=True).size().sort_index().to_numpy(
                dtype=float
            )
        )
        # sum_i n_i * gammahat[i,v] should be close to 0 for each feature v.
        weighted = (ni[:, None] * result.gammahat.values).sum(axis=0)
        np.testing.assert_allclose(weighted, 0.0, atol=1e-8)
