"""Tests for additive and multiplicative batch effects.

Ports of ``longCombat/R/addTest.R`` and ``longCombat/R/multTest.R``.

See ``DIFFERENCES_FROM_R.md`` for an important deviation in ``add_test``:
the R version uses a Kenward-Roger F-test from ``pbkrtest``; there is no
Kenward-Roger implementation in the Python scientific stack, so the default
here is a likelihood-ratio chi-squared test.
"""
from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats

from ._core import _validate_no_missing_data
from ._ranef import parse_ranef


def add_test(
    data: pd.DataFrame,
    batch_col: str,
    id_col: str,
    features: Sequence[str] | Sequence[int],
    formula: str,
    ranef: str,
    *,
    method: str = "lrt",
    verbose: bool = True,
) -> pd.DataFrame:
    """Test for additive batch effects in LME residuals, per feature.

    For each feature, fits a "full" LME with ``batch_col`` as a fixed effect
    and a "reduced" LME without it, then reports a likelihood-ratio
    chi-squared statistic comparing them.

    Parameters
    ----------
    data : pandas.DataFrame
    batch_col, id_col : str
    features : sequence of str or int
        Integer indices are 0-based (Python), unlike R's 1-based; see
        ``DIFFERENCES_FROM_R.md`` §9.
    formula : str
        Fixed-effects right-hand side (Patsy), must **not** include
        ``batch_col`` or random effects.
    ranef : str
        Random-effects specification in ``lme4`` notation (e.g.
        ``"(1|subid)"``).
    method : {'lrt', 'kr'}, default 'lrt'
        ``'lrt'`` performs a likelihood-ratio chi-squared test. ``'kr'``
        raises :class:`NotImplementedError`; see ``DIFFERENCES_FROM_R.md``.
    verbose : bool, default True

    Returns
    -------
    pandas.DataFrame
        Columns ``['feature', 'chi2', 'df', 'p_value']``, sorted by
        ``chi2`` descending. See ``DIFFERENCES_FROM_R.md`` for how these
        columns relate to R's Kenward-Roger output.

    Raises
    ------
    NotImplementedError
        If ``method='kr'``. Kenward-Roger is not available in Python; use
        the R package for that.
    """
    if method == "kr":
        raise NotImplementedError(
            "Kenward-Roger testing is not available in this Python port. "
            "Pass method='lrt' for the likelihood-ratio chi-squared test, "
            "or use the R package longCombat::addTest for KR p-values. "
            "See DIFFERENCES_FROM_R.md for details."
        )
    if method != "lrt":
        raise ValueError(f"method must be 'lrt' or 'kr', got {method!r}")

    feature_names = _resolve_feature_names(data, features)
    if batch_col not in data.columns:
        raise KeyError(f"batch_col={batch_col!r} not found in data.columns")
    if id_col not in data.columns:
        raise KeyError(f"id_col={id_col!r} not found in data.columns")

    work = data.copy()
    # Force categorical for deterministic dummy coding.
    uniques = pd.unique(work[batch_col])
    try:
        levels = sorted(uniques.tolist())
    except TypeError:
        levels = sorted(uniques.tolist(), key=repr)
    work[batch_col] = pd.Categorical(work[batch_col], categories=levels)
    n_batch = len(levels)
    if verbose:
        print(f"[add_test] found {n_batch} batches")
        print(f"[add_test] found {len(feature_names)} features")

    ranef_spec = parse_ranef(ranef)
    _validate_no_missing_data(
        work, [batch_col, id_col, ranef_spec.groups_col, *feature_names]
    )

    rows = []
    for v, feat in enumerate(feature_names, start=1):
        if verbose:
            print(f"[add_test] testing for additive batch effect for feature {v} ({feat})")
        full_rhs = f"{formula} + C(Q('{batch_col}'), Treatment)"
        reduced_rhs = formula
        full_fit = _fit_lme(work, feat, full_rhs, ranef_spec)
        reduced_fit = _fit_lme(work, feat, reduced_rhs, ranef_spec)

        # Likelihood-ratio test. Both models must be fit with ML (not REML)
        # for the LRT to be comparable. _fit_lme uses reml=False.
        chi2 = 2.0 * (full_fit.llf - reduced_fit.llf)
        df = n_batch - 1
        # Clip to 0 to guard against tiny negative values from non-monotone
        # optimizer paths.
        chi2 = max(chi2, 0.0)
        p_value = float(scipy_stats.chi2.sf(chi2, df))
        rows.append(
            {"feature": feat, "chi2": float(chi2), "df": int(df), "p_value": p_value}
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        "chi2", ascending=False, kind="stable"
    ).reset_index(drop=True)


def mult_test(
    data: pd.DataFrame,
    batch_col: str,
    id_col: str,
    features: Sequence[str] | Sequence[int],
    formula: str,
    ranef: str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Test for multiplicative batch effects via Fligner-Killeen per feature.

    Fits an LME with ``batch_col`` as a fixed effect, takes the conditional
    residuals, and applies ``scipy.stats.fligner`` across batches. This
    matches ``longCombat::multTest``.

    Returns
    -------
    pandas.DataFrame
        Columns ``['feature', 'chi2', 'df', 'p_value']``, sorted by
        ``chi2`` descending. Column ``df`` is ``n_batch - 1`` — the degrees
        of freedom of the Fligner-Killeen chi-squared statistic.
    """
    feature_names = _resolve_feature_names(data, features)
    if batch_col not in data.columns:
        raise KeyError(f"batch_col={batch_col!r} not found in data.columns")

    work = data.copy()
    uniques = pd.unique(work[batch_col])
    try:
        levels = sorted(uniques.tolist())
    except TypeError:
        levels = sorted(uniques.tolist(), key=repr)
    work[batch_col] = pd.Categorical(work[batch_col], categories=levels)
    n_batch = len(levels)
    if verbose:
        print(f"[mult_test] found {n_batch} batches")
        print(f"[mult_test] found {len(feature_names)} features")

    ranef_spec = parse_ranef(ranef)
    _validate_no_missing_data(
        work, [batch_col, ranef_spec.groups_col, *feature_names]
    )

    batch_codes = np.asarray(work[batch_col].cat.codes)
    groups = [np.where(batch_codes == i)[0] for i in range(n_batch)]

    rows = []
    for v, feat in enumerate(feature_names, start=1):
        if verbose:
            print(
                f"[mult_test] testing for multiplicative batch effect for feature {v} ({feat})"
            )
        rhs = f"{formula} + C(Q('{batch_col}'), Treatment)"
        fit = _fit_lme(work, feat, rhs, ranef_spec)
        resid = np.asarray(fit.resid, dtype=float)
        chi2, p = scipy_stats.fligner(*(resid[g] for g in groups))
        rows.append(
            {
                "feature": feat,
                "chi2": float(chi2),
                "df": int(n_batch - 1),
                "p_value": float(p),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        "chi2", ascending=False, kind="stable"
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------


def _resolve_feature_names(
    data: pd.DataFrame, features: Sequence[str] | Sequence[int]
) -> list[str]:
    if isinstance(features, (str, bytes)) or not hasattr(features, "__len__"):
        raise ValueError("features must be a non-empty sequence of column names or indices")
    features = list(features)
    if len(features) == 0:
        raise ValueError("features must be a non-empty sequence of column names or indices")
    first = features[0]
    if isinstance(first, (int, np.integer)) and not isinstance(first, bool):
        names = [str(data.columns[int(i)]) for i in features]
    else:
        names = [str(f) for f in features]
    missing = [f for f in names if f not in data.columns]
    if missing:
        raise KeyError(f"feature column(s) not found in data: {missing}")
    return names


def _fit_lme(work: pd.DataFrame, feature: str, rhs: str, ranef_spec):
    """Fit MixedLM with ML (not REML) for LRT comparability."""
    full_formula = f"Q('{feature}') ~ {rhs}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm(
            full_formula,
            data=work,
            groups=work[ranef_spec.groups_col],
            re_formula=ranef_spec.re_formula,
        )
        for method in (["lbfgs"], ["bfgs"], ["powell"]):
            try:
                return md.fit(reml=False, method=method)
            except (np.linalg.LinAlgError, ValueError):
                continue
    raise RuntimeError(
        f"All optimizers (lbfgs, bfgs, powell) failed to fit MixedLM for "
        f"feature {feature!r}."
    )
