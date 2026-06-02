"""Longitudinal ComBat harmonization (port of ``longCombat/R/longCombat.R``)."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ._ranef import parse_ranef


@dataclass(frozen=True)
class LongCombatResult:
    """Result of :func:`long_combat`.

    Attributes
    ----------
    data_combat : pandas.DataFrame
        Harmonized data. Columns are ``[id_col, time_col, batch_col,
        <feature1>.combat, <feature2>.combat, ...]`` in that order. The
        original rows are preserved in order.
    gammahat : pandas.DataFrame
        Method-of-moments estimate of per-batch additive effects on the
        standardized residuals. Shape ``(m, V)`` indexed by batch level,
        columns are feature names.
    delta2hat : pandas.DataFrame
        Method-of-moments estimate of per-batch multiplicative effects
        (variance) on the standardized residuals. Same shape as ``gammahat``.
    gammastarhat : pandas.DataFrame
        Empirical-Bayes-shrunk estimate of additive batch effects.
    delta2starhat : pandas.DataFrame
        Empirical-Bayes-shrunk estimate of multiplicative batch effects.
    """

    data_combat: pd.DataFrame
    gammahat: pd.DataFrame
    delta2hat: pd.DataFrame
    gammastarhat: pd.DataFrame
    delta2starhat: pd.DataFrame

    def __getitem__(self, key: str):
        # Mirrors the R function's named-list return so users coming from
        # R can write ``result["data_combat"]`` as well as ``result.data_combat``.
        if key in self.keys():
            return getattr(self, key)
        raise KeyError(key)

    def keys(self) -> tuple[str, ...]:
        return (
            "data_combat",
            "gammahat",
            "delta2hat",
            "gammastarhat",
            "delta2starhat",
        )


def long_combat(
    data: pd.DataFrame,
    batch_col: str,
    id_col: str,
    time_col: str,
    features: Sequence[str] | Sequence[int],
    formula: str,
    ranef: str,
    *,
    niter: int = 30,
    method: str = "REML",
    eb: bool = True,
    mean_only: bool = False,
    verbose: bool = True,
) -> LongCombatResult:
    """Harmonize longitudinal multi-batch data using longitudinal ComBat.

    This is a Python port of ``longCombat::longCombat`` (Beer et al. 2020).
    See ``DIFFERENCES_FROM_R.md`` in the source distribution for deviations
    from the R implementation.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format data frame. One row per (subject, timepoint).
    batch_col : str
        Name of the batch variable column. Any hashable dtype is accepted;
        levels are sorted alphabetically (matching R's ``as.factor`` default).
    id_col : str
        Name of the subject-identifier column. Passed through to the
        harmonized output; not itself used in the linear mixed model unless
        it appears in ``ranef``.
    time_col : str
        Name of the numeric within-subject time/visit column.
    features : sequence of str or sequence of int
        Feature column names, or integer column indices. Each feature is
        harmonized independently after a per-feature LME fit. **Integer
        indices are 0-based** (Python convention), unlike R ``longCombat``
        where they are 1-based: the R call ``features = 6:8`` corresponds to
        ``features=[5, 6, 7]`` here. Passing column names avoids the
        ambiguity. See ``DIFFERENCES_FROM_R.md`` §9.
    formula : str
        Patsy-style right-hand side describing the fixed effects, e.g.
        ``"age + diagnosis*time"``. Must **not** include ``batch_col`` or
        the random-effects term.
    ranef : str
        Random-effects specification in ``lme4`` notation, e.g.
        ``"(1|subid)"`` or ``"(1 + time|subid)"``. A single grouping factor
        only.
    niter : int, default 30
        Number of empirical-Bayes fixed-point iterations. Ignored when
        ``eb=False`` or ``mean_only=True``.
    method : {'REML', 'MSR'}, default 'REML'
        Method for the residual standard deviation used in standardization.
        ``'REML'`` uses the residual SD from the REML fit (conservative
        type-I error); ``'MSR'`` uses the empirical standard deviation of
        the conditional residuals divided by N (more powerful, less
        conservative). Matches the R package's ``method`` argument.
    eb : bool, default True
        If ``True``, apply empirical-Bayes shrinkage of the batch-effect
        estimates. If ``False``, use the raw method-of-moments estimates.
        Mirrors ``neuroCombat``'s ``eb`` argument. The R package does not
        expose this toggle; it is always ``True`` in R.
    mean_only : bool, default False
        If ``True``, only harmonize the means; multiplicative batch effects
        are fixed at 1 (no variance scaling). Mirrors ``neuroCombat``.
    verbose : bool, default True
        Print progress messages with the ``[long_combat]`` prefix.

    Returns
    -------
    LongCombatResult
        See :class:`LongCombatResult`.

    Raises
    ------
    ValueError
        If the relevant input columns contain missing values, if any batch
        has fewer than 2 observations, if ``method`` is unknown, or if
        ``features`` is empty.
    KeyError
        If a named column is not present in ``data``.

    Notes
    -----
    The algorithm exactly mirrors the R implementation: for each feature a
    mixed-effects model is fit with ``batch_col`` as a fixed effect, the
    full ``m``-vector of additive batch effects is recovered under the
    sum-to-L constraint ``Σ n_i γ_i = 0``, the residuals are standardized
    by the residual SD, and empirical Bayes with inverse-gamma priors
    (parameters fit by method of moments) shrinks the per-batch mean and
    variance estimates. The EB iteration is Jacobi-style with a fixed
    ``niter`` (no convergence check), matching the R code.
    """
    _validate_inputs(data, batch_col, id_col, time_col, method)

    feature_names = _resolve_feature_names(data, features)
    _validate_no_missing_data(data, [batch_col, id_col, time_col, *feature_names])

    work = data.copy()
    batch_levels, batch_codes, batch_row_indices, ni = _prepare_batch(
        work, batch_col
    )
    m = len(batch_levels)
    V = len(feature_names)
    L = len(work)

    if m < 2:
        raise ValueError(
            f"long_combat needs at least 2 batches; found {m} in {batch_col!r}"
        )
    if min(ni) < 2:
        single = [
            str(lvl) for lvl, n in zip(batch_levels, ni, strict=True) if n < 2
        ]
        raise ValueError(
            "The following batches have fewer than 2 observations: "
            f"{', '.join(single)}. long_combat needs at least 2 observations "
            "per batch to harmonize variance across batches. Remove these "
            "rows or merge the batches before running long_combat."
        )

    # Empirical-Bayes hyperpriors are estimated from the distribution of
    # per-batch moments *across features*. With V=1 that cross-feature
    # variance is undefined (R's `var()` returns NA), producing silently
    # NaN output. Fail loudly instead and point at the workaround.
    if V < 2 and (eb or mean_only):
        raise ValueError(
            "Empirical-Bayes longitudinal ComBat requires at least 2 "
            f"features (got {V}). The hyperpriors τ̄², D̄, and S̄² are "
            "estimated from the distribution of per-batch moments across "
            "features, which is undefined for V<2. Pass eb=False (with "
            "mean_only=False) to skip empirical-Bayes shrinkage for "
            "single-feature harmonization; mean_only also requires V≥2 "
            "because its additive shrinkage uses the same cross-feature "
            "hyperprior τ̄²."
        )

    if verbose:
        print(f"[long_combat] found {m} batches")
        print(f"[long_combat] found {V} features")
        print(f"[long_combat] found {L} total observations")

    ranef_spec = parse_ranef(ranef)

    # --- standardize data across features ---------------------------------
    if verbose:
        print("[long_combat] standardizing data across features...")

    sigma_estimates = np.empty(V, dtype=float)
    predicted = np.empty((L, V), dtype=float)
    batch_effects = np.empty((m - 1, V), dtype=float)

    for v, feat in enumerate(feature_names):
        if verbose:
            print(f"[long_combat] fitting lme model for feature {v + 1} ({feat})")
        fit = _fit_mixedlm(
            work, feat, formula, batch_col, ranef_spec,
        )
        sigma_estimates[v] = _residual_sd(fit, method)
        batch_effects[:, v] = _extract_batch_effects(
            fit, batch_col, batch_levels,
        )
        predicted[:, v] = np.asarray(fit.fittedvalues, dtype=float)

    # Broadcast sigmas to an (L, V) matrix.
    sigmas = np.broadcast_to(sigma_estimates, (L, V)).copy()

    # Recover full m-vector of batch effects under the sum-to-L constraint
    # Σ n_i γ_i = 0. lme4 (and statsmodels with treatment coding) drops the
    # first level; we solve for it:
    #   γ_1 = −(Σ_{i≥2} n_i · γ_i) / L
    gamma1hat = -(ni[1:] @ batch_effects) / L  # shape (V,)
    batch_effects_adjusted = np.vstack(
        [gamma1hat, batch_effects + gamma1hat]
    )  # (m, V)

    # Broadcast per-batch effects back to the full (L, V) matrix.
    batch_effects_expanded = batch_effects_adjusted[batch_codes, :]

    feat_matrix = work[feature_names].to_numpy(dtype=float, copy=True)
    data_std = (feat_matrix - predicted + batch_effects_expanded) / sigmas

    # --- method-of-moments hyperparameters --------------------------------
    if verbose:
        print("[long_combat] using method of moments to estimate hyperparameters")
    gammahat = np.empty((m, V), dtype=float)
    delta2hat = np.empty((m, V), dtype=float)
    for i, idx in enumerate(batch_row_indices):
        block = data_std[idx, :]
        gammahat[i, :] = block.mean(axis=0)
        # Unbiased (n-1) variance to match R's var() default.
        delta2hat[i, :] = block.var(axis=0, ddof=1)

    # --- empirical Bayes estimates ----------------------------------------
    if mean_only:
        # δ² fixed at 1; γ update is one-shot because δ² never changes.
        # γ*[i,v] = (n_i · τ² · γ̂ + 1 · γ̄) / (n_i · τ² + 1)
        gammabar = gammahat.mean(axis=1)
        tau2bar = gammahat.var(axis=1, ddof=1)
        gammastarhat_final = (
            (ni[:, None] * tau2bar[:, None] * gammahat + gammabar[:, None])
            / (ni[:, None] * tau2bar[:, None] + 1.0)
        )
        delta2starhat_final = np.ones((m, V), dtype=float)
    elif not eb:
        gammastarhat_final = gammahat.copy()
        delta2starhat_final = delta2hat.copy()
    else:
        gammastarhat_final, delta2starhat_final = _eb_iterate(
            data_std=data_std,
            batch_row_indices=batch_row_indices,
            ni=ni,
            gammahat=gammahat,
            delta2hat=delta2hat,
            niter=niter,
            verbose=verbose,
        )

    # --- adjust data for batch effects ------------------------------------
    if verbose:
        print("[long_combat] adjusting data for batch effects")
    gammastar_expanded = gammastarhat_final[batch_codes, :]
    delta2star_expanded = delta2starhat_final[batch_codes, :]

    data_combat_arr = (
        (sigmas / np.sqrt(delta2star_expanded))
        * (data_std - gammastar_expanded)
        + predicted
        - batch_effects_expanded
    )

    # --- package results --------------------------------------------------
    combat_feature_names = [f"{f}.combat" for f in feature_names]
    data_combat = pd.DataFrame(
        data_combat_arr,
        index=work.index,
        columns=combat_feature_names,
    )
    data_combat.insert(0, batch_col, data[batch_col].to_numpy())
    data_combat.insert(0, time_col, data[time_col].to_numpy())
    data_combat.insert(0, id_col, data[id_col].to_numpy())

    batch_index = pd.Index(batch_levels, name=batch_col)
    gammahat_df = pd.DataFrame(gammahat, index=batch_index, columns=feature_names)
    delta2hat_df = pd.DataFrame(delta2hat, index=batch_index, columns=feature_names)
    gammastar_df = pd.DataFrame(
        gammastarhat_final, index=batch_index, columns=feature_names
    )
    delta2star_df = pd.DataFrame(
        delta2starhat_final, index=batch_index, columns=feature_names
    )

    return LongCombatResult(
        data_combat=data_combat,
        gammahat=gammahat_df,
        delta2hat=delta2hat_df,
        gammastarhat=gammastar_df,
        delta2starhat=delta2star_df,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _validate_inputs(
    data: pd.DataFrame,
    batch_col: str,
    id_col: str,
    time_col: str,
    method: str,
) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    for col, role in ((batch_col, "batch_col"), (id_col, "id_col"), (time_col, "time_col")):
        if col not in data.columns:
            raise KeyError(f"{role}={col!r} not found in data.columns")
    if method not in ("REML", "MSR"):
        raise ValueError(f"method must be 'REML' or 'MSR', got {method!r}")


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


def _validate_no_missing_data(data: pd.DataFrame, cols: Iterable[str]) -> None:
    na_cols = [c for c in cols if data[c].isna().any()]
    if na_cols:
        raise ValueError(
            "Missing data in variables: "
            + ", ".join(na_cols)
            + "\n\nBefore running long_combat either impute the missing values, "
            "remove those rows, or remove columns with missing values if that "
            "variable is not in the model."
        )


def _prepare_batch(
    work: pd.DataFrame, batch_col: str
) -> tuple[list, np.ndarray, list[np.ndarray], np.ndarray]:
    """Sort batch levels, assign integer codes, and index rows per batch."""
    raw = work[batch_col]
    # Sort levels deterministically. For mixed types (rare), fall back to string.
    uniques = pd.unique(raw)
    try:
        levels = sorted(uniques.tolist())
    except TypeError:
        levels = sorted(uniques.tolist(), key=repr)
    cat = pd.Categorical(raw, categories=levels, ordered=False)
    codes = np.asarray(cat.codes)
    # Re-assign the column as categorical so that Patsy treatment-encodes it
    # with the deterministic level order we just established.
    work[batch_col] = cat
    row_indices = [np.where(codes == i)[0] for i in range(len(levels))]
    ni = np.array([len(idx) for idx in row_indices], dtype=float)
    return levels, codes, row_indices, ni


def _fit_mixedlm(
    work: pd.DataFrame,
    feature: str,
    formula: str,
    batch_col: str,
    ranef_spec,
):
    # Q()-quote batch_col so column names containing dots or spaces (legal in
    # R, common in neuroimaging tables) don't break the Patsy formula.
    rhs = f"{formula} + C(Q('{batch_col}'), Treatment)"
    full_formula = f"Q('{feature}') ~ {rhs}"
    fit = None
    with warnings.catch_warnings():
        # statsmodels emits warnings (ConvergenceWarning, "random-effects
        # covariance is singular") whenever the random-effects variance
        # collapses to 0 — an uninteresting boundary case that mirrors
        # behavior R tolerates silently.
        warnings.simplefilter("ignore")
        md = smf.mixedlm(
            full_formula,
            data=work,
            groups=work[ranef_spec.groups_col],
            re_formula=ranef_spec.re_formula,
        )
        # statsmodels' lbfgs can fail with a singular matrix at the
        # boundary (zero random-effects variance); fall back to bfgs then
        # powell, matching the spirit of lme4's "try another optimizer"
        # behavior.
        for method in (["lbfgs"], ["bfgs"], ["powell"]):
            try:
                fit = md.fit(reml=True, method=method)
                break
            except (np.linalg.LinAlgError, ValueError):
                continue
    if fit is None:
        raise RuntimeError(
            f"All optimizers (lbfgs, bfgs, powell) failed to fit MixedLM for "
            f"feature {feature!r}."
        )
    _warn_if_unreliable_fit(fit, feature, ranef_spec)
    return fit


def _warn_if_unreliable_fit(fit, feature: str, ranef_spec) -> None:
    """Warn when a mixed-model fit failed to converge.

    Near the random-effects-variance boundary the REML surface is nearly flat,
    the optimum is weakly identified, and ``statsmodels`` and ``lme4`` can
    settle on materially different solutions — so the harmonized values may
    differ from the R ``longCombat`` result by a few percent. This is the
    regime ``DIFFERENCES_FROM_R.md`` §1 warns about; surface it instead of
    using a bad fit silently.
    """
    if not getattr(fit, "converged", True):
        warnings.warn(
            f"MixedLM fit for feature {feature!r} (ranef "
            f"{ranef_spec.re_formula!r}) did not converge. Harmonized values "
            "for this feature may be unreliable and may differ noticeably "
            "from the R longCombat result. Consider a simpler random-effects "
            "structure such as a random intercept '(1|id)'.",
            stacklevel=3,
        )


def _residual_sd(fit, method: str) -> float:
    if method == "REML":
        # statsmodels MixedLM.scale is the residual variance estimate from
        # the (RE)ML fit, analogous to lme4's VarCorr 'Residual' sdcor squared.
        return float(np.sqrt(fit.scale))
    # MSR: empirical SD of the conditional residuals, dividing by N (not N-1)
    # to match the R package line-for-line.
    r = np.asarray(fit.resid, dtype=float)
    return float(np.sqrt(np.sum((r - r.mean()) ** 2) / len(r)))


def _extract_batch_effects(
    fit, batch_col: str, batch_levels: Sequence
) -> np.ndarray:
    """Return the (m-1,) vector of treatment-coded batch effect coefficients.

    The reference level (``batch_levels[0]``) is dropped by ``C(..., Treatment)``,
    so only levels 2..m appear in ``fe_params``. We look them up by name to
    avoid relying on column ordering, and return them in the same order as
    ``batch_levels[1:]``.
    """
    fe = fit.fe_params
    coefs = np.empty(len(batch_levels) - 1, dtype=float)
    for j, level in enumerate(batch_levels[1:]):
        key = f"C(Q('{batch_col}'), Treatment)[T.{level}]"
        if key not in fe.index:
            raise RuntimeError(
                f"Could not find batch coefficient {key!r} in model fit. "
                f"Available fixed-effect names: {list(fe.index)}"
            )
        coefs[j] = float(fe[key])
    return coefs


def _eb_iterate(
    *,
    data_std: np.ndarray,
    batch_row_indices: list[np.ndarray],
    ni: np.ndarray,
    gammahat: np.ndarray,
    delta2hat: np.ndarray,
    niter: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Jacobi-style EB iteration mirroring ``longCombat.R`` lines 130–163.

    Both ``γ*`` and ``δ²*`` are updated from the *previous* iteration's
    values of the other variable. No convergence check — runs exactly
    ``niter`` iterations.
    """
    m, V = gammahat.shape
    ni_col = ni[:, None]  # (m, 1)

    # Hyperpriors.
    gammabar = gammahat.mean(axis=1)
    tau2bar = gammahat.var(axis=1, ddof=1)
    Dbar = delta2hat.mean(axis=1)
    S2bar = delta2hat.var(axis=1, ddof=1)
    lambdabar = (Dbar**2 + 2.0 * S2bar) / S2bar
    thetabar = (Dbar**3 + Dbar * S2bar) / S2bar

    gammabar_col = gammabar[:, None]  # (m, 1)
    tau2bar_col = tau2bar[:, None]

    if verbose:
        print("[long_combat] using empirical Bayes to estimate batch effects...")
        print("[long_combat] initializing...")

    # Initial γ*, δ²*.
    gamma_star = (
        (ni_col * tau2bar_col * gammahat + delta2hat * gammabar_col)
        / (ni_col * tau2bar_col + delta2hat)
    )
    delta2_star = _update_delta2(
        data_std, batch_row_indices, gamma_star,
        ni_col, thetabar, lambdabar,
    )

    for b in range(1, niter + 1):
        if verbose:
            print(f"[long_combat] starting EM algorithm iteration {b}")
        # γ* update uses previous δ²*.
        gamma_new = (
            (ni_col * tau2bar_col * gammahat + delta2_star * gammabar_col)
            / (ni_col * tau2bar_col + delta2_star)
        )
        # δ²* update uses previous γ* (NOT gamma_new) — matches R exactly.
        delta2_new = _update_delta2(
            data_std, batch_row_indices, gamma_star,
            ni_col, thetabar, lambdabar,
        )
        gamma_star = gamma_new
        delta2_star = delta2_new

    return gamma_star, delta2_star


def _update_delta2(
    data_std: np.ndarray,
    batch_row_indices: list[np.ndarray],
    gamma_ref: np.ndarray,
    ni_col: np.ndarray,
    thetabar: np.ndarray,
    lambdabar: np.ndarray,
) -> np.ndarray:
    """Compute δ²* given a reference γ vector (shape (m, V))."""
    m, V = gamma_ref.shape
    out = np.empty((m, V), dtype=float)
    for i, idx in enumerate(batch_row_indices):
        block = data_std[idx, :]
        sq = ((block - gamma_ref[i, :]) ** 2).sum(axis=0)
        out[i, :] = (thetabar[i] + 0.5 * sq) / (ni_col[i, 0] / 2.0 + lambdabar[i] - 1.0)
    return out
