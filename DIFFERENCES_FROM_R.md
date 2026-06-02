# Differences from the R `longCombat` package

This document records every intentional deviation between `longcombat-py` and
the upstream R package [`jcbeer/longCombat`](https://github.com/jcbeer/longCombat).
Read this before comparing Python output to R output.

## 1. Mixed-effects optimizer: `statsmodels` vs `lme4`

**What changed.** R's `lme4::lmer` is called with `optimizer='bobyqa'`. The
Python port uses `statsmodels.regression.mixed_linear_model.MixedLM` with its
default optimizer (`lbfgs`).

**Why not just use BOBYQA in Python.** BOBYQA is available in Python
(`Py-BOBYQA`, `nlopt`), but using it inside `statsmodels` would not reproduce
`lme4`'s numerics. `lme4` optimizes the profiled REML deviance as a function
of the **Cholesky factors** of the random-effects covariance; `statsmodels`
uses a different parameterization and a different profiling of the
likelihood. The two packages converge to the same minimum of the same
likelihood (to tolerance) regardless of which derivative-free/quasi-Newton
optimizer is plugged in, because the objective surfaces are shaped
differently in each tool's parameter space.

**Practical impact.** On the seed-1 fixture in `tests/fixtures/` we
observe:

| Quantity | Max abs error vs R | Max rel error vs R |
|---|---|---|
| `gammahat`      | 1.6 × 10⁻⁵ | 1.6 × 10⁻⁵ |
| `delta2hat`     | 5.8 × 10⁻⁶ | 2.1 × 10⁻⁵ |
| `gammastarhat`  | 8.0 × 10⁻⁶ | 5.1 × 10⁻⁶ |
| `delta2starhat` | 3.7 × 10⁻⁶ | 1.0 × 10⁻⁵ |
| `data_combat`   | 4.5 × 10⁻⁵ | 8.7 × 10⁻⁴ |

All intermediate quantities agree to ~1 × 10⁻⁵ relative. The final
harmonized values (`data_combat`) hit a maximum relative error around
10⁻³ only at individual values very close to zero; the absolute error
stays at 10⁻⁵. This is a much tighter floor than `statsmodels` vs
`lme4` delivers on harder problems; on your own data the agreement may
be looser if the random-effects structure is near the variance-component
boundary.

**Random-slope models specifically.** Independent head-to-head testing
(fresh data generated in R, run through both packages on identical rows)
shows the agreement depends strongly on the random-effects structure:

| Random-effects structure | Typical max abs error on `data_combat` vs R |
|---|---|
| Random intercept `(1\|id)` (REML or MSR) | ~1e-6 – 1e-5 |
| Random slope `(1 + time\|id)`, weak/near-boundary slope variance | up to ~1e-1 (a few %) |

For random-slope models with a small or weakly identified slope variance,
the REML surface is nearly flat along that direction, and `lme4` (BOBYQA
on the Cholesky factor) and `statsmodels` (lbfgs/bfgs/powell) settle on
different-but-near-equally-likely covariance estimates that propagate into
the harmonized output. As of this version `long_combat` emits a
`UserWarning` when a per-feature fit fails to converge — the clearest
signal of this regime. Prefer the simplest random-effects structure your
design justifies (a random intercept is the canonical longitudinal-ComBat
setup); if you use random slopes and need R-level agreement, validate
against R.

## 2. `add_test`: likelihood-ratio test, not Kenward–Roger

**What changed.** R's `addTest` uses `pbkrtest::KRmodcomp`, which reports an
F statistic with Kenward–Roger-adjusted denominator degrees of freedom.
There is no Kenward–Roger implementation in Python's standard scientific
stack, so `longcombat-py`'s `add_test` returns a **likelihood-ratio
chi-squared test** by default.

**Returned columns are different.**

| R `addTest` column | Python `add_test` column |
|---|---|
| `Feature` | `feature` |
| `KR F(ndf, KRddf)` | `chi2` |
| `KRddf` | `df` |
| `KR p-value` | `p_value` |

**Impact on interpretation.** The LRT is known to be **anti-conservative**
for mixed-effects models relative to Kenward–Roger, especially with small
samples or complex random-effects structures. In practice this means:

- LRT p-values tend to be **smaller** than KR p-values for the same data.
- The **ranking** of features by test statistic is usually similar, but not
  guaranteed to be identical to R's.
- Features near the significance boundary may be flagged by LRT but not by
  KR.

Since `add_test` is diagnostic (it does not feed into `long_combat`), this
deviation does not affect harmonization results — only the interpretation
of which features appear to have significant additive batch effects.

Passing `method='kr'` raises `NotImplementedError`. If faithful KR p-values
are required, use R or call `long_combat` without running `add_test` first.

## 3. License

The upstream `longCombat/DESCRIPTION` file declares `License: MIT`, but the
upstream `longCombat/LICENSE.txt` file contains the full Artistic License
2.0 text. This port follows the license text that was actually distributed
with the package (Artistic 2.0), since a `LICENSE` file containing full
license text is a stronger legal signal than a metadata field.

## 4. `ref_batch` parameter

`neuroCombat` exposes a `ref_batch` argument that fixes one batch as the
reference (no harmonization applied, all other batches shifted toward it).
The R `longCombat` function does **not** have this; it always imposes the
sum-to-L constraint `Σ n_i γ_i = 0` on the recovered additive batch effects.
This port mirrors R's behavior and does not expose `ref_batch`.

## 5. Plotting

`longcombat-py`'s plotting helpers use `matplotlib` and return/operate on
`Axes` objects. The R versions operate on base graphics state. Plot output
is visually comparable but pixel-level identity is not a goal.

## 6. Missing data

Both versions error on missing data **in the columns used by the model**,
with a `ValueError` rather than R's `stop()`. The *scope* of the check
differs, though: R's `longCombat` errors if **any** column of `data`
contains `NA` (`sum(is.na(data)) > 0`), even a column the model never uses.
The Python port only checks the columns it actually uses (`batch_col`,
`id_col`, `time_col`, and the features for `long_combat`; `batch_col`,
`id_col`, the grouping factor, and the features for `add_test`/`mult_test`).
So a frame with `NaN` in an unused column runs in Python but errors in R.
Covariates referenced only in `formula` are not pre-validated; a `NaN`
there surfaces as an error from the model fit.

## 7. Single-feature harmonization

The R package silently returns ``NA`` for harmonized values when called
with a single feature, because the empirical-Bayes hyperpriors τ̄², D̄,
S̄² are estimated from the distribution of per-batch moments *across
features* (i.e. cross-feature variance), which is undefined for V=1.

The Python port instead raises a clear ``ValueError`` in this case and
points users to ``eb=False``, which harmonizes each feature using its own
method-of-moments estimates without the cross-feature EB shrinkage and
therefore works fine for V=1.

Note that ``mean_only=True`` *also* requires V≥2: its additive shrinkage
uses the same cross-feature hyperprior τ̄², so single-feature ``mean_only``
is rejected with the same ``ValueError`` (the ``eb=False`` escape hatch
applies only when ``mean_only=False``).

## 8. Factor levels

R's `as.factor` orders levels alphabetically (or by locale). The Python
port sorts batch levels using `pandas`' stable sort — for plain strings and
numbers this matches R's alphabetical ordering, which places the same batch
as the "reference level" dropped by the design matrix.

## 9. Integer feature indices are 0-based

When `features` is given as integer column indices, the Python port uses
**0-based** indexing (`data.columns[i]`), following Python convention. The R
package uses **1-based** indexing (`names(data)[features]`). So the R call
`features = 6:8` selects the same columns as `features=[5, 6, 7]` in this
port — *not* `features=[6, 7, 8]`. Passing column **names** avoids the
ambiguity entirely and is recommended when porting R code. This applies to
`long_combat`, `add_test`, and `mult_test`.
