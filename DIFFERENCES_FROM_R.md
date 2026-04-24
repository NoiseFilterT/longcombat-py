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

Both versions error on missing data in the columns used by the model. The
Python port matches this behavior and uses a `ValueError` rather than
R's `stop()`.

## 7. Single-feature harmonization

The R package silently returns ``NA`` for harmonized values when called
with a single feature, because the empirical-Bayes hyperpriors τ̄², D̄,
S̄² are estimated from the distribution of per-batch moments *across
features* (i.e. cross-feature variance), which is undefined for V=1.

The Python port instead raises a clear ``ValueError`` in this case and
points users to ``eb=False``, which harmonizes each feature using its own
method-of-moments estimates without the cross-feature EB shrinkage and
therefore works fine for V=1.

## 8. Factor levels

R's `as.factor` orders levels alphabetically (or by locale). The Python
port sorts batch levels using `pandas`' stable sort — for plain strings and
numbers this matches R's alphabetical ordering, which places the same batch
as the "reference level" dropped by the design matrix.
