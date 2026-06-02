# Evaluation Report — `longcombat-py` (R → Python port of `longCombat`)

**Date:** 2026-06-02
**Scope:** Independent assessment of how faithfully and correctly the Python
package `longcombat-py` (v0.1.0) reproduces the R package
[`jcbeer/longCombat`](https://github.com/jcbeer/longCombat) (Beer et al. 2020,
*NeuroImage*).
**Method:** Fresh, independent verification — datasets generated in R, run
through *both* the real R `longCombat` package (installed in
`tests/fixtures/r_lib/`) and the Python port on identical input rows, then
compared. Augmented by a 40-agent line-by-line fidelity review with adversarial
verification.

---

## 1. Verdict

**This is a high-quality, faithful, and unusually honest port.** The core
harmonization algorithm reproduces the R package to **~1×10⁻⁶** on every
intermediate and final quantity for the standard random-intercept model — i.e.
numerically indistinguishable from R for the use case the method was designed
for. The two diagnostic tests are correct (one is bit-for-bit identical to R).
The documentation is measured and accurate rather than aspirational: every
error-magnitude claimed in `DIFFERENCES_FROM_R.md` reproduces *exactly*.

The defects found are **edge cases**, not core-algorithm errors:
3 genuine bugs (a plotting mis-mapping, a silent-NaN path, an R↔Python
index-base mismatch) plus a cluster of robustness/usability gaps and one
documentation inaccuracy. None of them affect a default random-intercept
harmonization of clean, name-indexed data — the package's primary path.

| Area | Grade | One-line basis |
|---|---|---|
| Core algorithm fidelity | **A** | ~1e-6 vs R across 6 scenarios; sum-to-L constraint exact |
| Statistical tests | **A−** | `mult_test` == R to 1e-14; `add_test` ranking == R (documented test swap) |
| Documentation honesty | **A** | error table reproduces exactly; deviations disclosed |
| Robustness / edge cases | **B−** | silent-NaN path, index-base trap, opaque errors on NaN/bad names |
| Plotting | **B** | one real marker-scrambling bug; rest faithful |
| Packaging | **A** | example runs, deps correct, license consistent, types ship |

---

## 2. What I did (methodology)

R was available locally, so this is a *real* head-to-head, not a re-check of
the committed fixtures:

1. **`verify/gen_and_run_r.R`** generates six fresh longitudinal datasets in R
   (varying random-effects structure, batch count, formula, size, and the
   `REML`/`MSR` switch), runs the genuine `longCombat`, `addTest`, and
   `multTest` on each, and writes inputs + outputs to `verify/out/`.
2. **`verify/run_py_compare.py`** reads the *same input rows*, runs the Python
   port, and reports max abs/rel error for `gammahat`, `delta2hat`,
   `gammastarhat`, `delta2starhat`, and `data_combat`, plus test comparisons.
3. Targeted probes confirmed the load-bearing assumptions (e.g. that
   statsmodels `fittedvalues` includes the random-effect BLUPs) and isolated
   the one scenario that diverges.

Generating data **in R** and feeding the saved CSV to both tools is essential:
NumPy's and R's RNGs differ, so rebuilding "the same" dataset in each language
would compare different data. This harness avoids that.

The existing test suite (51 tests) passes in full.

---

## 3. Numerical fidelity — the headline result

Max **absolute** error, Python vs. genuine R `longCombat`, identical input rows:

| Scenario | RE structure | method | gammahat | delta2hat | gammastar | delta2star | data_combat |
|---|---|---|---|---|---|---|---|
| baseline | `(1\|subid)` | REML | 1.5e-6 | 1.2e-6 | 1.1e-6 | 8.9e-7 | **2.9e-6** |
| MSR | `(1\|subid)` | MSR | 8.6e-7 | 2.4e-6 | 8.6e-7 | 9.2e-7 | **5.9e-6** |
| 6 batches | `(1\|subid)` | REML | 1.7e-6 | 1.6e-6 | 1.2e-6 | 1.1e-6 | **9.6e-6** |
| simple formula | `(1\|subid)` | REML | 3.8e-6 | 5.1e-6 | 3.8e-6 | 3.3e-6 | **9.4e-6** |
| large (150×6, 8 feat) | `(1\|subid)` | REML | 5.1e-6 | 6.5e-6 | 3.3e-6 | 3.0e-6 | **4.3e-5** |
| **random slope** | `(1 + time\|subid)` | REML | 1.6e-2 | 2.4e-2 | 1.1e-2 | 1.6e-2 | **8.7e-2** |

**Reading this:** for random-intercept models — the standard longitudinal-ComBat
setup and what every package example uses — agreement is at the **1e-6** floor.
That is not "close," it is "the same answer up to optimizer round-off." The
`MSR` path, the multi-batch path, and the recovered constraint all match.

Supporting checks, all passing:
- **Sum-to-L constraint** `Σ nᵢ·γᵢ = 0` holds to **1.8e-14** (machine precision).
- **`fittedvalues` semantics**: statsmodels' fitted values include the BLUP
  random effects (`max|fitted − Xβ| = 2.31`, large), matching R's `fitted()`;
  the REML residual SD matches lme4 to 6 digits (1.49093 vs 1.490933).
- **Determinism**: repeated runs are bit-identical.
- **Effectiveness** (end-to-end): after harmonization, additive batch effects
  collapse from χ² ≈ 72–118 (p ~ 1e-16…1e-26) to χ² < 1 (p > 0.6), and the one
  real multiplicative effect drops from p = 7.5e-4 to p = 0.16. The method does
  what it claims.
- **`DIFFERENCES_FROM_R.md` error table** reproduces *exactly* against the
  committed seed-1 fixture (gammahat 1.6e-5, delta2hat 5.8e-6, gammastarhat
  8.0e-6, delta2starhat 3.7e-6, data_combat 4.5e-5).

### 3.1 The one real numerical caveat: random slopes

The random-slope scenario (`(1 + time|subid)`) diverges by up to **0.087** on
harmonized values — ~4 orders of magnitude looser than the random-intercept
floor. I traced it precisely:

- It is **feature-specific**, concentrated in features whose slope variance is
  small / weakly identified (one feature was flagged singular by lme4).
- For a well-conditioned feature in the *same* model, the port's random-effects
  covariance matched lme4 to 4 digits and reached the *same* REML optimum
  (verified by evaluating statsmodels' own objective at lme4's variance
  components: identical to 5 decimals).
- Where the slope variance is near the boundary, the REML surface is nearly
  flat along that direction, and lme4 (BOBYQA on the Cholesky factor) vs.
  statsmodels (lbfgs→bfgs→powell on a different parameterization) settle on
  different-but-near-equally-likely covariance estimates. Those small
  differences propagate through the BLUPs into the harmonized output.

This is **not a porting bug** — it is the inherent `lme4`-vs-`statsmodels`
optimizer difference that `DIFFERENCES_FROM_R.md §1` already warns about
("agreement may be looser if the random-effects structure is near the
variance-component boundary"). This report simply **quantifies** it: expect
1e-6 for random intercepts, but degraded agreement (up to ~1e-1 absolute, a few
percent) for random-slope models with weak slope variance. Worth promoting from
a footnote to a headline caveat, since random-slope users are exactly who would
be surprised.

---

## 4. Statistical tests

| Test | R method | Python method | Result |
|---|---|---|---|
| `mult_test` | `fligner.test(resid ~ batch)` | `scipy.stats.fligner` | **χ² match to 3e-14, p to 1e-16 — effectively identical** |
| `add_test` | Kenward–Roger F (`pbkrtest`) | likelihood-ratio χ² | Different statistic (documented); **feature ranking identical to R** |

- **`mult_test` is a textbook-faithful port.** That the residuals match R to
  machine precision is also strong independent evidence that the statsmodels
  conditional residuals equal lme4's `residuals()`.
- **`add_test`** correctly fits *both* models with ML (`reml=False`) — required
  for a valid fixed-effects LRT — uses `df = n_batch−1`, and computes
  `χ² = 2(llf_full − llf_reduced)`. Kenward–Roger has no Python implementation,
  so the LRT substitution is reasonable and disclosed. On the test data the LRT
  produced the *same feature ranking* as R's KR, which is what matters for a
  diagnostic. (LRT is known to be anti-conservative vs KR — documented.)

---

## 5. Findings (bugs & gaps)

Severities below reflect adversarial re-verification; all were reproduced
against the actual source and, where relevant, the live R package. None affect a
default random-intercept harmonization of clean, name-indexed data.

### High

**H1 — `traj_plot` scrambles point markers/colors when a subject's rows are not
time-sorted.** *(confirmed by reproduction)*
`_plots.py:312–317` zips `orig_idx` (original row order) against `rows`
(sorted by time), so per-observation markers/colors (including the default
batch-derived glyphs) are attached to the wrong points. Demonstrated: with input
rows out of time order, every point received the wrong color
(time0→red instead of green, etc.). Silent — produces a misleading plot.
Trivial fix: index the marker/color arrays by the time-sorted positions.

**H2 — Integer `features` indices are 0-based (Python) but 1-based in R —
undocumented.** *(confirmed in both languages)*
`features=[5,6,7]` selects `feat1,feat2,feat3` in Python but `batch,feat1,feat2`
in R (`names(data)[c(5,6,7)]`). A user porting R's `features=6:8` to
`features=[6,7,8]` either crashes with `IndexError` or, worse, silently
harmonizes the wrong, off-by-one columns. The validator can't catch it because
the shifted indices still resolve to real columns. Most users pass names (safe),
but the integer path is a documented input mode with a silent-wrong-result trap.
Fix: document the 0-based convention prominently, or accept 1-based to mirror R.

**H3 — `mean_only=True` with a single feature silently returns all-NaN.**
*(confirmed)*
The guard that turns single-feature EB into a clear `ValueError`
(`_core.py:183`) is conditioned `and not mean_only`, so the `mean_only` branch
falls through to `gammahat.var(axis=1, ddof=1)` — a 1-element variance = NaN —
and emits all-NaN `data_combat` with only a swallowed `RuntimeWarning`. The
error message the guard *would* have shown recommends `eb=False`, but `eb` is
ignored under `mean_only` (see L-cluster), so that workaround also yields NaN.
Fix: extend the `V<2` guard to the `mean_only` path.

### Medium

**M1 — `add_test`/`mult_test`/`long_combat` accept non-converged optimizer
fits.** The `lbfgs→bfgs→powell` fallback catches *exceptions* but never checks
`fit.converged`. A silently non-converged ML fit corrupts the `add_test` χ²
(and, in principle, the harmonization). It did not trigger on my six scenarios
(add_test matched R), but it is a real robustness gap for harder fits. Fix:
reject/loudly warn on non-convergence before using a fit.

**M2 — `add_test`/`mult_test` crash with an opaque `IndexError` on NaN in a
feature column.** Unlike `long_combat`, the test helpers don't validate missing
data; statsmodels drops NA rows from `endog/exog` but keeps the full-length
`groups`, raising `index N out of bounds` with no hint about NaN. Fix: reuse
`_validate_no_missing_data` in the test helpers.

**M3 — A `batch_col` (or covariate) name containing a dot or space crashes with
an opaque Patsy/SyntaxError.** `_core.py:393` interpolates `batch_col` raw into
`C({batch_col}, Treatment)` without `Q()` quoting (only the feature is quoted).
`scanner.site` → `NameError: name 'scanner' is not defined`; `scanner site` →
`SyntaxError`. These are legal R column names, so dotted-name data that works in
R crashes here. (Batch *values* with dots/spaces are fine.) Fix: `Q()`-quote
`batch_col`.

### Low (selected)

- **L1 — `DIFFERENCES_FROM_R.md §6 is inaccurate.** It says "both versions error
  on missing data in the columns used by the model." In fact R errors on NaN in
  **any** column (whole-frame `sum(is.na(data))`), even unused ones; Python only
  checks model columns. Confirmed live: an unused all-NaN column makes R `stop()`
  while Python runs to completion. Python's narrower check is arguably *better*,
  but the doc misdescribes the deviation.
- **L2 — lme4 `||` (uncorrelated random effects) is rejected with a misleading
  "multiple random-effects blocks" message.** `(1 + time || subid)` is one
  block; the parser misdiagnoses *why* it's unsupported.
- **L3 — `eb=False` is silently ignored when `mean_only=True`** (dispatch order);
  the `eb` docstring promises raw estimates with no caveat.
- **L4 — `ranef` parser is depth-unaware**: `split('|', 1)` and single-layer
  paren-stripping can mis-handle a `|` inside an LHS function call or
  redundant outer parens. Narrow, but undocumented.
- **L5 — `features` rejects `numpy.ndarray`, `pandas.Index`, and `range`** with
  "must be a non-empty list," despite the `Sequence` type hint; forces `list()`.
- **L6 — Tie-breaking in the test result sort is not stable** (pandas default
  quicksort) vs R's stable `order()`; equal statistics may order differently.

### Positively verified (no action)

`mult_test` faithful (1e-14); correlated intercept-slope covariance for
`(1 + time|subid)` correctly reproduced; `batch_boxplot` fit/residuals/ordering
faithful to R; the exact-key batch-coefficient extraction is **more robust** than
R's substring `grep` (immune to covariate-name collisions that make R crash);
API/signatures match docs exactly; example runs clean; license metadata
internally consistent; EB-iteration "matches R exactly" claim verified against R
source; type hints (`py.typed`) shipped.

---

## 6. Engineering quality

The port is not a transliteration — it is a thoughtful re-implementation
(1,251 LoC vs 549 in R) that adds real value over a literal translation:

- **Honest, testable docs.** `DIFFERENCES_FROM_R.md` enumerates 8 deviations;
  the quantitative claims reproduce exactly. This is rare and builds trust.
- **Better failure modes than R in several places** — clear `ValueError`s for
  single-feature EB and single-observation batches (R silently returns NaN /
  is terse), and an exact-key batch extraction that avoids an R footgun.
- **Sensible neuroCombat-aligned extensions** (`eb`, `mean_only`) that the R
  package lacks, kept mathematically consistent (verified for V≥2).
- **Real test coverage** (51 tests incl. a committed R-equivalence fixture) and
  CI.
- **Pragmatic optimizer fallback** (`lbfgs→bfgs→powell`) that handles the
  singular-boundary fits lme4 absorbs silently.

The recurring theme in the defects is **input-edge handling** (single feature +
`mean_only`, integer indices, NaN in helpers, exotic names, `||` syntax) rather
than the mathematics, which is sound.

---

## 7. Recommended fixes (priority order)

1. **H1** `traj_plot`: index marker/color arrays by time-sorted position. *(1-line fix, correctness)*
2. **H3** extend the `V<2` guard to the `mean_only` path. *(silent NaN → clear error)*
3. **H2** document the 0-based integer-index convention (or switch to 1-based). *(silent wrong result)*
4. **M3** `Q()`-quote `batch_col` in all three formula builders. *(crash on legal names)*
5. **M2** validate missing data in `add_test`/`mult_test`. *(opaque error)*
6. **M1** check `fit.converged` and warn/raise. *(silent bad fits)*
7. **L1** correct `DIFFERENCES_FROM_R.md §6`; **L2** improve the `||` message;
   promote the **random-slope agreement caveat** (§3.1) from a footnote to a
   prominent note.

---

## 7a. Fixes applied (this session)

The following were fixed in the working tree and covered by 9 new regression
tests (suite now 60/60; R-vs-Python numerics unchanged):

| ID | Fix | Files |
|---|---|---|
| H1 | `traj_plot` now indexes markers/colors by time-sorted position | `_plots.py` |
| H3 | single-feature `mean_only=True` raises a clear error instead of silent NaN | `_core.py` |
| M1 | non-converged fits now emit a `UserWarning` (fires on the worst random-slope diverger) | `_core.py` |
| M2 | `add_test`/`mult_test` validate missing data → clear `ValueError` (was opaque `IndexError`) | `_tests.py` |
| M3 | `batch_col` is `Q()`-quoted in all formula builders → dotted/spaced names work | `_core.py`, `_tests.py`, `_plots.py` |
| L1 | `DIFFERENCES_FROM_R.md §6` corrected (whole-frame vs model-column NaN check) | doc |
| L2 | lme4 `||` now gets a specific "uncorrelated random effects" message | `_ranef.py` |
| L3 | `mean_only`/V≥2 interaction documented | doc §7 |
| L5 | `features` accepts ndarray / `Index` / `range`, not just list/tuple | `_core.py`, `_tests.py` |
| L6 | test result sort is stable for ties | `_tests.py` |
| §3.1 | random-slope agreement caveat quantified and promoted in `DIFFERENCES_FROM_R.md §1` | doc |

| H2 | integer `features` indices documented as 0-based (kept Pythonic, per decision) — docstrings + `DIFFERENCES_FROM_R.md §9` | `_core.py`, `_tests.py`, doc |

**Deferred:** **L4** (depth-aware `ranef` split) left as a minor follow-up.

## 8. Reproduce

```bash
# R side: generate fresh data + run genuine longCombat/addTest/multTest
R_LIBS_USER=tests/fixtures/r_lib Rscript verify/gen_and_run_r.R
# Python side: run the port on identical rows and compare
venv/bin/python verify/run_py_compare.py
# existing suite
venv/bin/python -m pytest -q
```

Artifacts: `verify/gen_and_run_r.R`, `verify/run_py_compare.py`,
`verify/out/` (inputs, R outputs, `comparison_full.txt`).

---

## 9. Bottom line

For its central purpose — **harmonizing longitudinal multi-scanner data with a
random-intercept (or richer but well-identified) mixed model** — `longcombat-py`
faithfully reproduces R `longCombat` to numerical round-off and removes batch
effects as designed. It is safe to use as a drop-in for that workflow. The
caveats are real but bounded: degraded agreement for weakly-identified
random-slope models (inherent to the optimizer swap, now quantified), and a
handful of edge-case bugs that are easy to fix and easy to avoid. The
documentation's honesty is a standout. **Recommended, with the §7 fixes and the
random-slope caveat kept front of mind.**
