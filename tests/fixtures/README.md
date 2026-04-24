# R-equivalence fixtures

This directory holds CSV fixtures used by ``tests/test_r_equivalence.py``
to verify that ``longcombat-py`` produces outputs numerically close to the
reference R package [`jcbeer/longCombat`](https://github.com/jcbeer/longCombat).

## What's here

| File | Purpose |
|---|---|
| `regenerate.R` | Regenerates the CSV fixtures using R's `longCombat` package. |
| `r_input.csv` | The synthetic dataset R used (saved because Python's and R's RNGs differ — see note below). |
| `r_data_combat.csv` | R's harmonized output for `r_input.csv`. |
| `r_gammahat.csv`, `r_delta2hat.csv` | R's method-of-moments per-batch moment estimates. |
| `r_gammastarhat.csv`, `r_delta2starhat.csv` | R's EB-shrunk per-batch effect estimates. |

The CSVs are small (~32 KB total) and *are* checked into version control
so `tests/test_r_equivalence.py` runs on any machine without needing R
installed. Regenerate them only when the upstream R package changes or
you want to verify fixtures on a different platform.

If the CSVs are missing, `tests/test_r_equivalence.py` is skipped.

## How to regenerate

You need R (≥ 3.5) and the `lme4` and `longCombat` packages installed.
From the repository root:

```bash
# If you don't yet have longCombat installed, in R:
# install.packages("devtools")
# devtools::install_github("jcbeer/longCombat")

Rscript tests/fixtures/regenerate.R
```

This writes all six CSVs into `tests/fixtures/`.

## Why we save the R-generated dataset

Python's `numpy.random.default_rng(1)` and R's `set.seed(1)` produce
**different** sequences of random numbers. If we rebuilt the synthetic
dataset from seed 1 in each language, we'd be comparing `long_combat(data_python)`
to `longCombat(data_R)` — meaningless.

Instead, `regenerate.R` writes out `r_input.csv` alongside the outputs, and
the Python equivalence test reads `r_input.csv` as its input so both tools
are working on the *same rows*.

## Expected tolerance

See `DIFFERENCES_FROM_R.md` at the repository root for why bit-exact
agreement is not expected. The equivalence test uses a relative tolerance
on the order of `1e-3` for harmonized values, and wider tolerances for
variance-component-dependent quantities (`delta2hat`, `delta2starhat`) where
the `lme4` vs `statsmodels` optimizer difference is most visible.
