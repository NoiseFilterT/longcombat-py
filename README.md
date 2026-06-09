# longcombat-py

Python port of [jcbeer/longCombat](https://github.com/jcbeer/longCombat).
The API mirrors [`neuroCombat`](https://github.com/Jfortin1/neuroCombat) where
concepts overlap so the two can be used side-by-side.

Longitudinal ComBat uses an empirical Bayes method to harmonize the means and
variances of the residuals across batches in a linear mixed-effects model
framework. See:

> Beer JC, Tustison NJ, Cook PA, Davatzikos C, Sheline YI, Shinohara RT,
> Linn KA (2020). *Longitudinal ComBat: A method for harmonizing
> longitudinal multi-scanner imaging data.* NeuroImage, 220:117129.
> <https://doi.org/10.1016/j.neuroimage.2020.117129>

> **Deviations from the R original** — this port is *not* bit-identical to
> the R package. Please read
> [`DIFFERENCES_FROM_R.md`](https://github.com/NoiseFilterT/longcombat-py/blob/main/DIFFERENCES_FROM_R.md)
> before using, especially if comparing results to the R output.

## Install

```bash
pip install longcombat-py            # core
pip install "longcombat-py[plot]"    # + matplotlib for visualization helpers
```

## Quick start

Long-format `pandas.DataFrame` with one row per (subject, visit):

```python
import pandas as pd
from longcombat import long_combat

result = long_combat(
    data=df,
    batch_col="scanner",
    id_col="subid",
    time_col="visit",
    features=["feature1", "feature2", "feature3"],
    formula="age + diagnosis*visit",     # Patsy RHS, fixed effects
    ranef="(1|subid)",                   # lme4-style random effects
)
harmonized = result.data_combat
```

`long_combat` returns a `LongCombatResult` with:

- `.data_combat` — harmonized DataFrame
  (`[id_col, time_col, batch_col, feature1.combat, ...]`)
- `.gammahat`, `.delta2hat` — method-of-moments estimates of additive and
  multiplicative batch effects on the standardized residuals
- `.gammastarhat`, `.delta2starhat` — empirical-Bayes shrunk estimates

## API surface

| Function | R analog |
|---|---|
| `long_combat` | `longCombat` |
| `add_test` | `addTest` (LRT only — see `DIFFERENCES_FROM_R.md`) |
| `mult_test` | `multTest` |
| `batch_time_viz` | `batchTimeViz` |
| `batch_boxplot` | `batchBoxplot` |
| `traj_plot` | `trajPlot` |

Parameter naming mirrors `neuroCombat` where the concept exists
(`batch_col`, `eb`, `mean_only`). Additional concepts from `longCombat`
(`id_col`, `time_col`, `ranef`, `formula`, `niter`, `method`) preserve the
R names.

## License

Artistic License 2.0. See
[`LICENSE`](https://github.com/NoiseFilterT/longcombat-py/blob/main/LICENSE) and
[`NOTICE`](https://github.com/NoiseFilterT/longcombat-py/blob/main/NOTICE).
