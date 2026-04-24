"""Python port of longCombat/longCombat_example.R.

Simulates longitudinal multi-batch data with known additive and
multiplicative batch effects, then demonstrates the full longcombat-py
workflow: visualization, testing for batch effects, harmonization, and
post-harmonization verification.

Run from the repository root after installing the package:

    python examples/longcombat_example.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from longcombat import (
    add_test,
    batch_boxplot,
    batch_time_viz,
    long_combat,
    mult_test,
    traj_plot,
)


def simulate() -> pd.DataFrame:
    """Simulate 100 subjects × 5 timepoints, 8 scanners, 20 features.

    Mirrors the structure of longCombat_example.R (same overall design,
    though the RNG streams differ between R and numpy so specific values
    will differ).
    """
    rng = np.random.default_rng(1)
    n_subj, n_visit, n_feat = 100, 5, 20
    L = n_subj * n_visit

    subid = np.repeat(np.arange(n_subj), n_visit)
    age = np.repeat(rng.integers(20, 61, n_subj), n_visit).astype(float)
    diagnosis = np.repeat([0] * 50 + [1] * 50, n_visit)
    time = np.tile(np.arange(n_visit), n_subj).astype(float)

    # Four batch patterns over five timepoints; each subject draws one.
    batch_patterns = np.array(
        [
            [1, 1, 1, 2, 2],
            [3, 3, 4, 4, 5],
            [6, 6, 6, 6, 6],
            [7, 7, 8, 8, 8],
        ]
    )
    pattern = rng.integers(0, 4, n_subj)
    batch = np.concatenate([batch_patterns[p] for p in pattern])

    # Feature matrix (L, V) with batch-specific additive + multiplicative effects.
    features = rng.normal(size=(L, n_feat))

    gamma = rng.uniform(-5, 5, 8)
    tau = rng.uniform(0.1, 0.3, 8)
    batch_add = np.column_stack(
        [rng.normal(gamma[b], tau[b], n_feat) for b in range(8)]
    )

    # Multiplicative: inverse-gamma-distributed. scipy.stats.invgamma uses
    # shape=α, scale=β parameterization (mean = β/(α−1)).
    from scipy.stats import invgamma
    lambda_ = rng.choice([2, 3], 8)
    theta = rng.choice([0.5, 1.0], 8)
    batch_mult = np.column_stack(
        [
            invgamma.rvs(a=lambda_[b], scale=theta[b], size=n_feat, random_state=rng)
            for b in range(8)
        ]
    )

    # Apply batch effects to each row.
    for i in range(L):
        b = batch[i] - 1  # 1-indexed → 0-indexed
        features[i, :] = features[i, :] * batch_mult[:, b] + batch_add[:, b]

    # Add covariate effects.
    features = (
        features
        - 0.1 * age[:, None]
        + diagnosis[:, None]
        - 0.5 * time[:, None]
        - 2.0 * (diagnosis * time)[:, None]
    )

    # Per-subject random intercept shared across features.
    subj_re = rng.normal(size=n_subj)
    features = features + subj_re[subid, None]

    feature_names = [f"feature{i + 1}" for i in range(n_feat)]
    df = pd.DataFrame(
        {"subid": subid, "age": age, "diagnosis": diagnosis, "time": time, "batch": batch}
    )
    df[feature_names] = features
    return df


def main() -> None:
    import matplotlib.pyplot as plt

    print("Simulating data...")
    sim = simulate()
    feature_names = [c for c in sim.columns if c.startswith("feature")]
    print(f"  shape={sim.shape}, batches={sorted(sim['batch'].unique())}")

    # --- visualize batches over time -------------------------------------
    batch_time_viz(sim, batch_col="batch", time_col="time", verbose=False)
    plt.suptitle("Batches over time")
    plt.tight_layout()

    # --- boxplot of residuals by batch, for a single feature -------------
    fig, ax = plt.subplots()
    batch_boxplot(
        sim, batch_col="batch", feature="feature1",
        formula="age + diagnosis*time", ranef="(1|subid)",
        ax=ax, verbose=False,
    )
    ax.set_title("feature1 residuals by batch (before ComBat)")

    # --- trajectory plot --------------------------------------------------
    fig, ax = plt.subplots()
    traj_plot(
        sim, id_col="subid", time_col="time",
        feature="feature2", batch_col="batch",
        ax=ax, verbose=False,
    )
    ax.set_title("feature2 individual trajectories")

    # --- test for additive and multiplicative batch effects --------------
    print("\nTesting for additive batch effects (LRT) ...")
    add_table = add_test(
        sim, batch_col="batch", id_col="subid",
        features=feature_names,
        formula="age + diagnosis*time", ranef="(1|subid)",
        verbose=False,
    )
    print(add_table.head(5).to_string(index=False))

    print("\nTesting for multiplicative batch effects (Fligner-Killeen) ...")
    mult_table = mult_test(
        sim, batch_col="batch", id_col="subid",
        features=feature_names,
        formula="age + diagnosis*time", ranef="(1|subid)",
        verbose=False,
    )
    print(mult_table.head(5).to_string(index=False))

    # --- harmonize --------------------------------------------------------
    print("\nHarmonizing with longitudinal ComBat ...")
    result = long_combat(
        sim,
        batch_col="batch", id_col="subid", time_col="time",
        features=feature_names,
        formula="age + diagnosis*time", ranef="(1|subid)",
        verbose=False,
    )

    # Merge harmonized features back onto the original DataFrame so we can
    # test on the post-ComBat columns using the same batch/subid/time.
    combat_cols = [f"{f}.combat" for f in feature_names]
    merged = pd.concat(
        [sim, result.data_combat[combat_cols].reset_index(drop=True)], axis=1
    )

    print("\nRe-testing additive effects on harmonized data ...")
    add_after = add_test(
        merged, batch_col="batch", id_col="subid",
        features=combat_cols,
        formula="age + diagnosis*time", ranef="(1|subid)",
        verbose=False,
    )
    print(add_after.head(5).to_string(index=False))

    print("\nRe-testing multiplicative effects on harmonized data ...")
    mult_after = mult_test(
        merged, batch_col="batch", id_col="subid",
        features=combat_cols,
        formula="age + diagnosis*time", ranef="(1|subid)",
        verbose=False,
    )
    print(mult_after.head(5).to_string(index=False))

    # Summary: p-values should tend larger post-harmonization.
    import numpy as np
    before_add_median = float(np.median(add_table["p_value"]))
    after_add_median = float(np.median(add_after["p_value"]))
    before_mult_median = float(np.median(mult_table["p_value"]))
    after_mult_median = float(np.median(mult_after["p_value"]))
    print(
        "\n"
        f"Median additive p-value:       {before_add_median:.3g} → {after_add_median:.3g}\n"
        f"Median multiplicative p-value: {before_mult_median:.3g} → {after_mult_median:.3g}"
    )

    plt.show()


if __name__ == "__main__":
    main()
