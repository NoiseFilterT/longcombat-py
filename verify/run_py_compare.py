"""Independent verification harness (Python side).

Reads the input CSVs that the R harness generated, runs the Python port's
long_combat / add_test / mult_test on the SAME rows, and compares against the
R outputs. Prints a per-scenario error table and an overall verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from longcombat import add_test, long_combat, mult_test  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

SCENARIOS = [
    "s1_baseline", "s2_ranslope", "s3_msr",
    "s4_6batch", "s5_simpleformula", "s6_large",
]


def err(py: np.ndarray, r: np.ndarray) -> tuple[float, float]:
    py = np.asarray(py, dtype=float)
    r = np.asarray(r, dtype=float)
    abs = np.abs(py - r)
    max_abs = float(np.nanmax(abs))
    denom = np.maximum(np.abs(r), 1e-12)
    max_rel = float(np.nanmax(abs / denom))
    return max_abs, max_rel


def compare_scenario(name: str) -> dict:
    params = json.loads((OUT / f"{name}_params.json").read_text())
    feats = params["features"]
    df = pd.read_csv(OUT / f"{name}_input.csv")

    res = long_combat(
        data=df, batch_col="batch", id_col="subid", time_col="time",
        features=feats, formula=params["formula"], ranef=params["ranef"],
        niter=30, method=params["method"], eb=True, verbose=False,
    )

    # batch levels sorted (both R and Python order alphabetically)
    levels = sorted(df["batch"].unique().tolist())

    out = {"scenario": name, "method": params["method"], "ranef": params["ranef"]}

    for key, attr in [("gammahat", "gammahat"), ("delta2hat", "delta2hat"),
                      ("gammastarhat", "gammastarhat"), ("delta2starhat", "delta2starhat")]:
        r_df = pd.read_csv(OUT / f"{name}_{key}.csv", index_col=0)
        r_df.index = [str(i) for i in r_df.index]
        r_vals = r_df.loc[[str(l) for l in levels], feats].to_numpy()
        py_vals = getattr(res, attr).loc[levels, feats].to_numpy()
        out[key] = err(py_vals, r_vals)

    r_dc = pd.read_csv(OUT / f"{name}_data_combat.csv")
    combat_cols = [f"{f}.combat" for f in feats]
    out["data_combat"] = err(
        res.data_combat[combat_cols].to_numpy(), r_dc[combat_cols].to_numpy()
    )
    # also check that R and Python preserved the same id/time/batch ordering
    out["row_align_ok"] = bool(
        np.array_equal(res.data_combat["subid"].to_numpy(), r_dc["subid"].to_numpy())
        and np.array_equal(res.data_combat["batch"].astype(str).to_numpy(),
                           r_dc["batch"].astype(str).to_numpy())
    )
    # sanity: does the sum-to-L constraint hold on python gammahat-adjusted effects?
    return out


def fmt(pair: tuple[float, float]) -> str:
    a, r = pair
    return f"abs={a:.2e} rel={r:.2e}"


def main() -> int:
    print("=" * 96)
    print("INDEPENDENT R vs PYTHON COMPARISON  (fresh data, identical input rows)")
    print("=" * 96)
    rows = []
    worst_combat_abs = 0.0
    worst_moment_abs = 0.0
    for name in SCENARIOS:
        o = compare_scenario(name)
        rows.append(o)
        print(f"\n[{name}]  method={o['method']} ranef={o['ranef']}  row_align_ok={o['row_align_ok']}")
        for k in ["gammahat", "delta2hat", "gammastarhat", "delta2starhat", "data_combat"]:
            print(f"    {k:14s} {fmt(o[k])}")
        worst_combat_abs = max(worst_combat_abs, o["data_combat"][0])
        for k in ["gammahat", "delta2hat", "gammastarhat", "delta2starhat"]:
            worst_moment_abs = max(worst_moment_abs, o[k][0])

    print("\n" + "=" * 96)
    print("add_test / mult_test on s1_baseline")
    print("=" * 96)
    df1 = pd.read_csv(OUT / "s1_baseline_input.csv")
    feats1 = [f"feat{i}" for i in range(1, 6)]
    add_py = add_test(df1, batch_col="batch", id_col="subid", features=feats1,
                      formula="age + diagnosis*time", ranef="(1|subid)",
                      method="lrt", verbose=False)
    mult_py = mult_test(df1, batch_col="batch", id_col="subid", features=feats1,
                        formula="age + diagnosis*time", ranef="(1|subid)",
                        verbose=False)
    add_r = pd.read_csv(OUT / "s1_addtest_R.csv")
    mult_r = pd.read_csv(OUT / "s1_multtest_R.csv")

    print("\nadd_test  (Python LRT chi2  vs  R Kenward-Roger F):")
    print("  Python ranking:", list(add_py["feature"]))
    print("  R ranking:     ", list(add_r["Feature"]))
    add_r_sorted = add_r.copy()
    add_r_sorted.columns = ["Feature", "KRF", "KRddf", "KRp"]
    print(add_py.to_string(index=False))

    print("\nmult_test  (Python scipy.fligner  vs  R fligner.test) — should match closely:")
    mr = mult_r.copy()
    mr.columns = ["feature", "ChiSq", "DF", "p_value"]
    merged = mult_py.merge(mr, on="feature", suffixes=("_py", "_R"))
    merged["chi2_absdiff"] = (merged["chi2"] - merged["ChiSq"]).abs()
    merged["p_absdiff"] = (merged["p_value_py"] - merged["p_value_R"]).abs()
    print(merged[["feature", "chi2", "ChiSq", "chi2_absdiff",
                  "p_value_py", "p_value_R", "p_absdiff"]].to_string(index=False))
    mult_chi2_maxdiff = float(merged["chi2_absdiff"].max())

    # rank-correlation of add_test
    add_merged = add_py.merge(
        add_r_sorted.assign(feature=add_r_sorted["Feature"]), on="feature"
    )
    rank_py = add_py.reset_index(drop=True).reset_index().set_index("feature")["index"]
    rank_r = add_r_sorted.reset_index(drop=True).reset_index().set_index("Feature")["index"]
    common = rank_py.index
    spearman_ok = bool((rank_py.loc[common].rank().values ==
                        rank_r.loc[common].rank().values).all())

    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    print(f"  worst abs error on moments (gamma/delta) across all scenarios: {worst_moment_abs:.2e}")
    print(f"  worst abs error on data_combat across all scenarios:           {worst_combat_abs:.2e}")
    print(f"  mult_test chi2 max |Python - R|:                               {mult_chi2_maxdiff:.2e}")
    print(f"  add_test feature ranking Python==R:                            {spearman_ok}")
    all_align = all(o["row_align_ok"] for o in rows)
    print(f"  row alignment preserved in all scenarios:                      {all_align}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
