"""Visualization helpers (ports of ``batchTimeViz``, ``batchBoxplot``, ``trajPlot``).

Matplotlib is imported lazily at function-call time so the core package
works without it installed. Install the plot extra::

    pip install "longcombat-py[plot]"
"""
from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ._ranef import parse_ranef


def _import_mpl():
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for longcombat plotting helpers. "
            "Install with: pip install 'longcombat-py[plot]'"
        ) from e
    return plt


def batch_time_viz(
    data: pd.DataFrame,
    batch_col: str,
    time_col: str,
    *,
    xlabel: str = "time",
    ylabel: str = "batch",
    title: str = "",
    ax=None,
    verbose: bool = True,
):
    """Visualize which batches were active at which timepoints.

    Port of ``longCombat::batchTimeViz``. Each horizontal line shows the
    timepoints for one batch. Batches are stacked vertically and ordered
    by (earliest timepoint, duration).

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _import_mpl()
    uniques = pd.unique(data[batch_col])
    try:
        levels = sorted(uniques.tolist())
    except TypeError:
        levels = sorted(uniques.tolist(), key=repr)
    m = len(levels)
    if verbose:
        print(f"[batch_time_viz] found {m} batches")

    # Earliest timepoint + duration per batch → reorder.
    first = []
    duration = []
    for lvl in levels:
        t = data.loc[data[batch_col] == lvl, time_col].to_numpy()
        first.append(float(np.min(t)))
        duration.append(float(np.max(t) - np.min(t)))
    order = np.lexsort((duration, first))  # primary: first, secondary: duration
    levels = [levels[i] for i in order]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(3, 0.4 * m)))

    tmin = float(data[time_col].min())
    tmax = float(data[time_col].max())
    ax.set_xlim(tmin, tmax)
    ax.set_ylim(-0.5, m - 0.5)

    for i, lvl in enumerate(levels):
        t = np.sort(data.loc[data[batch_col] == lvl, time_col].to_numpy())
        ax.plot(t, np.full_like(t, i, dtype=float), marker="o", linestyle="-")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks(range(m))
    ax.set_yticklabels([str(lvl) for lvl in levels])
    if title:
        ax.set_title(title)
    return ax


def batch_boxplot(
    data: pd.DataFrame,
    batch_col: str,
    feature: str | int,
    formula: str,
    ranef: str,
    *,
    adjust_batch: bool = False,
    orderby: str = "mean",
    plot_means: bool = True,
    colors: str | Sequence[str] = "grey",
    xlabel: str = "batch",
    ylabel: str = "residuals",
    ylim: tuple[float, float] | None = None,
    title: str = "",
    ax=None,
    verbose: bool = True,
):
    """Boxplot of LME residuals per batch for a single feature.

    Port of ``longCombat::batchBoxplot``. Use ``adjust_batch=False`` to
    illustrate additive (plus multiplicative) batch effects, and
    ``adjust_batch=True`` with ``orderby='var'`` to isolate multiplicative
    effects.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _import_mpl()

    if isinstance(feature, (int, np.integer)) and not isinstance(feature, bool):
        feature_name = str(data.columns[int(feature)])
    else:
        feature_name = str(feature)
    if feature_name not in data.columns:
        raise KeyError(f"feature column {feature_name!r} not found in data")
    if orderby not in ("mean", "var"):
        raise ValueError(f"orderby must be 'mean' or 'var', got {orderby!r}")

    work = data.copy()
    uniques = pd.unique(work[batch_col])
    try:
        levels = sorted(uniques.tolist())
    except TypeError:
        levels = sorted(uniques.tolist(), key=repr)
    work[batch_col] = pd.Categorical(work[batch_col], categories=levels)
    n_batch = len(levels)
    if verbose:
        print(f"[batch_boxplot] found {n_batch} batches")
        print(f"[batch_boxplot] fitting lme model for feature {feature_name}")

    if isinstance(colors, str):
        colors_list: list[str] = [colors] * n_batch
    else:
        colors_list = list(colors)
        if len(colors_list) < n_batch:
            # recycle
            colors_list = [colors_list[i % len(colors_list)] for i in range(n_batch)]

    ranef_spec = parse_ranef(ranef)
    rhs = (
        f"{formula} + C({batch_col}, Treatment)" if adjust_batch else formula
    )
    full_formula = f"Q('{feature_name}') ~ {rhs}"
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
                fit = md.fit(reml=True, method=method)
                break
            except (np.linalg.LinAlgError, ValueError):
                fit = None
    if fit is None:
        raise RuntimeError(f"All optimizers failed for feature {feature_name!r}")
    resid = np.asarray(fit.resid, dtype=float)

    per_batch = [resid[np.asarray(work[batch_col]) == lvl] for lvl in levels]
    stats = [r.mean() if orderby == "mean" else r.var(ddof=1) for r in per_batch]
    order = np.argsort(stats)
    ordered_levels = [levels[i] for i in order]
    ordered_data = [per_batch[i] for i in order]
    ordered_colors = [colors_list[i] for i in order]

    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 0.5 * n_batch), 5))

    bp = ax.boxplot(
        ordered_data,
        tick_labels=[str(lvl) for lvl in ordered_levels],
        patch_artist=True,
        widths=0.6,
    )
    for patch, c in zip(bp["boxes"], ordered_colors, strict=True):
        patch.set_facecolor(c)

    if plot_means:
        means = [r.mean() for r in ordered_data]
        ax.plot(range(1, len(means) + 1), means, "rd", markersize=5)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if title:
        ax.set_title(title)
    return ax


def traj_plot(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    feature: str | int,
    batch_col: str,
    *,
    point_shape: Iterable | None = None,
    point_col: Iterable | None = None,
    line_col: Iterable | None = None,
    line_type: Iterable | None = None,
    xlabel: str = "time",
    ylabel: str = "feature",
    title: str = "",
    xlimits: tuple[float, float] | None = None,
    ylimits: tuple[float, float] | None = None,
    ax=None,
    verbose: bool = True,
):
    """Plot one trajectory per subject.

    Port of ``longCombat::trajPlot``. Each subject contributes one line
    connecting their (time, feature) points in time order. Point shapes
    can optionally encode batch membership.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = _import_mpl()

    if isinstance(feature, (int, np.integer)) and not isinstance(feature, bool):
        feature_name = str(data.columns[int(feature)])
    else:
        feature_name = str(feature)
    if feature_name not in data.columns:
        raise KeyError(f"feature column {feature_name!r} not found in data")

    subject_ids = list(pd.unique(data[id_col]))
    n = len(subject_ids)
    L = len(data)
    uniques = pd.unique(data[batch_col])
    try:
        batch_levels = sorted(uniques.tolist())
    except TypeError:
        batch_levels = sorted(uniques.tolist(), key=repr)
    m = len(batch_levels)
    if verbose:
        print(f"[traj_plot] found {n} unique subjects")
        print(f"[traj_plot] found {L} observations")
        print(f"[traj_plot] found {m} batches")

    # Default encodings.
    batch_code = {lvl: i % 26 for i, lvl in enumerate(batch_levels)}
    # matplotlib marker cycle roughly analogous to R's pch 1:25
    _markers = list("oXx+*sDdP^v<>12348pH|_.")
    if point_shape is None:
        shapes = np.array(
            [_markers[batch_code[b] % len(_markers)] for b in data[batch_col]]
        )
    else:
        shapes = np.asarray(list(point_shape))
        if len(shapes) != L:
            raise ValueError("point_shape length must equal len(data)")

    if point_col is None:
        pcols = np.array(["black"] * L)
    else:
        pcols = np.asarray([str(c) for c in point_col])
        if len(pcols) != L:
            raise ValueError("point_col length must equal len(data)")

    if line_col is None:
        lcols = np.array(["black"] * n)
    else:
        lcols = np.asarray([str(c) for c in line_col])
        if len(lcols) != n:
            raise ValueError("line_col length must equal number of unique subjects")

    if line_type is None:
        ltypes = np.array(["solid"] * n)
    else:
        ltypes = np.asarray([str(t) for t in line_type])
        if len(ltypes) != n:
            raise ValueError("line_type length must equal number of unique subjects")

    if xlimits is None:
        xlimits = (float(data[time_col].min()), float(data[time_col].max()))
    if ylimits is None:
        ylimits = (float(data[feature_name].min()), float(data[feature_name].max()))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    for i, sid in enumerate(subject_ids):
        mask = np.asarray(data[id_col] == sid)
        rows = data.loc[mask].sort_values(time_col)
        ax.plot(
            rows[time_col], rows[feature_name],
            color=lcols[i], linestyle=ltypes[i],
        )
        # Points: per-observation marker + color. matplotlib can't vectorize
        # different marker shapes in one scatter call; loop by row.
        orig_idx = np.where(mask)[0]
        for k, row in zip(orig_idx, rows.itertuples(index=False), strict=True):
            ax.plot(
                getattr(row, time_col), getattr(row, feature_name),
                marker=str(shapes[k]), color=pcols[k], linestyle="none",
            )

    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    return ax
