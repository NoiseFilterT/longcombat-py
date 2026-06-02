"""Smoke tests for plotting helpers. Uses the Agg backend via conftest env."""
from __future__ import annotations

import os

import matplotlib
import pytest

matplotlib.use("Agg", force=True)  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

os.environ.setdefault("MPLBACKEND", "Agg")

from longcombat import batch_boxplot, batch_time_viz, traj_plot  # noqa: E402


class TestBatchTimeViz:
    def test_runs_without_error(self, synthetic_data):
        fig, ax = plt.subplots()
        try:
            ax_out = batch_time_viz(
                synthetic_data, batch_col="batch", time_col="time",
                ax=ax, verbose=False,
            )
            assert ax_out is ax
        finally:
            plt.close(fig)

    def test_creates_own_axes_when_none(self, synthetic_data):
        ax = batch_time_viz(
            synthetic_data, batch_col="batch", time_col="time", verbose=False
        )
        assert ax is not None
        plt.close(ax.figure)


class TestBatchBoxplot:
    def test_runs_without_error(self, synthetic_data):
        fig, ax = plt.subplots()
        try:
            ax_out = batch_boxplot(
                synthetic_data, batch_col="batch", feature="feat1",
                formula="age + diagnosis*time", ranef="(1|subid)",
                ax=ax, verbose=False,
            )
            assert ax_out is ax
        finally:
            plt.close(fig)

    def test_adjust_batch_and_orderby_var(self, synthetic_data):
        fig, ax = plt.subplots()
        try:
            batch_boxplot(
                synthetic_data, batch_col="batch", feature="feat1",
                formula="age + diagnosis*time", ranef="(1|subid)",
                adjust_batch=True, orderby="var",
                colors=["tab:red", "tab:green", "tab:blue"],
                ax=ax, verbose=False,
            )
        finally:
            plt.close(fig)

    def test_invalid_orderby_raises(self, synthetic_data):
        fig, ax = plt.subplots()
        try:
            with pytest.raises(ValueError, match="orderby must be"):
                batch_boxplot(
                    synthetic_data, batch_col="batch", feature="feat1",
                    formula="age + diagnosis*time", ranef="(1|subid)",
                    orderby="foo",
                    ax=ax, verbose=False,
                )
        finally:
            plt.close(fig)


class TestTrajPlot:
    def test_runs_without_error(self, synthetic_data):
        fig, ax = plt.subplots()
        try:
            ax_out = traj_plot(
                synthetic_data, id_col="subid", time_col="time",
                feature="feat1", batch_col="batch",
                ax=ax, verbose=False,
            )
            assert ax_out is ax
        finally:
            plt.close(fig)

    def test_point_colors_track_rows_when_unsorted(self):
        """Per-observation point colors must follow their rows even when a
        subject's input rows are not already in time order."""
        import pandas as pd

        df = pd.DataFrame({
            "subid": [1, 1, 1], "time": [2.0, 0.0, 1.0],
            "batch": ["A", "B", "C"], "feat": [20.0, 0.0, 10.0],
        })
        # color keyed to each row: time2->red, time0->green, time1->blue
        point_col = ["red", "green", "blue"]
        expected = {2.0: "red", 0.0: "green", 1.0: "blue"}
        fig, ax = plt.subplots()
        try:
            traj_plot(
                df, id_col="subid", time_col="time", feature="feat",
                batch_col="batch", point_col=point_col, ax=ax, verbose=False,
            )
            pts = [
                (float(ln.get_xdata()[0]), ln.get_color())
                for ln in ax.get_lines()
                if ln.get_linestyle() == "None" and len(ln.get_xdata()) == 1
            ]
            assert pts, "no point markers drawn"
            for t, c in pts:
                assert c == expected[t], f"time {t}: got {c}, expected {expected[t]}"
        finally:
            plt.close(fig)
