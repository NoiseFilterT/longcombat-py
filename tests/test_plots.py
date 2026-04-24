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
