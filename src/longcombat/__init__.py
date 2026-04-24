"""longcombat-py: longitudinal ComBat harmonization in Python.

Python port of the R package ``longCombat`` by Beer et al. (2020).
See :mod:`longcombat._core` for the main ``long_combat`` function.
"""
from __future__ import annotations

from ._core import LongCombatResult, long_combat
from ._tests import add_test, mult_test

__all__ = [
    "LongCombatResult",
    "long_combat",
    "add_test",
    "mult_test",
]

__version__ = "0.0.0"


def __getattr__(name: str):
    # Lazy-import plotting helpers so matplotlib is optional.
    if name in {"batch_time_viz", "batch_boxplot", "traj_plot"}:
        from . import _plots

        return getattr(_plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
