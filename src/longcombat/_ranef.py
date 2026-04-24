"""Parse ``lme4``-style random-effects strings.

The upstream R package takes random-effect specifications as strings like
``"(1|subid)"`` or ``"(1 + time|subid)"`` (standard ``lme4`` syntax) and
passes them through to ``lme4::lmer``. This port uses
``statsmodels.regression.mixed_linear_model.MixedLM``, which takes a
grouping vector and a Patsy-style ``re_formula`` separately.

This module converts one into the other.
"""
from __future__ import annotations

import re
from typing import NamedTuple


class RanefSpec(NamedTuple):
    """A parsed ``lme4`` random-effects specification.

    Attributes
    ----------
    groups_col : str
        Name of the grouping factor column (right-hand side of ``|``).
    re_formula : str
        Patsy-style formula string for the random-effects design
        (left-hand side of ``|``). Passed to
        :meth:`statsmodels.MixedLM.from_formula`.
    """

    groups_col: str
    re_formula: str


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def parse_ranef(ranef: str) -> RanefSpec:
    """Parse an ``lme4``-style random-effects string.

    Parameters
    ----------
    ranef : str
        Random-effects specification in ``lme4`` notation. Supported forms
        are a single ``(expr | group)`` block — optionally with the outer
        parentheses already present — where ``group`` is a single column
        name and ``expr`` is the random-effects design (e.g. ``1``,
        ``1 + time``, ``0 + time``).

    Returns
    -------
    RanefSpec
        A named tuple ``(groups_col, re_formula)``.

    Raises
    ------
    ValueError
        If the string does not describe a single ``(expr | group)`` block,
        if the grouping term is not a simple column name, or if the
        left-hand side is empty.

    Notes
    -----
    ``lme4`` allows multiple random-effects blocks (e.g. ``(1|g1) +
    (1|g2)``) and crossed grouping factors. ``statsmodels``' ``MixedLM``
    supports only one grouping factor, so those forms are rejected here
    rather than silently approximated.
    """
    if not isinstance(ranef, str):
        raise TypeError(f"ranef must be a string, got {type(ranef).__name__}")

    s = ranef.strip()
    if not s:
        raise ValueError("ranef string is empty")

    # Reject multi-block forms early with a clear message.
    if _looks_like_multiple_blocks(s):
        raise ValueError(
            f"ranef {ranef!r} appears to specify multiple random-effects "
            "blocks; statsmodels.MixedLM only supports a single grouping "
            "factor in this port. If you need crossed or multiple "
            "grouping factors, use the R package directly."
        )

    # Strip at most one layer of outer parentheses.
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    if "|" not in s:
        raise ValueError(
            f"ranef {ranef!r} must contain '|' separating the "
            "random-effects expression from the grouping factor"
        )

    lhs, rhs = s.split("|", 1)
    lhs = lhs.strip()
    rhs = rhs.strip()

    if not lhs:
        raise ValueError(
            f"ranef {ranef!r} has an empty left-hand side; expected "
            "something like '1' or '1 + time'"
        )
    if not rhs:
        raise ValueError(
            f"ranef {ranef!r} has an empty grouping factor on the "
            "right-hand side of '|'"
        )
    if not _IDENTIFIER.match(rhs):
        raise ValueError(
            f"ranef grouping factor {rhs!r} must be a simple column name; "
            "expressions like 'f(x)' or 'a:b' as grouping factors are not "
            "supported"
        )

    return RanefSpec(groups_col=rhs, re_formula=lhs)


def _looks_like_multiple_blocks(s: str) -> bool:
    """Return True if ``s`` contains more than one ``|`` at depth 0 or 1.

    This is a heuristic check for forms like ``(1|g1) + (1|g2)``. A single
    top-level ``|`` (possibly inside one pair of parentheses) is allowed.
    """
    # Count pipes outside of nested parens; more than one → multiple blocks.
    depth = 0
    pipes_at_low_depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "|" and depth <= 1:
            pipes_at_low_depth += 1
    return pipes_at_low_depth > 1
