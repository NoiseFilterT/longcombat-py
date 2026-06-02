"""Tests for the lme4-style random-effects string parser."""
from __future__ import annotations

import pytest

from longcombat._ranef import RanefSpec, parse_ranef


class TestParseRanef:
    def test_random_intercept(self):
        spec = parse_ranef("(1|subid)")
        assert spec == RanefSpec(groups_col="subid", re_formula="1")

    def test_random_intercept_and_slope(self):
        spec = parse_ranef("(1 + time|subid)")
        assert spec.groups_col == "subid"
        assert spec.re_formula == "1 + time"

    def test_without_outer_parens(self):
        spec = parse_ranef("1|subid")
        assert spec == RanefSpec(groups_col="subid", re_formula="1")

    def test_handles_whitespace(self):
        spec = parse_ranef("  ( 1 + time | subid )  ")
        assert spec.groups_col == "subid"
        assert spec.re_formula == "1 + time"

    def test_rejects_multiple_blocks(self):
        with pytest.raises(ValueError, match="multiple random-effects blocks"):
            parse_ranef("(1|g1) + (1|g2)")

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            parse_ranef("")

    def test_rejects_no_pipe(self):
        with pytest.raises(ValueError, match="must contain '|'"):
            parse_ranef("(1 + time)")

    def test_rejects_empty_lhs(self):
        with pytest.raises(ValueError, match="empty left-hand side"):
            parse_ranef("(|subid)")

    def test_rejects_complex_grouping(self):
        with pytest.raises(ValueError, match="simple column name"):
            parse_ranef("(1|f(x))")

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            parse_ranef(123)  # type: ignore[arg-type]

    def test_rejects_uncorrelated_double_pipe(self):
        # lme4's '||' (uncorrelated) gets a specific message, not the generic
        # "multiple random-effects blocks" one.
        with pytest.raises(ValueError, match="uncorrelated random effects"):
            parse_ranef("(1 + time || subid)")
