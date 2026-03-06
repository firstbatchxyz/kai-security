"""Tests for the CVSS v3.1 calculator."""

import pytest

from kai.utils.cvss import (
    compute_score,
    parse_vector,
    score_to_severity,
    validate_and_compute,
)


class TestParseVector:
    def test_valid_vector(self):
        result = parse_vector("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H")
        assert result == {
            "AV": "N", "AC": "L", "PR": "N", "UI": "N",
            "S": "U", "C": "H", "I": "H", "A": "H",
        }

    def test_invalid_prefix(self):
        assert parse_vector("CVSS:2.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H") is None

    def test_missing_metric(self):
        assert parse_vector("CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H") is None

    def test_invalid_metric_value(self):
        assert parse_vector("CVSS:3.1/AV:X/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H") is None

    def test_empty_string(self):
        assert parse_vector("") is None

    def test_strips_whitespace(self):
        result = parse_vector("  CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H  ")
        assert result is not None


class TestComputeScore:
    """Test against known CVSS v3.1 reference scores."""

    def test_max_score(self):
        # AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H = 10.0
        metrics = {"AV": "N", "AC": "L", "PR": "N", "UI": "N",
                   "S": "C", "C": "H", "I": "H", "A": "H"}
        assert compute_score(metrics) == 10.0

    def test_scope_unchanged_high(self):
        # AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H = 9.8
        metrics = {"AV": "N", "AC": "L", "PR": "N", "UI": "N",
                   "S": "U", "C": "H", "I": "H", "A": "H"}
        assert compute_score(metrics) == 9.8

    def test_zero_impact(self):
        # All CIA = None → impact = 0 → score = 0
        metrics = {"AV": "N", "AC": "L", "PR": "N", "UI": "N",
                   "S": "U", "C": "N", "I": "N", "A": "N"}
        assert compute_score(metrics) == 0.0

    def test_medium_score(self):
        # AV:N/AC:H/PR:L/UI:R/S:U/C:L/I:L/A:N = 3.7
        metrics = {"AV": "N", "AC": "H", "PR": "L", "UI": "R",
                   "S": "U", "C": "L", "I": "L", "A": "N"}
        score = compute_score(metrics)
        assert 3.0 <= score <= 4.0

    def test_scope_changed_medium(self):
        # AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N = 6.1
        metrics = {"AV": "N", "AC": "L", "PR": "N", "UI": "R",
                   "S": "C", "C": "L", "I": "L", "A": "N"}
        assert compute_score(metrics) == 6.1


class TestScoreToSeverity:
    @pytest.mark.parametrize("score,expected", [
        (0.0, "None"),
        (0.1, "Low"),
        (3.9, "Low"),
        (4.0, "Medium"),
        (6.9, "Medium"),
        (7.0, "High"),
        (8.9, "High"),
        (9.0, "Critical"),
        (10.0, "Critical"),
    ])
    def test_severity_ranges(self, score, expected):
        assert score_to_severity(score) == expected


class TestValidateAndCompute:
    def test_valid_vector(self):
        result = validate_and_compute(
            "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        )
        assert result is not None
        assert result.score == 9.8
        assert result.severity == "Critical"
        assert result.vector == "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"

    def test_none_vector(self):
        assert validate_and_compute(None) is None

    def test_empty_vector(self):
        assert validate_and_compute("") is None

    def test_invalid_vector(self):
        assert validate_and_compute("not-a-vector") is None

    def test_ignores_claimed_score(self):
        # Even if claimed score is wrong, computed score is correct
        result = validate_and_compute(
            "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            claimed_score=5.0,
        )
        assert result is not None
        assert result.score == 9.8
