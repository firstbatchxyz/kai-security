"""CVSS v3.1 Base Score calculator and vector string parser.

Pure-Python implementation — no external dependencies.
Spec: https://www.first.org/cvss/v3.1/specification-document
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_VECTOR_RE = re.compile(
    r"^CVSS:3\.1"
    r"/AV:([NALP])"
    r"/AC:([LH])"
    r"/PR:([NLH])"
    r"/UI:([NR])"
    r"/S:([UC])"
    r"/C:([NLH])"
    r"/I:([NLH])"
    r"/A:([NLH])$"
)

# Metric value → numeric weight (CVSS v3.1 spec tables)
_AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}
_AC = {"L": 0.77, "H": 0.44}
_UI = {"N": 0.85, "R": 0.62}

# PR depends on Scope
_PR_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
_PR_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.50}

_CIA = {"N": 0.0, "L": 0.22, "H": 0.56}

SEVERITY_RANGES = {
    "Critical": (9.0, 10.0),
    "High": (7.0, 8.9),
    "Medium": (4.0, 6.9),
    "Low": (0.1, 3.9),
    "None": (0.0, 0.0),
}


def _roundup(x: float) -> float:
    """CVSS v3.1 roundup: smallest tenth >= x."""
    return math.ceil(x * 10) / 10


@dataclass(frozen=True)
class CVSSResult:
    vector: str
    score: float
    severity: str


def parse_vector(vector: str) -> dict[str, str] | None:
    """Parse a CVSS:3.1 vector string into metric components.

    Returns None if the vector is malformed.
    """
    m = _VECTOR_RE.match(vector.strip())
    if m is None:
        return None
    return {
        "AV": m.group(1),
        "AC": m.group(2),
        "PR": m.group(3),
        "UI": m.group(4),
        "S": m.group(5),
        "C": m.group(6),
        "I": m.group(7),
        "A": m.group(8),
    }


def compute_score(metrics: dict[str, str]) -> float:
    """Compute CVSS v3.1 base score from metric dict."""
    scope_changed = metrics["S"] == "C"
    pr_table = _PR_CHANGED if scope_changed else _PR_UNCHANGED

    isc_base = 1.0 - (
        (1.0 - _CIA[metrics["C"]])
        * (1.0 - _CIA[metrics["I"]])
        * (1.0 - _CIA[metrics["A"]])
    )

    if scope_changed:
        impact = 7.52 * (isc_base - 0.029) - 3.25 * (isc_base - 0.02) ** 15
    else:
        impact = 6.42 * isc_base

    if impact <= 0:
        return 0.0

    exploitability = (
        8.22
        * _AV[metrics["AV"]]
        * _AC[metrics["AC"]]
        * pr_table[metrics["PR"]]
        * _UI[metrics["UI"]]
    )

    if scope_changed:
        score = _roundup(min(1.08 * (impact + exploitability), 10.0))
    else:
        score = _roundup(min(impact + exploitability, 10.0))

    return score


def score_to_severity(score: float) -> str:
    """Map a numeric CVSS score to a severity label."""
    if score >= 9.0:
        return "Critical"
    if score >= 7.0:
        return "High"
    if score >= 4.0:
        return "Medium"
    if score > 0.0:
        return "Low"
    return "None"


def validate_and_compute(
    vector: str | None,
    claimed_score: float | None = None,
) -> CVSSResult | None:
    """Validate a CVSS vector and compute/correct the score.

    Returns a CVSSResult with the correct score derived from the
    vector, or None if the vector is invalid/missing.
    """
    if not vector:
        return None
    metrics = parse_vector(vector)
    if metrics is None:
        return None
    score = compute_score(metrics)
    severity = score_to_severity(score)
    return CVSSResult(vector=vector.strip(), score=score, severity=severity)
