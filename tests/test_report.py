"""Tests for the Markdown findings report."""

from __future__ import annotations

import json
from pathlib import Path

from kai.report import main, render_html, render_markdown, render_run
from kai.viewer.findings import load_findings

_EXPLOITS = [
    {
        "exploit_id": "e2",
        "status": "rejected",
        "confirmed": False,
        "hypothesis": "Fee truncation rounds small trades to zero.",
        "file": "contracts/Fees.sol",
        "function": "calcFee",
        "category": "theoretical_bounds",
        "cvss_score": 4.3,
    },
    {
        "exploit_id": "e1",
        "status": "verified",
        "confirmed": True,
        "hypothesis": "Reentrancy in withdraw drains the vault.",
        "file": "contracts/Vault.sol",
        "function": "withdraw",
        "category": "active_exploit",
        "severity": "critical",
        "cvss_score": 9.1,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "cvss_justification": {"AV": "remote attacker"},
        "poc_code": "contract Attacker { function pwn() external {} }",
        "patch": "-  call_before_update;\n+  update_before_call;",
        "attacker_role": "anyone",
        "prerequisite": "a non-zero deposit",
    },
]


def _write_run(dir_path: Path) -> None:
    (dir_path / "exploits.json").write_text(json.dumps(_EXPLOITS), encoding="utf-8")


def test_report_summary_table_and_order(tmp_path: Path) -> None:
    _write_run(tmp_path)
    md = render_markdown(load_findings(tmp_path), title="myrepo")

    assert md.startswith("# Security findings — myrepo")
    assert "**2 findings** · 1 critical · 1 medium" in md
    # Summary table header + both findings, confirmed-critical sorted first.
    assert "| CVSS | Severity | Finding | Category | Location | Status |" in md
    crit_at = md.index("Reentrancy in withdraw")
    med_at = md.index("Fee truncation")
    assert crit_at < med_at


def test_report_sections_and_code_fences(tmp_path: Path) -> None:
    _write_run(tmp_path)
    md = render_run(tmp_path)

    assert "## 1. Reentrancy in withdraw" in md
    assert "CVSS 9.1 (critical)" in md
    assert "- **Attacker:** anyone" in md
    # PoC fenced, patch fenced as a diff, CVSS breakdown table present.
    assert "```\ncontract Attacker" in md
    assert "```diff\n-  call_before_update;" in md
    assert "| AV | Network | remote attacker |" in md


def test_report_empty(tmp_path: Path) -> None:
    md = render_markdown([], title="empty")
    assert "No findings recorded for this run." in md


def test_main_writes_file(tmp_path: Path) -> None:
    _write_run(tmp_path)
    out = tmp_path / "report.md"
    rc = main([str(tmp_path), "-o", str(out)])
    assert rc == 0
    assert "Reentrancy in withdraw" in out.read_text(encoding="utf-8")


def test_main_rejects_non_dir(tmp_path: Path) -> None:
    assert main([str(tmp_path / "nope")]) == 2


def test_html_report_is_self_contained_and_styled(tmp_path: Path) -> None:
    _write_run(tmp_path)
    html = render_html(load_findings(tmp_path), title="myrepo")

    assert html.startswith("<!DOCTYPE html>")
    # Fully offline, and shares the viewer's design tokens (one design system).
    assert "http://" not in html and "https://" not in html
    assert "--accent:" in html  # tokens from kai.viewer.style
    assert 'class="finding sev-critical' in html
    assert "Reentrancy in withdraw" in html
    assert '<pre class="diff">' in html
    # The patch diff classes drive the +/- colouring.
    assert '<span class="del">' in html and '<span class="add">' in html


def test_main_format_html_writes_file(tmp_path: Path) -> None:
    _write_run(tmp_path)
    rc = main([str(tmp_path), "--format", "html"])
    assert rc == 0
    out = tmp_path / "report.html"
    assert out.exists()
    assert "Reentrancy in withdraw" in out.read_text(encoding="utf-8")
