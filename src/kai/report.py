"""Render a run's findings as a Markdown security report.

The no-browser companion to :mod:`kai.viewer`: same on-disk source
(``<run_dir>/exploits.json``), but a plain-text report you can pipe into CI,
paste into a PR, or read over SSH. Markdown renders on GitHub and stays
legible in a terminal, so one format serves both.

    python -m kai.report <run_dir> [--format md|html] [-o OUT]

``--format html`` renders a styled single-page document using the viewer's
design system (:mod:`kai.viewer.style`), so it matches ``kai view``.
"""

from __future__ import annotations

import argparse
import sys
from html import escape
from pathlib import Path

from ra.viewer import style

from kai.viewer.findings import Finding, load_findings

_SEVERITY_ORDER = ("critical", "high", "medium", "low", "none")


def _cell(text: str) -> str:
    """Make a value safe for a single Markdown table cell."""

    return str(text).replace("|", r"\|").replace("\n", " ").strip()


def _score(finding: Finding) -> str:
    return f"{finding.cvss_score:.1f}" if finding.cvss_score is not None else "—"


def _location(finding: Finding) -> str:
    file = finding.file.split("/")[-1] if finding.file else ""
    return f"{file}:{finding.function}" if finding.function else file


def _summary_line(findings: list[Finding]) -> str:
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    parts = [f"{counts[s]} {s}" for s in _SEVERITY_ORDER if counts.get(s)]
    n = len(findings)
    head = f"**{n} finding{'s' if n != 1 else ''}**"
    return f"{head} · {' · '.join(parts)}" if parts else head


def _summary_table(findings: list[Finding]) -> list[str]:
    rows = [
        "| CVSS | Severity | Finding | Category | Location | Status |",
        "|---|---|---|---|---|---|",
    ]
    for f in findings:
        status = f.status + (" ✓" if f.confirmed else "")
        rows.append(
            f"| {_score(f)} | {f.severity} | {_cell(f.title)} | "
            f"{_cell(f.category.replace('_', ' '))} | {_cell(_location(f))} | "
            f"{_cell(status)} |"
        )
    return rows


def _finding_section(idx: int, f: Finding) -> list[str]:
    out: list[str] = []
    sev = f" ({f.severity})" if f.severity != "none" else ""
    out.append(f"## {idx}. {f.title}  ·  CVSS {_score(f)}{sev}")
    out.append("")

    facts = [
        ("Location", f"{f.file} · `{f.function}()`" if f.function else f.file),
        ("Category", f.category.replace("_", " ")),
        ("Attacker", f.attacker_role),
        ("Precondition", f.prerequisite),
        ("Status", f.status + (" · confirmed" if f.confirmed else "")),
    ]
    for label, value in facts:
        if value:
            out.append(f"- **{label}:** {value}")
    out.append("")

    if f.hypothesis:
        out += ["**Why it's exploitable**", "", f.hypothesis, ""]
    if f.exploit_sketch:
        out += ["**Exploit sketch**", "", f.exploit_sketch, ""]

    if f.cvss_rows:
        out += ["**CVSS 3.1**" + (f" — `{f.cvss_vector}`" if f.cvss_vector else ""), ""]
        out += ["| Metric | Value | Justification |", "|---|---|---|"]
        for r in f.cvss_rows:
            out.append(f"| {r['metric']} | {r['value']} | {_cell(r['why'])} |")
        out.append("")

    if f.poc_code:
        out += ["**Proof of concept**", "", "```", f.poc_code, "```", ""]
    if f.patch:
        out += ["**Suggested patch**", "", "```diff", f.patch, "```", ""]
    if f.critic_summary:
        out += ["**Critic**", "", f.critic_summary, ""]
    return out


def render_markdown(findings: list[Finding], title: str = "") -> str:
    """Render a sorted findings list into a Markdown report."""

    lines = [f"# Security findings{f' — {title}' if title else ''}", ""]
    if not findings:
        lines += ["No findings recorded for this run.", ""]
        return "\n".join(lines)

    lines += [_summary_line(findings), ""]
    lines += _summary_table(findings)
    lines += ["", "---", ""]
    for idx, f in enumerate(findings, start=1):
        lines += _finding_section(idx, f)
    return "\n".join(lines).rstrip() + "\n"


def render_run(run_dir: Path) -> str:
    """Load ``<run_dir>/exploits.json`` and render the Markdown report."""

    run_dir = Path(run_dir)
    return render_markdown(load_findings(run_dir), title=run_dir.name)


# ---------------------------------------------------------------------------
# HTML (--format html): a styled single-page report document.
#
# Shares kai.viewer.style so it matches `kai view` exactly. Unlike the
# interactive viewer (master-detail + trace tabs), this is a linear, fully
# expanded document meant to be printed, attached, or shared. Static HTML,
# so every dynamic value is escaped server-side.
# ---------------------------------------------------------------------------

_REPORT_LAYOUT = """\
  header.doc { max-width: 820px; margin: 0 auto; padding: 22px 24px 14px; }
  header.doc h1 { margin: 0 0 6px; font-size: 22px; font-weight: 600; }
  header.doc .summary { font-size: 13px; color: var(--muted-2); }
  header.doc .summary b { color: var(--ink); }
  header.doc .summary .crit { color: var(--accent); font-weight: 600; }
  .toggle { float: right; border: 1px solid var(--rule-2); background: none; color: var(--muted-2);
    border-radius: 5px; cursor: pointer; font-size: 12px; padding: 3px 8px; }
  main.report { max-width: 820px; margin: 0 auto; padding: 8px 24px 64px; }
  .summary-table { margin: 18px 0 4px; }
  .finding { border-top: 1px solid var(--rule); padding-top: 22px; margin-top: 26px; }
  .finding:first-of-type { border-top: 0; margin-top: 14px; }
  .finding h2 { font-size: 18px; margin: 0 0 4px; font-weight: 600; line-height: 1.35; }
  .finding .where { font-size: 12.5px; color: var(--muted); margin-bottom: 14px; }
"""

_REPORT_CSS = style.base_css() + _REPORT_LAYOUT

_THEME_TOGGLE = (
    '<button class="toggle" onclick="document.documentElement.dataset.theme='
    "document.documentElement.dataset.theme==='dark'?'':'dark'\">◐ theme</button>"
)


def _bar(finding: Finding) -> str:
    if finding.cvss_score is None:
        return ""
    pct = max(0, min(100, round(finding.cvss_score / 10 * 100)))
    return f'<span class="bar"><i style="width:{pct}%"></i></span>'


def _summary_row(f: Finding) -> str:
    status = escape(f.status + (" ✓" if f.confirmed else ""))
    return (
        f'<tr class="sev-{escape(f.severity)}"><td class="cvss">'
        f'<span class="dot"></span><span class="score">{_score(f)}</span>{_bar(f)}</td>'
        f'<td class="ftitle">{escape(f.title)}</td>'
        f'<td class="cat">{escape(f.category.replace("_", " "))}</td>'
        f'<td class="loc">{escape(_location(f))}</td><td>{status}</td></tr>'
    )


def _html_diff(patch: str) -> str:
    lines = []
    for line in patch.split("\n"):
        cls = "add" if line.startswith("+") else "del" if line.startswith("-") else ""
        lines.append(f'<span class="{cls}">{escape(line)}</span>' if cls else escape(line))
    return '<pre class="diff">' + "\n".join(lines) + "</pre>"


def _html_finding(idx: int, f: Finding) -> str:
    sev = f" ({escape(f.severity)})" if f.severity != "none" else ""
    where = escape(f.file) + (f" · <code>{escape(f.function)}()</code>" if f.function else "")
    out = [
        f'<section class="finding sev-{escape(f.severity)}'
        f'{"" if f.confirmed else " unconf"}">',
        f'<h2><span class="dot"></span>{idx}. {escape(f.title)} · CVSS {_score(f)}{sev}</h2>',
        f'<div class="where">{where}</div>',
    ]

    kv = []
    for label, value in (
        ("Category", f.category.replace("_", " ")),
        ("Attacker", f.attacker_role),
        ("Precondition", f.prerequisite),
        ("Status", f.status + (" · confirmed" if f.confirmed else "")),
    ):
        if value:
            kv.append(f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>")
    if kv:
        out.append('<dl class="kv">' + "".join(kv) + "</dl>")

    if f.hypothesis:
        out += ['<div class="sec-label">Why it\'s exploitable</div>',
                f'<p class="prose">{escape(f.hypothesis)}</p>']
    if f.exploit_sketch:
        out += ['<div class="sec-label">Exploit sketch</div>',
                f'<p class="prose">{escape(f.exploit_sketch)}</p>']
    if f.cvss_rows:
        out.append('<div class="sec-label">CVSS 3.1 vector</div>')
        if f.cvss_vector:
            out.append(f'<div class="vector mono">{escape(f.cvss_vector)}</div>')
        rows = "".join(
            f'<span class="m">{escape(r["metric"])}</span>'
            f'<span class="v">{escape(r["value"])}</span>'
            f'<span class="why">{escape(r["why"])}</span>'
            for r in f.cvss_rows
        )
        out.append(f'<div class="cvss-grid">{rows}</div>')
    if f.poc_code:
        out += ['<div class="sec-label">Proof of concept</div>',
                f'<pre class="code">{escape(f.poc_code)}</pre>']
    if f.patch:
        out += ['<div class="sec-label">Suggested patch</div>', _html_diff(f.patch)]
    if f.critic_summary:
        out += ['<div class="sec-label">Critic</div>',
                f'<p class="prose">{escape(f.critic_summary)}</p>']
    out.append("</section>")
    return "\n".join(out)


def render_html(findings: list[Finding], title: str = "") -> str:
    """Render a styled, self-contained single-page HTML report document."""

    crit = sum(1 for f in findings if f.severity == "critical")
    n = len(findings)
    summary = f"<b>{n}</b> finding{'s' if n != 1 else ''}"
    if crit:
        summary += f' · <span class="crit">{crit} critical</span>'

    body = [
        '<header class="doc">',
        _THEME_TOGGLE,
        f'<h1 class="serif">Security findings{" — " + escape(title) if title else ""}</h1>',
        f'<div class="summary">{summary}</div>',
        "</header>",
        '<main class="report">',
    ]
    if not findings:
        body.append('<div class="empty">No findings recorded for this run.</div>')
    else:
        body.append(
            '<table class="summary-table"><thead><tr><th class="num">CVSS</th>'
            "<th>Finding</th><th>Category</th><th>Location</th><th>Status</th>"
            "</tr></thead><tbody>"
            + "".join(_summary_row(f) for f in findings)
            + "</tbody></table>"
        )
        body += [_html_finding(i, f) for i, f in enumerate(findings, start=1)]
    body.append("</main>")

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>kai — {escape(title) or 'findings'}</title>\n"
        f"<style>\n{_REPORT_CSS}</style></head>\n<body>\n"
        + "\n".join(body)
        + "\n</body></html>\n"
    )


def render_run_html(run_dir: Path) -> str:
    """Load ``<run_dir>/exploits.json`` and render the HTML report document."""

    run_dir = Path(run_dir)
    return render_html(load_findings(run_dir), title=run_dir.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kai.report",
        description="Render a run's findings as a security report.",
    )
    parser.add_argument("run_dir", help="run directory (a state/<run_id>/ dir)")
    parser.add_argument(
        "-f",
        "--format",
        choices=("md", "html"),
        default="md",
        help="md (Markdown, default) or html (styled single-page document)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="write to PATH (md: default stdout; html: default <run_dir>/report.html)",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the rendered file in a browser (html only)",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"error: {run_dir} is not a directory", file=sys.stderr)
        return 2

    if args.format == "md":
        markdown = render_run(run_dir)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(markdown, encoding="utf-8")
            print(out)
        else:
            sys.stdout.write(markdown)
        return 0

    target = Path(args.output) if args.output else run_dir / "report.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_run_html(run_dir), encoding="utf-8")
    print(target)
    if args.open:
        import webbrowser

        webbrowser.open(target.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
