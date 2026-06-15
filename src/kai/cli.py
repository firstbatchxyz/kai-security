"""The ``kai`` command-line entry point.

A thin dispatcher over the existing modules, giving the friendly verbs the
docs promise:

    kai audit <repo>          analyze a repository (setup → exploit pipeline)
    kai view <run_dir>        open a finished run as interactive HTML
    kai report <run_dir>      render a run's findings (Markdown, or --format html)

``kai pipeline`` / ``kai agent`` remain available as direct aliases into the
full :mod:`kai.main` interface. The distribution is published as
``kai-security``; the command and the import package stay ``kai``.
"""

from __future__ import annotations

import sys

_USAGE = """\
kai-security — automated vulnerability discovery, verification, and patching

usage: kai-security <command> [options]

commands:
  audit <repo>       Analyze a repository for vulnerabilities (setup → exploit)
  view <run_dir>     Open a finished run as interactive HTML (findings + trace)
  report <run_dir>   Render a run's findings as Markdown (default) or HTML

  pipeline           Full pipeline interface (kai audit is the friendly alias)
  agent              Run a single agent

Run `kai-security <command> -h` for command-specific options.
"""


def main(argv: list[str] | None = None) -> int:
    """Dispatch a ``kai`` subcommand. Returns a process exit code."""

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        sys.stdout.write(_USAGE)
        return 0

    command, rest = argv[0], argv[1:]

    if command in ("audit", "pipeline"):
        from kai.main import main as kai_main

        kai_main(["pipeline", *rest])
        return 0
    if command == "agent":
        from kai.main import main as kai_main

        kai_main(["agent", *rest])
        return 0
    if command == "view":
        from kai.viewer.__main__ import main as view_main

        return view_main(rest)
    if command == "report":
        from kai.report import main as report_main

        return report_main(rest)

    sys.stderr.write(f"kai-security: unknown command {command!r}\n\n")
    sys.stdout.write(_USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
