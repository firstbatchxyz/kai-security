"""The ``kai`` command-line entry point.

A thin dispatcher over the existing modules, giving the friendly verbs the
docs promise:

    kai audit <repo>          analyze a repository (setup → exploit pipeline)
    kai view <run_dir>        open a finished run as interactive HTML
    kai report <run_dir>      render a run's findings (Markdown, or --format html)

``kai pipeline`` / ``kai agent`` remain available as direct aliases into the
full :mod:`kai.main` interface.

**Umbrella plugins.** Beyond its built-in verbs, ``kai`` discovers commands
registered by other installed packages under the ``kai.plugins`` entry-point
group, so a sibling tool can plug in a namespace — e.g. ``kai evolve …`` once
``kai-evolve`` is installed. kai-security registers itself as the ``security``
plugin, so ``kai security audit`` is equivalent to ``kai audit``. See
``docs/umbrella.md``.

The distribution is published as ``kai-security``; the command and the import
package stay ``kai``.
"""

from __future__ import annotations

import sys
from importlib.metadata import EntryPoint, entry_points

_PLUGIN_GROUP = "kai.plugins"

# Verbs handled directly by this module (a plugin can't shadow them).
_BUILTINS = ("audit", "view", "report", "pipeline", "agent")

_USAGE_HEAD = """\
kai-security — automated vulnerability discovery, verification, and patching

usage: kai-security <command> [options]

commands:
  audit <repo>       Analyze a repository for vulnerabilities (setup → exploit)
  view <run_dir>     Open a finished run as interactive HTML (findings + trace)
  report <run_dir>   Render a run's findings as Markdown (default) or HTML

  pipeline           Full pipeline interface (kai audit is the friendly alias)
  agent              Run a single agent
"""

_USAGE_TAIL = "\nRun `kai-security <command> -h` for command-specific options.\n"


def _plugins() -> dict[str, EntryPoint]:
    """Commands registered by other packages under ``kai.plugins``.

    Built-in verbs always win, so a plugin can never shadow ``audit`` etc.
    """

    found = {ep.name: ep for ep in entry_points(group=_PLUGIN_GROUP)}
    return {name: ep for name, ep in found.items() if name not in _BUILTINS}


def _usage(plugins: dict[str, EntryPoint]) -> str:
    if not plugins:
        return _USAGE_HEAD + _USAGE_TAIL
    lines = "".join(f"  {name:<17}(plugin)\n" for name in sorted(plugins))
    return f"{_USAGE_HEAD}\nplugins:\n{lines}{_USAGE_TAIL}"


def main(argv: list[str] | None = None) -> int:
    """Dispatch a ``kai`` subcommand. Returns a process exit code."""

    argv = list(sys.argv[1:] if argv is None else argv)
    plugins = _plugins()
    if not argv or argv[0] in ("-h", "--help", "help"):
        sys.stdout.write(_usage(plugins))
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

    if command in plugins:
        return _run_plugin(command, plugins[command], rest)

    sys.stderr.write(f"kai-security: unknown command {command!r}\n\n")
    sys.stdout.write(_usage(plugins))
    return 2


def _run_plugin(name: str, ep: EntryPoint, rest: list[str]) -> int:
    """Invoke a plugin exactly like its own console script.

    Plugins follow the standard console-script convention: a zero-arg callable
    that reads ``sys.argv`` and returns an exit code (or ``None``). We point
    ``sys.argv`` at ``kai <name> <rest…>`` and call it, so any package that
    already ships a ``[project.scripts]`` entry point works as a kai plugin
    unchanged — no kai-specific signature required.
    """

    handler = ep.load()
    saved_argv = sys.argv
    sys.argv = [f"kai {name}", *rest]
    try:
        result = handler()
    finally:
        sys.argv = saved_argv
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
