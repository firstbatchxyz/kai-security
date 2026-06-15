"""Tests for the unified ``kai`` CLI dispatcher."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kai import cli


def _write_run(dir_path: Path) -> None:
    exploits = [
        {
            "exploit_id": "e1", "status": "verified", "confirmed": True,
            "hypothesis": "Reentrancy in withdraw drains the vault.",
            "file": "Vault.sol", "function": "withdraw", "category": "active_exploit",
            "severity": "critical", "cvss_score": 9.1,
        }
    ]
    (dir_path / "exploits.json").write_text(json.dumps(exploits), encoding="utf-8")


def test_help_and_no_args_print_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    assert "usage: kai-security <command>" in capsys.readouterr().out
    assert cli.main(["--help"]) == 0
    assert "audit" in capsys.readouterr().out


def test_unknown_command_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["bogus"]) == 2
    err = capsys.readouterr().err
    assert "unknown command 'bogus'" in err


@pytest.mark.parametrize(
    "command,expected",
    [
        (["audit", "/repo", "--verbose"], ["pipeline", "/repo", "--verbose"]),
        (["pipeline", "--recipe", "r.json"], ["pipeline", "--recipe", "r.json"]),
        (["agent", "setup", "--input", "{}"], ["agent", "setup", "--input", "{}"]),
    ],
)
def test_audit_pipeline_agent_delegate_to_kai_main(
    command: list[str], expected: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[list[str]] = []
    monkeypatch.setattr("kai.main.main", lambda argv: captured.append(argv))
    assert cli.main(command) == 0
    assert captured == [expected]


def test_view_delegates_and_writes_html(tmp_path: Path) -> None:
    _write_run(tmp_path)
    out = tmp_path / "v.html"
    assert cli.main(["view", str(tmp_path), "-o", str(out)]) == 0
    assert out.exists() and out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_report_delegates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_run(tmp_path)
    assert cli.main(["report", str(tmp_path)]) == 0
    assert "Security findings" in capsys.readouterr().out

    out = tmp_path / "r.html"
    assert cli.main(["report", str(tmp_path), "--format", "html", "-o", str(out)]) == 0
    assert out.exists()


class _FakeEntryPoint:
    def __init__(self, handler: object, name: str = "evolve") -> None:
        self._handler = handler
        self.name = name

    def load(self) -> object:
        return self._handler


def test_plugin_invoked_console_script_style(monkeypatch: pytest.MonkeyPatch) -> None:
    # Plugins are zero-arg callables that read sys.argv (the console-script
    # convention) — NOT functions taking an argv list.
    seen: dict[str, list[str]] = {}

    def handler() -> int:
        seen["argv"] = list(sys.argv)
        return 7

    monkeypatch.setattr(cli, "_plugins", lambda: {"evolve": _FakeEntryPoint(handler)})
    before = list(sys.argv)

    assert cli.main(["evolve", "run", "--x", "1"]) == 7
    # The dispatcher pointed sys.argv at the plugin's invocation, then restored.
    assert seen["argv"] == ["kai evolve", "run", "--x", "1"]
    assert sys.argv == before


def test_plugin_none_return_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "_plugins", lambda: {"evolve": _FakeEntryPoint(lambda: None)})
    assert cli.main(["evolve"]) == 0


def test_plugin_argv_restored_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> int:
        raise RuntimeError("plugin crashed")

    monkeypatch.setattr(cli, "_plugins", lambda: {"evolve": _FakeEntryPoint(boom)})
    before = list(sys.argv)
    with pytest.raises(RuntimeError):
        cli.main(["evolve", "x"])
    assert sys.argv == before  # restored even when the plugin raises


def test_builtin_wins_over_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    # A plugin can't shadow a built-in verb: _plugins() filters them out.
    fakes = [_FakeEntryPoint(None, name="audit"), _FakeEntryPoint(None, name="evolve")]
    monkeypatch.setattr("kai.cli.entry_points", lambda group: fakes)
    plugins = cli._plugins()
    assert "audit" not in plugins
    assert "evolve" in plugins


def test_usage_lists_plugins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "_plugins", lambda: {"evolve": _FakeEntryPoint(None)})
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "plugins:" in out and "evolve" in out


def test_security_plugin_is_registered() -> None:
    # kai-security registers itself under kai.plugins, so `kai security …` works.
    assert "security" in cli._plugins()
