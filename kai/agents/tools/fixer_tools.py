"""
Tools for FixerAgent - producing and registering code fixes.

This module exports a restricted tool set for fix generation:
- File listing for exploration
- Dependency graph queries for grounding fixes
- Build/test execution helpers
- Canonical diff generation + patch application
- Fix registration (stores fix artifacts on the agent instance)
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from kai.agents.tools.tools import (
    _get_current_agent,
    _normalize_agent_path,
    list_files,
    dependency_graph_snippet,
)
from kai.agents.tools.state_tools import write_and_compile


def _ensure_in_scope(abs_path: str) -> Optional[str]:
    """
    Return an error string if abs_path is outside the current agent scope.
    """
    agent = _get_current_agent()
    if agent and getattr(agent, "restricted_scope", False):
        if not any(
            abs_path.startswith(allowed)
            for allowed in getattr(agent, "allowed_paths", [])
        ):
            return f"Error: Access denied. File '{abs_path}' is outside assigned scope."
    return None


def generate_canonical_diff(
    file_path: str,
    new_content: str,
    old_content: Optional[str] = None,
    context_lines: int = 3,
) -> Dict[str, Any]:
    """
    Generate a stable (canonical) unified diff for a single file.

    Args:
        file_path: Path to the file (repo/workspace-relative or absolute).
        new_content: The full desired new file content.
        old_content: Optional override for old content. If omitted, the current file
            on disk is read (or treated as empty if it does not exist).
        context_lines: Unified diff context lines (default 3).

    Returns:
        Dict with:
        - file_path: str
        - changed: bool
        - unified_diff: str
        - error: Optional[str]
    """
    try:
        import difflib

        normalized = _normalize_agent_path(file_path)
        if normalized is None:
            return {"error": f"Error: Invalid path resolution for {file_path}"}
        abs_path = os.path.abspath(normalized)

        scope_err = _ensure_in_scope(abs_path)
        if scope_err:
            return {"error": scope_err}

        if old_content is None:
            if os.path.exists(abs_path) and os.path.isfile(abs_path):
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    old_text = f.read()
            else:
                old_text = ""
        else:
            old_text = old_content

        # Canonicalize line endings for diff generation
        old_text = old_text.replace("\r\n", "\n").replace("\r", "\n")
        new_text = (new_content or "").replace("\r\n", "\n").replace("\r", "\n")

        changed = old_text != new_text

        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()

        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=file_path,
                tofile=file_path,
                n=max(0, int(context_lines)),
                lineterm="",
            )
        )
        unified = "\n".join(diff_lines) + ("\n" if diff_lines else "")

        return {
            "file_path": file_path,
            "changed": changed,
            "unified_diff": unified,
        }
    except Exception as e:
        return {"error": str(e)}


_HUNK_RE = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@")


def _apply_unified_diff_to_text(
    old_text: str, unified_diff: str
) -> Union[str, Dict[str, Any]]:
    """
    Apply a unified diff (single-file) to old_text.

    This is a strict applier:
    - Context lines (' ') must match exactly.
    - Removed lines ('-') must match exactly.
    - Added lines ('+') are inserted.
    """
    old_text = (old_text or "").replace("\r\n", "\n").replace("\r", "\n")
    diff_text = (unified_diff or "").replace("\r\n", "\n").replace("\r", "\n")

    old_lines = old_text.splitlines()
    old_has_trailing_nl = old_text.endswith("\n")

    lines = diff_text.split("\n")
    # Strip possible trailing empty line after final newline
    if lines and lines[-1] == "":
        lines = lines[:-1]

    # Extract hunks
    i = 0
    hunks: list[dict[str, Any]] = []
    while i < len(lines):
        line = lines[i]
        if line.startswith("@@"):
            m = _HUNK_RE.match(line)
            if not m:
                return {"error": f"Invalid hunk header: {line}"}
            old_start = int(m.group(1))
            i += 1
            hunk_lines: list[str] = []
            while i < len(lines) and not lines[i].startswith("@@"):
                hunk_lines.append(lines[i])
                i += 1
            hunks.append({"old_start": old_start, "lines": hunk_lines})
            continue
        i += 1

    if not hunks:
        return {"error": "No hunks found in diff"}

    out: list[str] = []
    cursor = 0  # 0-based index into old_lines

    for h in hunks:
        old_start = max(1, int(h["old_start"]))
        target = old_start - 1
        if target < cursor or target > len(old_lines):
            return {
                "error": f"Hunk position out of range: old_start={old_start}, file_lines={len(old_lines)}"
            }

        out.extend(old_lines[cursor:target])
        cursor = target

        for hl in h["lines"]:
            if hl == r"\ No newline at end of file":
                continue
            if hl == "":
                return {"error": "Invalid diff line: empty line without prefix"}

            prefix = hl[0]
            text = hl[1:]

            if prefix == " ":
                if cursor >= len(old_lines) or old_lines[cursor] != text:
                    got = old_lines[cursor] if cursor < len(old_lines) else "<EOF>"
                    return {
                        "error": "Context mismatch while applying diff",
                        "expected": text,
                        "got": got,
                        "line_index": cursor + 1,
                    }
                out.append(text)
                cursor += 1
            elif prefix == "-":
                if cursor >= len(old_lines) or old_lines[cursor] != text:
                    got = old_lines[cursor] if cursor < len(old_lines) else "<EOF>"
                    return {
                        "error": "Removal mismatch while applying diff",
                        "expected": text,
                        "got": got,
                        "line_index": cursor + 1,
                    }
                cursor += 1
            elif prefix == "+":
                out.append(text)
            else:
                return {"error": f"Invalid diff line prefix '{prefix}' in: {hl}"}

    out.extend(old_lines[cursor:])

    new_text = "\n".join(out)
    if old_has_trailing_nl:
        new_text += "\n"
    return new_text


def update_file_with_diff(
    file_path: str,
    unified_diff: str,
    create_if_missing: bool = False,
) -> Dict[str, Any]:
    """
    Apply a unified diff to a file on disk.

    Args:
        file_path: Path to the target file (repo/workspace-relative or absolute).
        unified_diff: A unified diff string (single-file).
        create_if_missing: If True, treat missing file as empty and create it.

    Returns:
        Dict with:
        - applied: bool
        - file_path: str
        - bytes_written: int
        - error: Optional[str]
    """
    try:
        normalized = _normalize_agent_path(file_path)
        if normalized is None:
            return {
                "applied": False,
                "error": f"Error: Invalid path resolution for {file_path}",
            }
        abs_path = os.path.abspath(normalized)

        scope_err = _ensure_in_scope(abs_path)
        if scope_err:
            return {"applied": False, "error": scope_err}

        if os.path.exists(abs_path):
            if not os.path.isfile(abs_path):
                return {
                    "applied": False,
                    "error": f"Error: '{file_path}' is not a file",
                }
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                old_text = f.read()
        else:
            if not create_if_missing:
                return {
                    "applied": False,
                    "error": f"Error: File '{file_path}' does not exist",
                }
            old_text = ""

        applied = _apply_unified_diff_to_text(old_text, unified_diff)
        if isinstance(applied, dict) and "error" in applied:
            return {"applied": False, **applied}

        # At this point applied is guaranteed to be str (checked above)
        new_text: str = str(applied)

        parent = os.path.dirname(abs_path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_text)

        return {
            "applied": True,
            "file_path": file_path,
            "bytes_written": len(new_text.encode("utf-8", errors="replace")),
        }
    except Exception as e:
        return {"applied": False, "error": str(e)}


def register_fix(
    summary: str,
    reasoning: str,
    canonical_diff: str,
    files_changed: Optional[List[str]] = None,
    exploit_candidate: Optional[Dict[str, Any]] = None,
    verdict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Register a proposed fix (diff + reasoning) on the current agent instance.

    This does not apply changes by itself; use update_file_with_diff for that.

    Args:
        summary: One-paragraph summary of what was fixed.
        reasoning: Why this fix addresses the vulnerability / verdict.
        canonical_diff: A unified diff string (can be multi-file concatenated).
        files_changed: Optional list of changed file paths.
        exploit_candidate: Optional serialized ExploitCandidate (fallback if agent context missing).
        verdict: Optional serialized Verdict (fallback if agent context missing).

    Returns:
        Dict with:
        - registered: bool
        - fix_id: str
        - fix_count: int
        - message: str
    """
    agent = _get_current_agent()
    if agent is None:
        return {"registered": False, "error": "No agent context available"}

    if not hasattr(agent, "_registered_fixes"):
        agent._registered_fixes = []

    # Prefer agent-attached objects if present (matches VerifierAgent pattern)
    ec_obj = getattr(agent, "exploit_candidate", None)
    vd_obj = getattr(agent, "_verdict", None)

    fix_id = f"fix_{uuid.uuid4().hex}"
    record = {
        "fix_id": fix_id,
        "summary": summary,
        "reasoning": reasoning,
        "canonical_diff": canonical_diff,
        "files_changed": files_changed or [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "exploit_candidate": exploit_candidate
        or (ec_obj.model_dump() if ec_obj is not None else None),
        "verdict": verdict or (vd_obj.model_dump() if vd_obj is not None else None),
    }

    agent._registered_fixes.append(record)

    return {
        "registered": True,
        "fix_id": fix_id,
        "fix_count": len(agent._registered_fixes),
        "message": f"Registered fix {fix_id}. Total fixes: {len(agent._registered_fixes)}.",
    }


__all__ = [
    "list_files",
    "dependency_graph_snippet",
    "write_and_compile",
    "run_test",
    "update_file_with_diff",
    "generate_canonical_diff",
    "register_fix",
]
