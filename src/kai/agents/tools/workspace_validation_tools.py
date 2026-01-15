from __future__ import annotations

from typing import Any, Dict, Optional

from kai.agents.tools.tools import read_file, list_files, _get_current_agent, run_script
from kai.agents.tools import state_tools
from kai.schemas import WorkspaceValidationResult

__all__ = [
    "read_file",
    "list_files",
    "write_and_compile",
    "run_test",
    "run_script",
    "register_workspace_validation_result",
]


def write_and_compile(file_path: str, content: str) -> Dict[str, Any]:
    """
    Write a smoke test file into the already-provisioned workspace and compile it.

    Adds a convenience key `match_path` (workspace-relative path) and stores it on
    the agent instance for subsequent run_test calls.
    """

    agent = _get_current_agent()
    result = state_tools.write_and_compile(file_path, content)

    match_path: Optional[str] = None
    if isinstance(result, dict):
        p = result.get("path")
        if isinstance(p, str) and p.strip():
            # state_tools returns ".kai_workspace/<workspace-relative-path>"
            match_path = p.replace(".kai_workspace/", "", 1)
            if match_path.startswith("/"):
                match_path = match_path.lstrip("/")
            result["match_path"] = match_path

    if agent is not None and isinstance(match_path, str) and match_path.strip():
        setattr(agent, "_last_ws_validation_match_path", match_path.strip())

    return result


def run_test(
    match_contract: Optional[str] = None,
    match_test: Optional[str] = None,
    verbosity: int = 2,
    additional_args: Optional[str] = None,
    framework_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the smoke test in the already-provisioned workspace.

    If framework_kwargs does not specify "match_path", we will use the last
    match_path remembered from write_and_compile.
    """

    agent = _get_current_agent()
    fw = dict(framework_kwargs or {})
    if "match_path" not in fw:
        last_mp = (
            getattr(agent, "_last_ws_validation_match_path", None) if agent else None
        )
        if isinstance(last_mp, str) and last_mp.strip():
            fw["match_path"] = last_mp.strip()

    # For Python framework, use confcutdir to prevent project conftest.py files
    # from interfering with the smoke test
    framework = getattr(agent, "framework", None) if agent else None
    if framework == "python" and "confcutdir" not in fw:
        fw["confcutdir"] = "tests/poc"

    return state_tools.run_test(
        match_contract=match_contract,
        match_test=match_test,
        verbosity=verbosity,
        additional_args=additional_args,
        framework_kwargs=fw or None,
    )


def register_workspace_validation_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Register the final WorkspaceValidationResult for this validation run.

    The payload must match the WorkspaceValidationResult schema.

    Required fields:
    - preset: "lightweight" | "clean" | "writeable" | "sandbox"
    - workspace_path: str
    - smoke_test_relpath: str
    - framework: str (use "foundry" for now)
    - compiled: bool
    - compile_errors: List[str]
    - test_success: bool
    - tests_passed: int
    - tests_failed: int
    - raw_output: str
    - error: Optional[str]
    """

    agent = _get_current_agent()
    if agent is None:
        return {"registered": False, "error": "No agent context available"}

    try:
        result = WorkspaceValidationResult(**payload)
        # Store on agent instance for process consumption
        agent._registered_workspace_validation_result = result
        return {
            "registered": True,
            "message": "WorkspaceValidationResult registered. Stop now.",
        }
    except Exception as e:
        return {"registered": False, "error": f"Validation failed: {str(e)}"}
