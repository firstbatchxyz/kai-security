"""
Tools for StateAgent - finding state/ordering vulnerabilities.

Provides:
- Graph tools for understanding code (reused from shared tools)
- write_and_compile: Write file + immediate compilation feedback
- run_test: Run tests with parsed output (framework-agnostic)
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any

# Import shared tools
from kai.agents.tools.tools import (
    dependency_graph_loc,
    dependency_graph_slice,
    dependency_graph_paths,
    dependency_graph_neighbors,
    dependency_graph_resolve,
    dependency_graph_snippet,
    dependency_graph_callers,
    dependency_graph_callees,
    dependency_graph_explain,
    dependency_graph_protocol_entrypoints,
    _get_current_agent,
    _normalize_agent_path,
    _get_adapter,
    write_and_compile,  # Centralized in tools.py
    register_exploit,  # Unified exploit registration with auto-compile
)


def run_test(
    match_contract: Optional[str] = None,
    match_test: Optional[str] = None,
    verbosity: int = 3,
    additional_args: Optional[str] = None,
    framework_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run tests with parsed results (framework-agnostic).

    Uses the appropriate test runner based on the project framework
    (Foundry, Hardhat, Anchor, etc.).

    Args:
        match_contract: Filter by contract/module name pattern
        match_test: Filter by test function name pattern
        verbosity: Verbosity level (0-5), default 3 for traces on failure
        additional_args: Additional CLI arguments for the test runner

    Returns:
        {
            "success": bool,          # Test command succeeded
            "tests_passed": int,
            "tests_failed": int,
            "assertion_failures": List[str],  # Test names that had assertion failures
            "reverts": List[str],             # Test names that reverted
            "raw_output": str,
            "parsed_results": Dict[str, str]  # test_name -> "pass"|"fail"|"revert"
        }

    Example:
        result = run_test(match_contract="ExploitTest", match_test="test_drain")

        if result["tests_passed"] > 0 and result["assertion_failures"]:
            # Assertion failed = we proved the invariant can be broken
            print("Exploit confirmed!")
    """
    agent = _get_current_agent()
    if agent is None:
        return {"success": False, "error": "No agent context available"}

    # Use the provisioned workspace
    workspace_path = getattr(agent, "workspace_path", None)
    if not workspace_path:
        return {
            "success": False,
            "error": "No workspace provisioned. Set agent.workspace_path first.",
        }

    workspace = Path(workspace_path)

    # Get the adapter for framework-specific operations
    adapter = _get_adapter()

    # Run tests using the adapter
    test_result = adapter.run_test(
        workspace_path=workspace,
        match_contract=match_contract,
        match_test=match_test,
        verbosity=verbosity,
        additional_args=additional_args,
        framework_kwargs=framework_kwargs,
    )

    # Track test attempts
    if not hasattr(agent, "_test_attempts"):
        agent._test_attempts = 0
    agent._test_attempts += 1

    # Convert TestResult to dict and add attempt number
    result_dict = test_result.to_dict()
    result_dict["attempt"] = agent._test_attempts

    return result_dict


def patch_file(file_path: str, old_content: str, new_content: str) -> Dict[str, Any]:
    """
    Patch a file by replacing old_content with new_content, then recompile.

    Useful for fixing compilation errors without rewriting the entire file.
    Can only patch files in allowed test directories (framework-specific).

    Args:
        file_path: Path to file (must be in allowed test directory)
        old_content: Exact string to find and replace
        new_content: Replacement string

    Returns:
        Same as write_and_compile: {written, path, compiled, errors, ...}
    """
    agent = _get_current_agent()
    if agent is None:
        return {"written": False, "error": "No agent context available"}

    abs_path = _normalize_agent_path(file_path)
    if abs_path is None:
        return {"written": False, "error": f"Invalid path: {file_path}"}

    # Safety check - use adapter's allowed directories
    rel_path = (
        os.path.relpath(abs_path, agent.repo_path) if agent.repo_path else file_path
    )
    adapter = _get_adapter()
    allowed_dirs = adapter.get_allowed_patch_directories()
    if not any(rel_path.startswith(d) for d in allowed_dirs):
        return {
            "written": False,
            "error": f"Can only patch files in: {', '.join(allowed_dirs)}",
        }

    # Read current content
    try:
        current = Path(abs_path).read_text()
    except Exception as e:
        return {"written": False, "error": f"Failed to read file: {e}"}

    # Check if old_content exists
    if old_content not in current:
        return {
            "written": False,
            "error": f"Could not find content to replace. Looking for: {old_content[:100]}...",
        }

    # Replace
    new_file_content = current.replace(old_content, new_content, 1)

    # Write and compile
    return write_and_compile(file_path, new_file_content)


__all__ = [
    "dependency_graph_loc",
    "dependency_graph_slice",
    "dependency_graph_paths",
    "dependency_graph_neighbors",
    "dependency_graph_resolve",
    "dependency_graph_snippet",
    "dependency_graph_callers",
    "dependency_graph_callees",
    "dependency_graph_explain",
    "dependency_graph_protocol_entrypoints",
    "write_and_compile",
    "run_test",
    "patch_file",
    "register_exploit",
]
