"""
Common tools shared across all agents.

This module contains utility functions that are used by multiple agents
to avoid code duplication and ensure consistency.
"""

import contextvars
import os
import json
import subprocess
import shlex
import uuid
from typing import Optional, Union, Dict, Any, List, Literal

from kai.agents.utils import load_gitignore_spec, should_ignore_path
from kai.utils.dependency import GraphQueryEngine
from kai.utils.dependency.adapters import SolidityAdapter
from kai.utils.dependency.analysis import (
    FileSourceLoader,
    NodeRef,
    ContextSlice,
    EvidencePack,
)

# Context variable for current agent (async-safe)
_current_agent_var: contextvars.ContextVar = contextvars.ContextVar(
    "current_agent", default=None
)


def set_current_agent(agent):
    """Set the current agent for tools to access (async-safe)."""
    _current_agent_var.set(agent)


def _get_current_agent():
    """
    Get the current agent instance from contextvars.

    All agents using tools must call set_current_agent() before tool execution.
    This is handled automatically by BaseAgent._create_tool_executor().
    """
    return _current_agent_var.get()


def _normalize_agent_path(path: Optional[str]) -> Optional[str]:
    """
    Normalize user-provided paths so agents can reference files using either
    repo-relative paths (e.g. repos/<slug>/...) or working-dir relative paths.
    """
    if path is None:
        return None

    try:
        agent = _get_current_agent()
    except (NameError, TypeError):
        agent = None

    # Absolute paths stay as-is
    if path and os.path.isabs(path):
        return path

    normalized = os.path.normpath(path) if path else ""
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized == ".":
        normalized = ""

    if agent:
        repo_slug = (
            os.path.basename(agent.repo_path)
            if getattr(agent, "repo_path", None)
            else ""
        )
        if normalized:
            parts = normalized.split(os.sep)
            if len(parts) >= 2 and parts[0] == "repos" and parts[1] == repo_slug:
                remaining = os.path.join(*parts[2:]) if len(parts) > 2 else ""
                return os.path.join(agent.repo_path, remaining)

        base_dir = getattr(agent, "working_dir", agent.repo_path)
        if base_dir and normalized:
            return os.path.join(base_dir, normalized)
        return base_dir

    # Fallback: resolve relative to current directory
    if normalized:
        return os.path.abspath(normalized)
    return os.getcwd()


def _get_dependency_graph():
    """Retrieve the dependency graph attached to the current agent, if any."""
    agent = _get_current_agent()
    if agent and getattr(agent, "dependency_graph", None) is not None:
        return agent.dependency_graph
    return None


def _get_query_engine() -> Optional[GraphQueryEngine]:
    """
    Build a GraphQueryEngine for the current agent if a dependency graph is present.
    """
    graph = _get_dependency_graph()
    agent = _get_current_agent()
    if graph is None or agent is None:
        return None

    base_path = (
        getattr(agent, "repo_path", None)
        or getattr(agent, "working_dir", None)
        or os.getcwd()
    )
    adapter = SolidityAdapter()
    source_loader = FileSourceLoader(base_path)
    return GraphQueryEngine(graph=graph, adapter=adapter, source_loader=source_loader)


# =============================================================================
# GraphQueryEngine Tool Wrappers
# =============================================================================
# These functions expose the GraphQueryEngine methods as individual agent tools
# with proper typed returns (NodeRef, ContextSlice, EvidencePack).


def dependency_graph_resolve(
    ref: str,
    scope: Optional[str] = None,
) -> Union[List[NodeRef], Dict[str, str]]:
    """
    Resolve a symbol reference to node IDs in the dependency graph.

    This is the entry point for finding code elements. Returns ranked candidates
    where public entrypoints are prioritized.

    Args:
        ref: Symbol name to resolve (e.g., "withdraw", "Vault.deposit", or a node ID).
        scope: Optional scope to narrow the search (e.g., contract name).

    Returns:
        List of NodeRef objects representing matching code elements, ranked by relevance.
        Returns {"error": "..."} if the graph is unavailable.

    Examples:
        # Find all functions named "withdraw"
        results = graph_resolve("withdraw")

        # Find "deposit" within the "Vault" contract
        results = graph_resolve("deposit", scope="Vault")

        # Resolve by exact node ID (fast path)
        results = graph_resolve("Vault.deposit(uint256)")
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        return engine.resolve(ref, scope)
    except Exception as e:
        return {"error": str(e)}


def dependency_graph_loc(node_id: str) -> Union[Dict[str, Any], Dict[str, str]]:
    """
    Get the precise location (file, line span, signature) for a node.

    This is the anchor for all graph queries - every answer maps back to a location.

    Args:
        node_id: The ID of the node to locate.

    Returns:
        Dict with: id, kind, file, span (start/end lines), signature.
        Returns {"error": "..."} if the node doesn't exist or graph is unavailable.

    Examples:
        loc = graph_loc("Vault.deposit(uint256)")
        # Returns: {"id": "...", "file": "src/Vault.sol", "span": {"start": 45, "end": 60}, ...}
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        return engine.loc(node_id)
    except Exception as e:
        return {"error": str(e)}


def dependency_graph_snippet(
    file: str, start_line: int, end_line: int
) -> Union[str, Dict[str, str]]:
    """
    Pull a minimal code snippet from a file by line range.

    Uses a secure loader to prevent path traversal attacks.

    Args:
        file: The file path (relative to repo root).
        start_line: Start line number (1-indexed).
        end_line: End line number (1-indexed, inclusive).

    Returns:
        The source code string for the specified range.
        Returns {"error": "..."} if the file doesn't exist or graph is unavailable.

    Examples:
        code = graph_snippet("src/Vault.sol", 45, 60)
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        return engine.snippet(file, {"start": start_line, "end": end_line})
    except Exception as e:
        return {"error": str(e)}


def dependency_graph_neighbors(
    node_id: str,
    edge_kinds: List[str],
    direction: Literal["in", "out", "both"] = "out",
) -> Union[List[NodeRef], Dict[str, str]]:
    """
    Get neighbors of a node filtered by edge types and direction.

    This is the atomic local expansion primitive for graph traversal.

    Args:
        node_id: The ID of the node to expand from.
        edge_kinds: List of edge types to follow. Valid types:
            - "calls": Function call edges
            - "reads": State variable read edges
            - "writes": State variable write edges
            - "inherits": Inheritance edges
            - "imports": Import edges
            - "defines": Container definition edges
            - "accepts": Modifier/interface usage edges
            - "uses_type": Type usage edges
            - "emits": Event emission edges
        direction: "in" (incoming), "out" (outgoing), or "both".

    Returns:
        List of NodeRef objects for neighboring nodes.
        Returns {"error": "..."} if the graph is unavailable.

    Examples:
        # Find all functions that call "withdraw"
        callers = graph_neighbors("Vault.withdraw(uint256)", ["calls"], "in")

        # Find what "deposit" reads and writes
        deps = graph_neighbors("Vault.deposit(uint256)", ["reads", "writes"], "out")
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        return engine.neighbors(node_id, edge_kinds, direction)
    except Exception as e:
        return {"error": str(e)}


def dependency_graph_callers(func_id: str) -> Union[List[NodeRef], Dict[str, str]]:
    """
    Find all functions that call a given function.

    Shortcut for graph_neighbors(func_id, ["calls"], "in").

    Args:
        func_id: The ID of the function to find callers for.

    Returns:
        List of NodeRef objects representing caller functions.
        Returns {"error": "..."} if the graph is unavailable.

    Examples:
        callers = graph_callers("Vault.withdraw(uint256)")
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        return engine.callers(func_id)
    except Exception as e:
        return {"error": str(e)}


def dependency_graph_callees(func_id: str) -> Union[List[NodeRef], Dict[str, str]]:
    """
    Find all functions called by a given function.

    Shortcut for graph_neighbors(func_id, ["calls"], "out").

    Args:
        func_id: The ID of the function to find callees for.

    Returns:
        List of NodeRef objects representing called functions.
        Returns {"error": "..."} if the graph is unavailable.

    Examples:
        callees = graph_callees("Vault.deposit(uint256)")
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        return engine.callees(func_id)
    except Exception as e:
        return {"error": str(e)}


def _sanitize_window(
    start: int, end: Optional[int], total: int, max_window: int = 200
) -> tuple[int, int]:
    """
    Normalize pagination window to prevent unbounded output.
    """
    start = max(start, 0)
    if end is None or end <= start:
        end = start + 50
    end = min(end, start + max_window, total)
    return start, end


def dependency_graph_public_entrypoints(
    start: int = 0, end: Optional[int] = 50
) -> Union[Dict[str, Any], Dict[str, str]]:
    """
    Get all public/external function entrypoints in the dependency graph.

    Returns functions with "public" or "external" visibility (excluding constructors).
    Includes ALL public functions - both protocol code and libraries.

    For protocol-only entrypoints (excluding libraries), use dependency_graph_protocol_entrypoints().

    Pagination:
        - Results are windowed with start/end (0-based, end exclusive).
        - Maximum window size is capped to avoid huge token outputs.

    Returns:
        Dict with:
            results: List[NodeRef]
            total: int
            window: [start, end]
            truncated: bool
        Returns {"error": "..."} if the graph is unavailable.

    Examples:
        # Get all public entry points (including libraries)
        entrypoints = dependency_graph_public_entrypoints()
        for ep in entrypoints["results"]:
            print(f"{ep.container}.{ep.name} - {ep.signature}")
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        # Get public entrypoint IDs from the graph
        entrypoint_ids = engine.graph.public_entrypoints()
        total = len(entrypoint_ids)
        start, end = _sanitize_window(start, end, total)
        sliced = entrypoint_ids[start:end]
        # Convert to NodeRef using the engine's helper
        results = [engine._to_ref(engine.graph.node(nid)) for nid in sliced]
        return {
            "results": results,
            "total": total,
            "window": [start, end],
            "truncated": end < total,
        }
    except Exception as e:
        return {"error": str(e)}


def dependency_graph_protocol_entrypoints(
    start: int = 0, end: Optional[int] = 50
) -> Union[Dict[str, Any], Dict[str, str]]:
    """
    Get public/external entrypoints that belong to the protocol (not libraries).

    This is the primary tool for identifying the attack surface. It filters out:
    - Library code (OpenZeppelin, Solmate, forge-std, etc.)
    - Test files

    Use this instead of dependency_graph_public_entrypoints() when you want
    only the protocol's own functions, not inherited library functions.

    Pagination:
        - Results are windowed with start/end (0-based, end exclusive).
        - Maximum window size is capped to avoid huge token outputs.

    Returns:
        Dict with:
            results: List[NodeRef]
            total: int
            window: [start, end]
            truncated: bool
        Returns {"error": "..."} if the graph is unavailable.

    Examples:
        # Get protocol attack surface
        entrypoints = dependency_graph_protocol_entrypoints()
        for ep in entrypoints["results"]:
            print(f"{ep.container}.{ep.name} - {ep.signature}")
            # e.g., "Vault.deposit - deposit(uint256)"
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        protocol_eps = engine.protocol_entrypoints()
        total = len(protocol_eps)
        start, end = _sanitize_window(start, end, total)
        results = protocol_eps[start:end]
        return {
            "results": results,
            "total": total,
            "window": [start, end],
            "truncated": end < total,
        }
    except Exception as e:
        return {"error": str(e)}


def dependency_graph_paths(
    src_ids: List[str],
    dst_ids: List[str],
    edge_kinds: List[str],
    max_depth: int = 5,
) -> Union[List[List[NodeRef]], Dict[str, str]]:
    """
    Find all paths between source and destination nodes via specified edge types.

    Uses BFS to enumerate bounded paths. Useful for reachability analysis.

    Args:
        src_ids: List of source node IDs to start from.
        dst_ids: List of destination node IDs to reach.
        edge_kinds: List of edge types to traverse (e.g., ["calls"]).
        max_depth: Maximum path length (default 5).

    Returns:
        List of paths, where each path is a List[NodeRef].
        Returns {"error": "..."} if the graph is unavailable.

    Examples:
        # Find all call paths from public functions to a vulnerable sink
        paths = graph_paths(
            src_ids=["Vault.deposit(uint256)", "Vault.withdraw(uint256)"],
            dst_ids=["Vault._transfer(address,uint256)"],
            edge_kinds=["calls"],
            max_depth=5
        )
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        return engine.paths(src_ids, dst_ids, edge_kinds, max_depth)
    except Exception as e:
        return {"error": str(e)}


def dependency_graph_data_paths(
    entrypoints: List[str],
    symbol_id: str,
    mode: Literal["read", "write"] = "write",
) -> Union[List[Dict[str, Any]], Dict[str, str]]:
    """
    Trace data flow from entrypoints to a state variable access.

    Finds paths: Entrypoint -> ... -> Function that reads/writes symbol.

    Args:
        entrypoints: List of public function IDs to trace from.
        symbol_id: The state variable node ID to trace to.
        mode: "read" to find read paths, "write" to find write paths (default).

    Returns:
        List of dicts with: entrypoint, accessor, symbol, steps, length.
        Returns {"error": "..."} if the graph is unavailable.

    Examples:
        # Find all paths where public functions can write to "balances"
        paths = graph_data_paths(
            entrypoints=["Vault.deposit(uint256)", "Vault.withdraw(uint256)"],
            symbol_id="Vault.balances",
            mode="write"
        )
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        return engine.data_paths(entrypoints, symbol_id, mode)
    except Exception as e:
        return {"error": str(e)}


def dependency_graph_slice(
    seeds: List[str],
    policy: str = "standard",
) -> Union[ContextSlice, Dict[str, str]]:
    """
    Build a justified context slice from seed nodes.

    Expands seeds to include necessary context (type definitions, callees, parents)
    to prevent hallucination. Each included node has a justification.

    Args:
        seeds: List of node IDs to build context around.
        policy: Expansion policy. "standard" includes types, callees, and parents.

    Returns:
        ContextSlice with: nodes (List[NodeRef]), files (List[str]), justification (Dict).
        Returns {"error": "..."} if the graph is unavailable.

    Examples:
        # Get context for analyzing a function
        ctx = graph_slice(["Vault.deposit(uint256)"])
        # ctx.nodes contains the function, its callees, type definitions used
        # ctx.justification explains why each node is included
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        return engine.slice(seeds, policy)
    except Exception as e:
        return {"error": str(e)}


def dependency_graph_explain(path: List[str]) -> Union[EvidencePack, Dict[str, str]]:
    """
    Generate verifiable evidence for a path/trace.

    Takes a path and returns code snippets and
    edge metadata that prove the path exists. Used by verifiers.

    Args:
        path: List of node IDs representing a path (e.g., from graph_paths).

    Returns:
        EvidencePack with: item, trace, edges, snippets.
        Returns {"error": "..."} if the graph is unavailable.

    Examples:
        # Prove a call path exists
        paths = graph_paths(src_ids, dst_ids, ["calls"])
        if paths:
            evidence = graph_explain([ref.id for ref in paths[0]])
            # evidence.snippets contains the actual code
            # evidence.trace shows exact file:line locations
    """
    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available to the agent"}

    try:
        return engine.explain(path)
    except Exception as e:
        return {"error": str(e)}


def read_file(
    file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None
) -> str:
    """
    Read a file with a given path, optionally specifying a line range.

    Args:
        file_path: The path to the file.
        start_line: Optional starting line number (1-indexed, inclusive).
        end_line: Optional ending line number (1-indexed, inclusive).

    Returns:
        The content of the file (or specified line range), or an error message if the file cannot be read.

    Examples:
        read_file("foo.rs")              # Full file
        read_file("foo.rs", 100, 150)    # Lines 100-150 only
    """
    try:
        # Resolve relative paths relative to agent's working_dir
        try:
            agent = _get_current_agent()
            normalized = _normalize_agent_path(file_path)
            if normalized is None:
                return f"Error: Invalid path resolution for {file_path}"
            file_path = normalized
        except (NameError, TypeError):
            pass

        # Now convert to absolute path for scope validation
        abs_path = os.path.abspath(file_path)

        # Scope validation (if _get_current_agent is available)
        try:
            agent = _get_current_agent()
            if agent and hasattr(agent, "restricted_scope") and agent.restricted_scope:
                if not any(
                    abs_path.startswith(allowed) for allowed in agent.allowed_paths
                ):
                    return f"Error: Access denied. File '{file_path}' is outside assigned scope."
        except (NameError, TypeError):
            # _get_current_agent not defined or returns None, skip scope validation
            pass

        # Ensure the file path is properly resolved
        if not os.path.exists(abs_path):
            return f"Error: File {file_path} does not exist"

        if not os.path.isfile(abs_path):
            return f"Error: {file_path} is not a file"

        with open(abs_path, "r") as f:
            if start_line is None and end_line is None:
                # Read entire file
                return f.read()
            else:
                # Read specific line range
                lines = f.readlines()
                total_lines = len(lines)

                # Empty files should just return empty content, regardless of range.
                if total_lines == 0:
                    return ""

                # Validate line numbers
                if start_line is not None and (
                    start_line < 1 or start_line > total_lines
                ):
                    return f"Error: start_line {start_line} is out of range (file has {total_lines} lines)"

                # If end_line is past EOF, clamp it to the file length instead of erroring.
                if end_line is not None:
                    if end_line < 1:
                        return f"Error: end_line {end_line} is out of range (file has {total_lines} lines)"
                    if end_line > total_lines:
                        end_line = total_lines

                if (
                    start_line is not None
                    and end_line is not None
                    and start_line > end_line
                ):
                    return f"Error: start_line {start_line} cannot be greater than end_line {end_line}"

                # Extract line range (convert to 0-indexed)
                start_idx = (start_line - 1) if start_line else 0
                end_idx = end_line if end_line else total_lines

                return "".join(lines[start_idx:end_idx])

    except PermissionError:
        return f"Error: Permission denied accessing {file_path}"
    except Exception as e:
        return f"Error: {e}"


def list_files(path: Optional[str] = None, depth: int = 2) -> str:
    """
    Display all files and directories as a tree structure.

    Example output:
    ```
    ./
    ├── user.md
    └── entities/
        ├── 452_willow_creek_dr.md
        └── frank_miller_plumbing.md
    ```

    Args:
        path: Optional path to the directory to display. If None, uses current working directory.
        depth: Maximum depth to traverse. Default is 2.
           depth=0 shows only the root directory contents,
           depth=1 shows root and one level of subdirectories, etc.
               Set to a large number (e.g., 10) for deep exploration.

    Returns:
        A string representation of the directory tree.

    Examples:
        # List current directory with default depth of 2
        tree = list_files()

        # List specific directory with custom depth
        tree = list_files(path="bft", depth=1)

        # Deep exploration
        tree = list_files(depth=10)
    """
    try:
        # Use agent's working_dir if available and no path specified, otherwise os.getcwd()
        if path is None:
            try:
                agent = _get_current_agent()
                dir_path = agent.working_dir if agent else os.getcwd()
            except (NameError, TypeError):
                dir_path = os.getcwd()
        else:
            normalized = _normalize_agent_path(path)
            if normalized is None:
                return f"Error: Invalid path resolution for {path}"
            dir_path = normalized

        # Scope validation (if _get_current_agent is available)
        try:
            agent = _get_current_agent()
            if agent and hasattr(agent, "restricted_scope") and agent.restricted_scope:
                abs_path = os.path.abspath(dir_path)
                if not any(
                    abs_path.startswith(allowed) for allowed in agent.allowed_paths
                ):
                    return f"Error: Access denied. Directory '{dir_path}' is outside assigned scope."
        except (NameError, TypeError):
            # _get_current_agent not defined or returns None, skip scope validation
            pass

        # Load gitignore patterns
        gitignore_spec = load_gitignore_spec(dir_path)

        def build_tree(start_path, prefix="", is_last=True, current_depth=0):
            """Recursively build tree structure"""
            entries = []
            try:
                items = sorted(os.listdir(start_path))
                # Filter out hidden files, __pycache__, and gitignored items
                filtered_items = []
                for item in items:
                    if item.startswith(".") or item == "__pycache__":
                        continue
                    item_path = os.path.join(start_path, item)
                    if should_ignore_path(item_path, dir_path, gitignore_spec):
                        continue
                    filtered_items.append(item)
                items = filtered_items
            except PermissionError:
                return f"{prefix}[Permission Denied]\n"

            if not items:
                return ""

            for i, item in enumerate(items):
                item_path = os.path.join(start_path, item)
                is_last_item = i == len(items) - 1

                # Choose the right prefix characters
                if is_last_item:
                    current_prefix = prefix + "└── "
                    extension = prefix + "    "
                else:
                    current_prefix = prefix + "├── "
                    extension = prefix + "│   "

                if os.path.isdir(item_path):
                    # Check if we've reached the depth limit
                    if depth is not None and current_depth >= depth:
                        entries.append(f"{current_prefix}{item}/ [...]\n")
                    else:
                        # Check if directory is empty (considering gitignore)
                        try:
                            dir_contents = []
                            for f in os.listdir(item_path):
                                if f.startswith(".") or f == "__pycache__":
                                    continue
                                f_path = os.path.join(item_path, f)
                                if should_ignore_path(f_path, dir_path, gitignore_spec):
                                    continue
                                dir_contents.append(f)

                            if not dir_contents:
                                entries.append(f"{current_prefix}{item}/ (empty)\n")
                            else:
                                entries.append(f"{current_prefix}{item}/\n")
                                # Recursively add subdirectory contents
                                entries.append(
                                    build_tree(
                                        item_path,
                                        extension,
                                        is_last_item,
                                        current_depth + 1,
                                    )
                                )
                        except PermissionError:
                            entries.append(
                                f"{current_prefix}{item}/ [Permission Denied]\n"
                            )
                else:
                    entries.append(f"{current_prefix}{item}\n")

            return "".join(entries)

        # Start with the root directory
        tree = f"./\n{build_tree(dir_path)}"
        return tree.rstrip()  # Remove trailing newline

    except Exception as e:
        return f"Error: {e}"


def forge_test(
    test_script_path: Optional[str] = None,
    working_dir: Optional[str] = None,
    match_contract: Optional[str] = None,
    match_test: Optional[str] = None,
    additional_args: Optional[str] = None,
    output_json: bool = True,
) -> dict:
    """
    Run forge test with flexible parameters to support various repository structures.

    This function supports repositories with multiple sub-repositories and test directories.
    You can specify the working directory where forge should run, and use various
    matching patterns to target specific tests.

    Args:
        test_script_path: The path pattern to match test files (uses --match-path).
                         Can be a glob pattern like "test/*.t.sol" or specific file.
        working_dir: The directory to run the forge command from. Useful when the repo
                    has multiple sub-repos with their own foundry.toml files.
                    If None, uses the current working directory.
        match_contract: Contract name pattern to match (uses --match-contract).
        match_test: Test function name pattern to match (uses --match-test).
        additional_args: Any additional forge test arguments as a string.
        output_json: Whether to output JSON format (default True, uses --json flag).

    Returns:
        A dictionary containing the test results. If JSON parsing fails, returns
        {"stdout": <output>, "stderr": <errors>} with the raw output.

    Examples:
        # Run test in a sub-repository
        forge_test(test_script_path="test/MyTest.t.sol", working_dir="ve33")

        # Run specific test function
        forge_test(match_test="test_exploit", working_dir="cl")

        # Run with multiple filters
        forge_test(
            test_script_path="test/*.t.sol",
            match_contract="ExploitTest",
            working_dir="ve33"
        )
    """
    try:
        # Default working dir for agent-driven runs.
        # Blackbox (native tool calling) frequently omits `working_dir` or passes "".
        wd = (working_dir or "").strip() if isinstance(working_dir, str) else ""
        if not wd:
            agent = _get_current_agent()
            if agent is not None:
                wd = (getattr(agent, "repo_path", None) or "") or (
                    getattr(agent, "working_dir", None) or ""
                )
        if not wd:
            wd = os.getcwd()

        # Build the forge test command
        cmd_parts: List[str] = ["forge", "test"]

        # Add match patterns
        if test_script_path:
            cmd_parts.extend(["--match-path", test_script_path])
        if match_contract:
            cmd_parts.extend(["--match-contract", match_contract])
        if match_test:
            cmd_parts.extend(["--match-test", match_test])

        # Add JSON output flag if requested
        if output_json:
            cmd_parts.append("--json")

        # Add any additional arguments
        if additional_args:
            # Use shlex.split to preserve quoted args.
            cmd_parts.extend(shlex.split(additional_args))

        # Run the command
        p = subprocess.run(cmd_parts, text=True, capture_output=True, cwd=wd)

        # Always include metadata so callers can reason about failures.
        # Keep the payload lightweight (truncate large outputs).
        stdout = p.stdout or ""
        stderr = p.stderr or ""
        meta: Dict[str, Any] = {
            "returncode": p.returncode,
            "stdout": stdout[:8000] if len(stdout) > 8000 else stdout,
            "stderr": stderr[:8000] if len(stderr) > 8000 else stderr,
            "cwd": wd,
            "command": cmd_parts,
        }

        # Try to parse JSON output if requested
        if output_json:
            try:
                parsed = json.loads(stdout)
                # Preserve the original JSON structure for agents, but also provide
                # returncode/stdout/stderr for reliability and debugging.
                if isinstance(parsed, dict):
                    parsed.update(meta)
                    parsed["json_parsed"] = True
                    return parsed
                return {
                    "parsed_json": parsed,
                    "json_parsed": True,
                    **meta,
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw output
                return {
                    "error": "Failed to parse JSON output",
                    "json_parsed": False,
                    **meta,
                }
        else:
            # Return raw output for non-JSON mode
            return meta

    except Exception as e:
        return {"error": str(e), "stdout": "", "stderr": "", "returncode": -1}


def cargo_test(
    working_dir: Optional[str] = None,
    package: Optional[str] = None,
    test_name: Optional[str] = None,
    release: bool = False,
    additional_args: Optional[str] = None,
    output_json: bool = False,
) -> dict:
    """
    Run cargo test with flexible parameters to support various Rust project structures.

    This function supports Rust workspaces with multiple packages and can run specific
    tests or all tests in a package.

    Args:
        working_dir: The directory to run the cargo command from. Useful when the repo
                    has multiple sub-projects or you want to run from a specific location.
                    If None, uses the current working directory.
        package: Package name to test (uses -p or --package flag).
                For example: "sp1-prover", "recursion-circuit", etc.
        test_name: Optional specific test name or pattern to run.
                  If None, runs all tests.
        release: Whether to run tests in release mode (optimized).
                Default is False (runs in debug mode).
        additional_args: Any additional cargo test arguments as a string.
                        For example: "--no-fail-fast", "--test-threads=1", etc.
        output_json: Whether to request JSON output format (uses --format json).
                    Note: This requires nightly Rust or the test to support it.

    Returns:
        A dictionary containing the test results:
        - stdout: The standard output from cargo test
        - stderr: The standard error from cargo test
        - returncode: The exit code (0 for success, non-zero for failure)
        - If JSON output is requested and parsing succeeds, returns parsed JSON.

    Examples:
        # Run all tests in a workspace
        result = cargo_test()

        # Run tests for a specific package
        result = cargo_test(package="sp1-prover")

        # Run a specific test in a package
        result = cargo_test(package="sp1-prover", test_name="test_uninitialized_memory")

        # Run tests in release mode with specific package
        result = cargo_test(package="recursion-circuit", release=True)

        # Run from a specific directory
        result = cargo_test(working_dir="crates/prover")

        # Run with additional flags
        result = cargo_test(package="sp1-prover", additional_args="--no-fail-fast --test-threads=1")
    """
    try:
        # Build the cargo test command
        cmd_parts = ["cargo", "test"]

        # Add package filter if specified
        if package:
            cmd_parts.extend(["-p", package])

        # Add test name filter if specified
        if test_name:
            cmd_parts.append(test_name)

        # Add release flag if requested
        if release:
            cmd_parts.append("--release")

        # Add JSON output flag if requested
        if output_json:
            cmd_parts.extend(["--", "--format", "json"])

        # Add any additional arguments
        if additional_args:
            # If additional_args contains test-specific flags (after --), handle carefully
            if "--" in additional_args:
                # Split and add appropriately
                cmd_parts.extend(additional_args.split())
            else:
                cmd_parts.extend(additional_args.split())

        # Resolve working_dir relative to agent's working_dir
        resolved_dir = working_dir
        if working_dir is not None:
            try:
                agent = _get_current_agent()
                if agent and not os.path.isabs(working_dir):
                    resolved_dir = os.path.join(agent.working_dir, working_dir)
            except (NameError, TypeError):
                pass
        else:
            try:
                agent = _get_current_agent()
                resolved_dir = agent.working_dir if agent else os.getcwd()
            except (NameError, TypeError):
                resolved_dir = os.getcwd()

        # Run the command
        result = subprocess.run(
            cmd_parts, capture_output=True, text=True, cwd=resolved_dir
        )

        # Try to parse JSON output if requested
        if output_json:
            try:
                return {
                    "parsed_json": json.loads(result.stdout),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw output with error note
                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "error": "Failed to parse JSON output",
                }
        else:
            # Return raw output for non-JSON mode
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

    except Exception as e:
        return {"error": str(e), "stdout": "", "stderr": "", "returncode": -1}


def anchor_test(
    working_dir: Optional[str] = None,
    test_name: Optional[str] = None,
    skip_build: bool = False,
    skip_deploy: bool = False,
    skip_local_validator: bool = False,
    additional_args: Optional[str] = None,
) -> dict:
    """
    Run anchor test for Solana programs using the Anchor framework.

    This function executes integration tests for Solana programs. By default, the 'anchor test'
    command starts a local validator, builds the program, deploys it, runs tests, and then
    stops the validator. You can skip individual steps using the flags.

    Args:
        working_dir: The directory to run the anchor command from. Useful when the repo
                    has multiple Anchor projects or you want to run from a specific location.
                    If None, uses the current working directory.
        test_name: Optional specific test name or pattern to run.
                  If None, runs all tests.
        skip_build: Whether to skip building the program (uses --skip-build).
                   Default is False.
        skip_deploy: Whether to skip deploying the program (uses --skip-deploy).
                    Default is False. Useful if program is already deployed.
        skip_local_validator: Whether to skip starting local validator (uses --skip-local-validator).
                             Default is False. Use this if you have a validator already running.
        additional_args: Any additional anchor test arguments as a string.
                        For example: "--detach" to keep validator running after tests.

    Returns:
        A dictionary containing the test results:
        - stdout: The standard output from anchor test
        - stderr: The standard error from anchor test
        - returncode: The exit code (0 for success, non-zero for failure)

    Examples:
        # Run all tests (starts validator, builds, deploys, tests, stops validator)
        result = anchor_test()

        # Run tests with existing validator
        result = anchor_test(skip_local_validator=True)

        # Run specific test
        result = anchor_test(test_name="test_initialize")

        # Run from specific directory
        result = anchor_test(working_dir="programs/my-program")

        # Skip build and deploy (useful for quick test iterations)
        result = anchor_test(skip_build=True, skip_deploy=True)

        # Run with additional flags
        result = anchor_test(additional_args="--detach")
    """
    try:
        # Build the anchor test command
        cmd_parts = ["anchor", "test"]

        # Add skip flags if requested
        if skip_build:
            cmd_parts.append("--skip-build")
        if skip_deploy:
            cmd_parts.append("--skip-deploy")
        if skip_local_validator:
            cmd_parts.append("--skip-local-validator")

        # Add test name filter if specified
        if test_name:
            cmd_parts.append(test_name)

        # Add any additional arguments
        if additional_args:
            cmd_parts.extend(additional_args.split())

        # Resolve working_dir relative to agent's working_dir
        resolved_dir = working_dir
        if working_dir is not None:
            try:
                agent = _get_current_agent()
                if agent and not os.path.isabs(working_dir):
                    resolved_dir = os.path.join(agent.working_dir, working_dir)
            except (NameError, TypeError):
                pass
        else:
            try:
                agent = _get_current_agent()
                resolved_dir = agent.working_dir if agent else os.getcwd()
            except (NameError, TypeError):
                resolved_dir = os.getcwd()

        # Run the command
        result = subprocess.run(
            cmd_parts, capture_output=True, text=True, cwd=resolved_dir
        )

        # Return raw output
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except Exception as e:
        return {"error": str(e), "stdout": "", "stderr": "", "returncode": -1}


def ctest(
    build_dir: str,
    test_regex: Optional[str] = None,
    parallel: bool = True,
    verbose: bool = False,
    additional_args: Optional[str] = None,
) -> dict:
    """
    Run tests for a C++ project using CTest.

    CTest is CMake's test runner. Tests must be registered with CMake (add_test() in CMakeLists.txt)
    and the project must be built before running tests.

    Args:
        build_dir: The directory containing the CMake build files (where you ran cmake_build).
        test_regex: Optional regex pattern to filter tests by name.
                   For example: "unit_.*" to run only unit tests.
        parallel: If True, runs tests in parallel using available CPU cores.
                 Default is True.
        verbose: If True, enables verbose output (shows test output even for passing tests).
                Default is False.
        additional_args: Any additional ctest arguments as a string.
                        For example: "--rerun-failed --output-on-failure"

    Returns:
        A dictionary containing the test results:
        - stdout: The standard output from ctest
        - stderr: The standard error from ctest
        - returncode: The exit code (0 for success, non-zero for failure)

    Examples:
        # Run all tests in parallel
        result = ctest(build_dir="monad/build")

        # Run specific test pattern with verbose output
        result = ctest(build_dir="monad/build", test_regex="unit_.*", verbose=True)

        # Run tests serially (no parallelization)
        result = ctest(build_dir="monad/build", parallel=False)

        # Run with custom flags
        result = ctest(build_dir="monad/build", additional_args="--output-on-failure --timeout 300")
    """
    try:
        # Build the ctest command
        cmd_parts = ["ctest"]

        # Add test regex filter if specified
        if test_regex:
            cmd_parts.extend(["-R", test_regex])

        # Add parallel flag if requested
        if parallel:
            cmd_parts.append("--parallel")

        # Add verbose flag if requested
        if verbose:
            cmd_parts.append("--verbose")

        # Add any additional arguments
        if additional_args:
            cmd_parts.extend(additional_args.split())

        # Resolve build_dir relative to agent's working_dir
        resolved_build_dir = build_dir
        try:
            agent = _get_current_agent()
            if agent and not os.path.isabs(build_dir):
                resolved_build_dir = os.path.join(agent.working_dir, build_dir)
        except (NameError, TypeError):
            pass

        # Run the command from the build directory
        result = subprocess.run(
            cmd_parts, capture_output=True, text=True, cwd=resolved_build_dir
        )

        # Return raw output
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except Exception as e:
        return {"error": str(e), "stdout": "", "stderr": "", "returncode": -1}


def update_file(file_path: str, old_content: str, new_content: str) -> Union[bool, str]:
    """
    Simple find-and-replace update method for files.

    This is an easier alternative to write_to_file() that doesn't require
    creating git-style diffs. It performs a simple string replacement.

    Parameters
    ----------
    file_path : str
        Path to the file to update.
    old_content : str
        The exact text to find and replace in the file.
    new_content : str
        The text to replace old_content with.

    Returns
    -------
    Union[bool, str]
        True if successful, error message string if failed.

    Examples
    --------
    # Add a new row to a table
    old = "| TKT-1056  | 2024-09-25 | Late Delivery   | Resolved |"
    new = "| TKT-1056  | 2024-09-25 | Late Delivery   | Resolved |\\n| TKT-1057  | 2024-11-11 | Damaged Item    | Open     |"
    result = update_file("user.md", old, new)
    """
    try:
        # Resolve relative paths relative to agent's working_dir
        normalized = _normalize_agent_path(file_path)
        if normalized is None:
            return f"Error: Invalid path resolution for {file_path}"
        file_path = normalized

        # Now convert to absolute path for scope validation
        abs_path = os.path.abspath(file_path)

        # Scope validation
        agent = _get_current_agent()
        if agent and hasattr(agent, "restricted_scope") and agent.restricted_scope:
            if not any(abs_path.startswith(allowed) for allowed in agent.allowed_paths):
                return f"Error: Access denied. File '{file_path}' is outside assigned scope."

        # Read the current file content
        if not os.path.exists(abs_path):
            return f"Error: File '{file_path}' does not exist"

        if not os.path.isfile(abs_path):
            return f"Error: '{file_path}' is not a file"

        with open(abs_path, "r") as f:
            current_content = f.read()

        # Check if old_content exists in the file
        if old_content not in current_content:
            # Provide helpful context about what wasn't found
            preview_length = 50
            preview = (
                old_content[:preview_length] + "..."
                if len(old_content) > preview_length
                else old_content
            )
            return f"Error: Could not find the specified content in the file. Looking for: '{preview}'"

        # Perform the replacement (only first occurrence)
        updated_content = current_content.replace(old_content, new_content, 1)

        # Check if replacement actually changed anything
        if updated_content == current_content:
            return "Error: No changes were made to the file"

        # Write the updated content back
        with open(abs_path, "w") as f:
            f.write(updated_content)

        return True

    except PermissionError:
        return f"Error: Permission denied writing to '{file_path}'"
    except Exception as e:
        return f"Error: Unexpected error - {str(e)}"


def create_file(file_path: str, content: str = "") -> bool:
    """
    Create a new file in the file system with the given content (if any).
    If the file already exists, overwrite it with the new content.

    Args:
        file_path: The path to the file.
        content: The content of the file.

    Returns:
        True if the file was created successfully, False otherwise.
    """
    temp_file_path = None
    try:
        # Resolve relative paths relative to agent's working_dir
        normalized = _normalize_agent_path(file_path)
        if normalized is None:
            raise Exception(f"Error: Invalid path resolution for {file_path}")
        file_path = normalized

        # Now convert to absolute path for scope validation
        abs_path = os.path.abspath(file_path)

        # Scope validation
        agent = _get_current_agent()
        if agent and hasattr(agent, "restricted_scope") and agent.restricted_scope:
            if not any(abs_path.startswith(allowed) for allowed in agent.allowed_paths):
                raise Exception(
                    f"Error: Access denied. File '{file_path}' is outside assigned scope."
                )

        # Create parent directories if they don't exist
        parent_dir = os.path.dirname(abs_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # Create a unique temporary file name in the same directory as the target file
        # This ensures the temp file is within the sandbox's allowed path
        target_dir = os.path.dirname(abs_path) or "."
        temp_file_path = os.path.join(target_dir, f"temp_{uuid.uuid4().hex[:8]}.txt")

        with open(temp_file_path, "w") as f:
            f.write(content)

        # Move the content to the final destination
        with open(abs_path, "w") as f:
            f.write(content)
        os.remove(temp_file_path)
        return True
    except Exception as e:
        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        raise Exception(f"Error creating file {file_path}: {e}")


# =============================================================================
# Framework Detection and Tool Adapters
# =============================================================================


def _get_agent_framework() -> str:
    """
    Get the tool framework from the current agent context.

    Checks master_context.frameworks for supported tool frameworks (foundry, hardhat, etc.),
    then falls back to agent.framework attribute if set.

    Returns:
        Framework name (defaults to "foundry" if not available)
    """
    from kai.utils.tool_adapters import get_supported_frameworks

    agent = _get_current_agent()
    if agent is None:
        return "foundry"

    # Check master_context.frameworks for supported tool framework
    master_context = getattr(agent, "master_context", None)
    if master_context:
        frameworks = getattr(master_context, "frameworks", None) or []
        supported = set(get_supported_frameworks())
        for fw in frameworks:
            fw_lower = fw.lower()
            if fw_lower in supported:
                return fw_lower

    # Try framework attribute directly on agent
    framework = getattr(agent, "framework", None)
    if framework:
        return framework.lower()

    return "foundry"


def _get_adapter():
    """Get the tool adapter for the current agent's framework."""
    from kai.utils.tool_adapters import get_tool_adapter

    return get_tool_adapter(_get_agent_framework())


def write_and_compile(file_path: str, content: str) -> Dict[str, Any]:
    """
    Write a test file to the agent workspace and compile it.

    Tests are written to the provisioned workspace's test/ directory.
    The workspace has remappings to access the main repo's contracts.

    Args:
        file_path: Test file name (e.g., "MyExploit.t.sol")
        content: The test file content

    Returns:
        {
            "written": bool,
            "path": str,
            "workspace": str,
            "compiled": bool,
            "errors": List[str],  # Parsed error messages
            "raw_output": str,    # Full compiler output
            "attempt": int        # Compilation attempt number
        }

    Example:
        result = write_and_compile("MyTest.t.sol", '''
        // SPDX-License-Identifier: MIT
        pragma solidity ^0.8.0;
        import "forge-std/Test.sol";
        import "contracts/MyContract.sol";

        contract MyTest is Test {
            function test_example() public {
                assertTrue(true);
            }
        }
        ''')

        if result["compiled"]:
            # Ready to run
            pass
        else:
            # Fix errors in result["errors"]
            pass
    """
    from pathlib import Path

    agent = _get_current_agent()
    if agent is None:
        return {"written": False, "error": "No agent context available"}

    # Use the provisioned workspace
    workspace_path = getattr(agent, "workspace_path", None)
    if not workspace_path:
        return {
            "written": False,
            "error": "No workspace provisioned. Set agent.workspace_path first.",
        }

    workspace = Path(workspace_path)

    # Get the adapter for framework-specific operations
    adapter = _get_adapter()

    # Normalize the test path using the adapter
    abs_path = adapter.normalize_test_path(file_path, workspace)
    rel_path = abs_path.relative_to(workspace)

    # Create parent directories and write file
    try:
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content)
    except Exception as e:
        return {"written": False, "error": f"Failed to write file: {e}"}

    # Compile using the adapter
    rel_test_path = f".kai_workspace/{rel_path.as_posix()}"
    compile_result = adapter.compile(workspace)

    # Track compilation attempts
    if not hasattr(agent, "_compile_attempts"):
        agent._compile_attempts = 0
    agent._compile_attempts += 1

    return {
        "written": True,
        "path": rel_test_path,
        "workspace": str(workspace),
        "compiled": compile_result.success,
        "errors": compile_result.errors,
        "warnings": getattr(compile_result, "warnings", []),
        "raw_output": compile_result.raw_output,
        "attempt": agent._compile_attempts,
    }


def register_exploit(
    exploit_found: bool,
    reasoning: str,
    poc_path: Optional[str] = None,
    poc_code: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Register an exploit finding with automatic PoC compilation.

    This is the unified registration tool used by State, Quant, and Gamified agents.
    If poc_code and poc_path are provided, the PoC is compiled first. Registration
    fails if compilation fails, giving the agent a chance to fix errors.

    Args:
        exploit_found: True if you found a way to violate the invariant/exploit a gap
        reasoning: Explanation of your analysis and conclusion
        poc_path: Path to the PoC test file (e.g., "test/poc/Exploit.t.sol")
        poc_code: Full code of the PoC (required if exploit_found=True)

    Returns:
        On success: {"registered": True, "compiled": bool, "exploit_count": int, ...}
        On compile failure: {"registered": False, "compile_errors": [...], ...}

    Example (exploit found):
        register_exploit(
            exploit_found=True,
            reasoning="The mint() function lacks role check when...",
            poc_path="test/poc/MintExploit.t.sol",
            poc_code="// SPDX-License-Identifier: MIT\\npragma solidity..."
        )

    Example (no exploit - verified safe):
        register_exploit(
            exploit_found=False,
            reasoning="All paths to mint() are guarded by onlyRole(MINTER_ROLE)..."
        )
    """
    agent = _get_current_agent()
    if agent is None:
        return {"registered": False, "error": "No agent context available"}

    # Initialize exploit registry if not present
    if not hasattr(agent, "_registered_exploits"):
        agent._registered_exploits = []
    if not hasattr(agent, "_exploit_candidates"):
        agent._exploit_candidates = []

    compiled = False
    compile_result = None

    # If exploit found, require PoC and compile it
    if exploit_found:
        if not poc_code or not poc_path:
            return {
                "registered": False,
                "error": "exploit_found=True requires both poc_path and poc_code",
                "message": "Provide the PoC code and path to register the exploit.",
            }

        # Use write_and_compile from this module
        compile_result = write_and_compile(poc_path, poc_code)

        if not compile_result.get("compiled"):
            # Return full compile result so agent has all info to debug
            return {
                "registered": False,
                "message": "PoC failed to compile. Fix the errors and try again.",
                **compile_result,  # Include written, path, workspace, errors, raw_output, etc.
            }

        compiled = True

    # Build exploit record
    exploit_record = {
        "exploit_found": exploit_found,
        "reasoning": reasoning,
        "poc_path": poc_path,
        "poc_code": poc_code,
        "compiled": compiled,
    }
    agent._registered_exploits.append(exploit_record)

    # If exploit found, also add to exploit_candidates for dispatcher
    if exploit_found:
        from kai.schemas import ExploitCandidate

        mission = getattr(agent, "mission", None)
        mission_id = mission.mission_id if mission else "unknown"
        worker_id = getattr(agent, "execution_id", f"agent_{id(agent)}")

        # Determine invariant_id and invariant_ids from mission context
        invariant = getattr(mission, "invariant", None) if mission else None
        invariant_cluster = (
            getattr(mission, "invariant_cluster", None) if mission else None
        )

        if invariant:
            # State/Quant agents have a single target invariant
            invariant_id = invariant.id
            invariant_ids = [invariant.id]
        elif invariant_cluster and len(invariant_cluster) > 0:
            # Gamified agents have an invariant cluster
            invariant_ids = [inv.id for inv in invariant_cluster]
            invariant_id = invariant_ids[0]  # Primary is first in cluster
        else:
            invariant_id = "unknown"
            invariant_ids = []

        exploit_candidate = ExploitCandidate(
            mission_id=mission_id,
            worker_id=worker_id,
            invariant_id=invariant_id,
            invariant_ids=invariant_ids,
            mechanism=reasoning[:200] if len(reasoning) > 200 else reasoning,
            poc_code=poc_code or "",
            target_file=poc_path or "",
            target_function="",
            description=reasoning,
            compiled=compiled,
            logs=[f"registered_by_{type(agent).__name__}"],
        )
        agent._exploit_candidates.append(exploit_candidate)

    return {
        "registered": True,
        "compiled": compiled,
        "type": "exploit" if exploit_found else "verification",
        "exploit_count": len(agent._exploit_candidates),
        "finding_count": len(agent._registered_exploits),
        "message": f"Registered {'exploit' if exploit_found else 'verification'}. "
        f"Total exploits: {len(agent._exploit_candidates)}. "
        "Continue exploring or register more findings.",
    }


# =============================================================================
# Tool Schema Helpers
# =============================================================================

# Tools that need framework-specific descriptions from adapters
ADAPTER_DESCRIBED_TOOLS = {
    "write_and_compile",
    "run_test",
    "patch_file",
    "register_exploit",
}


def get_tool_description(tool_fn, adapter=None) -> str:
    """
    Get the description for a tool, using adapter if needed.

    For tools in ADAPTER_DESCRIBED_TOOLS, uses adapter.get_tool_description()
    to get framework-specific descriptions. Otherwise uses the tool's docstring.
    """
    tool_name = tool_fn.__name__

    # Get description from adapter if available and tool needs it
    if adapter is not None and tool_name in ADAPTER_DESCRIBED_TOOLS:
        desc = adapter.get_tool_description(tool_name)
        if desc is not None:
            return desc.strip()

    # Fall back to docstring
    return (tool_fn.__doc__ or f"Tool: {tool_name}").strip()
