"""Tools for the BlackboxAgent.

Key property: **per-run isolation**.

The original forge-overlay approach reused a single overlay directory and accumulated
broken harnesses across runs, which caused cross-run compilation failures.

These tools provision a fresh adapter workspace per harness run to ensure failures
in one experiment never poison the next.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from kai.agents.tools.tools import (
    _get_current_agent,
    dependency_graph_protocol_entrypoints,
    dependency_graph_public_entrypoints,
    list_files,
)
from kai.schemas import Observation


def _ensure_agent():
    agent = _get_current_agent()
    if agent is None:
        raise RuntimeError("Blackbox tools require an active agent context.")
    return agent


def _get_agent_framework() -> str:
    """Get tool framework from current agent context."""

    from kai.utils.tool_adapters import get_supported_frameworks

    agent = _ensure_agent()

    master_context = getattr(agent, "master_context", None)
    if master_context:
        frameworks = getattr(master_context, "frameworks", None) or []
        supported = set(get_supported_frameworks())
        for fw in frameworks:
            fw_lower = str(fw).lower()
            if fw_lower in supported:
                return fw_lower

    framework = getattr(agent, "framework", None)

    if framework:
        return str(framework).lower()

    raise ValueError("No framework found for agent.")


def _get_tool_adapter():
    from kai.utils.tool_adapters import get_tool_adapter

    return get_tool_adapter(_get_agent_framework())


def _get_workspace_adapter():
    from kai.utils.workspace import get_workspace_adapter

    return get_workspace_adapter(_get_agent_framework())


def _provision_fresh_workspace(run_id: Optional[str] = None) -> Path:
    """Provision a fresh per-run workspace under the blackbox campaigns_dir."""

    agent = _ensure_agent()

    campaigns_dir = getattr(agent, "campaigns_dir", None)
    if not campaigns_dir:
        raise RuntimeError(
            "Blackbox agent campaigns_dir is not set. This should be attached by BlackboxProcess."
        )

    mc = getattr(agent, "master_context", None)
    if mc is None or not getattr(mc, "root_path", None):
        raise RuntimeError("Blackbox agent master_context.root_path is required.")

    run_id = (run_id or "").strip() or uuid.uuid4().hex
    runs_root = Path(campaigns_dir) / "runs" / run_id
    workspace = runs_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    master = Path(mc.root_path)
    ws_adapter = _get_workspace_adapter()
    ws_adapter.provision_lightweight(workspace, master, mc, logger=None)

    # Expose the last provisioned workspace so subsequent tool calls can reuse it.
    setattr(agent, "_last_blackbox_workspace", str(workspace))
    setattr(agent, "workspace_path", str(workspace))  # compatibility with other tooling

    return workspace


def write_and_compile(file_path: str, content: str) -> Dict[str, Any]:
    """Write a test harness into a fresh isolated workspace and compile it."""

    agent = _ensure_agent()
    tool_adapter = _get_tool_adapter()

    workspace = _provision_fresh_workspace()

    abs_path = tool_adapter.normalize_test_path(file_path, workspace)
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(content)

    # Store default match_path for run_test so the model can omit it.
    rel_match_path = abs_path.relative_to(workspace).as_posix()
    setattr(agent, "_last_blackbox_match_path", rel_match_path)

    compile_result = tool_adapter.compile(workspace)

    return {
        "written": True,
        "path": str(abs_path),
        "workspace": str(workspace),
        "match_path": rel_match_path,
        "compiled": compile_result.success,
        "errors": compile_result.errors,
        "warnings": compile_result.warnings,
        "raw_output": compile_result.raw_output,
    }


def run_test(
    match_contract: Optional[str] = None,
    match_test: Optional[str] = None,
    verbosity: int = 3,
    additional_args: Optional[str] = None,
    framework_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run tests in the most recently provisioned blackbox workspace."""

    agent = _ensure_agent()
    tool_adapter = _get_tool_adapter()

    workspace_path = getattr(agent, "_last_blackbox_workspace", None) or getattr(
        agent, "workspace_path", None
    )
    if not workspace_path:
        # If the model calls run_test without calling write_and_compile first, still
        # create a clean workspace (so failures don’t come from some stale workspace).
        workspace = _provision_fresh_workspace()
        workspace_path = str(workspace)

    workspace = Path(workspace_path)

    fw = dict(framework_kwargs or {})
    if "match_path" not in fw:
        last_match = getattr(agent, "_last_blackbox_match_path", None)
        if last_match:
            fw["match_path"] = last_match

    test_result = tool_adapter.run_test(
        workspace_path=workspace,
        match_contract=match_contract,
        match_test=match_test,
        verbosity=verbosity,
        additional_args=additional_args,
        framework_kwargs=fw or None,
    )

    if not hasattr(agent, "_blackbox_test_attempts"):
        agent._blackbox_test_attempts = 0
    agent._blackbox_test_attempts += 1

    payload = test_result.to_dict()
    payload["attempt"] = agent._blackbox_test_attempts
    payload["workspace"] = str(workspace)
    payload["framework"] = _get_agent_framework()
    payload["framework_kwargs"] = fw
    return payload


def add_observation(
    description: str,
    affected_functions: List[str],
    affected_files: List[str],
    logs: List[str],
    anomaly_type: Optional[str] = None,
    repro_command: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Record an observation of anomalous behavior."""

    agent = _ensure_agent()

    mission_id = getattr(agent, "execution_id", None) or agent.agent_id

    obs = Observation(
        agent_id=agent.agent_id,
        mission_id=mission_id,
        description=description,
        affected_functions=affected_functions,
        affected_files=affected_files,
        logs=logs,
        anomaly_type=anomaly_type,
        repro_command=repro_command,
        seed=seed,
    )

    bucket = getattr(agent, "blackbox_observations", None)
    if bucket is None:
        bucket = []
        setattr(agent, "blackbox_observations", bucket)
    bucket.append(obs)

    return {
        "status": "observation_added",
        "observation_id": len(bucket),
        "total_observations": len(bucket),
    }


__all__ = [
    "list_files",
    "dependency_graph_public_entrypoints",
    "dependency_graph_protocol_entrypoints",
    "write_and_compile",
    "run_test",
    "add_observation",
]
