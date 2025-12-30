from pathlib import Path

import pytest  # type: ignore[import-not-found]

from kai.agents import settings
from kai.processes.workspace_validation import WorkspaceValidationProcess
from kai.schemas import WorkspacePreset, WorkspaceValidationInput
from tests.test_processes_profiler import (
    _load_master_context,
    _normalize_master_context_paths,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_workspace_validation_integration_bbp_master_fixture():
    """
    Real integration smoke test against the materialized Ethena BBP master workspace.

    This follows the same fixture + path normalization pattern as the other
    integration-style tests in kai/tests/ (e.g. profiler/adapter_chooser).
    """
    import shutil

    project_root = Path(__file__).resolve().parents[2]
    mc = _load_master_context("bbp_master_context.json")
    mc = _normalize_master_context_paths(mc, project_root)

    repo_root = Path(mc.root_path)
    if not repo_root.exists():
        pytest.skip(
            f"MasterContext root_path not found at {repo_root}. "
            "Materialize the testbed (envsetup) before running this test."
        )

    api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
    if not api_key:
        pytest.skip(
            "Requires OPENROUTER_API_KEY or OPENAI_API_KEY for WorkspaceValidationAgent run"
        )

    if shutil.which("forge") is None:
        pytest.skip(
            "Requires Foundry forge installed to run real workspace validation."
        )

    proc = WorkspaceValidationProcess(context=mc)
    out = await proc.execute(
        WorkspaceValidationInput(
            master_context=mc,
            presets=[
                WorkspacePreset.LIGHTWEIGHT,
                WorkspacePreset.CLEAN,
                WorkspacePreset.WRITEABLE,
                WorkspacePreset.SANDBOX,
            ],
            timeout_compile_s=120,
            timeout_test_s=120,
        )
    )

    if not out.success:
        details = "\n\n".join(
            [
                f"{r.preset.value}: compiled={r.compiled}, test_success={r.test_success}\n{r.raw_output}"
                for r in out.results
            ]
        )
        header = out.error_message or "Workspace validation failed"
        raise AssertionError(f"{header}\n\n{details}")

    assert out.results
    for r in out.results:
        assert r.compiled is True, f"{r.preset}: compile failed: {r.compile_errors}"
        assert r.test_success is True, f"{r.preset}: test failed: {r.raw_output}"
