from pathlib import Path

import pytest  # type: ignore[import-not-found]

from kai.processes.adapter_chooser import AdapterChooserProcess
from kai.schemas import AdapterChooserInput, Framework, Language
from tests.test_processes_profiler import (
    _load_master_context,
    _normalize_master_context_paths,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_adapter_chooser_selects_foundry(monkeypatch):
    project_root = Path(__file__).resolve().parents[2]
    mc = _load_master_context("bbp_master_context.json")
    mc = _normalize_master_context_paths(mc, project_root)

    repo_root = Path(mc.root_path)
    if not repo_root.exists():
        pytest.skip(
            f"MasterContext root_path not found at {repo_root}. "
            "Run envsetup or provide a materialized testbed repository."
        )

    process = AdapterChooserProcess(mc)

    result = await process.execute(AdapterChooserInput(model_name="z-ai/glm-4.7"))

    assert result.success is True
    assert result.choice is not None
    assert Language.SOLIDITY in result.choice.languages
    assert Language.JAVASCRIPT in result.choice.languages
    assert Framework.FOUNDRY in result.choice.frameworks
    assert Framework.NODE in result.choice.frameworks
    assert "SolidityAdapter" in result.choice.adapters
    # Node adapter mapping intentionally None
    assert None in result.choice.adapters
