import os
import shutil
from pathlib import Path

import pytest

from kai.agents import settings
from kai.processes.envsetup import EnvironmentSetupProcess
from kai.schemas import EnvironmentSetupInput, EnvironmentSetupOutput, MasterContext


# Default repo for setup runs (Solidity)
BBP_REPO_URL = "https://github.com/ethena-labs/bbp-public-assets.git"


@pytest.fixture
def anyio_backend():
    # Restrict anyio to asyncio backend to avoid requiring trio
    return "asyncio"


@pytest.mark.anyio
async def test_envsetup_process_integration(monkeypatch):
    api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
    if not api_key:
        pytest.skip("Requires OPENROUTER_API_KEY or OPENAI_API_KEY for real setup run")

    repo_url = os.getenv("BBP_REPO_URL", BBP_REPO_URL)
    repo_path_override = os.getenv("BBP_REPO_PATH")
    model_name = os.getenv("SETUP_MODEL", settings.SETUP_DEFAULT_MODEL)

    # Persist outputs under the repository testbed for inspection
    repo_root = Path(__file__).resolve().parents[2]
    testbed_root = repo_root / "testbed"
    testbed_root.mkdir(exist_ok=True)
    monkeypatch.setattr(
        EnvironmentSetupProcess, "_project_root", lambda self: repo_root
    )

    process = EnvironmentSetupProcess(
        MasterContext(root_path="/tmp", compile_success=True)
    )
    slug = process._repo_slug(repo_url)
    slug_root = testbed_root / slug
    inputs_dir = slug_root / "inputs"
    master_dir = slug_root / "master"

    # Clean previous run remnants
    if slug_root.exists():
        for root, dirs, files in os.walk(slug_root):
            for name in files:
                os.chmod(Path(root) / name, 0o700)
            for name in dirs:
                os.chmod(Path(root) / name, 0o700)
        os.chmod(slug_root, 0o700)
        shutil.rmtree(slug_root, ignore_errors=True)

    result = await process.execute(
        EnvironmentSetupInput(
            repo_url=repo_url,
            num_turns=int(os.getenv("SETUP_TURNS", settings.DEFAULT_MAX_TURNS)),
            model_name=model_name,
            use_openai=bool(
                settings.OPENAI_API_KEY and not settings.OPENROUTER_API_KEY
            ),
            repo_path_override=repo_path_override,
        )
    )

    assert isinstance(result, EnvironmentSetupOutput)
    assert result.success is True
    mc = result.master_context
    assert isinstance(mc, MasterContext)
    assert Path(result.master_repo_path).exists()
    # Allow the agent to set a working root inside the master copy (e.g., contracts/),
    # but still require it to live under the testbed root.
    mc_root = Path(mc.root_path).resolve()
    assert mc_root.exists()
    assert testbed_root in mc_root.parents
    assert inputs_dir.exists()
    assert master_dir.exists()
    # Master should include at least all files from inputs (allow extra build artifacts)
    for root, _, files in os.walk(inputs_dir):
        rel_root = Path(root).relative_to(inputs_dir)
        for f in files:
            rel_file = rel_root / f
            assert (master_dir / rel_file).exists(), f"Missing in master: {rel_file}"
    # Master copy must be read-only
    assert not os.access(result.master_repo_path, os.W_OK)
