import json
import os
from pathlib import Path

import pytest

from kai.processes.profiler import ProfilerProcess
from kai.schemas import MasterContext, ProfilerInput
from kai.agents import settings

# Defaults for live profiler test; overridable via env vars
DEFAULT_PROFILER_REPO_URL = "https://github.com/ethena-labs/bbp-public-assets.git"
DEFAULT_MASTER_CONTEXT_FILENAME = "bbp_master_context.json"


@pytest.fixture
def anyio_backend():
    # Restrict anyio to asyncio backend to avoid requiring trio
    return "asyncio"


@pytest.fixture(autouse=True)
def disable_mongo_logging():
    """No-op since MongoDBHandler is removed."""
    yield


def _load_master_context(master_context_filename: str) -> MasterContext:
    """Load a MasterContext fixture by short name."""
    fixture_path = (
        Path(__file__).resolve().parent / "fixtures" / master_context_filename
    )
    data = json.loads(fixture_path.read_text())
    return MasterContext(**data)


def _normalize_master_context_paths(
    mc: MasterContext, project_root: Path
) -> MasterContext:
    """
    Convert any relative paths in the fixture to absolute paths anchored at the project root
    and coerce root_path to the most specific existing directory.
    """

    def _norm(p: str | None) -> str | None:
        if not p:
            return None
        path = Path(p)
        return str(path if path.is_absolute() else project_root / path)

    mc.root_path = _norm(mc.root_path) or mc.root_path
    mc.artifacts_path = _norm(mc.artifacts_path) or mc.artifacts_path
    mc.src_path = _norm(mc.src_path) or mc.src_path
    mc.lib_path = _norm(mc.lib_path) or mc.lib_path
    mc.test_path = _norm(mc.test_path) or mc.test_path

    # If root_path is too generic (e.g., just "testbed"), infer a tighter root
    candidates = []
    for p in [mc.src_path, mc.lib_path, mc.test_path, mc.artifacts_path]:
        if p:
            path_obj = Path(p)
            # For common build/output dirs, use parent as repo root candidate
            if path_obj.name in {"out", "artifacts"}:
                candidates.append(path_obj.parent)
            else:
                candidates.append(path_obj.parent)
    for candidate in candidates:
        if candidate.exists():
            mc.root_path = str(candidate)
            break

    return mc


@pytest.mark.anyio
async def test_profiler_process_live_with_fixture():
    """
    Runs the real ProfilerProcess + ProfilerAgent using the selected MasterContext fixture,
    the glm model from settings.MAIN_DEFAULT_MODEL, and the actual dependency graph builder.

    You can override defaults with:
      PROFILER_REPO_URL - repository URL (defaults to Ethena BBP repo)
      PROFILER_MASTER_CONTEXT_NAME - fixture name: bbp (default) or swan
    """
    repo_url = os.getenv("PROFILER_REPO_URL", DEFAULT_PROFILER_REPO_URL)
    master_context_filename = os.getenv(
        "PROFILER_MASTER_CONTEXT_FILENAME", DEFAULT_MASTER_CONTEXT_FILENAME
    )

    project_root = Path(__file__).resolve().parents[2]
    mc = _load_master_context(master_context_filename)
    mc = _normalize_master_context_paths(mc, project_root)

    repo_root = Path(mc.root_path)
    if not repo_root.exists():
        pytest.skip(
            f"MasterContext root_path not found at {repo_root}. "
            f"Run envsetup for {repo_url} or adjust PROFILER_REPO_URL/PROFILER_MASTER_CONTEXT_NAME."
        )

    process = ProfilerProcess(mc)
    result = await process.execute(
        ProfilerInput(
            master_context=mc,
            num_turns=16,
            model_name=settings.MAIN_DEFAULT_MODEL,
            use_openai=bool(
                settings.OPENAI_API_KEY and not settings.OPENROUTER_API_KEY
            ),
        )
    )

    assert result.success is True
    assert result.protocol_manifesto is not None
    assert result.response is not None
    assert result.total_tokens.get("prompt_tokens", 0) > 0
