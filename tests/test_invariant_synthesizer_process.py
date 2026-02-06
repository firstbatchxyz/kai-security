import json
from pathlib import Path
from typing import List

import pytest

from kai.processes.invariant_synthesizer import InvariantSynthesizerProcess
from kai.schemas import (
    CampaignBrief,
    Observation,
    InvariantSynthesizerInput,
)
from kai.utils.dependency.builders import SolidityBuilder
from kai.agents import settings


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _load_campaign_brief() -> CampaignBrief:
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    brief_path = fixtures_dir / "campaign_brief_bbp.json"
    brief_data = json.loads(brief_path.read_text())
    return CampaignBrief(**brief_data)


def _load_observations() -> List[Observation]:
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    obs_path = fixtures_dir / "blackbox_observations_bbp.json"
    obs_data = json.loads(obs_path.read_text())
    return [Observation(**o) for o in obs_data]


@pytest.mark.anyio
async def test_invariant_synthesizer_process_real_run():
    """
    Real test for InvariantSynthesizerProcess.
    Builds a real graph and runs the real agent (requires API keys in env).
    """
    api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
    if not api_key:
        pytest.skip("Requires OPENROUTER_API_KEY or OPENAI_API_KEY for real agent run")

    brief = _load_campaign_brief()
    assert brief.master_context is not None
    repo_root = Path(brief.master_context.root_path)

    if not repo_root.exists():
        pytest.skip(f"MasterContext root_path not found at {repo_root}")

    # 1. Build real graph
    # Match BlackboxProcess grounding logic
    slither_kwargs = {"compile_force_framework": "foundry"}
    builder = SolidityBuilder()
    graph = builder.build(str(repo_root), slither_kwargs=slither_kwargs)

    observations = _load_observations()
    process = InvariantSynthesizerProcess(context=brief.master_context)

    # Only test with a subset of observations to save tokens/time in test
    input_data = InvariantSynthesizerInput(
        observations=observations[:2],
        master_context=brief.master_context,
        dependency_graph=graph,
        protocol_manifesto=None,
        model_name="z-ai/glm-4.7",  # Fast/cheap model
        use_openai=False,
    )

    # 2. Execute process (real agent run)
    output = await process.execute(input_data)

    # 3. Assertions
    assert output.success is True
    assert output.stats["seen"] == 2

    assert len(output.invariants) == output.stats["converted"]

    # Only run these if we actually got invariants
    if output.invariants:
        for inv in output.invariants:
            assert inv.source == "observation_llm"
            assert inv.rule
            assert inv.type
            assert inv.id.startswith("INV_OBS_")
