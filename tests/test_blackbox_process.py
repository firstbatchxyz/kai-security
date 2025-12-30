import json
from pathlib import Path

import pytest  # type: ignore[import-not-found]

from kai.processes.blackbox import BlackboxProcess
from kai.schemas import CampaignBrief, BlackboxInput
from logger import logging
from logger.mongo_adapter import MongoDBHandler


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def disable_mongo_logging():
    logger = logging.getLogger()
    removed = []
    for h in list(logger.handlers):
        if isinstance(h, MongoDBHandler):
            logger.removeHandler(h)
            removed.append(h)
    yield


def _load_campaign_brief() -> CampaignBrief:
    """
    Load the BBP CampaignBrief fixture (v2).

    Note: this fixture points at a real BBP repo checkout under testbed/.
    """
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    brief_path = fixtures_dir / "campaign_brief_bbp.json"
    brief_data = json.loads(brief_path.read_text())
    return CampaignBrief(**brief_data)


@pytest.mark.anyio
async def test_blackbox_process_saves_conversation_and_returns_findings(
    monkeypatch, tmp_path
):
    brief = _load_campaign_brief()
    assert brief.master_context is not None
    repo_root = Path(brief.master_context.root_path)
    if not repo_root.exists():
        pytest.skip(f"MasterContext root_path not found at {repo_root}")

    # Clean external campaigns output for this campaign to avoid stale harnesses.
    project_root = Path(__file__).resolve().parents[2]
    external_campaigns_root = (
        project_root / "output" / "campaigns" / brief.campaign_id / repo_root.name
    )
    if external_campaigns_root.exists():
        for p in external_campaigns_root.rglob("*.t.sol"):
            try:
                p.unlink()
            except Exception:
                pass

    process = BlackboxProcess(brief.master_context)

    repo_campaigns_before = (
        {p for p in (repo_root / "campaigns").glob("*.t.sol")}
        if (repo_root / "campaigns").exists()
        else set()
    )
    repo_test_before = (
        {p for p in (repo_root / "test").glob("*.t.sol")}
        if (repo_root / "test").exists()
        else set()
    )

    result = await process.execute(
        BlackboxInput(
            campaign_brief=brief,
            num_turns=brief.budget.max_turns_per_agent,
            model_name="anthropic/claude-opus-4.5",
            # Blackbox runs via OpenRouter (OpenAI-compatible), not OpenAI direct.
            use_openai=False,
        )
    )

    assert result.success is True
    assert result.response is not None
    assert result.response.master_context is not None
    assert isinstance(result.observations, list)
    assert result.observations, (
        "Expected at least one observation for a passing blackbox run"
    )
    assert result.estimated_cost >= 0
    assert "prompt_tokens" in result.total_tokens

    # Ensure we didn't leave harness files inside the target repo.
    repo_campaigns_after = (
        {p for p in (repo_root / "campaigns").glob("*.t.sol")}
        if (repo_root / "campaigns").exists()
        else set()
    )
    repo_test_after = (
        {p for p in (repo_root / "test").glob("*.t.sol")}
        if (repo_root / "test").exists()
        else set()
    )
    assert repo_campaigns_after == repo_campaigns_before
    assert repo_test_after == repo_test_before
