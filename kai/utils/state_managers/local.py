from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

from kai.schemas import (
    ActorMatrix,
    CampaignBrief,
    ExploitCandidate,
    Fix,
    Invariant,
    Mission,
    Observation,
    ProtocolManifesto,
    Verdict,
)
from kai.state_manager import KaiStateManager


class LocalStateManager(KaiStateManager):
    """
    Local JSON-backed KaiStateManager.

    Currently, only `save_conversation` is used by tests. Other methods are implemented
    as minimal no-ops to satisfy the abstract interface without expanding scope.
    """

    def __init__(
        self,
        execution_id: str,
        *,
        output_dir: Optional[str | Path] = None,
    ):
        super().__init__(execution_id=execution_id)
        self._output_dir = Path(output_dir) if output_dir is not None else None
        
        # Internal ID mappings (same as ExecutorStateManager for consistency)
        self._invariant_id_map: Dict[str, str] = {}
        self._campaign_id_map: Dict[str, str] = {}

    def _project_root(self) -> Path:
        # kai/utils/state_managers/local.py -> .../kai/utils/state_managers -> .../kai/utils -> .../kai -> <repo_root>
        return Path(__file__).resolve().parents[3]

    def _conversations_root(self) -> Path:
        if self._output_dir is not None:
            return self._output_dir
        return self._project_root() / "output" / "conversations"

    def _normalize_relative_convo_path(self, rel: str) -> Path:
        """
        Normalize caller-provided relative path so it is rooted under output/conversations.

        Examples:
        - "conversations/fixer_convo.json" -> "fixer_convo.json"
        - "output/conversations/fixer_convo.json" -> "fixer_convo.json"
        """
        p = Path(rel)
        parts = list(p.parts)
        if parts[:1] == ["conversations"]:
            parts = parts[1:]
        if parts[:2] == ["output", "conversations"]:
            parts = parts[2:]
        return Path(*parts) if parts else Path("conversation.json")

    async def update_state(
        self, state: Literal["setup", "profiler", "invariant"]
    ) -> None:
        return None

    async def save_campaigns(
        self, campaigns: List[CampaignBrief], invariant_id_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        return {}

    async def save_dependency_graph(self, graph_data: Dict[str, Any]) -> bool:
        return True

    async def save_actor_matrix(self, actor_matrix: ActorMatrix) -> bool:
        return True

    async def save_protocol_manifesto(self, manifesto: ProtocolManifesto) -> bool:
        return True

    async def save_master_context(self, context: Any) -> bool:
        return True

    async def save_invariants(self, invariants: List[Invariant]) -> Dict[str, str]:
        return {}

    async def save_missions(
        self,
        missions: List[Mission],
        campaign_id_map: Optional[Dict[str, str]] = None,
        invariant_id_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        return {}

    async def update_mission_status(
        self,
        mission_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> bool:
        return True

    async def save_exploit_candidate(self, candidate: ExploitCandidate) -> bool:
        return True

    async def save_verdict(self, verdict: Verdict) -> bool:
        return True

    async def save_fix(self, fix: Fix) -> bool:
        return True

    async def save_observations(self, observations: List[Observation]) -> bool:
        return True

    async def save_conversation(
        self,
        agent_id: str,
        agent_type: str,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Optional[str]:
        """
        Save an agent's conversation history to a JSON file.

        If `metadata` contains a `conversation_path` key, we treat it as a caller-provided
        relative path (e.g. "conversations/fixer_convo.json") and normalize it under
        output/conversations.
        """
        try:
            convo_root = self._conversations_root()
            convo_root.mkdir(parents=True, exist_ok=True)

            requested = metadata.get("conversation_path") or metadata.get("path")
            if isinstance(requested, str) and requested.strip():
                rel = self._normalize_relative_convo_path(requested.strip())
                out_path = convo_root / rel
            else:
                out_path = convo_root / f"{agent_type}_{agent_id}.json"

            out_path.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "execution_id": self.execution_id,
                "agent_id": agent_id,
                "agent_type": agent_type,
                "metadata": metadata,
                "messages": messages,
            }

            def _json_default(obj: Any):
                # Make persistence robust to common non-JSON-native types (Enums, Paths,
                # and Pydantic models). Fallback to string for anything else.
                if isinstance(obj, Enum):
                    return obj.value
                if isinstance(obj, Path):
                    return str(obj)
                if isinstance(obj, BaseModel):
                    return obj.model_dump(mode="json")
                return str(obj)

            out_path.write_text(
                json.dumps(payload, indent=2, default=_json_default), encoding="utf-8"
            )
            return str(out_path)
        except Exception:
            return None
