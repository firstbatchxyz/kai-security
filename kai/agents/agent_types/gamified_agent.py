"""
GamifiedAgent - Adversarial reasoning agent for invariant cluster analysis.

Discovers exploitation opportunities by reasoning about gaps BETWEEN
invariants in a cluster. Produces ExploitCandidates with action sequences.

Key differences from other agents:
- Receives a CLUSTER of related invariants, not a single invariant
- Identifies unguarded states between invariant boundaries
- Produces concrete exploit hypotheses via register_hypothesis tool
"""

from pathlib import Path
from typing import Optional, List

from kai.agents.base import BaseAgent
from kai.agents.utils import AgentType
from kai.agents import settings
from kai.schemas import (
    AgentResponse,
    MasterContext,
    Mission,
    Invariant,
    ChatMessage,
    Role,
    ActorMatrix,
    VarVocabEntry,
    ProtocolManifesto,
    ExploitCandidate,
)

# Path to toolcalling prompt template
TOOLCALLING_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "gamified_agent_prompt.txt"
)


class GamifiedAgentResult:
    """Result container for GamifiedAgent."""

    def __init__(
        self,
        mission_id: str,
        cluster_id: str,
        exploit_candidates: Optional[List[ExploitCandidate]] = None,
    ):
        self.mission_id = mission_id
        self.cluster_id = cluster_id
        self.exploit_candidates = exploit_candidates or []

    @property
    def has_findings(self) -> bool:
        return len(self.exploit_candidates) > 0

    def to_dict(self) -> dict:
        return {
            "mission_id": self.mission_id,
            "cluster_id": self.cluster_id,
            "exploit_candidates": [ec.model_dump() for ec in self.exploit_candidates],
        }


class GamifiedAgent(BaseAgent):
    """
    Agent that discovers exploitation opportunities through adversarial reasoning.

    Unlike invariant-checking agents, GamifiedAgent:
    1. Receives a cluster of related invariants
    2. Discovers what "winning" means from context
    3. Identifies gaps between invariants
    4. Produces concrete exploit hypotheses

    Domain-agnostic: works for Solidity, Rust, C++, etc.
    """

    def __init__(
        self,
        mission: Mission,
        master_context: MasterContext,
        invariant_cluster: List[Invariant],
        vars_in_scope: Optional[List[VarVocabEntry]] = None,
        actor_matrix: Optional[ActorMatrix] = None,
        protocol_manifesto: Optional[ProtocolManifesto] = None,
        dependency_graph=None,
        max_tool_turns: Optional[int] = None,
        repo_path: Optional[str] = None,
        use_vllm: bool = False,
        model: Optional[str] = settings.GAMIFIED_DEFAULT_MODEL,
        use_openai: bool = False,
        execution_id: Optional[str] = None,
    ):
        super().__init__(
            max_tool_turns=max_tool_turns,
            repo_path=repo_path,
            use_vllm=use_vllm,
            model=model,
            agent_type=AgentType.GAMIFIED,
            use_openai=use_openai,
        )

        self.mission = mission
        self.master_context = master_context
        self.invariant_cluster = invariant_cluster
        self.vars_in_scope = vars_in_scope or []
        self.actor_matrix = actor_matrix
        self.protocol_manifesto = protocol_manifesto
        self.dependency_graph = dependency_graph
        self.workspace_path: Optional[str] = repo_path

        # Result tracking (populated by register_hypothesis tool)
        self._exploit_candidates: List[ExploitCandidate] = []

        if execution_id:
            self.execution_id = execution_id

    def set_toolcalling_prompt(self, cluster_id: str = "default"):
        """
        Replace system prompt with the gamified agent template.

        Args:
            cluster_id: Identifier for this invariant cluster
        """
        try:
            template = TOOLCALLING_PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            template = (
                "You are GamifiedAgent. Discover exploitation opportunities "
                "by analyzing gaps between invariants."
            )

        # Format invariant cluster
        inv_lines = []
        for inv in self.invariant_cluster:
            inv_lines.append(f"- {inv.id} ({inv.type.value}): {inv.rule}")
            if inv.explanation:
                inv_lines.append(f"  Explanation: {inv.explanation}")
            if inv.target_function_ids:
                inv_lines.append(
                    f"  Target functions: {', '.join(inv.target_function_ids)}"
                )
            if inv.target_var_ids:
                inv_lines.append(f"  Target variables: {', '.join(inv.target_var_ids)}")
        invariant_cluster_text = (
            "\n".join(inv_lines) if inv_lines else "No invariants provided"
        )

        # Format variables in scope
        vars_lines = []
        for v in self.vars_in_scope:
            writers = ", ".join(v.writers) if v.writers else "none"
            readers = ", ".join(v.readers) if v.readers else "none"
            vars_lines.append(
                f"- {v.name} (in {v.container}): writers=[{writers}], readers=[{readers}]"
            )
        vars_text = "\n".join(vars_lines) if vars_lines else "No variables in scope"

        # Format actor context
        actor_lines = []
        if self.actor_matrix:
            for role in self.actor_matrix.roles:
                privs = [p.name for p in role.privileges[:5]]
                if len(role.privileges) > 5:
                    privs.append(f"... +{len(role.privileges) - 5} more")
                actor_lines.append(
                    f"- {role.name} (trust: {role.trust}): can call [{', '.join(privs)}]"
                )
        actor_text = (
            "\n".join(actor_lines) if actor_lines else "No actor information available"
        )

        # Format system context from manifesto
        context_lines = []
        if self.protocol_manifesto:
            if self.protocol_manifesto.name:
                context_lines.append(f"System: {self.protocol_manifesto.name}")
            if self.protocol_manifesto.purpose:
                context_lines.append(
                    f"Purpose: {self.protocol_manifesto.purpose[:200]}"
                )
            if self.protocol_manifesto.domain:
                context_lines.append(f"Domain: {self.protocol_manifesto.domain}")
            if self.protocol_manifesto.key_concepts:
                concepts = list(self.protocol_manifesto.key_concepts.keys())[:5]
                context_lines.append(f"Key concepts: {', '.join(concepts)}")
        system_context = (
            "\n".join(context_lines) if context_lines else "No system context available"
        )

        # Substitute template variables
        replacements = {
            "{{max_tool_turns}}": str(self.max_tool_turns),
            "{{invariant_cluster}}": invariant_cluster_text,
            "{{vars_in_scope}}": vars_text,
            "{{actor_context}}": actor_text,
            "{{system_context}}": system_context,
        }

        prompt = template
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)

        self.system_prompt = prompt
        self.messages = [ChatMessage(role=Role.SYSTEM, content=prompt)]

    def check_termination(self, response: str, python_code: str) -> bool:
        """GamifiedAgent terminates when model stops calling tools."""
        return False

    def get_tools_module(self) -> str:
        """Return the tools module for GamifiedAgent."""
        return "kai.agents.tools.gamified_tools"

    def extract_final_result(
        self, thoughts: str, python_code: str, response: str
    ) -> AgentResponse:
        """
        Extract the final result from GamifiedAgent's work.

        Reads exploit candidates from register_hypothesis tool calls.
        """
        import json
        from datetime import datetime

        mission_id = self.mission.mission_id if self.mission else "unknown"
        cluster_id = (
            f"cluster_{hash(tuple(inv.id for inv in self.invariant_cluster)) % 10000}"
        )

        # Build result from tool-populated state
        self.gamified_result = GamifiedAgentResult(
            mission_id=mission_id,
            cluster_id=cluster_id,
            exploit_candidates=self._exploit_candidates,
        )

        # Save findings to JSON
        if self._exploit_candidates:
            output_dir = (
                Path(self.repo_path).parent if self.repo_path else Path("output")
            )
            output_dir = output_dir / "findings" / "gamified"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{mission_id}_{timestamp}.json"

            output_data = {
                "mission_id": mission_id,
                "cluster_id": cluster_id,
                "invariants_in_cluster": [inv.id for inv in self.invariant_cluster],
                "agent_type": "gamified",
                "timestamp": timestamp,
                "result": self.gamified_result.to_dict(),
            }

            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

        return AgentResponse(
            thoughts=thoughts,
            python_block=python_code,
            master_context=self.master_context,
        )

    def get_result(self) -> Optional[GamifiedAgentResult]:
        """Get the full GamifiedAgentResult after chat() completes."""
        return getattr(self, "gamified_result", None)

    def get_exploit_candidates(self) -> List[ExploitCandidate]:
        """Get all exploit candidates discovered."""
        if not hasattr(self, "gamified_result"):
            return []
        return self.gamified_result.exploit_candidates
