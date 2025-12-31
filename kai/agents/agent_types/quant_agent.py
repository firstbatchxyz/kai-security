"""
QuantAgent - Invariant-driven numeric/mathematical exploitation worker.

Given a numeric invariant (solvency, conservation, equations), finds input values
that violate it. Produces compiled PoCs or returns None if no violation found.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

from kai.agents.base import BaseAgent
from kai.agents.utils import AgentType
from kai.schemas import (
    AgentResponse,
    MasterContext,
    Mission,
    ExploitCandidate,
    Invariant,
    ChatMessage,
    Role,
)

# Path to toolcalling prompt template
TOOLCALLING_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "quant_agent_prompt.txt"
)


class QuantAgentResult:
    """Result container for QuantAgent - supports multiple findings."""

    def __init__(
        self,
        mission_id: str,
        findings: Optional[List[Dict[str, Any]]] = None,
        test_attempts: int = 0,
        compile_attempts: int = 0,
    ):
        self.mission_id = mission_id
        self.findings = findings or []
        self.test_attempts = test_attempts
        self.compile_attempts = compile_attempts

    @property
    def exploits(self) -> List[Dict[str, Any]]:
        """Return only findings where exploit_found=True."""
        return [f for f in self.findings if f.get("exploit_found")]

    @property
    def verifications(self) -> List[Dict[str, Any]]:
        """Return only findings where exploit_found=False (verification tests)."""
        return [f for f in self.findings if not f.get("exploit_found")]

    @property
    def has_exploit(self) -> bool:
        """Return True if any exploit was found."""
        return len(self.exploits) > 0

    def to_exploit_candidates(self, worker_id: str) -> List[ExploitCandidate]:
        """Convert all exploits to ExploitCandidates."""
        candidates = []
        for f in self.exploits:
            candidates.append(
                ExploitCandidate(
                    mission_id=self.mission_id,
                    worker_id=worker_id,
                    invariant_id=f.get("invariant_id", ""),
                    mechanism=f.get("mechanism", "numeric_violation"),
                    poc_code=f.get("poc_code", ""),
                    target_file=f.get("poc_path", ""),
                    target_function="",
                    description=f.get("reasoning", ""),
                    compiled=True,
                    logs=[],
                )
            )
        return candidates


class QuantAgent(BaseAgent):
    """
    Agent that attempts to break numeric/mathematical invariants.

    Given a mission with an invariant, it:
    1. Analyzes the target using dependency graph tools
    2. Identifies arithmetic operations and potential vulnerabilities
    3. Hypothesizes attack vectors (overflow, precision loss, boundary conditions)
    4. Writes and compiles PoC tests with concrete numeric values
    5. Returns ExploitCandidate if successful, None otherwise
    """

    def __init__(
        self,
        mission: Mission,
        master_context: MasterContext,
        dependency_graph=None,
        max_tool_turns: Optional[int] = None,
        repo_path: Optional[str] = None,
        use_vllm: bool = False,
        model: Optional[str] = None,
        use_openai: bool = False,
        execution_id: Optional[str] = None,
    ):
        # Initialize with minimal system prompt - will be replaced by set_toolcalling_prompt()
        super().__init__(
            max_tool_turns=max_tool_turns,
            repo_path=repo_path,
            use_vllm=use_vllm,
            model=model,
            agent_type=AgentType.QUANT,
            use_openai=use_openai,
        )

        # Store mission context
        self.mission = mission
        self.master_context = master_context
        self.dependency_graph = dependency_graph
        self.workspace_path: Optional[str] = repo_path

        if execution_id:
            self.execution_id = execution_id

    def set_toolcalling_prompt(
        self,
        invariant: Invariant,
        actor_context: str = "No actor context available.",
    ):
        """
        Replace system prompt with the native toolcalling template.

        This should be called before chat_with_tools() to set up
        the proper prompt format for native OpenAI tool calling.

        Args:
            invariant: The invariant to embed in the prompt
            actor_context: Formatted string of relevant actor roles
        """
        try:
            template = TOOLCALLING_PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            template = (
                "You are QuantAgent. Find numeric invariant violations and write PoCs."
            )

        # Substitute template variables
        replacements = {
            "{{max_tool_turns}}": str(self.max_tool_turns),
            "{{invariant_label}}": invariant.label,
            "{{invariant_type}}": invariant.type.value if invariant.type else "unknown",
            "{{invariant_rule}}": invariant.rule,
            "{{invariant_explanation}}": invariant.explanation or "N/A",
            "{{target_function_ids}}": ", ".join(invariant.target_function_ids)
            if invariant.target_function_ids
            else "N/A",
            "{{target_var_ids}}": ", ".join(invariant.target_var_ids)
            if invariant.target_var_ids
            else "N/A",
            "{{actor_context}}": actor_context,
        }
        prompt = template
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)

        # Replace system message
        self.system_prompt = prompt
        self.messages = [ChatMessage(role=Role.SYSTEM, content=prompt)]

    def check_termination(self, response: str, python_code: str) -> bool:
        """QuantAgent with native tool calling terminates when model stops calling tools."""
        # Termination is handled by chat_with_tools() when no more tool calls
        return False

    def get_tools_module(self) -> str:
        """Return the tools module for QuantAgent."""
        return "kai.agents.tools.quant_tools"

    def extract_final_result(
        self, thoughts: str, python_code: str, response: str
    ) -> AgentResponse:
        """
        Extract the final result from QuantAgent's work.

        Reads findings from _registered_exploits populated by register_exploit tool.
        """
        import json
        from datetime import datetime

        # Get findings from register_exploit tool calls
        registered = getattr(self, "_registered_exploits", [])
        mission_id = self.mission.mission_id if self.mission else "unknown"

        # Store result for later retrieval
        self.quant_result = QuantAgentResult(
            mission_id=mission_id,
            findings=registered,
            test_attempts=getattr(self, "_test_attempts", 0),
            compile_attempts=getattr(self, "_compile_attempts", 0),
        )

        # Save findings to JSON
        if registered:
            output_dir = (
                Path(self.repo_path).parent if self.repo_path else Path("output")
            )
            output_dir = output_dir / "findings"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{mission_id}_{timestamp}.json"

            output_data = {
                "mission_id": mission_id,
                "invariant_id": self.mission.invariant_id if self.mission else None,
                "invariant_rule": self.mission.invariant.rule
                if self.mission and self.mission.invariant
                else None,
                "agent_type": "quant",
                "timestamp": timestamp,
                "test_attempts": getattr(self, "_test_attempts", 0),
                "compile_attempts": getattr(self, "_compile_attempts", 0),
                "findings": registered,
            }

            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2, default=str)

        return AgentResponse(
            thoughts=thoughts,
            python_block=python_code,
            master_context=self.master_context,
        )

    def get_exploit_candidates(self) -> List[ExploitCandidate]:
        """
        Get all ExploitCandidates found.

        Call this after chat() completes.
        """
        if not hasattr(self, "quant_result"):
            return []
        return self.quant_result.to_exploit_candidates(self.agent_id)

    def get_result(self) -> Optional[QuantAgentResult]:
        """Get the full QuantAgentResult after chat() completes."""
        return getattr(self, "quant_result", None)
