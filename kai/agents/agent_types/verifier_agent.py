"""
VerifierAgent - Validates exploit findings from State/Quant agents.

Analyzes PoC code, economic feasibility, and determines validity + severity.
Outputs a structured Verdict.
"""

from pathlib import Path
from typing import Optional

from kai.agents.base import BaseAgent
from kai.agents.utils import AgentType
from kai.agents import settings
from kai.schemas import (
    AgentResponse,
    MasterContext,
    ExploitCandidate,
    Invariant,
    Verdict,
    VerdictSeverity,
    ChatMessage,
    Role,
)

# Path to toolcalling prompt template
TOOLCALLING_PROMPT_PATH = (
    Path(__file__).parent.parent.parent / "prompts" / "verifier_agent_prompt.txt"
)


class VerifierAgentResult:
    """Result container for VerifierAgent."""

    def __init__(
        self,
        exploit_candidate: ExploitCandidate,
        verdict: Optional[Verdict] = None,
    ):
        self.exploit_candidate = exploit_candidate
        self.verdict = verdict

    @property
    def is_valid(self) -> bool:
        """Return True if verdict confirms exploit is valid."""
        return self.verdict.is_valid if self.verdict else False

    @property
    def severity(self) -> Optional[VerdictSeverity]:
        """Return severity if verdict exists."""
        return self.verdict.severity if self.verdict else None


class VerifierAgent(BaseAgent):
    """
    Agent that validates exploit findings from State/Quant agents.

    Given an ExploitCandidate, it:
    1. Analyzes the PoC code for mock/fake components
    2. Verifies the exploit targets real implementations
    3. Evaluates economic feasibility
    4. Determines if it's a known limitation or real bug
    5. Assigns severity and outputs a Verdict
    """

    def __init__(
        self,
        exploit_candidate: ExploitCandidate,
        invariant: Invariant,
        master_context: MasterContext,
        dependency_graph=None,
        max_tool_turns: Optional[int] = None,
        repo_path: Optional[str] = None,
        use_vllm: bool = False,
        model: Optional[str] = settings.VERIFIER_DEFAULT_MODEL,
        use_openai: bool = False,
        execution_id: Optional[str] = None,
    ):
        # Initialize with minimal system prompt - will be replaced by set_toolcalling_prompt()
        super().__init__(
            max_tool_turns=max_tool_turns or settings.VERIFIER_MAX_TURNS,
            repo_path=repo_path,
            use_vllm=use_vllm,
            model=model,
            agent_type=AgentType.VERIFIER,
            use_openai=use_openai,
        )

        # Store context
        self.exploit_candidate = exploit_candidate
        self.invariant = invariant
        self.master_context = master_context
        self.dependency_graph = dependency_graph
        self.workspace_path: Optional[str] = None

        if execution_id:
            self.execution_id = execution_id

        # Will be populated by submit_verdict tool
        self._verdict: Optional[Verdict] = None

    def set_toolcalling_prompt(self):
        """
        Replace system prompt with the native toolcalling template.
        """
        try:
            template = TOOLCALLING_PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            template = (
                "You are VerifierAgent. Validate exploit findings and output a Verdict."
            )

        # Substitute template variables
        replacements = {
            "{{max_tool_turns}}": str(self.max_tool_turns),
            "{{mission_id}}": self.exploit_candidate.mission_id,
            "{{invariant_id}}": self.exploit_candidate.invariant_id,
            "{{invariant_rule}}": self.invariant.rule if self.invariant else "N/A",
            "{{mechanism}}": self.exploit_candidate.mechanism,
            "{{poc_path}}": self.exploit_candidate.target_file or "N/A",
            "{{worker_reasoning}}": self.exploit_candidate.description,
            "{{poc_code}}": self.exploit_candidate.poc_code or "No PoC code provided",
        }
        prompt = template
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)

        # Replace system message
        self.system_prompt = prompt
        self.messages = [ChatMessage(role=Role.SYSTEM, content=prompt)]

    def check_termination(self, response: str, python_code: str) -> bool:
        """VerifierAgent terminates when verdict is submitted or no more tool calls."""
        return self._verdict is not None

    def get_tools_module(self) -> str:
        """Return the tools module for VerifierAgent."""
        return "kai.agents.tools.verifier_tools"

    def extract_final_result(
        self, thoughts: str, python_code: str, response: str
    ) -> AgentResponse:
        """
        Extract the final result from VerifierAgent's work.

        Reads verdict from _verdict populated by submit_verdict tool.
        Verdict persistence is handled by Dispatcher.export_results().
        """
        # Store result for later retrieval
        self.verifier_result = VerifierAgentResult(
            exploit_candidate=self.exploit_candidate,
            verdict=self._verdict,
        )

        return AgentResponse(
            thoughts=thoughts,
            python_block=python_code,
            master_context=self.master_context,
        )

    def get_verdict(self) -> Optional[Verdict]:
        """Get the Verdict after chat() completes."""
        return self._verdict

    def get_result(self) -> Optional[VerifierAgentResult]:
        """Get the full VerifierAgentResult after chat() completes."""
        return getattr(self, "verifier_result", None)
