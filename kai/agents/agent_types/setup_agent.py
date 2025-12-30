import json
from typing import Optional

from kai.agents.base import BaseAgent
from kai.agents.utils import AgentType, check_done
from kai.schemas import AgentResponse, MasterContext


class SetupAgent(BaseAgent):
    """Agent for setting up a codebase and emitting MasterContext."""

    def __init__(
        self,
        max_tool_turns: Optional[int] = None,
        repo_path: Optional[str] = None,
        use_vllm: bool = False,
        model: Optional[str] = None,
        use_openai: bool = False,
        execution_id: Optional[str] = None,
    ):
        super().__init__(
            max_tool_turns=max_tool_turns,
            repo_path=repo_path,
            use_vllm=use_vllm,
            model=model,
            agent_type=AgentType.SETUP,
            use_openai=use_openai,
        )

        if execution_id:
            self.execution_id = execution_id

    @staticmethod
    def _extract_master_context(response: str) -> Optional[MasterContext]:
        """
        Parse MasterContext JSON embedded inside <done>...</done>.
        """
        if "<done>" not in response or "</done>" not in response:
            return None

        payload = response.split("<done>")[1].split("</done>")[0].strip()
        if not payload:
            return None

        try:
            data = json.loads(payload)
            return MasterContext(**data)
        except Exception:
            return None

    def check_termination(self, response: str, python_code: str) -> bool:
        """
        Setup agent terminates when it produces a done block (with or without MasterContext).
        """
        done_present = check_done(response)
        return bool(done_present and not python_code)

    def get_tools_module(self) -> str:
        """
        Get the tools module for setup agent.
        """
        return "kai.agents.tools.setup_tools"

    def extract_final_result(
        self, thoughts: str, python_code: str, response: str
    ) -> AgentResponse:
        """
        Extract the final result for setup agent, including MasterContext if present.
        """
        # Prefer registered context from tool call
        master_context = getattr(self, "_registered_master_context", None)

        # Fallback to parsing <done> block for backward compatibility
        if master_context is None:
            master_context = self._extract_master_context(response)

        return AgentResponse(
            thoughts=thoughts,
            python_block=python_code,
            master_context=master_context,
        )
