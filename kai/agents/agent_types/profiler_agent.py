import json
from typing import Optional

from kai.agents.base import BaseAgent
from kai.agents.utils import AgentType, check_done
from kai.schemas import AgentResponse, MasterContext, ProtocolManifesto


class ProfilerAgent(BaseAgent):
    """
    Agent that profiles a repository and emits a ProtocolManifesto.
    """

    def __init__(
        self,
        master_context: MasterContext,
        dependency_graph=None,
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
            agent_type=AgentType.PROFILER,
            use_openai=use_openai,
        )

        # Expose master context and dependency graph for tools and prompt grounding
        self.master_context = master_context
        self.dependency_graph = dependency_graph

        if execution_id:
            self.execution_id = execution_id

    @staticmethod
    def _extract_manifesto(response: str) -> Optional[ProtocolManifesto]:
        """
        Parse ProtocolManifesto JSON embedded inside <done>...</done>.
        """
        if "<done>" not in response or "</done>" not in response:
            return None

        payload = response.split("<done>")[1].split("</done>")[0].strip()
        if not payload:
            return None

        try:
            data = json.loads(payload)
            return ProtocolManifesto(**data)
        except Exception:
            return None

    def check_termination(self, response: str, python_code: str) -> bool:
        """
        Profiler agent terminates when it produces a done block (with or without manifesto).
        """
        done_present = check_done(response)
        return bool(done_present and not python_code)

    def get_tools_module(self) -> str:
        """
        Get the tools module for profiler agent.
        """
        return "kai.agents.tools.profiler_tools"

    def extract_final_result(
        self, thoughts: str, python_code: str, response: str
    ) -> AgentResponse:
        """
        Extract the final result for profiler agent, including ProtocolManifesto if present.
        """
        # Prefer registered manifesto from tool call
        manifesto = getattr(self, "_registered_protocol_manifesto", None)

        # Fallback to parsing <done> block for backward compatibility
        if manifesto is None:
            manifesto = self._extract_manifesto(response)

        return AgentResponse(
            thoughts=thoughts,
            python_block=python_code,
            protocol_manifesto=manifesto,
            master_context=self.master_context,
        )
