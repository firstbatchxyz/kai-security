import re
from pathlib import Path
from typing import Optional

from kai.agents.agent_types import ProfilerAgent
from kai.processes.base import BaseProcess
from kai.schemas import (
    AgentResponse,
    ProfilerInput,
    ProfilerOutput,
    ProtocolManifesto,
)
from kai.utils.dependency.builders import SolidityBuilder
from kai.utils.dependency.graph import DependencyGraph


def build_dependency_graph(project_root: str) -> DependencyGraph:
    """
    Build a dependency graph for the project root.
    """
    return SolidityBuilder().build(project_root)


class ProfilerProcess(BaseProcess[ProfilerInput, ProfilerOutput]):
    """
    Process to run the ProfilerAgent and generate a ProtocolManifesto.
    """

    async def execute(self, input_data: ProfilerInput) -> ProfilerOutput:
        ctx = input_data.master_context
        repo_path = ctx.root_path

        dependency_graph: Optional[DependencyGraph] = input_data.dependency_graph
        graph_error: Optional[str] = None

        if dependency_graph is None:
            try:
                dependency_graph = build_dependency_graph(repo_path)
            except Exception as e:
                graph_error = str(e)
                # Fall back to an empty graph so tools remain available
                try:
                    dependency_graph = DependencyGraph(repo_path)
                except Exception:
                    raise ValueError("Failed to build dependency graph")

        agent = ProfilerAgent(
            master_context=ctx,
            dependency_graph=dependency_graph,
            repo_path=repo_path,
            model=input_data.model_name,
            max_tool_turns=input_data.num_turns,
            use_openai=input_data.use_openai,
            execution_id=input_data.execution_id,
        )

        response: Optional[AgentResponse] = None
        exception_msg = ""

        try:
            user_prompt = (
                "Profile the repository and produce a ProtocolManifesto.\n"
                "Use the provided MasterContext as authoritative repo info:\n"
                f"{ctx.model_dump_json(indent=2)}"
            )
            response = await agent.chat_with_tools(user_prompt)

            # If the agent terminated without registering a manifesto, nudge it.
            if response is not None and response.protocol_manifesto is None:
                retry_prompt = (
                    "FORMAT REQUIREMENT: You must call register_protocol_manifesto({...}) "
                    "with your findings. Call it now to finish."
                )
                response = await agent.chat_with_tools(retry_prompt)
        except Exception as e:
            exception_msg = str(e)
        finally:
            try:
                await agent.close()
            except Exception as e:
                raise e

        # Conversation saving handled by dispatcher via state_manager

        manifesto: Optional[ProtocolManifesto] = (
            response.protocol_manifesto if response else None
        )
        # Treat graph errors as non-fatal; if we got a manifesto, mark success.
        success = response is not None and manifesto is not None

        error_message = None
        if not success:
            error_message = (
                graph_error
                or exception_msg
                or "Profiler agent did not produce a ProtocolManifesto"
            )

        return ProfilerOutput(
            response=response,
            protocol_manifesto=manifesto,
            estimated_cost=agent.estimated_cost,
            total_tokens=agent.total_tokens,
            success=success,
            error_message=error_message,
            repo_path=repo_path,
        )

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def _repo_slug(self, repo_path: str) -> str:
        name = Path(repo_path).name or "repo"
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
        return safe_name
