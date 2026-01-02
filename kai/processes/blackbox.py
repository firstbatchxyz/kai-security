import json
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from kai.agents.agent_types import BlackboxAgent
from kai.processes.base import BaseProcess
from kai.schemas import (
    AgentResponse,
    BlackboxInput,
    BlackboxOutput,
    Observation,
)
from kai.utils.dependency.builders import SolidityBuilder
from kai.utils.dependency.graph import DependencyGraph


class BlackboxProcess(BaseProcess[BlackboxInput, BlackboxOutput]):
    """
    Process to run the BlackboxAgent and emit observations.
    """

    async def execute(self, input_data: BlackboxInput) -> BlackboxOutput:
        brief = input_data.campaign_brief
        ctx = brief.master_context
        if ctx is None:
            return BlackboxOutput(
                response=None,
                observations=[],
                estimated_cost=0.0,
                total_tokens={},
                success=False,
                error_message="CampaignBrief.master_context is required for BlackboxProcess",
                repo_path="",
            )
        ctx_root_path = ctx.root_path
        # Treat MasterContext.root_path as the build/analysis root. In Foundry flows, this
        # is the `forge --root ...` directory and is the only path guaranteed to have
        # the same compilation context as the campaign brief.
        repo_path = ctx_root_path

        # Ensure Foundry artifacts (cache/out) do not write into the target repo.
        # Some repos configure cache_path/out inside the repo and may have restrictive perms.
        campaign_label = getattr(brief, "label", None) or "CMP_UNKNOWN"
        # Use the MasterContext.root_path basename for output paths to keep tests stable
        # even when we normalize repo_path for tooling.
        repo_slug = self._repo_slug(ctx_root_path)
        foundry_root = (
            self._project_root()
            / "output"
            / "campaigns"
            / str(campaign_label)
            / repo_slug
            / "_foundry"
        )
        foundry_cache = foundry_root / "cache"
        foundry_out = foundry_root / "out"
        foundry_cache.mkdir(parents=True, exist_ok=True)
        foundry_out.mkdir(parents=True, exist_ok=True)

        prev_env_cache = os.environ.get("FOUNDRY_CACHE_PATH")
        prev_env_out = os.environ.get("FOUNDRY_OUT")
        os.environ["FOUNDRY_CACHE_PATH"] = str(foundry_cache)
        os.environ["FOUNDRY_OUT"] = str(foundry_out)

        try:
            graph_warning: Optional[str] = None
            try:
                dependency_graph = self._build_dependency_graph(
                    repo_path,
                    frameworks=(ctx.frameworks or []),
                )
            except Exception as e:
                # Blackbox should still run even if dependency-graph tooling fails.
                # Some targets are "contracts-only" directories without a compilation framework.
                graph_warning = f"Dependency graph build failed: {e}"
                dependency_graph = DependencyGraph(Path(repo_path).resolve())

            # Enforce brief + caller budgeting: blackbox should not exceed either.
            max_turns = min(
                int(input_data.num_turns),
                int(
                    getattr(getattr(brief, "budget", None), "max_turns_per_worker", 0)
                    or 0
                )
                or int(getattr(brief, "budget_turns", 0) or 0)
                or 0,
            )
            if max_turns <= 0:
                max_turns = int(input_data.num_turns)

            agent = BlackboxAgent(
                campaign_brief=brief,
                dependency_graph=dependency_graph,
                repo_path=repo_path,
                model=input_data.model_name,
                max_tool_turns=max_turns,
                use_openai=input_data.use_openai,
                execution_id=input_data.execution_id,
            )

            # External campaigns directory (outside the target repo)
            campaigns_root = (
                self._project_root()
                / "output"
                / "campaigns"
                / str(campaign_label)
                / repo_slug
                / agent.agent_id
            )
            campaigns_root.mkdir(parents=True, exist_ok=True)
            setattr(agent, "campaigns_dir", str(campaigns_root))

            response: Optional[AgentResponse] = None
            prefix = "blackbox"
            exception_msg = ""

            try:
                brief_dump = brief.model_dump(exclude={"dependency_graph"})
                user_prompt = (
                    "Run the blackbox campaign using the provided briefing.\n"
                    "Focus on the entrypoints_subset, respect the budget, "
                    "and record observations using the provided tools.\n"
                    "Record observations incrementally as you learn things (including negative results); do not wait until the end.\n"
                    "Do NOT emit <done>.\n"
                    f"CampaignBrief (dependency_graph omitted):\n"
                    f"{json.dumps(brief_dump, indent=2)}"
                )
                response = await agent.chat_with_tools(user_prompt)
            except Exception as e:
                exception_msg = str(e)
                prefix = "error_blackbox"
            finally:
                try:
                    await agent.close()
                except Exception:
                    pass

            # Save conversation via state manager
            convo_path = await self._save_conversation(
                agent_id=agent.agent_id,
                agent_type="blackbox",
                messages=[m.model_dump() for m in agent.messages],
                metadata={
                    "repo_path": repo_path,
                    "campaign_label": campaign_label,
                    "estimated_cost": agent.estimated_cost,
                    "total_tokens": agent.total_tokens,
                    "time_spent": agent.time_spent,
                    "model": agent.model,
                    "prefix": prefix,
                },
            )

            observations: List[Observation] = getattr(
                agent, "blackbox_observations", []
            )

            success = response is not None and not exception_msg
            error_message = exception_msg or None
            if response is None and not error_message:
                error_message = "Blackbox agent did not return a response"
            if success and graph_warning and not error_message:
                error_message = graph_warning

            # Save structured findings next to the conversation file for easy consumption.
            try:
                if convo_path:
                    # Keep results adjacent to the convo file but avoid double extensions:
                    # blackbox_<id>.json -> blackbox_<id>.results.json
                    if convo_path.endswith(".json"):
                        results_path = convo_path[: -len(".json")] + ".results.json"
                    else:
                        results_path = f"{convo_path}.results.json"
                    payload = {
                        "agent_id": agent.agent_id,
                        "repo_path": repo_path,
                        "campaign_label": campaign_label,
                        "success": success,
                        "error_message": error_message,
                        "estimated_cost": agent.estimated_cost,
                        "total_tokens": agent.total_tokens,
                        "observations": [o.model_dump() for o in observations],
                        "conversation_path": convo_path,
                    }
                    with open(results_path, "w") as f:
                        json.dump(payload, f, indent=2)
            except Exception:
                # Never fail the run due to result persistence issues.
                pass

            return BlackboxOutput(
                response=response,
                observations=observations,
                estimated_cost=agent.estimated_cost,
                total_tokens=agent.total_tokens,
                success=success,
                error_message=error_message,
                repo_path=repo_path,
            )
        finally:
            # Restore env to avoid cross-test leakage
            if prev_env_cache is None:
                os.environ.pop("FOUNDRY_CACHE_PATH", None)
            else:
                os.environ["FOUNDRY_CACHE_PATH"] = prev_env_cache
            if prev_env_out is None:
                os.environ.pop("FOUNDRY_OUT", None)
            else:
                os.environ["FOUNDRY_OUT"] = prev_env_out

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def _repo_slug(self, repo_path: str) -> str:
        name = Path(repo_path).name or "repo"
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
        return safe_name

    def _build_dependency_graph(
        self, repo_path: str, frameworks: Optional[List[str]] = None
    ):
        """
        Build a real dependency graph so GraphQueryEngine tools are usable.
        """
        try:
            # Slither treats directory targets as invalid unless it knows which compilation
            # framework to use. Our MasterContext already carries that signal.
            fw = {str(x).lower() for x in (frameworks or [])}

            # Heuristic fallback: Foundry projects often have lib/test/out directories
            # even when foundry.toml is absent (forge defaults).
            repo = Path(repo_path).resolve()
            looks_like_foundry = (
                repo.is_dir() and (repo / "lib").exists() and (repo / "test").exists()
            )

            slither_kwargs: Optional[Dict[str, Any]] = None
            if "foundry" in fw or looks_like_foundry:
                slither_kwargs = {"compile_force_framework": "foundry"}

            if slither_kwargs:
                return SolidityBuilder().build(repo_path, slither_kwargs=slither_kwargs)

            return SolidityBuilder().build(repo_path)
        except Exception as e:
            # Propagate so caller can mark the run as failed
            raise RuntimeError(str(e))
