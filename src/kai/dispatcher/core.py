"""
Dispatcher: Mission control for Kai v2.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from kai.agents import settings
from kai.state_manager import KaiStateManager

from kai.schemas import (
    ActorMatrix,
    CampaignBrief,
    CampaignBudget,
    ExploitCandidate,
    Fix,
    Invariant,
    InvariantType,
    MasterContext,
    Mission,
    MissionAgentType,
    Observation,
    ProtocolManifesto,
    Verdict,
    VerdictSeverity,
    VerifierProcessInput,
    WorkspacePreset,
)
from kai.utils.dependency.graph import DependencyGraph

from kai.dispatcher.planner import MissionPlanner
from kai.dispatcher.workspace import WorkspaceManager
from kai.dispatcher.agent_factories import AGENT_FACTORIES as DEFAULT_AGENT_FACTORIES

# Type alias for shutdown trigger callable
ShutdownTrigger = Callable[[], bool]

if TYPE_CHECKING:
    from kai.agents.base import BaseAgent

logger = logging.getLogger(__name__)

# Type alias for agent factory function
AgentFactory = Callable[..., "BaseAgent"]


@dataclass
class DispatcherConfig:
    """Configuration for Dispatcher."""

    max_concurrent_agents: int = settings.MAX_CONCURRENT_AGENTS
    max_invariants_per_cluster: int = 5
    max_campaigns: int = 10
    include_exploration: bool = True
    default_budget: CampaignBudget = field(default_factory=CampaignBudget)
    workspace_dir: str = "./kai_workspaces"  # TODO: check if it respects workspace_dir
    # Model settings for agents
    model: str = settings.MAIN_DEFAULT_MODEL
    verifier_model: str = settings.VERIFIER_DEFAULT_MODEL
    fixer_model: str = settings.FIXER_DEFAULT_MODEL
    fallback_model: str = settings.FALLBACK_MODEL  # Fallback when primary model fails
    use_openai: bool = False
    # Rollout saving
    save_rollouts: bool = False
    rollouts_dir: Optional[str] = None  # If None, uses workspace_dir/rollouts
    # Output directory for results (invariants.json, report.json, etc.)
    # If None, derives from rollouts_dir parent or falls back to workspace_dir
    output_dir: Optional[str] = None
    # Setup agent settings
    setup_model: str = settings.SETUP_DEFAULT_MODEL
    setup_max_turns: int = settings.SETUP_MAX_TURNS
    # Profiler agent settings
    profiler_max_turns: int = settings.PROFILER_MAX_TURNS
    # Main agent settings (state, quant, blackbox, gamified)
    main_agent_max_turns: int = settings.DEFAULT_MAX_TURNS
    # Fixer agent settings
    fixer_max_turns: int = settings.DEFAULT_MAX_TURNS
    # Verifier agent settings
    verifier_max_turns: int = settings.VERIFIER_MAX_TURNS
    # Invariant synthesizer settings
    invariant_synth_max_turns: int = settings.INVARIANT_SYNTH_MAX_TURNS
    # Invariant process settings
    max_concurrent_invariant_chunks: int = 10  # Max parallel LLM calls for vocab chunks
    # Workspace validation settings
    validation_max_turns: int = settings.VALIDATION_MAX_TURNS
    # Disable gamified agents (useful for BountyBench)
    disable_gamified: bool = False
    # Extra instructions to pass to agents (e.g., CWE hints)
    extra_instructions: Optional[str] = None
    # Skip workspace validation (useful when context is pre-validated)
    skip_workspace_validation: bool = False
    # Number of blackbox missions to create (for HTTP/exploratory testing)
    blackbox_missions: int = 1


class Dispatcher:
    """
    Mission control: turns global knowledge into precise agent missions.

    Orchestrates preprocessing, mission planning, and agent dispatch.

    Agent factories are callables that create BaseAgent instances for missions.
    Factory signature: (mission: Mission, workspace_path: str) -> BaseAgent

    Example:
        def create_state_agent(mission: Mission, workspace: str) -> BaseAgent:
            return StateAgent(repo_path=workspace, max_tool_turns=mission.max_turns)

        dispatcher = Dispatcher(
            agent_factories={MissionAgentType.STATE: create_state_agent},
            config=DispatcherConfig(),
        )
    """

    def __init__(
        self,
        agent_factories: Optional[Dict[MissionAgentType, AgentFactory]] = None,
        config: Optional[DispatcherConfig] = None,
        shutdown_trigger: Optional[ShutdownTrigger] = None,
        state_manager: Optional[KaiStateManager] = None,
    ):
        # Use default factories if none provided
        self.agent_factories = agent_factories or dict(DEFAULT_AGENT_FACTORIES)
        self.config = config or DispatcherConfig()

        self.logger = logger.getChild("Dispatcher")

        # External state manager for persistence (mongo/s3/local)
        self._state_manager = state_manager

        # Shutdown trigger - callable that returns True when shutdown requested
        self._shutdown_trigger = shutdown_trigger
        self._shutdown_requested = False
        self._shutdown_reason: Optional[str] = None

        # State populated by boot()
        self.master_context: Optional[MasterContext] = None
        self.dependency_graph: Optional[DependencyGraph] = None
        self.protocol_manifesto: Optional[ProtocolManifesto] = None
        self.actor_matrix: Optional[ActorMatrix] = None
        # Store invariants keyed by ID (Pydantic models are not reliably hashable)
        self.invariants: Dict[str, Invariant] = {}

        # Mission queue: (priority, mission_id, mission)
        self.mission_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_missions: Dict[str, Mission] = {}

        # Results
        self.campaigns: List[CampaignBrief] = []
        self.completed_missions: List[Mission] = []
        self.exploit_candidates: List[ExploitCandidate] = []
        self.verdicts: List[Verdict] = []
        self.fixes: List[Fix] = []

        # Aggregated token/cost tracking across all agents
        self.total_tokens: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        self.total_cost: float = 0.0
        # Per-phase breakdown for detailed reporting
        self.token_usage_by_phase: Dict[str, Dict[str, Any]] = {}

        self._workspace_manager = WorkspaceManager(
            workspace_dir=self.config.workspace_dir, logger=self.logger
        )
        self._planner: Optional[MissionPlanner] = None

    async def _persist(self, coro) -> bool:
        """Safely call state manager method. No-op if no state manager."""
        if not self._state_manager:
            return True
        try:
            return await coro
        except Exception as e:
            self.logger.warning(f"State persistence failed: {e}")
            return False

    def _save_rollout(
        self,
        agent: Any,
        rollout_type: str,
        identifier: str,
    ) -> None:
        """
        Save agent conversation rollout to disk.

        Args:
            agent: The agent with messages attribute
            rollout_type: Type of rollout (e.g., "mission", "verifier")
            identifier: Unique identifier (e.g., mission_id)
        """
        if not self.config.save_rollouts:
            return

        import json
        from pathlib import Path

        # Determine rollouts directory
        rollouts_dir = self.config.rollouts_dir
        if not rollouts_dir:
            rollouts_dir = str(Path(self.config.workspace_dir) / "rollouts")

        # Create directory structure: rollouts/{type}/{identifier}.json
        rollout_path = Path(rollouts_dir) / rollout_type
        rollout_path.mkdir(parents=True, exist_ok=True)

        # Extract messages from agent
        messages = getattr(agent, "messages", [])
        if not messages:
            return

        # Serialize messages
        serialized = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                serialized.append(msg.model_dump())
            elif hasattr(msg, "__dict__"):
                serialized.append(msg.__dict__)
            else:
                serialized.append(str(msg))

        # Build rollout data
        rollout_data = {
            "identifier": identifier,
            "type": rollout_type,
            "model": getattr(agent, "model", "unknown"),
            "agent_type": str(getattr(agent, "agent_type", "unknown")),
            "messages": serialized,
            "total_tokens": getattr(agent, "total_tokens", {}),
            "estimated_cost": getattr(agent, "estimated_cost", 0.0),
        }

        # Write to file
        output_file = rollout_path / f"{identifier}.json"
        try:
            with open(output_file, "w") as f:
                json.dump(rollout_data, f, indent=2, default=str)
            self.logger.debug(f"Saved rollout: {output_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save rollout {identifier}: {e}")

    def _save_verifier_rollout(
        self,
        mission_id: str,
        messages: List[Any],
        model: str,
        total_tokens: Dict[str, int],
        estimated_cost: float,
    ) -> None:
        """
        Save verifier conversation rollout to disk.

        Similar to _save_rollout but takes messages directly instead of agent object.
        """
        if not self.config.save_rollouts:
            return

        import json
        from pathlib import Path

        # Determine rollouts directory
        rollouts_dir = self.config.rollouts_dir
        if not rollouts_dir:
            rollouts_dir = str(Path(self.config.workspace_dir) / "rollouts")

        # Create directory structure: rollouts/verifier/{mission_id}.json
        rollout_path = Path(rollouts_dir) / "verifier"
        rollout_path.mkdir(parents=True, exist_ok=True)

        # Serialize messages
        serialized = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                serialized.append(msg.model_dump())
            elif hasattr(msg, "__dict__"):
                serialized.append(msg.__dict__)
            else:
                serialized.append(str(msg))

        # Build rollout data
        rollout_data = {
            "identifier": f"verify_{mission_id}",
            "type": "verifier",
            "model": model,
            "agent_type": "verifier",
            "messages": serialized,
            "total_tokens": total_tokens,
            "estimated_cost": estimated_cost,
        }

        # Write to file
        output_file = rollout_path / f"verify_{mission_id}.json"
        try:
            with open(output_file, "w") as f:
                json.dump(rollout_data, f, indent=2, default=str)
            self.logger.debug(f"Saved verifier rollout: {output_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save verifier rollout {mission_id}: {e}")

    def _aggregate_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        phase: str,
        agent_type: str,
    ) -> None:
        """
        Core method to aggregate token usage into dispatcher totals.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cost: Estimated cost in USD
            phase: Phase name (e.g., "boot", "run_loop")
            agent_type: Type of agent/process (e.g., "state", "verifier")
        """
        # Update dispatcher totals
        self.total_tokens["prompt_tokens"] += prompt_tokens
        self.total_tokens["completion_tokens"] += completion_tokens
        self.total_cost += cost

        # Track per-phase breakdown
        if phase not in self.token_usage_by_phase:
            self.token_usage_by_phase[phase] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0.0,
                "by_agent_type": {},
            }

        phase_data = self.token_usage_by_phase[phase]
        phase_data["prompt_tokens"] += prompt_tokens
        phase_data["completion_tokens"] += completion_tokens
        phase_data["cost"] += cost

        # Track by agent type within phase
        if agent_type not in phase_data["by_agent_type"]:
            phase_data["by_agent_type"][agent_type] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost": 0.0,
                "count": 0,
            }

        agent_type_data = phase_data["by_agent_type"][agent_type]
        agent_type_data["prompt_tokens"] += prompt_tokens
        agent_type_data["completion_tokens"] += completion_tokens
        agent_type_data["cost"] += cost
        agent_type_data["count"] += 1

    def _aggregate_agent_usage(
        self,
        agent: Any,
        phase: str,
        agent_type: str = "unknown",
    ) -> None:
        """
        Aggregate token usage and cost from an agent into dispatcher totals.

        Args:
            agent: The agent with total_tokens and estimated_cost attributes
            phase: Phase name (e.g., "boot", "run_loop")
            agent_type: Type of agent (e.g., "state", "quant", "verifier")
        """
        agent_tokens = getattr(agent, "total_tokens", {})
        self._aggregate_usage(
            prompt_tokens=agent_tokens.get("prompt_tokens", 0),
            completion_tokens=agent_tokens.get("completion_tokens", 0),
            cost=getattr(agent, "estimated_cost", 0.0),
            phase=phase,
            agent_type=agent_type,
        )

    def _aggregate_process_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        phase: str,
        agent_type: str = "process",
    ) -> None:
        """
        Aggregate token usage from a process (not an agent) into dispatcher totals.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            cost: Estimated cost in USD
            phase: Phase name (e.g., "boot")
            agent_type: Type of process (e.g., "profiler", "actor", "invariant")
        """
        self._aggregate_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            phase=phase,
            agent_type=agent_type,
        )

    async def boot(
        self,
        repo_url: Optional[str] = None,
        repo_path: Optional[str] = None,
        model_name: str = settings.MAIN_DEFAULT_MODEL,
        use_openai: bool = False,
        master_context: Optional[MasterContext] = None,
    ) -> bool:
        """
        Run preprocess chain to populate global knowledge.

        Chain: EnvironmentSetup - StaticAnalysis - Profiler - ActorProcess - InvariantProcess

        Args:
            repo_url: Git URL of the repository
            repo_path: Local path to the repository (overrides repo_url)
            model_name: Model to use for agent inference
            use_openai: Whether to use OpenAI API directly
            master_context: Pre-built MasterContext (BountyBench mode, skips EnvironmentSetup)
        """
        self.logger.info("Booting Dispatcher...")

        from kai.processes.envsetup import EnvironmentSetupProcess
        from kai.processes.profiler import ProfilerProcess
        from kai.processes.actors import ActorProcess
        from kai.processes.invariants import InvariantProcess
        from kai.schemas import (
            EnvironmentSetupInput,
            ProfilerInput,
            ActorMatrixInput,
            InvariantProcessInput,
        )

        try:
            # Always run EnvironmentSetupProcess to discover paths
            # If master_context is provided (BountyBench mode), use its root_path
            self.logger.info("Step 1/6: Environment Setup...")

            # Determine repo path - use provided master_context.root_path if available
            effective_repo_path = repo_path
            if master_context and master_context.root_path:
                effective_repo_path = master_context.root_path
                self.logger.info(f"Using provided root_path: {effective_repo_path}")

            env_process = EnvironmentSetupProcess(
                MasterContext(
                    root_path=effective_repo_path or "./", compile_success=True
                )
            )
            env_input = EnvironmentSetupInput(
                repo_url=repo_url or "",
                num_turns=self.config.setup_max_turns,
                model_name=self.config.setup_model,
                use_openai=use_openai,
                repo_path_override=effective_repo_path,
            )
            env_output = await env_process.run(env_input)

            if not env_output.success or not env_output.master_context:
                self.logger.error(
                    f"Environment setup failed: {env_output.error_message}"
                )
                return False

            self.master_context = env_output.master_context

            # Preserve adapter from provided master_context if set
            if master_context and master_context.adapter:
                self.master_context.adapter = master_context.adapter

            self.logger.info(f"MasterContext ready: {self.master_context.root_path}")

            # Persist master context
            await self._persist(
                self._state_manager.save_master_context(self.master_context)
                if self._state_manager
                else None
            )

            # Persist state transition
            await self._persist(
                self._state_manager.update_state("setup")
                if self._state_manager
                else None
            )

            # Build DependencyGraph BEFORE workspace validation to avoid cache conflicts
            # (Workspace validation runs forge which creates cache files)
            self.logger.info("Step 2/6: Building DependencyGraph...")
            self.dependency_graph = await self._build_dependency_graph()
            if not self.dependency_graph:
                self.logger.error("Static analysis failed")
                return False

            # Persist dependency graph
            await self._persist(
                self._state_manager.save_dependency_graph(
                    self.dependency_graph.to_dict()
                )
                if self._state_manager
                else None
            )

            if not self.config.skip_workspace_validation:
                self.logger.info("Step 3/6: Workspace Validation...")
                from kai.processes.workspace_validation import (
                    WorkspaceValidationProcess,
                )
                from kai.schemas import WorkspaceValidationInput
                from kai.utils.workspace import get_workspace_adapter

                # Get validation presets from adapter (framework-specific)
                if not self.master_context or not self.master_context.frameworks:
                    raise RuntimeError("MasterContext.frameworks is required for workspace validation")
                framework = self.master_context.frameworks[0]
                adapter = get_workspace_adapter(framework)
                validation_presets = adapter.get_validation_presets()

                ws_output = await WorkspaceValidationProcess(
                    context=self.master_context, workspace_dir=self.config.workspace_dir
                ).run(
                    WorkspaceValidationInput(
                        master_context=self.master_context,
                        presets=validation_presets,
                        timeout_compile_s=120,
                        timeout_test_s=120,
                        max_turns=self.config.validation_max_turns,
                    )
                )
                if not ws_output.success:
                    self.logger.error(
                        ws_output.error_message or "Workspace validation failed"
                    )
                    # Log per-preset summary for debugging
                    for r in ws_output.results:
                        self.logger.error(
                            f"WorkspaceValidation {r.preset.value}: "
                            f"compiled={r.compiled}, test_success={r.test_success}, "
                            f"workspace={r.workspace_path}, error={r.error}"
                        )
                    return False

                self.logger.info("Workspace validation passed")
            else:
                self.logger.info(
                    "Skipping workspace validation (config.skip_workspace_validation=True)"
                )

            self.logger.info("Step 4/6: Profiler...")
            profiler_process = ProfilerProcess(context=self.master_context)
            profiler_input = ProfilerInput(
                master_context=self.master_context,
                dependency_graph=self.dependency_graph,
                num_turns=self.config.profiler_max_turns,
                model_name=model_name,
                use_openai=use_openai,
            )
            profiler_output = await profiler_process.run(profiler_input)

            if profiler_output.success and profiler_output.protocol_manifesto:
                self.protocol_manifesto = profiler_output.protocol_manifesto
                self.logger.info(
                    f"ProtocolManifesto ready: {self.protocol_manifesto.name}"
                )
                await self._persist(
                    self._state_manager.save_protocol_manifesto(self.protocol_manifesto)
                    if self._state_manager
                    else None
                )
            else:
                self.logger.warning("Profiler failed, continuing without manifesto")

            # Persist state transition
            await self._persist(
                self._state_manager.update_state("profiler")
                if self._state_manager
                else None
            )

            self.logger.info("Step 5/6: Actor Analysis...")
            actor_process = ActorProcess(context=self.master_context)
            actor_input = ActorMatrixInput(
                master_context=self.master_context,
                dependency_graph=self.dependency_graph,
                protocol_manifesto=self.protocol_manifesto,
                model_name=model_name,
                use_openai=use_openai,
            )
            actor_output = await actor_process.run(actor_input)

            if not actor_output.success or not actor_output.actor_matrix:
                self.logger.error(
                    f"Actor analysis failed: {actor_output.error_message}"
                )
                return False

            self.actor_matrix = actor_output.actor_matrix
            self.logger.info(f"ActorMatrix ready: {len(self.actor_matrix.roles)} roles")

            # Persist actor matrix
            await self._persist(
                self._state_manager.save_actor_matrix(self.actor_matrix)
                if self._state_manager
                else None
            )

            self.logger.info("Step 6/6: Invariant Analysis...")
            inv_process = InvariantProcess(context=self.master_context)
            inv_input = InvariantProcessInput(
                master_context=self.master_context,
                dependency_graph=self.dependency_graph,
                actor_matrix=self.actor_matrix,
                protocol_manifesto=self.protocol_manifesto,
                model_name=model_name,
                use_openai=use_openai,
                max_concurrent_chunks=self.config.max_concurrent_invariant_chunks,
            )
            inv_output = await inv_process.run(inv_input)

            if inv_output.success:
                self.invariants = {inv.id: inv for inv in inv_output.invariants}
                self.logger.info(f"Invariants ready: {len(self.invariants)} invariants")

                # Save invariants to file immediately
                invariants_data = [inv.model_dump() for inv in self.invariants.values()]
                # Determine output directory: output_dir > rollouts_dir parent > workspace_dir
                if self.config.output_dir:
                    output_dir = Path(self.config.output_dir)
                elif self.config.rollouts_dir:
                    output_dir = Path(self.config.rollouts_dir).parent
                else:
                    output_dir = Path(self.config.workspace_dir)
                invariants_file = output_dir / "invariants.json"
                invariants_file.parent.mkdir(parents=True, exist_ok=True)
                with open(invariants_file, "w") as f:
                    json.dump(invariants_data, f, indent=2, default=str)
                self.logger.info(f"Invariants saved to {invariants_file}")
            else:
                self.logger.warning(
                    f"Invariant analysis failed: {inv_output.error_message}"
                )

            # Persist state transition and invariants
            await self._persist(
                self._state_manager.update_state("invariant")
                if self._state_manager
                else None
            )
            await self._persist(
                self._state_manager.save_invariants(list(self.invariants.values()))
                if self._state_manager
                else None
            )

            # Initialize planner (actual planning happens in run_loop phases)
            self._planner = MissionPlanner(
                dependency_graph=self.dependency_graph,
                actor_matrix=self.actor_matrix,
                max_invariants_per_cluster=self.config.max_invariants_per_cluster,
                max_campaigns=self.config.max_campaigns,
                include_exploration=self.config.include_exploration,
                default_budget=self.config.default_budget,
                master_context=self.master_context,
            )

            self.logger.info(
                f"Boot complete: {len(self.invariants)} invariants, planner ready"
            )
            return True

        except Exception as e:
            self.logger.error(f"Boot failed: {e}", exc_info=True)
            return False

    async def _build_dependency_graph(self) -> Optional[DependencyGraph]:
        """
        Run static analysis to build DependencyGraph.

        Uses the appropriate builder based on MasterContext.adapter.
        """
        import os
        import uuid
        from pathlib import Path
        from kai.utils.dependency.builders import get_builder

        if not self.master_context:
            return None

        master_root = Path(self.master_context.root_path).resolve()
        analysis_root = master_root

        # Tree-sitter builders (python, javascript, c) only read files - no workspace needed.
        # Only Solidity (Slither) needs a writable workspace for compilation caches.
        adapter = (self.master_context.adapter or "solidity").lower()
        needs_writable_workspace = adapter == "solidity"

        if needs_writable_workspace and not os.access(str(master_root), os.W_OK):
            # Slither/CryticCompile needs write access for compilation caches.
            # Provision a writable workspace copy when master is read-only.
            try:
                ws_id = f"analysis_{uuid.uuid4().hex[:8]}"
                ws_path = self._workspace_manager.provision(
                    workspace_id=ws_id,
                    master_path=str(master_root),
                    preset=WorkspacePreset.CLEAN,
                    master_context=self.master_context,
                )
                analysis_root = Path(ws_path).resolve()
                self.logger.info(
                    f"Provisioned analysis workspace for DependencyGraph: {analysis_root}"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to provision analysis workspace; falling back to master root: {e}"
                )
                analysis_root = master_root

        try:
            builder = get_builder(self.master_context.adapter)
            graph = builder.build(analysis_root)
            self.logger.info(f"Built graph with {len(graph._nodes)} nodes")
            return graph
        except Exception as e:
            self.logger.error(f"Failed to build DependencyGraph: {e}")
        return None

    def _check_shutdown(self) -> bool:
        """
        Check if shutdown has been requested.

        Returns True if shutdown triggered, False otherwise.
        """
        if self._shutdown_requested:
            return True

        if self._shutdown_trigger and self._shutdown_trigger():
            self._shutdown_requested = True
            self.logger.warning("Shutdown triggered by external signal")
            return True

        return False

    @property
    def shutdown_requested(self) -> bool:
        """Whether shutdown has been requested."""
        return self._shutdown_requested

    @property
    def shutdown_reason(self) -> Optional[str]:
        """Reason for shutdown, if available."""
        return self._shutdown_reason

    def request_shutdown(self, reason: str = "Manual shutdown") -> None:
        """
        Request graceful shutdown of the dispatcher.

        Args:
            reason: Human-readable reason for shutdown
        """
        self._shutdown_requested = True
        self._shutdown_reason = reason
        self.logger.info(f"Shutdown requested: {reason}")

    async def run_loop(self) -> None:
        """
        Three-phase execution:
        - Phase 0: Blackbox → observations → new invariants
        - Phase 1: State/Quant (with updated invariants)
        - Phase 2: Gamified on clusters
        """
        # Phase 0: Blackbox
        if self.config.include_exploration and self._planner:
            self.logger.info("Phase 0: Blackbox...")
            await self._queue_blackbox_missions()
            await self._drain_mission_queue()
            self.logger.info(f"Phase 0 done: {len(self.invariants)} invariants")

        if self._shutdown_requested:
            return

        # Phase 1: State/Quant (plan with all invariants including any from blackbox)
        self.logger.info("Phase 1: State/Quant...")
        await self._plan_state_quant_missions()
        await self._drain_mission_queue()
        self.logger.info(f"Phase 1 done: {len(self.completed_missions)} missions")

        if self._shutdown_requested:
            return

        # Phase 2: Gamified
        if self.invariants and self._planner:
            self.logger.info("Phase 2: Gamified...")
            await self._queue_gamified_missions()
            await self._drain_mission_queue()
            self.logger.info("Phase 2 done")

        self.logger.info(f"Total: {len(self.exploit_candidates)} candidates")
        await self._fix_verified_exploits()

    async def _drain_mission_queue(self) -> None:
        """Run missions from queue until empty."""
        while not self.mission_queue.empty() or self.active_missions:
            if self._check_shutdown():
                while self.active_missions:
                    await asyncio.sleep(0.5)
                break

            while (
                len(self.active_missions) < self.config.max_concurrent_agents
                and not self.mission_queue.empty()
            ):
                _, _, mission = await self.mission_queue.get()
                self.active_missions[mission.mission_id] = mission
                asyncio.create_task(self._execute_mission(mission))

            await asyncio.sleep(0.1)

    async def _queue_blackbox_missions(self) -> None:
        """Queue blackbox missions (phase 0)."""
        if not self._planner:
            return
        campaign, missions = self._planner.build_blackbox_campaign(
            num_missions=self.config.blackbox_missions
        )
        self.campaigns.append(campaign)
        await self._persist(
            self._state_manager.save_campaigns([campaign])
            if self._state_manager
            else None
        )
        if missions:
            await self._persist(
                self._state_manager.save_missions(missions)
                if self._state_manager
                else None
            )
        for mission in missions:
            self.mission_queue.put_nowait((0, mission.mission_id, mission))

    async def _plan_state_quant_missions(self) -> None:
        """Plan state/quant missions with all invariants (phase 1)."""
        if not self._planner or not self.invariants:
            return
        campaigns, missions = self._planner.plan(
            invariants=list(self.invariants.values())
        )
        self.campaigns.extend(campaigns)
        if campaigns:
            await self._persist(
                self._state_manager.save_campaigns(campaigns)
                if self._state_manager
                else None
            )
        if missions:
            await self._persist(
                self._state_manager.save_missions(missions)
                if self._state_manager
                else None
            )
        for mission in missions:
            self.mission_queue.put_nowait((1, mission.mission_id, mission))

    async def _queue_gamified_missions(self) -> None:
        """Queue gamified missions from invariant clusters (phase 2)."""
        if not self._planner:
            return
        if self.config.disable_gamified:
            self.logger.info("Gamified missions disabled via config")
            return
        campaigns, missions = self._planner.build_gamified_campaigns(
            list(self.invariants.values())
        )
        self.campaigns.extend(campaigns)
        if campaigns:
            await self._persist(
                self._state_manager.save_campaigns(campaigns)
                if self._state_manager
                else None
            )
        if missions:
            await self._persist(
                self._state_manager.save_missions(missions)
                if self._state_manager
                else None
            )
        for mission in missions:
            self.mission_queue.put_nowait((2, mission.mission_id, mission))

    async def _execute_mission(self, mission: Mission) -> None:
        """Execute a single mission with an agent."""
        mission.status = "in_progress"
        await self._persist(
            self._state_manager.update_mission_status(mission.mission_id, "in_progress")
            if self._state_manager
            else None
        )
        # Note: mission already added to active_missions in run_loop

        factory = self.agent_factories.get(mission.agent_type)
        if not factory:
            self.logger.warning(f"No agent factory for type {mission.agent_type}")
            mission.status = "failed"
            await self._persist(
                self._state_manager.update_mission_status(
                    mission.mission_id,
                    "failed",
                    error=f"No agent factory for type {mission.agent_type}",
                )
                if self._state_manager
                else None
            )
            self.active_missions.pop(mission.mission_id, None)
            self.completed_missions.append(mission)
            return

        agent = None
        try:
            # Provision workspace
            workspace_path = self._provision_workspace(mission)

            # Create agent instance via factory with full context
            agent = factory(
                mission=mission,
                workspace_path=workspace_path,
                master_context=self.master_context,
                dependency_graph=self.dependency_graph,
                actor_matrix=self.actor_matrix,
                model=self.config.model,
                use_openai=self.config.use_openai,
                execution_id=mission.mission_id,
                extra_instructions=self.config.extra_instructions,
            )

            self.logger.info(
                f"Executing {mission.mission_id} with {mission.agent_type.value}"
            )

            # Agent-type specific execution
            if mission.agent_type == MissionAgentType.STATE:
                await agent.chat_with_tools("Begin.")
                await self._handle_state_agent_result(mission, agent)

            elif mission.agent_type == MissionAgentType.QUANT:
                await agent.chat_with_tools("Begin.")
                await self._handle_state_agent_result(mission, agent)

            elif mission.agent_type == MissionAgentType.BLACKBOX:
                await agent.chat_with_tools("Begin.")
                await self._handle_blackbox_agent_result(mission, agent)

            elif mission.agent_type == MissionAgentType.GAMIFIED:
                await agent.chat_with_tools("Begin.")
                await self._handle_gamified_agent_result(mission, agent)

            else:
                raise ValueError(f"Unsupported agent type: {mission.agent_type}")

            mission.status = "completed"
            await self._persist(
                self._state_manager.update_mission_status(
                    mission.mission_id, "completed"
                )
                if self._state_manager
                else None
            )

        except Exception as e:
            self.logger.error(
                f"Mission {mission.mission_id} failed: {e}", exc_info=True
            )
            mission.status = "failed"
            await self._persist(
                self._state_manager.update_mission_status(
                    mission.mission_id, "failed", error=str(e)
                )
                if self._state_manager
                else None
            )

        finally:
            # Save rollout before closing agent
            if agent is not None:
                self._save_rollout(agent, "missions", mission.mission_id)
                # Aggregate token usage from this agent
                self._aggregate_agent_usage(
                    agent=agent,
                    phase="run_loop",
                    agent_type=mission.agent_type.value if mission.agent_type else "unknown",
                )
                try:
                    await agent.close()
                except Exception:
                    pass
            self.active_missions.pop(mission.mission_id, None)
            self.completed_missions.append(mission)
            # Cleanup workspace
            self._cleanup_workspace(mission)

    async def _handle_state_agent_result(self, mission: Mission, agent: Any) -> None:
        """
        Extract and handle results from StateAgent/QuantAgent.

        These agents implement get_exploit_candidates() which returns registered exploits.
        """
        candidates = agent.get_exploit_candidates()

        if not candidates:
            self.logger.info(f"No exploit candidates from {mission.mission_id}")
            return

        self.logger.info(
            f"StateAgent {mission.mission_id} found {len(candidates)} exploit candidate(s)"
        )

        for candidate in candidates:
            # Tag candidate with mission context
            if hasattr(candidate, "mission_id") and not candidate.mission_id:
                candidate.mission_id = mission.mission_id
            if (
                hasattr(candidate, "invariant_id")
                and not candidate.invariant_id
                and mission.invariant
            ):
                candidate.invariant_id = mission.invariant.id

            self.exploit_candidates.append(candidate)

            # Persist exploit candidate
            await self._persist(
                self._state_manager.save_exploit_candidate(candidate)
                if self._state_manager
                else None
            )

            # Verify compiled candidates
            if candidate.compiled and self.master_context:
                await self._verify_candidate(candidate)

    async def _verify_candidate(self, candidate: ExploitCandidate) -> Optional[Verdict]:
        """
        Verify an exploit candidate using VerifierProcess.

        Args:
            candidate: The exploit candidate to verify

        Returns:
            Verdict if verification completed, None if failed
        """
        if not self.master_context:
            return None

        from kai.processes.verifier import VerifierProcess

        # Get the invariant for this candidate
        invariant = self.invariants.get(candidate.invariant_id)
        if not invariant:
            self.logger.warning(
                f"No invariant found for {candidate.invariant_id}, creating placeholder"
            )
            invariant = Invariant(
                id=candidate.invariant_id,
                type=InvariantType.OTHER,
                rule=f"Unknown invariant: {candidate.invariant_id}",
            )

        self.logger.info(f"Verifying exploit candidate: {candidate.mission_id}")

        try:
            process = VerifierProcess(context=self.master_context)
            process_input = VerifierProcessInput(
                exploit_candidate=candidate,
                invariant=invariant,
                master_context=self.master_context,
                dependency_graph=self.dependency_graph,
                model_name=self.config.verifier_model,
                fallback_model=self.config.fallback_model,
                use_openai=self.config.use_openai,
                max_turns=self.config.verifier_max_turns,
            )

            output = await process.run(process_input)

            # Save verifier rollout if messages available
            if self.config.save_rollouts and output.agent_messages:
                self._save_verifier_rollout(
                    candidate.mission_id,
                    output.agent_messages,
                    output.agent_model or "unknown",
                    output.total_tokens,
                    output.estimated_cost,
                )

            # Aggregate verifier token usage
            self._aggregate_process_usage(
                prompt_tokens=output.total_tokens.get("prompt_tokens", 0),
                completion_tokens=output.total_tokens.get("completion_tokens", 0),
                cost=output.estimated_cost,
                phase="run_loop",
                agent_type="verifier",
            )

            if output.success and output.verdict:
                verdict = output.verdict
                self.verdicts.append(verdict)

                # Persist verdict
                await self._persist(
                    self._state_manager.save_verdict(verdict)
                    if self._state_manager
                    else None
                )

                if verdict.is_valid:
                    self.logger.info(
                        f"VERIFIED: {candidate.mission_id} - "
                        f"{verdict.severity.value.upper()} - {verdict.vulnerability_class}"
                    )
                else:
                    self.logger.info(
                        f"REJECTED: {candidate.mission_id} - {verdict.rejection_reason}"
                    )

                return verdict
            else:
                self.logger.warning(
                    f"Verifier did not submit verdict for {candidate.mission_id}: "
                    f"{output.error_message}"
                )
                return None

        except Exception as e:
            self.logger.error(f"Verification failed for {candidate.mission_id}: {e}")
            return None

    async def _fix_verified_exploits(self) -> None:
        """
        Generate fixes for all verified exploits.

        Called after all missions complete. Iterates over verdicts with is_valid=True
        and runs FixerAgent on each.
        """
        valid_verdicts = [v for v in self.verdicts if v.is_valid]

        if not valid_verdicts:
            self.logger.info("No verified exploits to fix")
            return

        self.logger.info(
            f"Generating fixes for {len(valid_verdicts)} verified exploit(s)..."
        )

        for verdict in valid_verdicts:
            # Find the corresponding exploit candidate
            candidate = next(
                (
                    c
                    for c in self.exploit_candidates
                    if c.mission_id == verdict.mission_id
                    and c.invariant_id == verdict.invariant_id
                ),
                None,
            )

            if not candidate:
                self.logger.warning(
                    f"No exploit candidate found for verdict {verdict.mission_id}"
                )
                continue

            fixes = await self._fix_single_exploit(candidate, verdict)
            for fix in fixes:
                self.fixes.append(fix)
                # Embed fix into verdict
                verdict.fixes.append(fix)
                # Persist fix
                await self._persist(
                    self._state_manager.save_fix(fix) if self._state_manager else None
                )

        self.logger.info(f"Generated {len(self.fixes)} fix(es) for verified exploits")

    async def _fix_single_exploit(
        self, candidate: ExploitCandidate, verdict: Verdict
    ) -> List[Fix]:
        """
        Generate fixes for a single verified exploit using FixerAgent.

        Args:
            candidate: The exploit candidate
            verdict: The verification verdict

        Returns:
            List of Fix objects (may be empty if fixer failed)
        """
        import uuid

        if not self.master_context:
            return []

        from kai.agents.agent_types.fixer_agent import FixerAgent

        self.logger.info(f"Fixing exploit: {candidate.mission_id}")

        agent = None
        try:
            # Provision WRITEABLE workspace for fixer (needs to modify contracts)
            workspace_path = self._workspace_manager.provision(
                workspace_id=f"fixer_{candidate.mission_id}",
                master_path=self.master_context.root_path,
                preset=WorkspacePreset.WRITEABLE,
            )

            # Create fixer agent with required context
            agent = FixerAgent(
                exploit_candidate=candidate,
                verdict=verdict,
                repo_path=workspace_path,
                dependency_graph=self.dependency_graph,
                max_tool_turns=self.config.fixer_max_turns,
                model=self.config.fixer_model,
                use_openai=self.config.use_openai,
            )

            # Run the fixer
            await agent.chat_with_tools("Begin.")

            # Extract all registered fixes
            registered_fixes = getattr(agent, "_registered_fixes", [])

            if not registered_fixes:
                self.logger.warning(
                    f"Fixer did not register any fixes for {candidate.mission_id}"
                )
                return []

            # Convert all registered fixes to Fix objects
            fixes = []
            for fix_record in registered_fixes:
                fix = Fix(
                    fix_id=fix_record.get("fix_id", f"fix_{uuid.uuid4().hex}"),
                    mission_id=candidate.mission_id,
                    invariant_id=candidate.invariant_id,
                    summary=fix_record.get("summary", ""),
                    reasoning=fix_record.get("reasoning", ""),
                    canonical_diff=fix_record.get("canonical_diff", ""),
                    files_changed=fix_record.get("files_changed", []),
                    compiled=fix_record.get("compiled", False),
                    tests_passed=fix_record.get("tests_passed", False),
                )
                fixes.append(fix)

            self.logger.info(
                f"FIX GENERATED: {candidate.mission_id} - {len(fixes)} fix(es)"
            )
            return fixes

        except Exception as e:
            self.logger.error(f"Fix generation failed for {candidate.mission_id}: {e}")
            return []

        finally:
            # Save fixer rollout before cleanup
            if agent is not None:
                self._save_rollout(agent, "fixer", f"fixer_{candidate.mission_id}")
                try:
                    await agent.close()
                except Exception:
                    pass
            # Cleanup fixer workspace
            try:
                self._workspace_manager.cleanup(f"fixer_{candidate.mission_id}")
            except Exception:
                pass

    async def _handle_blackbox_agent_result(self, mission: Mission, agent: Any) -> None:
        """
        Extract and handle results from BlackboxAgent.

        BlackboxAgent implements get_observations() which returns recorded observations.
        """
        observations = agent.get_observations()

        if not observations:
            self.logger.info(f"No observations from {mission.mission_id}")
            return

        self.logger.info(
            f"BlackboxAgent {mission.mission_id} recorded {len(observations)} observation(s)"
        )

        # Persist observations
        await self._persist(
            self._state_manager.save_observations(observations)
            if self._state_manager
            else None
        )

        new_invariants: List[Invariant] = []
        for obs in observations:
            # Synthesize invariant from observation
            new_inv = await self._synthesize_invariant(obs)
            if new_inv and new_inv.id not in self.invariants:
                self.logger.info(f"New invariant discovered: {new_inv.id}")
                self.invariants[new_inv.id] = new_inv
                new_invariants.append(new_inv)
                await self._schedule_missions_for_invariant(new_inv)

        if new_invariants:
            await self._persist(
                self._state_manager.save_invariants(new_invariants)
                if self._state_manager
                else None
            )

    async def _handle_gamified_agent_result(self, mission: Mission, agent: Any) -> None:
        """
        Extract and handle results from GamifiedAgent.

        GamifiedAgent discovers exploitation opportunities by reasoning about gaps
        between invariants in a cluster. It implements get_exploit_candidates() which
        returns registered exploits (same as StateAgent/QuantAgent).
        """
        candidates = agent.get_exploit_candidates()

        if not candidates:
            self.logger.info(
                f"No exploit candidates from gamified {mission.mission_id}"
            )
            return

        self.logger.info(
            f"GamifiedAgent {mission.mission_id} found {len(candidates)} exploit candidate(s)"
        )

        for candidate in candidates:
            # Tag candidate with mission context
            if hasattr(candidate, "mission_id") and not candidate.mission_id:
                candidate.mission_id = mission.mission_id

            # Gamified agents work on clusters, so invariant_id might be "gap_exploit"
            # or set by the register_finding tool

            self.exploit_candidates.append(candidate)

            # Persist exploit candidate
            await self._persist(
                self._state_manager.save_exploit_candidate(candidate)
                if self._state_manager
                else None
            )

            # Verify compiled candidates
            if candidate.compiled and self.master_context:
                await self._verify_candidate(candidate)

    async def _synthesize_invariant(
        self, observation: Observation
    ) -> Optional[Invariant]:
        """
        Synthesize a tentative invariant from an observation.

        Uses LLM to convert unstructured logs into a rule.
        """
        if not self.master_context or not self.dependency_graph:
            return None

        from kai.processes.invariant_synthesizer import InvariantSynthesizerProcess
        from kai.schemas import InvariantSynthesizerInput

        process = InvariantSynthesizerProcess(context=self.master_context)
        input_data = InvariantSynthesizerInput(
            observations=[observation],
            master_context=self.master_context,
            dependency_graph=self.dependency_graph,
            protocol_manifesto=self.protocol_manifesto,
            model_name=self.config.model,
            use_openai=self.config.use_openai,
            max_turns_per_observation=self.config.invariant_synth_max_turns,
        )

        try:
            output = await process.run(input_data)
            if output.success and output.invariants:
                return output.invariants[0]
        except Exception as e:
            self.logger.error(f"Failed to synthesize invariant: {e}")

        return None

    async def _schedule_missions_for_invariant(self, invariant: Invariant) -> None:
        """Schedule new missions for a dynamically discovered invariant."""
        if not self._planner:
            self.logger.warning("Cannot schedule missions: planner not initialized")
            return

        base_id = len(self.completed_missions) + self.mission_queue.qsize()
        missions = self._planner.create_missions_for_invariant(invariant, base_id)

        if missions:
            await self._persist(
                self._state_manager.save_missions(missions)
                if self._state_manager
                else None
            )

        for mission in missions:
            self.mission_queue.put_nowait((1, mission.mission_id, mission))

    def _provision_workspace(self, mission: Mission) -> str:
        if not self.master_context:
            raise RuntimeError("MasterContext not initialized")
        return self._workspace_manager.provision_for_mission(
            mission, self.master_context
        )

    def _cleanup_workspace(self, mission: Mission) -> None:
        self._workspace_manager.cleanup_for_mission(mission)

    def get_verified_exploits(
        self, min_severity: VerdictSeverity = VerdictSeverity.LOW
    ) -> List[Verdict]:
        """Get verified exploits at or above a minimum severity."""
        severity_order = [
            VerdictSeverity.INFORMATIONAL,
            VerdictSeverity.LOW,
            VerdictSeverity.MEDIUM,
            VerdictSeverity.HIGH,
            VerdictSeverity.CRITICAL,
        ]
        min_idx = severity_order.index(min_severity)

        return [
            v
            for v in self.verdicts
            if v.is_valid and severity_order.index(v.severity) >= min_idx
        ]

    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics."""
        verified = [v for v in self.verdicts if v.is_valid]
        rejected = [v for v in self.verdicts if not v.is_valid]

        severity_counts: Dict[str, int] = {}
        for verdict in verified:
            sev = verdict.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_candidates": len(self.exploit_candidates),
            "verified_count": len(verified),
            "rejected_count": len(rejected),
            "by_severity": severity_counts,
            "rejection_reasons": [
                v.rejection_reason for v in rejected if v.rejection_reason
            ],
        }

    def export_results(self, output_path: str) -> None:
        """
        Export all dispatcher results to a JSON file.

        Includes campaigns, missions, exploit candidates, verdicts, and stats.

        Args:
            output_path: Path to the output JSON file
        """
        # Build summary
        summary = {
            "total_campaigns": len(self.campaigns),
            "total_missions": len(self.completed_missions),
            "successful_missions": len(
                [m for m in self.completed_missions if m.status == "completed"]
            ),
            "failed_missions": len(
                [m for m in self.completed_missions if m.status == "failed"]
            ),
            "total_exploit_candidates": len(self.exploit_candidates),
            "total_verdicts": len(self.verdicts),
            "verified_exploits": len([v for v in self.verdicts if v.is_valid]),
            "rejected_exploits": len([v for v in self.verdicts if not v.is_valid]),
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost,
        }

        # Serialize campaigns
        campaigns_data = [c.model_dump() for c in self.campaigns]

        # Serialize missions
        missions_data = [m.model_dump() for m in self.completed_missions]

        # Group exploit candidates by mission
        exploits_by_mission: Dict[str, List[Dict]] = {}
        for candidate in self.exploit_candidates:
            mission_id = candidate.mission_id
            if mission_id not in exploits_by_mission:
                exploits_by_mission[mission_id] = []
            exploits_by_mission[mission_id].append(candidate.model_dump())

        # Serialize verdicts with full details
        verdicts_data = [v.model_dump() for v in self.verdicts]

        # Build final report
        report = {
            "summary": summary,
            "verification_stats": self.get_verification_stats(),
            "campaigns": campaigns_data,
            "missions": missions_data,
            "exploits_by_mission": exploits_by_mission,
            "verdicts": verdicts_data,
        }

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save invariants to a separate file
        invariants_data = [inv.model_dump() for inv in self.invariants.values()]
        invariants_file = output_file.parent / "invariants.json"
        with open(invariants_file, "w") as f:
            json.dump(invariants_data, f, indent=2, default=str)

        self.logger.info(f"Results exported to {output_path}")
        self.logger.info(f"Invariants exported to {invariants_file} ({len(invariants_data)} invariants)")
