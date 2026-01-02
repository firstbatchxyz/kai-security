"""
Standalone Blackbox Evaluation Runner.

Enables running the Blackbox Agent -> Invariant Synthesizer -> Evaluation pipeline
independently of the Dispatcher.
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.evaluator import BlackboxEvaluator
from evaluation.schemas import BlackboxEvaluationReport
from kai.schemas import (
    CampaignBrief,
    CampaignBudget,
    EntrypointsSubset,
    Invariant,
    MasterContext,
    Observation,
)
from kai.utils.dependency.graph import DependencyGraph

logger = logging.getLogger(__name__)


class BlackboxEvaluationRunner:
    """
    Standalone runner for evaluating the Blackbox Agent -> Invariant Synthesizer pipeline.
    Can be used independently of the Dispatcher.
    """

    def __init__(
        self,
        repo_path: str,
        baseline_invariants: List[Invariant],
        model_name: str = "openai/gpt-5.2",
        use_openai: bool = False,
        output_dir: str = "./evaluation_output",
    ):
        """
        Initialize the runner.

        Note: Duplicate detection is handled post-hoc by PostHocDeduplicator.

        Args:
            repo_path: Path to the target repository.
            baseline_invariants: Baseline invariants for reference.
            model_name: Model for Blackbox Agent and Invariant Synthesizer.
            use_openai: Whether to use OpenAI API directly.
            output_dir: Directory for output artifacts.
        """
        self.repo_path = str(Path(repo_path).resolve())
        self.baseline_invariants = baseline_invariants
        self.baseline_invariants_dict = {inv.id: inv for inv in baseline_invariants}
        self.model_name = model_name
        self.use_openai = use_openai
        self.output_dir = Path(output_dir)

        self.execution_id = f"eval_{uuid.uuid4().hex[:8]}"
        self.logger = logger.getChild("BlackboxRunner")

        # State
        self.dependency_graph: Optional[DependencyGraph] = None
        self.master_context: Optional[MasterContext] = None
        self.observations: List[Observation] = []
        self.synthesized_invariants: List[Invariant] = []
        self.evaluator: Optional[BlackboxEvaluator] = None

        # Ensure output dir exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_full_pipeline(
        self,
        num_turns: int = 50,
        campaign_label: Optional[str] = None,
    ) -> BlackboxEvaluationReport:
        """
        Execute the full Blackbox -> Synthesizer -> Evaluation pipeline.

        Steps:
        1. Build DependencyGraph from repo
        2. Run Blackbox Agent to collect observations
        3. Synthesize invariants from each observation
        4. Evaluate each synthesized invariant (validity check)
        5. Generate and return report

        Args:
            num_turns: Budget for Blackbox Agent.
            campaign_label: Optional campaign label for tracking.

        Returns:
            Complete evaluation report.
        """
        campaign_label = campaign_label or f"eval_campaign_{uuid.uuid4().hex[:8]}"
        self.logger.info(f"Starting full pipeline evaluation: {campaign_label}")

        # Step 1: Build dependency graph
        self.logger.info("Step 1/4: Building dependency graph...")
        await self._build_dependency_graph()

        # Step 2: Run Blackbox Agent
        self.logger.info("Step 2/4: Running Blackbox Agent...")
        await self._run_blackbox_agent(num_turns, campaign_label)

        # Step 3 & 4: Synthesize and evaluate
        self.logger.info("Step 3/4: Synthesizing and evaluating invariants...")
        report = await self._synthesize_and_evaluate()

        # Save artifacts
        self.logger.info("Step 4/4: Saving artifacts...")
        self.save_artifacts()

        return report

    async def _build_dependency_graph(self) -> None:
        """Build dependency graph from the repository."""
        from kai.utils.dependency.builders import SolidityBuilder

        try:
            repo = Path(self.repo_path).resolve()

            # Check if it looks like a Foundry project
            looks_like_foundry = (
                repo.is_dir() and (repo / "lib").exists() and (repo / "test").exists()
            )

            if looks_like_foundry:
                self.dependency_graph = SolidityBuilder().build(
                    self.repo_path,
                    slither_kwargs={"compile_force_framework": "foundry"},
                )
            else:
                self.dependency_graph = SolidityBuilder().build(self.repo_path)

            self.logger.info(
                f"Built dependency graph with {len(self.dependency_graph._nodes)} nodes"
            )

            # Create minimal MasterContext
            self.master_context = MasterContext(
                root_path=self.repo_path,
                compile_success=True,
                adapter="solidity",
                frameworks=["foundry"] if looks_like_foundry else [],
            )

        except Exception as e:
            self.logger.warning(f"Failed to build dependency graph: {e}")
            # Create empty graph as fallback
            self.dependency_graph = DependencyGraph(Path(self.repo_path).resolve())
            self.master_context = MasterContext(
                root_path=self.repo_path,
                compile_success=False,
            )

    async def _run_blackbox_agent(
        self,
        num_turns: int,
        campaign_label: str,
    ) -> None:
        """Run the Blackbox Agent to collect observations."""
        from kai.agents.agent_types import BlackboxAgent

        # Create campaign brief
        brief = CampaignBrief(
            label=campaign_label,
            kind="blackbox_evaluation",
            invariant_ids=[],
            entrypoints_subset=EntrypointsSubset(function_ids=[]),
            workspace_preset="clean",
            budget=CampaignBudget(max_turns_per_worker=num_turns),
            master_context=self.master_context,
        )

        # Setup Foundry environment
        repo_slug = self._repo_slug(self.repo_path)
        foundry_root = self.output_dir / "foundry" / campaign_label / repo_slug
        foundry_cache = foundry_root / "cache"
        foundry_out = foundry_root / "out"
        foundry_cache.mkdir(parents=True, exist_ok=True)
        foundry_out.mkdir(parents=True, exist_ok=True)

        prev_env_cache = os.environ.get("FOUNDRY_CACHE_PATH")
        prev_env_out = os.environ.get("FOUNDRY_OUT")
        os.environ["FOUNDRY_CACHE_PATH"] = str(foundry_cache)
        os.environ["FOUNDRY_OUT"] = str(foundry_out)

        try:
            agent = BlackboxAgent(
                campaign_brief=brief,
                dependency_graph=self.dependency_graph,
                repo_path=self.repo_path,
                model=self.model_name,
                max_tool_turns=num_turns,
                use_openai=self.use_openai,
                execution_id=self.execution_id,
            )

            # Setup campaigns directory
            campaigns_root = self.output_dir / "campaigns" / campaign_label / repo_slug
            campaigns_root.mkdir(parents=True, exist_ok=True)
            setattr(agent, "campaigns_dir", str(campaigns_root))

            # Run the agent
            brief_dump = brief.model_dump(exclude={"dependency_graph"})
            user_prompt = (
                "Run the blackbox campaign using the provided briefing.\n"
                "Focus on exploring the codebase and finding anomalies.\n"
                "Record observations incrementally as you learn things.\n"
                "Do NOT emit <done>.\n"
                f"CampaignBrief:\n{json.dumps(brief_dump, indent=2)}"
            )

            await agent.chat_with_tools(user_prompt)

            # Extract observations
            self.observations = getattr(agent, "blackbox_observations", [])
            self.logger.info(f"Collected {len(self.observations)} observations")

        finally:
            # Restore environment
            if prev_env_cache is None:
                os.environ.pop("FOUNDRY_CACHE_PATH", None)
            else:
                os.environ["FOUNDRY_CACHE_PATH"] = prev_env_cache
            if prev_env_out is None:
                os.environ.pop("FOUNDRY_OUT", None)
            else:
                os.environ["FOUNDRY_OUT"] = prev_env_out

    async def _synthesize_and_evaluate(self) -> BlackboxEvaluationReport:
        """Synthesize invariants from observations and evaluate them."""
        from kai.processes.invariant_synthesizer import InvariantSynthesizerProcess
        from kai.schemas import InvariantSynthesizerInput

        # Initialize evaluator
        self._init_evaluator()

        if not self.observations:
            self.logger.warning("No observations to synthesize")
            return self.evaluator.generate_report()

        # Create minimal master context if needed
        if not self.master_context:
            self.master_context = MasterContext(
                root_path=self.repo_path,
                compile_success=True,
                adapter="solidity",
            )

        # Synthesize and evaluate each observation
        for obs in self.observations:
            # Record observation
            self.evaluator.record_observation(obs)

            # Synthesize invariant
            try:
                synth_process = InvariantSynthesizerProcess(context=self.master_context)
                synth_input = InvariantSynthesizerInput(
                    observations=[obs],
                    master_context=self.master_context,
                    dependency_graph=self.dependency_graph,
                    model_name=self.model_name,
                    use_openai=self.use_openai,
                    max_turns_per_observation=5,
                )

                synth_output = await synth_process.run(synth_input)

                if synth_output.success and synth_output.invariants:
                    invariant = synth_output.invariants[0]
                    self.synthesized_invariants.append(invariant)

                    # Evaluate the invariant
                    await self.evaluator.evaluate_synthesized_invariant(
                        invariant=invariant,
                        source_observation=obs,
                    )
                else:
                    self.evaluator.record_synthesis_failure(
                        obs=obs,
                        error=synth_output.error_message
                        or "Synthesis returned no invariants",
                    )

            except Exception as e:
                self.logger.error(f"Synthesis failed for observation: {e}")
                self.evaluator.record_synthesis_failure(obs=obs, error=str(e))

        return self.evaluator.generate_report()

    def _init_evaluator(self) -> None:
        """Initialize the evaluator."""
        self.evaluator = BlackboxEvaluator(
            execution_id=self.execution_id,
            existing_invariants=self.baseline_invariants_dict,
            repo_path=self.repo_path,
        )

    def _repo_slug(self, repo_path: str) -> str:
        """Generate safe repo slug for paths."""
        name = Path(repo_path).name or "repo"
        return re.sub(r"[^A-Za-z0-9._-]", "-", name)

    def save_artifacts(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save all artifacts (observations, synthesized invariants, report) to files.

        Args:
            output_dir: Override output directory.

        Returns:
            Dict of artifact_name -> file_path.
        """
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        artifacts = {}

        # Save observations
        if self.observations:
            obs_path = out_dir / f"observations_{timestamp}.json"
            with open(obs_path, "w") as f:
                json.dump(
                    [obs.model_dump() for obs in self.observations],
                    f,
                    indent=2,
                    default=str,
                )
            artifacts["observations"] = str(obs_path)

        # Save synthesized invariants
        if self.synthesized_invariants:
            inv_path = out_dir / f"synthesized_invariants_{timestamp}.json"
            with open(inv_path, "w") as f:
                json.dump(
                    [inv.model_dump() for inv in self.synthesized_invariants],
                    f,
                    indent=2,
                    default=str,
                )
            artifacts["synthesized_invariants"] = str(inv_path)

        # Save report
        if self.evaluator:
            report_path = out_dir / f"evaluation_report_{timestamp}.json"
            self.evaluator.export_report(str(report_path))
            artifacts["report"] = str(report_path)

        self.logger.info(f"Saved {len(artifacts)} artifacts to {out_dir}")
        return artifacts
