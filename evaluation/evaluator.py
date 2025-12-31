"""
Blackbox Evaluator - Main Facade.

Coordinates metrics collection and reporting for synthesized invariants.

Note: Duplicate detection is handled post-hoc by PostHocDeduplicator.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from evaluation.metrics_collector import BlackboxMetricsCollector
from evaluation.schemas import BlackboxEvaluationReport
from kai.schemas import Invariant, Observation

logger = logging.getLogger(__name__)


class BlackboxEvaluator:
    """
    Main evaluator facade for Blackbox Agent pipeline.

    Coordinates metrics collection and reporting.

    Note: Duplicate detection is handled post-hoc by PostHocDeduplicator.
    """

    def __init__(
        self,
        execution_id: str,
        existing_invariants: Dict[str, Invariant],
        repo_path: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            execution_id: Unique identifier for this evaluation run.
            existing_invariants: Dict of invariant_id -> Invariant for baseline reference.
            repo_path: Path to the repository being evaluated.
        """
        self.execution_id = execution_id
        self.repo_path = repo_path
        self.baseline_invariants = existing_invariants

        self.logger = logger.getChild("BlackboxEvaluator")

        self.metrics_collector = BlackboxMetricsCollector(
            execution_id=execution_id,
            baseline_invariant_ids=list(existing_invariants.keys()),
        )

    def record_observation(self, obs: Observation) -> str:
        """
        Record an observation for evaluation tracking.

        Args:
            obs: The observation to record.

        Returns:
            Generated observation ID.
        """
        return self.metrics_collector.record_observation(obs)

    async def evaluate_synthesized_invariant(
        self,
        invariant: Invariant,
        source_observation: Observation,
    ) -> None:
        """
        Record a synthesized invariant.

        Duplicate detection is handled post-hoc by PostHocDeduplicator.

        Args:
            invariant: The synthesized invariant.
            source_observation: The observation that led to this invariant.
        """
        obs_id = self.metrics_collector._generate_observation_id(source_observation)

        # Record synthesis success
        self.metrics_collector.record_synthesis_result(
            obs_id=obs_id,
            invariant=invariant,
        )

        self.logger.info(f"Recorded invariant {invariant.label}")

    def record_synthesis_failure(
        self,
        obs: Observation,
        error: str,
    ) -> None:
        """
        Record that synthesis failed for an observation.

        Args:
            obs: The observation.
            error: Error message.
        """
        obs_id = self.metrics_collector._generate_observation_id(obs)
        self.metrics_collector.record_synthesis_result(
            obs_id=obs_id,
            invariant=None,
            error=error,
        )

    def generate_report(self) -> BlackboxEvaluationReport:
        """Generate the final evaluation report."""
        return self.metrics_collector.generate_report(
            repo_path=self.repo_path,
        )

    def export_report(self, output_path: str) -> None:
        """
        Export evaluation report to JSON file.

        Args:
            output_path: Path to write the report.
        """
        report = self.generate_report()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)

        self.logger.info(f"Blackbox evaluation report exported to {output_path}")
