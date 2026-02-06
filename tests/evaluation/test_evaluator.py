"""Tests for BlackboxEvaluator.

Uses real invariants and observations from the Ethena BBP protocol analysis.
"""

import pytest

from evaluation.evaluator import BlackboxEvaluator
from kai.schemas import Invariant, InvariantType


class TestEvaluatorInitialization:
    """Test evaluator initialization with real data."""

    def test_init_with_baseline(self, sample_baseline_invariants):
        """Test initialization with real baseline invariants."""
        evaluator = BlackboxEvaluator(
            execution_id="test_001",
            existing_invariants=sample_baseline_invariants,
        )

        assert evaluator.execution_id == "test_001"
        assert len(evaluator.baseline_invariants) == len(sample_baseline_invariants)


class TestObservationRecording:
    """Test observation recording with real observations."""

    def test_record_observation(self, sample_baseline_invariants, real_observations):
        """Test that real observations are recorded correctly."""
        evaluator = BlackboxEvaluator(
            execution_id="test_003",
            existing_invariants=sample_baseline_invariants,
        )

        # Use first real observation
        obs = real_observations[0]
        obs_id = evaluator.record_observation(obs)

        assert obs_id is not None
        assert len(obs_id) == 16  # Hash length
        assert evaluator.metrics_collector._metrics.total_observations == 1

    def test_record_multiple_observations(
        self, sample_baseline_invariants, real_observations
    ):
        """Test recording multiple real observations."""
        evaluator = BlackboxEvaluator(
            execution_id="test_003b",
            existing_invariants=sample_baseline_invariants,
        )

        for obs in real_observations:
            evaluator.record_observation(obs)

        assert evaluator.metrics_collector._metrics.total_observations == len(
            real_observations
        )


class TestInvariantEvaluation:
    """Test invariant evaluation pipeline with real data."""

    @pytest.mark.asyncio
    async def test_invariant_recorded(
        self, sample_baseline_invariants, real_observations
    ):
        """Test that invariant is recorded."""
        evaluator = BlackboxEvaluator(
            execution_id="test_004",
            existing_invariants=sample_baseline_invariants,
        )

        obs = real_observations[0]
        evaluator.record_observation(obs)

        invariant = Invariant(
            id="NEW_UNIQUE_001",
            type=InvariantType.LIVENESS,
            rule="Unique test function must be callable for non-zero amounts",
            target_function_ids=["uniqueTestFunc123"],
            target_var_ids=["uniqueAmount456"],
            confidence=0.75,
        )

        await evaluator.evaluate_synthesized_invariant(
            invariant=invariant,
            source_observation=obs,
        )

        # Check metrics were recorded
        assert evaluator.metrics_collector._metrics.total_synthesized == 1


class TestSynthesisFailure:
    """Test synthesis failure recording."""

    def test_record_synthesis_failure(
        self, sample_baseline_invariants, real_observations
    ):
        """Test that synthesis failure is recorded with real observation."""
        evaluator = BlackboxEvaluator(
            execution_id="test_007",
            existing_invariants=sample_baseline_invariants,
        )

        obs = real_observations[0]
        evaluator.record_observation(obs)
        evaluator.record_synthesis_failure(
            obs=obs,
            error="Synthesis LLM returned empty response",
        )

        metrics = evaluator.metrics_collector._metrics
        assert metrics.observations_without_synthesis == 1


class TestReportGeneration:
    """Test report generation with real data."""

    @pytest.mark.asyncio
    async def test_generate_report(self, sample_baseline_invariants, real_observations):
        """Test report generation with real data."""
        evaluator = BlackboxEvaluator(
            execution_id="test_008",
            existing_invariants=sample_baseline_invariants,
            repo_path="/path/to/ethena/repo",
        )

        # Record all real observations
        for obs in real_observations:
            evaluator.record_observation(obs)

        # Evaluate an invariant
        invariant = Invariant(
            id="REP_001",
            type=InvariantType.LIVENESS,
            rule="Function must be callable under normal conditions",
            target_function_ids=["someUniqueFunc"],
            confidence=0.7,
        )

        await evaluator.evaluate_synthesized_invariant(
            invariant=invariant,
            source_observation=real_observations[0],
        )

        report = evaluator.generate_report()

        assert report.execution_id == "test_008"
        assert report.repo_path == "/path/to/ethena/repo"
        assert report.metrics.total_observations == len(real_observations)
        assert len(report.records) == len(real_observations)
        assert len(report.baseline_invariant_ids) == len(sample_baseline_invariants)


class TestRealDataIntegration:
    """Integration tests with real Ethena data."""

    def test_evaluator_handles_real_invariant_types(self, real_invariants):
        """Test that evaluator properly handles all real invariant types."""
        evaluator = BlackboxEvaluator(
            execution_id="test_real_types",
            existing_invariants=real_invariants,
        )

        # Should have loaded all invariants
        assert len(evaluator.baseline_invariants) == len(real_invariants)

    def test_real_observation_formats(
        self, sample_baseline_invariants, real_observations
    ):
        """Test that real observations are properly handled."""
        evaluator = BlackboxEvaluator(
            execution_id="test_real_obs",
            existing_invariants=sample_baseline_invariants,
        )

        for obs in real_observations:
            obs_id = evaluator.record_observation(obs)
            assert obs_id is not None

            # Verify observation has expected fields
            assert obs.description is not None
            assert obs.affected_functions is not None
            assert obs.affected_files is not None
