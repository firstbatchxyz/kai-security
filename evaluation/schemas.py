"""
Pydantic models for the Blackbox Agent Evaluation system.

Defines data structures for:
- Observation and invariant evaluation records
- Aggregated metrics
- Complete evaluation reports

Note: Duplicate detection is handled post-hoc by PostHocDeduplicator,
which uses its own dataclasses defined in post_hoc_deduplicator.py.
"""

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class EvaluationRecord(BaseModel):
    """Unified evaluation record tracking observation → invariant."""

    # Observation info
    observation_id: str  # worker_id + mission_id hash
    mission_id: str
    worker_id: str
    description_preview: str
    anomaly_type: Optional[str] = None

    # Synthesis result
    synthesis_success: bool = False
    synthesis_error: Optional[str] = None

    # Invariant info (populated if synthesis succeeded)
    invariant_id: Optional[str] = None
    invariant_type: Optional[str] = None
    rule: Optional[str] = None
    confidence: Optional[float] = None
    target_function_count: int = 0
    target_var_count: int = 0
    target_file_count: int = 0

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BlackboxEvaluationMetrics(BaseModel):
    """Aggregated metrics for Blackbox Agent evaluation.

    Note: Duplicate detection metrics are tracked separately by PostHocDeduplicator.
    """

    # Observation metrics
    total_observations: int = 0
    observations_with_synthesis: int = 0
    observations_without_synthesis: int = 0
    observation_to_invariant_rate: float = 0.0  # % of observations -> invariants

    # Synthesis metrics
    total_synthesized: int = 0


class BlackboxEvaluationReport(BaseModel):
    """Complete evaluation report for Blackbox Agent pipeline.

    Note: Duplicate detection is handled separately by PostHocDeduplicator
    which produces its own DeduplicationResult output.
    """

    # Metadata
    execution_id: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    repo_path: Optional[str] = None

    # Aggregated metrics
    metrics: BlackboxEvaluationMetrics = Field(
        default_factory=BlackboxEvaluationMetrics
    )

    # Detailed records
    records: List[EvaluationRecord] = Field(default_factory=list)

    # Baseline information (for reference, deduplication done post-hoc)
    baseline_invariant_count: int = 0
    baseline_invariant_ids: List[str] = Field(default_factory=list)
