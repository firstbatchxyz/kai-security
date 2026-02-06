"""Tests for PostHocDeduplicator.

Uses real invariants from the Ethena BBP protocol analysis as fixtures.
Tests the parallel bulk deduplication approach.
"""

import pytest

from evaluation.post_hoc_deduplicator import (
    PostHocDeduplicator,
)
from kai.schemas import Invariant, InvariantType


class TestDeduplicatorInitialization:
    """Test deduplicator initialization."""

    def test_init_with_baseline(self, real_invariants_list):
        """Test initialization with real baseline invariants."""
        deduplicator = PostHocDeduplicator(
            baseline_invariants=real_invariants_list,
        )

        assert len(deduplicator.baseline_invariants) == len(real_invariants_list)
        assert deduplicator.batch_size == 10  # Default
        assert deduplicator.max_concurrent == 5  # Default

    def test_init_with_custom_settings(self, real_invariants_list):
        """Test initialization with custom batch size and concurrency."""
        deduplicator = PostHocDeduplicator(
            baseline_invariants=real_invariants_list,
            batch_size=20,
            max_concurrent=10,
        )

        assert deduplicator.batch_size == 20
        assert deduplicator.max_concurrent == 10


class TestBulkDeduplication:
    """Test bulk deduplication functionality."""

    @pytest.mark.asyncio
    async def test_no_baselines_all_novel(self):
        """Test that with no baselines, all candidates are novel."""
        deduplicator = PostHocDeduplicator(baseline_invariants=[])

        candidates = [
            Invariant(
                id=f"CANDIDATE_{i}",
                type=InvariantType.ACCESS,
                rule=f"Some access control rule {i}",
                target_function_ids=[f"func_{i}"],
                confidence=0.8,
            )
            for i in range(5)
        ]

        result = await deduplicator.deduplicate(candidates)

        assert result.total_candidates == 5
        assert result.novel_count == 5
        assert result.duplicate_count == 0
        assert result.total_llm_calls == 0
        assert len(result.novel_invariants) == 5

    @pytest.mark.asyncio
    async def test_parallel_processing(self, real_invariants_list, mock_llm_client):
        """Test that candidates are processed in parallel."""
        create_client, get_pricing = mock_llm_client()

        deduplicator = PostHocDeduplicator(
            baseline_invariants=real_invariants_list,
            batch_size=10,
            max_concurrent=3,
            _client_factory=create_client,
            _pricing_factory=get_pricing,
        )

        candidates = [
            Invariant(
                id=f"CANDIDATE_{i}",
                type=InvariantType.LIVENESS,
                rule=f"Unique liveness rule {i}",
                target_function_ids=[f"uniqueFunc_{i}"],
                confidence=0.7,
            )
            for i in range(5)
        ]

        result = await deduplicator.deduplicate(candidates)

        # Should have processed all candidates
        assert result.total_candidates == 5
        assert result.total_llm_calls > 0

    @pytest.mark.asyncio
    async def test_duplicate_detection(
        self, real_invariants_list, mock_llm_client, mock_duplicate_response
    ):
        """Test that duplicates are correctly identified."""
        create_client, get_pricing = mock_llm_client(mock_duplicate_response)

        first_inv = real_invariants_list[0]

        deduplicator = PostHocDeduplicator(
            baseline_invariants=real_invariants_list,
            batch_size=10,
            max_concurrent=5,
            _client_factory=create_client,
            _pricing_factory=get_pricing,
        )

        # Create a candidate that should be marked as duplicate
        candidates = [
            Invariant(
                id="DUPLICATE_CANDIDATE",
                type=first_inv.type,
                rule="Very similar to existing invariant",
                target_function_ids=first_inv.target_function_ids,
                target_var_ids=first_inv.target_var_ids,
                confidence=0.85,
            ),
        ]

        result = await deduplicator.deduplicate(candidates)

        assert result.duplicate_count == 1
        assert result.novel_count == 0
        assert len(result.duplicate_invariants) == 1

    @pytest.mark.asyncio
    async def test_novel_detection(
        self, real_invariants_list, mock_llm_client, mock_non_duplicate_response
    ):
        """Test that novel invariants are correctly identified."""
        create_client, get_pricing = mock_llm_client(mock_non_duplicate_response)

        deduplicator = PostHocDeduplicator(
            baseline_invariants=real_invariants_list,
            batch_size=10,
            max_concurrent=5,
            _client_factory=create_client,
            _pricing_factory=get_pricing,
        )

        candidates = [
            Invariant(
                id="NOVEL_CANDIDATE",
                type=InvariantType.LIVENESS,
                rule="A completely unique liveness invariant",
                target_function_ids=["uniqueFunc"],
                confidence=0.7,
            ),
        ]

        result = await deduplicator.deduplicate(candidates)

        assert result.novel_count == 1
        assert result.duplicate_count == 0
        assert len(result.novel_invariants) == 1


class TestBatching:
    """Test the batching mechanism."""

    @pytest.mark.asyncio
    async def test_correct_batch_count(self, mock_llm_client):
        """Test that baselines are batched correctly."""
        # Create 25 baseline invariants
        baselines = [
            Invariant(
                id=f"BASELINE_{i}",
                type=InvariantType.ACCESS,
                rule=f"Baseline rule {i}",
                target_function_ids=[f"func_{i}"],
                confidence=0.8,
            )
            for i in range(25)
        ]

        create_client, get_pricing = mock_llm_client()

        deduplicator = PostHocDeduplicator(
            baseline_invariants=baselines,
            batch_size=10,
            max_concurrent=5,
            _client_factory=create_client,
            _pricing_factory=get_pricing,
        )

        candidates = [
            Invariant(
                id="BATCH_TEST",
                type=InvariantType.ACCESS,
                rule="Testing batching",
                target_function_ids=["testFunc"],
                confidence=0.7,
            ),
        ]

        result = await deduplicator.deduplicate(candidates)

        # With 25 baselines and batch_size=10, should make 3 LLM calls per candidate
        # (unless early exit on duplicate)
        assert result.total_llm_calls <= 3

    @pytest.mark.asyncio
    async def test_early_exit_on_duplicate(self, mock_llm_client):
        """Test that processing stops early when duplicate is found."""
        # Create 50 baseline invariants
        baselines = [
            Invariant(
                id=f"BASELINE_{i}",
                type=InvariantType.ACCESS,
                rule=f"Baseline rule {i}",
                target_function_ids=[f"func_{i}"],
                confidence=0.8,
            )
            for i in range(50)
        ]

        # Response marks first baseline as duplicate
        duplicate_response = """{
            "results": [
                {"baseline_id": "BASELINE_0", "is_duplicate": true, "reasoning": "Same property"}
            ]
        }"""
        create_client, get_pricing = mock_llm_client(duplicate_response)

        deduplicator = PostHocDeduplicator(
            baseline_invariants=baselines,
            batch_size=10,
            max_concurrent=5,
            _client_factory=create_client,
            _pricing_factory=get_pricing,
        )

        candidates = [
            Invariant(
                id="EARLY_EXIT_TEST",
                type=InvariantType.ACCESS,
                rule="This will be a duplicate",
                target_function_ids=["testFunc"],
                confidence=0.7,
            ),
        ]

        result = await deduplicator.deduplicate(candidates)

        # Should have exited early after first batch
        assert result.duplicate_count == 1
        # Should have made only 1 LLM call (found duplicate in first batch)
        assert result.total_llm_calls == 1


class TestCostTracking:
    """Test LLM cost tracking."""

    @pytest.mark.asyncio
    async def test_cost_accumulation(
        self, real_invariants_list, mock_llm_client_with_usage
    ):
        """Test that costs accumulate across all candidates."""
        create_client, get_pricing = mock_llm_client_with_usage(
            prompt_tokens=100, completion_tokens=50
        )

        deduplicator = PostHocDeduplicator(
            baseline_invariants=real_invariants_list,
            batch_size=10,
            max_concurrent=5,
            _client_factory=create_client,
            _pricing_factory=get_pricing,
        )

        candidates = [
            Invariant(
                id=f"COST_TEST_{i}",
                type=InvariantType.ACCESS,
                rule=f"Cost tracking test {i}",
                target_function_ids=[f"func_{i}"],
                confidence=0.7,
            )
            for i in range(3)
        ]

        result = await deduplicator.deduplicate(candidates)

        # Should have tracked costs
        assert result.total_llm_cost > 0
        assert result.total_llm_tokens["prompt_tokens"] > 0
        assert result.total_llm_tokens["completion_tokens"] > 0


class TestOutputFormat:
    """Test output format and serialization."""

    @pytest.mark.asyncio
    async def test_output_dict_format(self, mock_llm_client):
        """Test that output dict has expected format."""
        baselines = [
            Invariant(
                id="BASELINE_1",
                type=InvariantType.ACCESS,
                rule="Baseline rule",
                target_function_ids=["func1"],
                confidence=0.8,
            ),
        ]

        create_client, get_pricing = mock_llm_client()

        deduplicator = PostHocDeduplicator(
            baseline_invariants=baselines,
            batch_size=10,
            max_concurrent=5,
            _client_factory=create_client,
            _pricing_factory=get_pricing,
        )

        candidates = [
            Invariant(
                id="CANDIDATE_1",
                type=InvariantType.LIVENESS,
                rule="Novel candidate rule",
                target_function_ids=["uniqueFunc"],
                confidence=0.7,
            ),
        ]

        result = await deduplicator.deduplicate(candidates)
        output = result.to_output_dict()

        # Check structure
        assert "metadata" in output
        assert "novel_invariants" in output
        assert "duplicates" in output

        # Check metadata fields
        assert "total_candidates" in output["metadata"]
        assert "duplicates_removed" in output["metadata"]
        assert "novel_count" in output["metadata"]
        assert "processing_time_seconds" in output["metadata"]
        assert "llm_cost" in output["metadata"]


class TestResponseParsing:
    """Test parsing of LLM responses."""

    def test_parse_valid_json(self, real_invariants_list):
        """Test parsing valid JSON response."""
        deduplicator = PostHocDeduplicator(
            baseline_invariants=real_invariants_list,
        )

        content = """
        {
            "results": [
                {"baseline_id": "INV_1", "is_duplicate": false, "reasoning": "Different"},
                {"baseline_id": "INV_2", "is_duplicate": true, "reasoning": "Same property"}
            ]
        }
        """

        batch_ids = ["INV_1", "INV_2"]
        results = deduplicator._parse_batch_response(content, batch_ids)

        assert len(results) == 2
        assert results[0] == ("INV_1", False, "Different")
        assert results[1] == ("INV_2", True, "Same property")

    def test_parse_markdown_wrapped(self, real_invariants_list):
        """Test parsing response wrapped in markdown."""
        deduplicator = PostHocDeduplicator(
            baseline_invariants=real_invariants_list,
        )

        content = """
        ```json
        {
            "results": [
                {"baseline_id": "INV_1", "is_duplicate": false, "reasoning": "Not duplicate"}
            ]
        }
        ```
        """

        batch_ids = ["INV_1"]
        results = deduplicator._parse_batch_response(content, batch_ids)

        assert len(results) == 1
        assert results[0] == ("INV_1", False, "Not duplicate")

    def test_parse_missing_baseline_defaults_false(self, real_invariants_list):
        """Test that missing baselines default to not duplicate."""
        deduplicator = PostHocDeduplicator(
            baseline_invariants=real_invariants_list,
        )

        content = """
        {
            "results": [
                {"baseline_id": "INV_1", "is_duplicate": false, "reasoning": "Checked"}
            ]
        }
        """

        # INV_2 is missing from response
        batch_ids = ["INV_1", "INV_2"]
        results = deduplicator._parse_batch_response(content, batch_ids)

        assert len(results) == 2
        assert results[0] == ("INV_1", False, "Checked")
        assert results[1] == ("INV_2", False, "Not included in LLM response")

    def test_parse_invalid_json_defaults_false(self, real_invariants_list):
        """Test that invalid JSON defaults all to not duplicate."""
        deduplicator = PostHocDeduplicator(
            baseline_invariants=real_invariants_list,
        )

        content = "This is not valid JSON"
        batch_ids = ["INV_1", "INV_2"]

        results = deduplicator._parse_batch_response(content, batch_ids)

        assert len(results) == 2
        assert not results[0][1]  # is_duplicate = False
        assert not results[1][1]  # is_duplicate = False


class TestRealDataIntegration:
    """Integration tests with real Ethena data."""

    @pytest.mark.asyncio
    async def test_with_real_baseline(
        self, real_invariants_list, mock_llm_client, mock_non_duplicate_response
    ):
        """Test deduplication with real Ethena baseline invariants."""
        create_client, get_pricing = mock_llm_client(mock_non_duplicate_response)

        deduplicator = PostHocDeduplicator(
            baseline_invariants=real_invariants_list,
            batch_size=10,
            max_concurrent=5,
            _client_factory=create_client,
            _pricing_factory=get_pricing,
        )

        # Create candidates that don't overlap with real baseline
        candidates = [
            Invariant(
                id=f"NOVEL_{i}",
                type=InvariantType.LIVENESS,
                rule=f"Novel liveness property {i} not in baseline",
                target_function_ids=[f"novelFunc_{i}"],
                confidence=0.7,
            )
            for i in range(3)
        ]

        result = await deduplicator.deduplicate(candidates)

        assert result.total_candidates == 3
        assert result.processing_time_seconds >= 0
        assert result.model_name == "z-ai/glm-4.7"  # Default
