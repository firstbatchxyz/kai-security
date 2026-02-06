"""Tests for BlackboxEvaluationRunner.

Uses real fixtures from the Ethena BBP protocol analysis.
"""

import tempfile
from pathlib import Path

import pytest

from evaluation.runner import BlackboxEvaluationRunner


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestRunnerInitialization:
    """Test runner initialization with real Ethena invariants."""

    def test_init_with_real_baseline(
        self, sample_baseline_invariants_list, temp_output_dir
    ):
        """Test initialization with real baseline invariants from Ethena."""
        runner = BlackboxEvaluationRunner(
            repo_path="/path/to/repo",
            baseline_invariants=sample_baseline_invariants_list,
            output_dir=temp_output_dir,
        )

        assert len(runner.baseline_invariants) == len(sample_baseline_invariants_list)
        assert runner.repo_path == str(Path("/path/to/repo").resolve())
        assert runner.output_dir == Path(temp_output_dir)

    def test_init_with_full_real_invariants(
        self, real_invariants_list, temp_output_dir
    ):
        """Test initialization with full set of real Ethena invariants."""
        runner = BlackboxEvaluationRunner(
            repo_path="/path/to/repo",
            baseline_invariants=real_invariants_list,
            output_dir=temp_output_dir,
        )

        assert len(runner.baseline_invariants) == len(real_invariants_list)
        # Verify invariants have expected structure
        for inv in runner.baseline_invariants:
            assert inv.id is not None
            assert inv.rule is not None
            assert len(inv.rule) > 10


class TestRepoSlug:
    """Test repository slug generation."""

    def test_repo_slug_simple(self, sample_baseline_invariants_list, temp_output_dir):
        """Test simple repo slug generation."""
        runner = BlackboxEvaluationRunner(
            repo_path="/path/to/my-repo",
            baseline_invariants=sample_baseline_invariants_list,
            output_dir=temp_output_dir,
        )

        slug = runner._repo_slug("/path/to/my-repo")
        assert slug == "my-repo"

    def test_repo_slug_special_chars(
        self, sample_baseline_invariants_list, temp_output_dir
    ):
        """Test repo slug with special characters."""
        runner = BlackboxEvaluationRunner(
            repo_path="/path/to/repo",
            baseline_invariants=sample_baseline_invariants_list,
            output_dir=temp_output_dir,
        )

        slug = runner._repo_slug("/path/to/repo with spaces & special!")
        # Special chars should be replaced with dashes
        assert " " not in slug
        assert "&" not in slug
        assert "!" not in slug

    def test_repo_slug_ethena_style(
        self, sample_baseline_invariants_list, temp_output_dir
    ):
        """Test repo slug with Ethena-style path."""
        runner = BlackboxEvaluationRunner(
            repo_path="/testbed/master/ethena-labs-ethena-2024-bbp",
            baseline_invariants=sample_baseline_invariants_list,
            output_dir=temp_output_dir,
        )

        slug = runner._repo_slug("/testbed/master/ethena-labs-ethena-2024-bbp")
        assert slug == "ethena-labs-ethena-2024-bbp"
