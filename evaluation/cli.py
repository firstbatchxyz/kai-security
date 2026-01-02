#!/usr/bin/env python3
"""
Blackbox Agent Evaluation CLI.

Usage examples:

# Run full pipeline on a repo:
python -m evaluation.cli run \
    --repo-path ./target-repo \
    --baseline-invariants ./baseline.json \
    --output-dir ./eval_output \
    --num-turns 50

# Post-hoc deduplication (fast, parallel):
python -m evaluation.cli deduplicate \
    --invariants ./synthesized.json \
    --baseline ./baseline.json \
    --output ./novel_invariants.json \
    --batch-size 10 \
    --max-concurrent 5
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import List

from kai.schemas import Invariant


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    When verbose is True, shows DEBUG logs for evaluation code only.
    Third-party libraries (httpcore, httpx, openai, etc.) are kept at WARNING.
    """
    # Always keep root logger at INFO to avoid third-party DEBUG spam
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Silence noisy third-party loggers
    noisy_loggers = [
        "httpcore",
        "httpx",
        "openai",
        "httpcore.connection",
        "httpcore.http11",
        "openai._base_client",
        "asyncio",
        "urllib3",
    ]
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Set our evaluation loggers to DEBUG if verbose
    if verbose:
        our_loggers = [
            "evaluation",
            "evaluation.runner",
            "evaluation.evaluator",
            "evaluation.post_hoc_deduplicator",
            "kai",
        ]
        for logger_name in our_loggers:
            logging.getLogger(logger_name).setLevel(logging.DEBUG)


def load_invariants(path: str) -> List[Invariant]:
    """Load invariants from JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    invariants = []
    if isinstance(data, list):
        for item in data:
            invariants.append(Invariant(**item))
    elif isinstance(data, dict):
        if "invariants" in data:
            for item in data["invariants"]:
                invariants.append(Invariant(**item))
        else:
            # Assume it's a single invariant or dict of invariants
            for key, value in data.items():
                if isinstance(value, dict) and "rule" in value:
                    invariants.append(Invariant(**value))

    return invariants


async def cmd_run(args: argparse.Namespace) -> int:
    """Run the full Blackbox -> Synthesizer -> Evaluation pipeline."""
    from evaluation.runner import BlackboxEvaluationRunner

    # Load baseline invariants
    baseline_invariants = load_invariants(args.baseline_invariants)
    print(f"Loaded {len(baseline_invariants)} baseline invariants")

    # Create runner
    runner = BlackboxEvaluationRunner(
        repo_path=args.repo_path,
        baseline_invariants=baseline_invariants,
        model_name=args.model,
        use_openai=args.use_openai,
        output_dir=args.output_dir,
    )

    # Run full pipeline
    print(f"Running full pipeline on {args.repo_path}...")
    report = await runner.run_full_pipeline(
        num_turns=args.num_turns,
        campaign_label=args.campaign_label,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Execution ID: {report.execution_id}")
    print(f"Total Observations: {report.metrics.total_observations}")
    print(f"Observations with Synthesis: {report.metrics.observations_with_synthesis}")
    print(f"Total Synthesized: {report.metrics.total_synthesized}")
    print(
        f"Observation -> Invariant Rate: {report.metrics.observation_to_invariant_rate:.2%}"
    )
    print("=" * 60)
    print("\nNote: Run 'deduplicate' command for post-hoc duplicate detection.")

    # Save artifacts
    artifacts = runner.save_artifacts()
    print("\nArtifacts saved:")
    for name, path in artifacts.items():
        print(f"  - {name}: {path}")

    return 0


async def cmd_deduplicate(args: argparse.Namespace) -> int:
    """Post-hoc deduplication of synthesized invariants."""
    from evaluation.post_hoc_deduplicator import PostHocDeduplicator

    # Load data
    baseline_invariants = load_invariants(args.baseline)
    print(f"Loaded {len(baseline_invariants)} baseline invariants")

    synthesized_invariants = load_invariants(args.invariants)
    print(f"Loaded {len(synthesized_invariants)} synthesized invariants")

    # Create deduplicator
    deduplicator = PostHocDeduplicator(
        baseline_invariants=baseline_invariants,
        model_name=args.model,
        use_openai=args.use_openai,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
    )

    # Run deduplication
    print(
        f"Running parallel deduplication "
        f"(batch_size={args.batch_size}, max_concurrent={args.max_concurrent})..."
    )
    result = await deduplicator.deduplicate(synthesized_invariants)

    # Print summary
    print("\n" + "=" * 60)
    print("DEDUPLICATION SUMMARY")
    print("=" * 60)
    print(f"Total Candidates: {result.total_candidates}")
    print(f"Duplicates Removed: {result.duplicate_count}")
    print(f"Novel Invariants: {result.novel_count}")
    print(f"Novel Rate: {result.novel_count / max(result.total_candidates, 1):.2%}")
    print(f"Processing Time: {result.processing_time_seconds:.1f}s")
    print(f"LLM Calls: {result.total_llm_calls}")
    print(f"LLM Cost: ${result.total_llm_cost:.4f}")
    print("=" * 60)

    # Save output
    output_data = result.to_output_dict()
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\nOutput saved to: {args.output}")
    print(f"Novel invariants: {len(result.novel_invariants)}")

    return 0


async def dispatch_command(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate command handler."""
    if args.command == "run":
        return await cmd_run(args)
    elif args.command == "deduplicate":
        return await cmd_deduplicate(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Blackbox Agent Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # run: Full pipeline
    run_parser = subparsers.add_parser(
        "run",
        help="Run full Blackbox -> Synthesizer -> Evaluation pipeline",
    )
    run_parser.add_argument(
        "--repo-path",
        required=True,
        help="Path to target repository",
    )
    run_parser.add_argument(
        "--baseline-invariants",
        required=True,
        help="Path to JSON file with baseline invariants",
    )
    run_parser.add_argument(
        "--output-dir",
        default="./evaluation_output",
        help="Directory for output artifacts (default: ./evaluation_output)",
    )
    run_parser.add_argument(
        "--num-turns",
        type=int,
        default=50,
        help="Budget for Blackbox Agent (default: 50)",
    )
    run_parser.add_argument(
        "--model",
        default="openai/gpt-5.2",
        help="Model for Blackbox Agent (default: openai/gpt-5.2)",
    )
    run_parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API directly instead of OpenRouter",
    )
    run_parser.add_argument(
        "--campaign-label",
        default=None,
        help="Campaign label for tracking (auto-generated if not provided)",
    )

    # deduplicate: Post-hoc bulk deduplication with parallel LLM calls
    dedup_parser = subparsers.add_parser(
        "deduplicate",
        help="Post-hoc deduplication of synthesized invariants (fast, parallel)",
    )
    dedup_parser.add_argument(
        "--invariants",
        required=True,
        help="Path to JSON file with synthesized invariants to deduplicate",
    )
    dedup_parser.add_argument(
        "--baseline",
        required=True,
        help="Path to JSON file with baseline invariants to compare against",
    )
    dedup_parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON file for novel invariants",
    )
    dedup_parser.add_argument(
        "--model",
        default="z-ai/glm-4.7",
        help="Model for semantic comparison (default: z-ai/glm-4.7)",
    )
    dedup_parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Use OpenAI API directly instead of OpenRouter",
    )
    dedup_parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of baseline invariants per LLM call (default: 10)",
    )
    dedup_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent LLM calls (default: 5)",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    try:
        return asyncio.run(dispatch_command(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
