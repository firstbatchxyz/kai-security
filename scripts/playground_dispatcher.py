#!/usr/bin/env python
"""
Playground: Dispatcher E2E Demo

Minimal example showing the full Kai v2 pipeline:
1. Boot (preprocessing: setup → graph → profiler → actors → invariants → planning)
2. Run Loop (execute missions with STATE/QUANT agents)
3. Verification (inline after each exploit found)
4. Fixing (after all missions complete, generates patches for verified exploits)

Usage:
    uv run python scripts/playground_dispatcher.py --repo-path ./master/your-contracts
    uv run python scripts/playground_dispatcher.py --repo-path ./master/your-contracts --model anthropic/claude-3-5-sonnet
"""

import asyncio
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from kai.agents import settings
from kai.dispatcher.core import Dispatcher, DispatcherConfig
from kai.schemas import CampaignBudget

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("playground")


async def run_dispatcher_demo(
    repo_path: str,
    model: str = settings.MAIN_DEFAULT_MODEL,
    max_concurrent: int = 2,
    max_turns: int = settings.MAX_TOOL_TURNS,
) -> None:
    """
    Run the full dispatcher pipeline and print results.
    """
    repo_path = str(Path(repo_path).resolve())
    repo_name = Path(repo_path).name

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "output" / "playground" / f"{repo_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("DISPATCHER PLAYGROUND")
    print(f"{'=' * 70}")
    print(f"Repository: {repo_path}")
    print(f"Model: {model}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 70}\n")

    # Configure dispatcher
    config = DispatcherConfig(
        max_concurrent_agents=max_concurrent,
        max_invariants_per_cluster=5,
        max_campaigns=10,
        include_exploration=False,  # Focus on STATE/QUANT agents
        default_budget=CampaignBudget(
            max_missions=10,
            max_agents=4,
            max_turns_per_agent=max_turns,
        ),
        workspace_dir=str(output_dir / "workspaces"),
        model=model,
        use_openai=False,
    )

    # Create dispatcher (no state manager for simplicity)
    dispatcher = Dispatcher(config=config)

    # =========================================================================
    # PHASE 1: BOOT
    # =========================================================================
    print("\n[PHASE 1] BOOT - Running preprocessing pipeline...")
    start_time = datetime.now()

    success = await dispatcher.boot(
        repo_path=repo_path,
        model_name=model,
        use_openai=False,
    )

    if not success:
        print("❌ Boot failed!")
        return

    print(f"✓ Boot complete in {(datetime.now() - start_time).seconds}s")
    print(f"  - Invariants: {len(dispatcher.invariants)}")
    print(f"  - Campaigns: {len(dispatcher.campaigns)}")
    print(f"  - Missions queued: {dispatcher.mission_queue.qsize()}")

    # Show discovered invariants
    if dispatcher.invariants:
        print("\n  Invariants discovered:")
        for inv in list(dispatcher.invariants.values())[:5]:
            inv_type = inv.type.value if inv.type else "?"
            rule_preview = inv.rule[:60] + "..." if len(inv.rule) > 60 else inv.rule
            print(f"    [{inv_type}] {inv.id}")
            print(f"        {rule_preview}")
        if len(dispatcher.invariants) > 5:
            print(f"    ... and {len(dispatcher.invariants) - 5} more")

    # =========================================================================
    # PHASE 2: RUN LOOP (includes inline verification + post-loop fixing)
    # =========================================================================
    print(f"\n[PHASE 2] RUN LOOP - Executing missions...")

    await dispatcher.run_loop()

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # =========================================================================
    # PHASE 3: RESULTS
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    print(f"\nDuration: {duration:.1f}s")
    print(f"Missions completed: {len(dispatcher.completed_missions)}")
    print(f"Exploit candidates: {len(dispatcher.exploit_candidates)}")
    print(f"Verdicts: {len(dispatcher.verdicts)}")
    print(f"Fixes generated: {len(dispatcher.fixes)}")

    # Show verified exploits
    verified = [v for v in dispatcher.verdicts if v.is_valid]
    rejected = [v for v in dispatcher.verdicts if not v.is_valid]

    if verified:
        print(f"\n{'=' * 70}")
        print(f"VERIFIED EXPLOITS ({len(verified)})")
        print(f"{'=' * 70}")
        for i, verdict in enumerate(verified):
            severity = verdict.severity.value.upper() if verdict.severity else "?"
            vuln_class = verdict.vulnerability_class or "unknown"
            print(f"\n[{i + 1}] {severity} - {vuln_class}")
            print(f"    Mission: {verdict.mission_id}")
            print(f"    Invariant: {verdict.invariant_id}")
            if verdict.reasoning:
                reasoning = (
                    verdict.reasoning[:100] + "..."
                    if len(verdict.reasoning) > 100
                    else verdict.reasoning
                )
                print(f"    Reasoning: {reasoning}")

            # Show associated fixes
            if verdict.fixes:
                print(f"    Fixes: {len(verdict.fixes)}")
                for fix in verdict.fixes:
                    summary = (
                        fix.summary[:60] + "..."
                        if len(fix.summary) > 60
                        else fix.summary
                    )
                    print(f"      - {fix.fix_id}: {summary}")
                    print(
                        f"        Compiled: {fix.compiled}, Tests passed: {fix.tests_passed}"
                    )

    if rejected:
        print(f"\n{'-' * 70}")
        print(f"REJECTED ({len(rejected)})")
        print(f"{'-' * 70}")
        for verdict in rejected[:3]:
            reason = verdict.rejection_reason or "No reason"
            reason = reason[:80] + "..." if len(reason) > 80 else reason
            print(f"  - {verdict.mission_id}: {reason}")
        if len(rejected) > 3:
            print(f"  ... and {len(rejected) - 3} more")

    # Show fixes summary
    if dispatcher.fixes:
        print(f"\n{'=' * 70}")
        print(f"FIXES GENERATED ({len(dispatcher.fixes)})")
        print(f"{'=' * 70}")
        for fix in dispatcher.fixes:
            print(f"\n[{fix.fix_id}]")
            print(f"  Mission: {fix.mission_id}")
            print(f"  Summary: {fix.summary}")
            print(
                f"  Files: {', '.join(fix.files_changed) if fix.files_changed else 'N/A'}"
            )
            print(f"  Compiled: {fix.compiled} | Tests passed: {fix.tests_passed}")
            if fix.canonical_diff:
                diff_preview = (
                    fix.canonical_diff[:200] + "..."
                    if len(fix.canonical_diff) > 200
                    else fix.canonical_diff
                )
                print(f"  Diff preview:\n{diff_preview}")

    # Export full results
    report_path = output_dir / "results.json"
    dispatcher.export_results(str(report_path))

    # Also save fixes separately
    if dispatcher.fixes:
        fixes_path = output_dir / "fixes.json"
        with open(fixes_path, "w") as f:
            json.dump(
                [fix.model_dump() for fix in dispatcher.fixes], f, indent=2, default=str
            )
        print(f"\nFixes saved to: {fixes_path}")

    print(f"\nFull report: {report_path}")
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Dispatcher E2E Playground")
    parser.add_argument(
        "--repo-path",
        type=str,
        required=True,
        help="Path to target repository",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=settings.MAIN_DEFAULT_MODEL,
        help=f"Model to use (default: {settings.MAIN_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=2,
        help="Max concurrent agents (default: 2)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=settings.MAX_TOOL_TURNS,
        help=f"Max turns per agent (default: {settings.MAX_TOOL_TURNS})",
    )

    args = parser.parse_args()

    # Resolve path
    repo_path = Path(args.repo_path)
    if not repo_path.is_absolute():
        repo_path = PROJECT_ROOT / repo_path

    if not repo_path.exists():
        print(f"Error: Repository not found: {repo_path}")
        sys.exit(1)

    asyncio.run(
        run_dispatcher_demo(
            repo_path=str(repo_path),
            model=args.model,
            max_concurrent=args.concurrent,
            max_turns=args.max_turns,
        )
    )


if __name__ == "__main__":
    main()
