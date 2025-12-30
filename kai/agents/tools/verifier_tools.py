"""
Tools for VerifierAgent - validating exploit findings.

Provides:
- Graph tools for verifying claims about real code
- Test execution tools to validate PoC
- submit_verdict: Output the final verdict on an exploit candidate
"""

from typing import Optional, Dict, Any, Literal

# Import shared tools for code analysis
from kai.agents.tools.tools import (
    dependency_graph_resolve,
    dependency_graph_snippet,
    dependency_graph_neighbors,
    _get_current_agent,
)

# Import test execution tools from state_tools
from kai.agents.tools.state_tools import (
    write_and_compile,
    run_test,
)


def submit_verdict(
    is_valid: bool,
    severity: Literal["critical", "high", "medium", "low", "informational"],
    reasoning: str,
    test_passed: bool = False,
    uses_mock_contracts: bool = False,
    is_economically_feasible: bool = True,
    is_known_limitation: bool = False,
    targets_real_implementation: bool = True,
    vulnerability_class: str = "",
    rejection_reason: Optional[str] = None,
    attack_cost_estimate: Optional[str] = None,
    attacker_profit_estimate: Optional[str] = None,
    cost_benefit_ratio: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Submit the final verdict on an exploit candidate.

    Call this tool after running the PoC test and analyzing the results.
    This must be called before the agent terminates.

    Args:
        is_valid: True if this is a real, exploitable vulnerability
        severity: One of "critical", "high", "medium", "low", "informational"
        reasoning: Full analysis and justification for the verdict
        test_passed: True if the PoC test compiled and passed (from run_test result)
        uses_mock_contracts: True if PoC uses fake/hostile test contracts
        is_economically_feasible: True if attack is profitable or cheap enough to execute
        is_known_limitation: True if this is a documented design tradeoff, not a bug
        targets_real_implementation: True if PoC tests actual code, not just interfaces
        vulnerability_class: Category (e.g., "reentrancy", "access_control", "donation_attack")
        rejection_reason: If invalid, specific reason for rejection
        attack_cost_estimate: Estimated cost to attacker (e.g., "$1M donation")
        attacker_profit_estimate: Estimated attacker gain (e.g., "Can extract $10K")
        cost_benefit_ratio: Ratio of cost to benefit (e.g., "1000:1 loss ratio")

    Returns:
        Confirmation that verdict was submitted

    Example (valid exploit, test passed):
        submit_verdict(
            is_valid=True,
            severity="high",
            reasoning="The withdraw() function is vulnerable to reentrancy...",
            test_passed=True,
            vulnerability_class="reentrancy",
            targets_real_implementation=True,
            is_economically_feasible=True,
            attacker_profit_estimate="Can drain entire vault balance"
        )

    Example (invalid - test failed):
        submit_verdict(
            is_valid=False,
            severity="informational",
            reasoning="PoC test did not pass - the exploit does not work",
            test_passed=False,
            rejection_reason="Test failed to demonstrate the vulnerability"
        )

    Example (invalid - mock contract):
        submit_verdict(
            is_valid=False,
            severity="informational",
            reasoning="PoC creates HostileToken that burns on transfer...",
            test_passed=True,
            uses_mock_contracts=True,
            targets_real_implementation=False,
            rejection_reason="Uses mock contract that violates ERC20 assumptions"
        )
    """
    from kai.schemas import Verdict, VerdictSeverity

    agent = _get_current_agent()
    if agent is None:
        return {"submitted": False, "error": "No agent context available"}

    # Get exploit candidate info from agent
    exploit_candidate = getattr(agent, "exploit_candidate", None)
    if exploit_candidate is None:
        return {"submitted": False, "error": "No exploit candidate in agent context"}

    # Map severity string to enum
    severity_map = {
        "critical": VerdictSeverity.CRITICAL,
        "high": VerdictSeverity.HIGH,
        "medium": VerdictSeverity.MEDIUM,
        "low": VerdictSeverity.LOW,
        "informational": VerdictSeverity.INFORMATIONAL,
    }
    verdict_severity = severity_map.get(severity.lower(), VerdictSeverity.INFORMATIONAL)

    # Build verdict
    verdict = Verdict(
        mission_id=exploit_candidate.mission_id,
        invariant_id=exploit_candidate.invariant_id,
        agent_id=exploit_candidate.agent_id,
        is_valid=is_valid,
        severity=verdict_severity,
        uses_mock_contracts=uses_mock_contracts,
        is_economically_feasible=is_economically_feasible,
        is_known_limitation=is_known_limitation,
        targets_real_implementation=targets_real_implementation,
        vulnerability_class=vulnerability_class,
        reasoning=reasoning,
        rejection_reason=rejection_reason,
        attack_cost_estimate=attack_cost_estimate,
        attacker_profit_estimate=attacker_profit_estimate,
        cost_benefit_ratio=cost_benefit_ratio,
        poc_path=exploit_candidate.target_file,
        test_passed=test_passed,
    )

    # Store verdict in agent
    agent._verdict = verdict

    return {
        "submitted": True,
        "verdict": {
            "is_valid": is_valid,
            "severity": severity,
            "vulnerability_class": vulnerability_class,
            "test_passed": test_passed,
            "uses_mock_contracts": uses_mock_contracts,
            "is_economically_feasible": is_economically_feasible,
        },
        "message": f"Verdict submitted: {'VALID' if is_valid else 'INVALID'} - {severity.upper()} (test_passed={test_passed})",
    }


__all__ = [
    "dependency_graph_resolve",
    "dependency_graph_snippet",
    "dependency_graph_neighbors",
    "write_and_compile",
    "run_test",
    "submit_verdict",
]
