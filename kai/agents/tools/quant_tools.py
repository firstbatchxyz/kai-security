"""
Tools for QuantAgent - finding numeric/mathematical vulnerabilities.

Provides:
- Graph tools for understanding code (reused from shared tools)
- write_and_compile: Write file + immediate compilation feedback
- run_test: Run tests with parsed output (framework-agnostic)
- analyze_arithmetic: Identify arithmetic operations and potential vulnerabilities
- compute_boundary_values: Generate edge case test values
"""

import re
from typing import Optional, Dict, Any

# Import shared tools
from kai.agents.tools.tools import (
    dependency_graph_loc,
    dependency_graph_slice,
    dependency_graph_neighbors,
    dependency_graph_resolve,
    dependency_graph_snippet,
    dependency_graph_callers,
    dependency_graph_callees,
    dependency_graph_protocol_entrypoints,
    _get_current_agent,
    write_and_compile,
    register_exploit,
)

# Import shared test execution tools from state_tools
from kai.agents.tools.state_tools import run_test, patch_file


def analyze_arithmetic(function_id: str) -> Dict[str, Any]:
    """
    Analyze arithmetic operations in a function for potential numeric vulnerabilities.

    Examines the function's code to identify:
    - Arithmetic operations (+, -, *, /, %, **)
    - Division operations (potential precision loss)
    - Unchecked blocks (potential overflow/underflow)
    - Type casts (potential truncation)
    - Comparison operations with numeric values

    Args:
        function_id: The ID of the function to analyze (e.g., "Vault.deposit(uint256)")

    Returns:
        {
            "function_id": str,
            "arithmetic_ops": List[Dict],  # {op, line, context}
            "divisions": List[Dict],        # {line, context, potential_issue}
            "unchecked_blocks": List[Dict], # {start_line, end_line}
            "type_casts": List[Dict],       # {from_type, to_type, line}
            "risk_indicators": List[str],   # High-level risk descriptions
            "code_snippet": str,
            "error": Optional[str]
        }

    Example:
        result = analyze_arithmetic("Vault.convertToShares(uint256)")
        # Returns analysis of arithmetic in the function including
        # potential precision loss from division operations
    """
    agent = _get_current_agent()
    if agent is None:
        return {"error": "No agent context available"}

    # Get function location
    from kai.agents.tools.tools import _get_query_engine

    engine = _get_query_engine()
    if engine is None:
        return {"error": "Dependency graph is not available"}

    try:
        loc = engine.loc(function_id)
        if isinstance(loc, dict) and "error" in loc:
            return {"error": loc["error"]}

        # Get code snippet
        file_path = loc.get("file", "")
        span = loc.get("span", {})
        start_line = span.get("start", 1)
        end_line = span.get("end", start_line + 50)

        code = engine.snippet(file_path, {"start": start_line, "end": end_line})
        if isinstance(code, dict) and "error" in code:
            return {"error": code["error"]}

        # Analyze the code for arithmetic patterns
        result: Dict[str, Any] = {
            "function_id": function_id,
            "file": file_path,
            "lines": f"{start_line}-{end_line}",
            "arithmetic_ops": [],
            "divisions": [],
            "unchecked_blocks": [],
            "type_casts": [],
            "risk_indicators": [],
            "code_snippet": code,
        }

        lines = code.split("\n")
        in_unchecked = False
        unchecked_start = None

        for i, line in enumerate(lines, start=start_line):
            stripped = line.strip()

            # Track unchecked blocks
            if "unchecked" in stripped and "{" in stripped:
                in_unchecked = True
                unchecked_start = i
            if in_unchecked and "}" in stripped:
                result["unchecked_blocks"].append(
                    {"start_line": unchecked_start, "end_line": i}
                )
                in_unchecked = False

            # Find arithmetic operations
            # Division (highest risk for precision loss)
            if "/" in stripped and "//" not in stripped and "/*" not in stripped:
                result["divisions"].append(
                    {
                        "line": i,
                        "context": stripped,
                        "in_unchecked": in_unchecked,
                        "potential_issue": "Division may cause precision loss (truncation)",
                    }
                )
                result["arithmetic_ops"].append(
                    {"op": "/", "line": i, "context": stripped}
                )

            # Multiplication
            if "*" in stripped and "**" not in stripped and "/*" not in stripped:
                result["arithmetic_ops"].append(
                    {"op": "*", "line": i, "context": stripped}
                )
                if in_unchecked:
                    result["risk_indicators"].append(
                        f"Line {i}: Multiplication in unchecked block - potential overflow"
                    )

            # Addition/Subtraction in unchecked
            if in_unchecked:
                if "+" in stripped:
                    result["arithmetic_ops"].append(
                        {"op": "+", "line": i, "context": stripped}
                    )
                    result["risk_indicators"].append(
                        f"Line {i}: Addition in unchecked block - potential overflow"
                    )
                if "-" in stripped and "->" not in stripped:
                    result["arithmetic_ops"].append(
                        {"op": "-", "line": i, "context": stripped}
                    )
                    result["risk_indicators"].append(
                        f"Line {i}: Subtraction in unchecked block - potential underflow"
                    )

            # Modulo
            if "%" in stripped:
                result["arithmetic_ops"].append(
                    {"op": "%", "line": i, "context": stripped}
                )

            # Exponentiation
            if "**" in stripped:
                result["arithmetic_ops"].append(
                    {"op": "**", "line": i, "context": stripped}
                )
                result["risk_indicators"].append(
                    f"Line {i}: Exponentiation - high overflow risk"
                )

            # Type casts (Solidity-style)
            cast_patterns = [
                r"uint(\d+)\(",
                r"int(\d+)\(",
                r"uint\(",
                r"int\(",
            ]
            for pattern in cast_patterns:
                if re.search(pattern, stripped):
                    result["type_casts"].append({"line": i, "context": stripped})
                    result["risk_indicators"].append(
                        f"Line {i}: Type cast - potential truncation or sign issues"
                    )

        # Add high-level risk assessment
        if result["divisions"]:
            result["risk_indicators"].insert(
                0,
                f"Found {len(result['divisions'])} division operations - check for precision loss",
            )
        if result["unchecked_blocks"]:
            result["risk_indicators"].insert(
                0,
                f"Found {len(result['unchecked_blocks'])} unchecked blocks - overflow/underflow possible",
            )

        return result

    except Exception as e:
        return {"error": str(e)}


def compute_boundary_values(
    param_type: str,
    context: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate boundary and edge case values for testing numeric parameters.

    Computes values that are commonly problematic for numeric types:
    - Zero and one
    - Maximum and minimum values
    - Values near overflow boundaries
    - Common decimal bases

    Args:
        param_type: The numeric type (e.g., "uint256", "int128", "u64", "i32", "float")
        context: Optional context about the parameter's purpose (e.g., "amount", "shares", "fee")

    Returns:
        {
            "type": str,
            "values": List[Dict],  # {value, description, risk}
            "overflow_pairs": List[Dict],  # {a, b, op} where a op b overflows
            "context_specific": List[Dict],  # Values specific to the context
            "test_assertions": List[str],   # Suggested assertions to test
        }

    Example:
        result = compute_boundary_values("uint256", context="shares")
        # Returns boundary values relevant for share calculations
    """
    result: Dict[str, Any] = {
        "type": param_type,
        "values": [],
        "overflow_pairs": [],
        "context_specific": [],
        "test_assertions": [],
    }

    # Normalize type
    type_lower = param_type.lower().strip()

    # Define type ranges
    type_info = {
        # Solidity unsigned integers
        "uint256": {"bits": 256, "signed": False, "max": 2**256 - 1, "min": 0},
        "uint128": {"bits": 128, "signed": False, "max": 2**128 - 1, "min": 0},
        "uint64": {"bits": 64, "signed": False, "max": 2**64 - 1, "min": 0},
        "uint32": {"bits": 32, "signed": False, "max": 2**32 - 1, "min": 0},
        "uint16": {"bits": 16, "signed": False, "max": 2**16 - 1, "min": 0},
        "uint8": {"bits": 8, "signed": False, "max": 2**8 - 1, "min": 0},
        # Solidity signed integers
        "int256": {
            "bits": 256,
            "signed": True,
            "max": 2**255 - 1,
            "min": -(2**255),
        },
        "int128": {
            "bits": 128,
            "signed": True,
            "max": 2**127 - 1,
            "min": -(2**127),
        },
        "int64": {"bits": 64, "signed": True, "max": 2**63 - 1, "min": -(2**63)},
        "int32": {"bits": 32, "signed": True, "max": 2**31 - 1, "min": -(2**31)},
        # Rust unsigned integers
        "u256": {"bits": 256, "signed": False, "max": 2**256 - 1, "min": 0},
        "u128": {"bits": 128, "signed": False, "max": 2**128 - 1, "min": 0},
        "u64": {"bits": 64, "signed": False, "max": 2**64 - 1, "min": 0},
        "u32": {"bits": 32, "signed": False, "max": 2**32 - 1, "min": 0},
        "u16": {"bits": 16, "signed": False, "max": 2**16 - 1, "min": 0},
        "u8": {"bits": 8, "signed": False, "max": 2**8 - 1, "min": 0},
        "usize": {"bits": 64, "signed": False, "max": 2**64 - 1, "min": 0},
        # Rust signed integers
        "i256": {
            "bits": 256,
            "signed": True,
            "max": 2**255 - 1,
            "min": -(2**255),
        },
        "i128": {
            "bits": 128,
            "signed": True,
            "max": 2**127 - 1,
            "min": -(2**127),
        },
        "i64": {"bits": 64, "signed": True, "max": 2**63 - 1, "min": -(2**63)},
        "i32": {"bits": 32, "signed": True, "max": 2**31 - 1, "min": -(2**31)},
        "isize": {"bits": 64, "signed": True, "max": 2**63 - 1, "min": -(2**63)},
    }

    # Get type info
    info = type_info.get(type_lower)
    if info is None:
        # Default to uint256-like
        info = {"bits": 256, "signed": False, "max": 2**256 - 1, "min": 0}
        result["values"].append(
            {
                "note": f"Unknown type '{param_type}', using uint256 defaults",
            }
        )

    max_val = info["max"]
    min_val = info["min"]
    is_signed = info["signed"]

    # Basic boundary values
    result["values"].extend(
        [
            {
                "value": "0",
                "description": "Zero",
                "risk": "Division by zero, empty state",
            },
            {
                "value": "1",
                "description": "One (smallest positive)",
                "risk": "Rounding to zero",
            },
            {"value": "2", "description": "Two", "risk": "Off-by-one errors"},
            {
                "value": str(max_val),
                "description": f"Maximum {param_type}",
                "risk": "Overflow on any addition",
            },
            {
                "value": str(max_val - 1),
                "description": "Max - 1",
                "risk": "Overflow on addition of 2+",
            },
        ]
    )

    if is_signed:
        result["values"].extend(
            [
                {
                    "value": str(min_val),
                    "description": f"Minimum {param_type}",
                    "risk": "Underflow on any subtraction",
                },
                {
                    "value": "-1",
                    "description": "Negative one",
                    "risk": "Sign issues in unsigned contexts",
                },
            ]
        )

    # Powers of 2 (common in bit manipulation)
    for exp in [64, 128, 192]:
        if 2**exp <= max_val:
            result["values"].append(
                {
                    "value": str(2**exp),
                    "description": f"2^{exp}",
                    "risk": "Bit boundary",
                }
            )

    # Common decimal bases
    decimal_bases = [
        (10**6, "1e6 (USDC decimals)"),
        (10**8, "1e8 (BTC decimals)"),
        (10**18, "1e18 (ETH/ERC20 decimals)"),
    ]
    for val, desc in decimal_bases:
        if val <= max_val:
            result["values"].append(
                {"value": str(val), "description": desc, "risk": "Decimal precision"}
            )

    # Overflow pairs
    half_max = max_val // 2
    result["overflow_pairs"].extend(
        [
            {
                "a": str(max_val),
                "b": "1",
                "op": "+",
                "description": "Max + 1 overflows",
            },
            {
                "a": str(half_max + 1),
                "b": str(half_max + 1),
                "op": "+",
                "description": "Two large values overflow",
            },
            {
                "a": str(max_val),
                "b": "2",
                "op": "*",
                "description": "Max * 2 overflows",
            },
        ]
    )

    if is_signed:
        result["overflow_pairs"].append(
            {
                "a": str(min_val),
                "b": "1",
                "op": "-",
                "description": "Min - 1 underflows",
            }
        )

    # Context-specific values
    if context:
        context_lower = context.lower()
        if any(word in context_lower for word in ["share", "supply", "total"]):
            result["context_specific"].extend(
                [
                    {
                        "value": "1",
                        "description": "First depositor/share attack setup",
                        "risk": "Share inflation attack",
                    },
                    {
                        "value": str(10**18),
                        "description": "Standard 1 token with 18 decimals",
                        "risk": "Rounding in share calculations",
                    },
                ]
            )
            result["test_assertions"].append("Assert shares > 0 for non-zero deposits")
            result["test_assertions"].append(
                "Assert assets >= deposits for all depositors"
            )

        if any(word in context_lower for word in ["fee", "rate", "percent"]):
            result["context_specific"].extend(
                [
                    {
                        "value": "0",
                        "description": "Zero fee",
                        "risk": "Division handling",
                    },
                    {
                        "value": str(10**18),
                        "description": "100% (if 1e18 = 100%)",
                        "risk": "Full extraction",
                    },
                    {
                        "value": str(10**18 + 1),
                        "description": ">100%",
                        "risk": "Fee > principal",
                    },
                ]
            )
            result["test_assertions"].append("Assert fee <= principal")

        if any(word in context_lower for word in ["amount", "balance", "deposit"]):
            result["context_specific"].extend(
                [
                    {
                        "value": "1",
                        "description": "Dust amount",
                        "risk": "Rounds to zero",
                    },
                ]
            )
            result["test_assertions"].append(
                "Assert balance change >= expected for all transfers"
            )

    return result


__all__ = [
    # Graph analysis tools
    "dependency_graph_loc",
    "dependency_graph_slice",
    "dependency_graph_neighbors",
    "dependency_graph_resolve",
    "dependency_graph_snippet",
    "dependency_graph_callers",
    "dependency_graph_callees",
    "dependency_graph_protocol_entrypoints",
    # Test execution tools
    "write_and_compile",
    "run_test",
    "patch_file",
    "register_exploit",
    # Quant-specific tools
    "analyze_arithmetic",
    "compute_boundary_values",
]
