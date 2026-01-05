"""
Tools for GamifiedAgent - gap discovery with self-verification.

Tools:
- register_exploit: Register exploit or verification finding (unified, with auto-compile)
- write_and_compile: Write test file and compile it
- run_test: Run tests to verify hypothesis
- dependency_graph_snippet: Get function source code
- dependency_graph_neighbors: See what a function reads/writes/calls
- dependency_graph_callers: Find callers of a function
- dependency_graph_callees: Find functions called by a function
- dependency_graph_resolve: Find node ID from name
"""

from kai.agents.tools.tools import (
    dependency_graph_snippet,
    dependency_graph_neighbors,
    dependency_graph_callers,
    dependency_graph_callees,
    dependency_graph_resolve,
    register_exploit,  # Unified exploit registration with auto-compile
)
from kai.agents.tools.state_tools import (
    write_and_compile,
    run_test,
)


__all__ = [
    "register_exploit",
    "write_and_compile",
    "run_test",
    "dependency_graph_snippet",
    "dependency_graph_callees",
    "dependency_graph_neighbors",
    "dependency_graph_callers",
    "dependency_graph_resolve",
]
