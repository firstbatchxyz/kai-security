"""
Domain adapters for language/framework-specific security analysis.

This module provides pluggable adapters that encapsulate domain-specific
knowledge, enabling Kai to support multiple languages and frameworks.

Currently supported:
- Solidity (Foundry, Hardhat, Truffle, Brownie)

To add support for a new language:
1. Create a new adapter inheriting from DomainAdapter
2. Implement all abstract methods with language-specific patterns
3. Add the adapter to ADAPTER_REGISTRY

Usage:
    from kai.utils.dependency.adapters import DomainAdapter, SolidityAdapter, get_adapter

    # Explicit adapter
    adapter = SolidityAdapter()

    # Get adapter by name
    adapter = get_adapter("solidity")

    # Use in analysis
    roles = get_actor_roles(graph, adapter=adapter)
"""

from typing import Literal

from .base import DomainAdapter
from .solidity import SolidityAdapter
from .python import PythonAdapter
from .javascript import JavaScriptAdapter
from .c import CAdapter
from .rust import RustAdapter

# Literal type for structured output validation
AdapterType = Literal["solidity", "python", "javascript", "c", "rust"]

# Registry mapping adapter names to classes
ADAPTER_REGISTRY: dict[str, type[DomainAdapter]] = {
    "solidity": SolidityAdapter,
    "python": PythonAdapter,
    "javascript": JavaScriptAdapter,
    "c": CAdapter,
    "rust": RustAdapter,
}


def get_adapter(name: str) -> DomainAdapter:
    """
    Get an adapter instance by name.

    Args:
        name: Adapter name (e.g., "solidity", "python", "javascript", "typescript", "c")

    Returns:
        Instantiated adapter

    Raises:
        ValueError: If adapter name is unknown
    """
    name_lower = name.lower()

    # Map aliases to canonical adapter names
    alias_map = {
        "typescript": "javascript",  # TypeScript uses JavaScript adapter
        "ts": "javascript",
        "js": "javascript",
        "py": "python",
        "sol": "solidity",
        "rs": "rust",
        "cargo": "rust",
    }
    name_lower = alias_map.get(name_lower, name_lower)

    if name_lower not in ADAPTER_REGISTRY:
        available = list(ADAPTER_REGISTRY.keys())
        raise ValueError(f"Unknown adapter '{name}'. Available: {available}")
    return ADAPTER_REGISTRY[name_lower]()


__all__ = [
    # Abstract base
    "DomainAdapter",
    # Type and registry
    "AdapterType",
    "ADAPTER_REGISTRY",
    "get_adapter",
    # Concrete adapters
    "SolidityAdapter",
    "PythonAdapter",
    "JavaScriptAdapter",
    "CAdapter",
    "RustAdapter",
]
