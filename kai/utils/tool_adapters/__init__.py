"""
Tool adapters for framework-specific build and test operations.

Each adapter handles the specifics of compiling and testing for a particular
framework (Foundry, Hardhat, Anchor, etc.).
"""

from typing import Optional

from kai.utils.tool_adapters.base import (
    ToolAdapter,
    CompileResult,
    TestResult,
)
from kai.utils.tool_adapters.foundry import FoundryToolAdapter
from kai.utils.tool_adapters.cargo import CargoToolAdapter
from kai.utils.tool_adapters.cmake import CMakeToolAdapter

__all__ = [
    "ToolAdapter",
    "CompileResult",
    "TestResult",
    "FoundryToolAdapter",
    "CargoToolAdapter",
    "CMakeToolAdapter",
    "get_tool_adapter",
    "get_supported_frameworks",
]

# Registry of tool adapters by framework name
_ADAPTERS = {
    "foundry": FoundryToolAdapter,
    "forge": FoundryToolAdapter,  # Alias
    # Future adapters:
    # "hardhat": HardhatToolAdapter,
    "cargo": CargoToolAdapter,
    "cmake": CMakeToolAdapter,
}

# Singleton cache for adapter instances
_adapter_cache: dict = {}


def get_tool_adapter(framework: Optional[str] = None) -> ToolAdapter:
    """
    Get the appropriate tool adapter for a framework.
    Uses a singleton cache to avoid creating multiple instances.
    """
    # Default to foundry
    framework = (framework or "foundry").lower()

    # Check cache
    if framework in _adapter_cache:
        return _adapter_cache[framework]

    # Look up adapter class
    adapter_cls = _ADAPTERS.get(framework)
    if adapter_cls is None:
        supported = ", ".join(sorted(set(_ADAPTERS.keys())))
        raise ValueError(f"Unsupported framework: {framework}. Supported: {supported}")

    # Create and cache instance
    adapter = adapter_cls()
    _adapter_cache[framework] = adapter

    return adapter


def get_supported_frameworks() -> list:
    """
    Get list of supported frameworks.

    Returns:
        List of framework names
    """
    return sorted(set(_ADAPTERS.keys()))
