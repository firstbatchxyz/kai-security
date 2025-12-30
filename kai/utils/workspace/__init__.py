"""
Workspace adapters for framework-specific workspace provisioning.

Each adapter handles the specifics of setting up a workspace for a particular
framework (Foundry, Hardhat, etc.).
"""

from kai.utils.workspace.base import WorkspaceAdapter
from kai.utils.workspace.foundry import FoundryWorkspaceAdapter
from kai.utils.workspace.cargo import CargoWorkspaceAdapter
from kai.utils.workspace.cmake import CMakeWorkspaceAdapter

__all__ = [
    "WorkspaceAdapter",
    "FoundryWorkspaceAdapter",
    "CargoWorkspaceAdapter",
    "CMakeWorkspaceAdapter",
    "get_workspace_adapter",
    "get_supported_frameworks",
]


_ADAPTERS = {
    "foundry": FoundryWorkspaceAdapter,
    "forge": FoundryWorkspaceAdapter,  # Alias
    "cargo": CargoWorkspaceAdapter,
    "cmake": CMakeWorkspaceAdapter,
}

_adapter_cache: dict = {}


def get_workspace_adapter(framework: str) -> WorkspaceAdapter:
    """
    Get the appropriate workspace adapter for a framework.

    Args:
        framework: Framework name (e.g., "foundry", "hardhat")

    Returns:
        WorkspaceAdapter instance for the framework

    Raises:
        ValueError: If framework is not supported
    """
    framework_lower = framework.lower()
    if framework_lower in _adapter_cache:
        return _adapter_cache[framework_lower]

    adapter_cls = _ADAPTERS.get(framework_lower)
    if adapter_cls is None:
        supported = ", ".join(sorted(set(_ADAPTERS.keys())))
        raise ValueError(f"Unsupported framework: {framework}. Supported: {supported}")

    adapter = adapter_cls()
    _adapter_cache[framework_lower] = adapter
    return adapter


def get_supported_frameworks() -> list:
    """Return supported workspace frameworks."""
    return sorted(set(_ADAPTERS.keys()))
