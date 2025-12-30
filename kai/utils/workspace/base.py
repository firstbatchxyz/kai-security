"""
Base workspace adapter interface.

All framework-specific adapters inherit from WorkspaceAdapter.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any

from kai.schemas import MasterContext, WorkspacePreset


class WorkspaceAdapter(ABC):
    """
    Abstract base class for workspace adapters.

    Each framework (Foundry, Hardhat, etc.) implements its own adapter
    to handle framework-specific workspace setup.
    """

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Return the framework name (e.g., 'foundry', 'hardhat')."""
        ...

    @abstractmethod
    def provision_lightweight(
        self,
        workspace: Path,
        master: Path,
        master_context: MasterContext,
        logger: Optional[Any] = None,
    ) -> str:
        """
        Create a lightweight workspace with remappings to the master repo.

        This doesn't copy source files - just creates a minimal project that
        references the original code via remappings/symlinks. Ideal for PoC tests.

        Args:
            workspace: Path to the workspace directory (already created)
            master: Path to the master/source repository
            master_context: MasterContext with repo metadata
            logger: Optional logger for debug output

        Returns:
            Path to the provisioned workspace as string
        """
        ...

    @abstractmethod
    def provision_full(
        self,
        workspace: Path,
        master: Path,
        master_context: MasterContext,
        preset: WorkspacePreset,
        logger: Optional[Any] = None,
    ) -> str:
        """
        Create a full workspace by copying files from master.

        Args:
            workspace: Path to the workspace directory (already created)
            master: Path to the master/source repository
            master_context: MasterContext with repo metadata
            preset: Workspace preset (CLEAN, WRITEABLE, SANDBOX)
            logger: Optional logger for debug output

        Returns:
            Path to the provisioned workspace as string
        """
        ...

    @abstractmethod
    def detect_remappings(self, master: Path) -> str:
        """
        Detect and generate import remappings for the workspace.

        Reads existing remappings from config files and adjusts paths
        to point from workspace to master directory.

        Args:
            master: Path to the master/source repository

        Returns:
            Formatted remappings string for the framework's config
        """
        ...

    def infer_src_path(self, master: Path) -> Path:
        """
        Infer the source directory path for the master repo.

        Override in subclasses for framework-specific conventions.

        Args:
            master: Path to the master/source repository

        Returns:
            Path to the source directory
        """
        # Check common patterns
        if (master / "contracts").exists():
            return master / "contracts"
        if (master / "src").exists():
            return master / "src"
        return master

    def get_runtime_writable_paths(
        self, project_root: Path, master_context: MasterContext
    ) -> list[Path]:
        """
        Return build/cache paths that must remain writable in the golden master.

        Envsetup makes the master read-only to enforce immutability, but some toolchains
        (compilers/test runners) need to write build artifacts and caches. This method
        lets each framework declare which *project-local* directories must stay writable.

        Implementations must only return paths under `project_root` (no absolute/external paths).
        """
        return []
