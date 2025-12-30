"""
Workspace provisioning/cleanup for Dispatcher missions.

Framework-agnostic orchestration layer that delegates to workspace adapters.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from kai.schemas import MasterContext, Mission, WorkspacePreset
from kai.utils.workspace import (
    get_workspace_adapter,
    get_supported_frameworks,
    WorkspaceAdapter,
)


class WorkspaceManager:
    """
    Manages workspace provisioning and cleanup for missions.

    Delegates framework-specific logic to WorkspaceAdapters.
    """

    def __init__(self, *, workspace_dir: str, logger=None) -> None:
        self.workspace_dir = workspace_dir
        self.logger = logger
        self._adapters: dict[str, WorkspaceAdapter] = {}

    def _get_adapter(self, framework: str) -> WorkspaceAdapter:
        """Get or create adapter for framework."""
        if framework not in self._adapters:
            self._adapters[framework] = get_workspace_adapter(framework)
        return self._adapters[framework]

    def _detect_framework(
        self, master: Path, master_context: Optional[MasterContext] = None
    ) -> str:
        """
        Detect the framework from master_context or by inspecting the repo.

        Args:
            master: Path to the master repository
            master_context: Optional MasterContext with framework info

        Returns:
            Framework name (defaults to "foundry" if not detected)
        """
        supported = set(get_supported_frameworks())

        # Check MasterContext first (pick first supported workspace framework)
        if master_context and master_context.frameworks:
            for fw in master_context.frameworks:
                fw_lower = str(fw).lower()
                if fw_lower == "forge":
                    fw_lower = "foundry"
                if fw_lower in supported:
                    return fw_lower

        # Detect by config files
        if (master / "foundry.toml").exists() and "foundry" in supported:
            return "foundry"
        if (master / "Cargo.toml").exists() and "cargo" in supported:
            return "cargo"
        if (master / "CMakeLists.txt").exists() and "cmake" in supported:
            return "cmake"

        # Optional: keep heuristics for other ecosystems, but only if adapters exist.
        if (
            (master / "hardhat.config.js").exists()
            or (master / "hardhat.config.ts").exists()
        ) and "hardhat" in supported:
            return "hardhat"
        if (master / "truffle-config.js").exists() and "truffle" in supported:
            return "truffle"

        # Default to foundry for Solidity projects
        return "foundry"

    def provision(
        self,
        workspace_id: str,
        master_path: str,
        preset: WorkspacePreset = WorkspacePreset.LIGHTWEIGHT,
        master_context: Optional[MasterContext] = None,
    ) -> str:
        """
        Provision a workspace.

        Args:
            workspace_id: Unique identifier for the workspace
            master_path: Path to the master/source repository
            preset: Workspace preset to use
            master_context: Optional MasterContext (will be inferred if not provided)

        Presets:
        - CLEAN: copy essential dirs + configs
        - WRITEABLE: currently same as CLEAN (workspace is writeable either way)
        - SANDBOX: full project copy
        - LIGHTWEIGHT: minimal project with remappings (no file copy)

        Returns:
            Path to the provisioned workspace
        """
        master = Path(master_path)
        workspace_base = Path(self.workspace_dir)
        workspace = workspace_base / workspace_id

        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        # Detect framework
        framework = self._detect_framework(master, master_context)
        adapter = self._get_adapter(framework)

        # Create MasterContext if not provided
        if master_context is None:
            src_path = adapter.infer_src_path(master)
            master_context = MasterContext(
                root_path=str(master),
                compile_success=True,
                src_path=str(src_path),
                frameworks=[framework],
            )

        if preset == WorkspacePreset.LIGHTWEIGHT:
            return adapter.provision_lightweight(
                workspace, master, master_context, self.logger
            )

        return adapter.provision_full(
            workspace, master, master_context, preset, self.logger
        )

    def provision_for_mission(
        self, mission: Mission, master_context: MasterContext
    ) -> str:
        """
        Provision a workspace for a mission.

        Convenience wrapper that extracts params from Mission object.
        """
        return self.provision(
            workspace_id=mission.mission_id,
            master_path=master_context.root_path,
            preset=mission.workspace_preset,
            master_context=master_context,
        )

    def cleanup(self, workspace_id: str) -> None:
        """Clean up a workspace by ID."""
        workspace = Path(self.workspace_dir) / workspace_id
        if workspace.exists():
            try:
                shutil.rmtree(workspace)
                if self.logger:
                    self.logger.debug(f"Cleaned up workspace: {workspace}")
            except OSError as e:
                if self.logger:
                    self.logger.warning(f"Failed to cleanup workspace {workspace}: {e}")

    def cleanup_for_mission(self, mission: Mission) -> None:
        """Clean up a mission's workspace."""
        self.cleanup(mission.mission_id)
