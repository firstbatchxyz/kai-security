"""
Cargo workspace adapter.

Setup-only scope: provision a writable workspace where `cargo build/test` can run.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional

from kai.schemas import MasterContext, WorkspacePreset
from kai.utils.workspace.base import WorkspaceAdapter


class CargoWorkspaceAdapter(WorkspaceAdapter):
    @property
    def framework_name(self) -> str:
        return "cargo"

    def provision_lightweight(
        self,
        workspace: Path,
        master: Path,
        master_context: MasterContext,
        logger: Optional[Any] = None,
    ) -> str:
        # For setup-only scope, treat LIGHTWEIGHT as a minimal copy.
        return self.provision_full(
            workspace=workspace,
            master=master,
            master_context=master_context,
            preset=WorkspacePreset.CLEAN,
            logger=logger,
        )

    def provision_full(
        self,
        workspace: Path,
        master: Path,
        master_context: MasterContext,
        preset: WorkspacePreset,
        logger: Optional[Any] = None,
    ) -> str:
        if preset == WorkspacePreset.SANDBOX:
            shutil.copytree(master, workspace, dirs_exist_ok=True)
            self._make_writable(workspace)
            return str(workspace)

        # CLEAN/WRITEABLE: copy most of the repo, skipping build artifacts
        skip_dirs = {
            "target",
            "node_modules",
            "__pycache__",
            "build",
            "out",
            "cache",
            "dist",
            ".venv",
        }

        for item in master.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                if item.name not in skip_dirs:
                    shutil.copytree(item, workspace / item.name)

        for item in master.iterdir():
            if item.is_file() and not item.name.startswith("."):
                shutil.copy2(item, workspace / item.name)

        self._make_writable(workspace)
        return str(workspace)

    def detect_remappings(self, master: Path) -> str:
        # Not applicable for Cargo.
        return ""

    def get_runtime_writable_paths(
        self, project_root: Path, master_context: MasterContext
    ) -> list[Path]:
        # Cargo writes build artifacts under target/
        try:
            return [(project_root / "target").resolve()]
        except Exception:
            return []

    def infer_src_path(self, master: Path) -> Path:
        if (master / "src").exists():
            return master / "src"
        return master

    def _make_writable(self, root: Path) -> None:
        try:
            root.chmod(root.stat().st_mode | 0o200)
        except Exception:
            pass
        for current, dirs, files in os.walk(root):
            for d in dirs:
                p = Path(current) / d
                try:
                    p.chmod(p.stat().st_mode | 0o200)
                except Exception:
                    pass
            for f in files:
                p = Path(current) / f
                try:
                    p.chmod(p.stat().st_mode | 0o200)
                except Exception:
                    pass
