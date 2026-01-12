"""
Python workspace adapter.

Handles Python-specific workspace provisioning including:
- Virtual environment creation
- Source directory symlinking
- Dependency installation
- Test directory setup
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Any, List

from kai.schemas import MasterContext, WorkspacePreset
from kai.utils.workspace.base import WorkspaceAdapter


class PythonWorkspaceAdapter(WorkspaceAdapter):
    """Workspace adapter for Python projects."""

    _uv_bin: Optional[str] = None  # Lazily initialized

    @classmethod
    def _find_or_install_uv(cls) -> str:
        """
        Find uv binary, installing it if not available.

        This is separated from __init__ to avoid side effects in constructor.

        Returns:
            Path to uv binary

        Raises:
            RuntimeError: If uv cannot be found or installed
        """
        uv_bin = shutil.which("uv")
        if uv_bin:
            return uv_bin

        # Install uv via official installer
        try:
            subprocess.run(
                ["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            # Update PATH and find uv
            home = os.path.expanduser("~")
            uv_path = os.path.join(home, ".local", "bin", "uv")
            if os.path.exists(uv_path):
                return uv_path
            uv_bin = shutil.which("uv")
            if uv_bin:
                return uv_bin
        except Exception:
            pass

        raise RuntimeError(
            "uv is required for Python workspace provisioning but could not be installed. "
            "Install uv manually: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )

    def _get_uv_bin(self) -> str:
        """Get uv binary path, initializing lazily if needed."""
        if self._uv_bin is None:
            self._uv_bin = self._find_or_install_uv()
        return self._uv_bin

    @property
    def framework_name(self) -> str:
        return "python"

    def provision_lightweight(
        self,
        workspace: Path,
        master: Path,
        master_context: MasterContext,
        logger: Optional[Any] = None,
    ) -> str:
        """
        Create a lightweight Python workspace with symlinks to master.

        Creates a virtual environment and symlinks source directories.
        """
        # Create directory structure
        (workspace / "tests" / "poc").mkdir(parents=True, exist_ok=True)

        # Create virtual environment
        self._create_venv(workspace, logger)

        # Symlink source directories
        self._setup_source_symlinks(workspace, master, master_context)

        # Copy config files
        self._copy_config_files(workspace, master)

        # Install dependencies - raise error if installation fails
        deps_installed = self._install_dependencies(workspace, logger)
        if not deps_installed:
            raise RuntimeError(
                "Failed to install project dependencies. "
                "All dependency sources (requirements.txt, pyproject.toml, setup.py) failed. "
                "Tests will likely fail with ModuleNotFoundError. "
                "Check the logs above for specific errors."
            )

        # Create conftest.py for pytest to find modules
        self._create_conftest(workspace, master, master_context)

        if logger:
            logger.debug(f"Provisioned LIGHTWEIGHT Python workspace: {workspace}")

        return str(workspace)

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
        """
        # Exclude patterns for Python projects
        exclude_dirs = {
            ".venv",
            "venv",
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            ".tox",
            "dist",
            "build",
            "*.egg-info",
            "kai_workspaces",
        }

        if preset == WorkspacePreset.SANDBOX:
            # Copy everything except excluded dirs
            self._copy_with_excludes(master, workspace, exclude_dirs)
        else:
            # CLEAN/WRITEABLE - selective copy
            self._copy_with_excludes(master, workspace, exclude_dirs)

        # Create fresh venv
        self._create_venv(workspace, logger)

        # Install dependencies - raise error if installation fails
        deps_installed = self._install_dependencies(workspace, logger)
        if not deps_installed:
            raise RuntimeError(
                "Failed to install project dependencies. "
                "All dependency sources (requirements.txt, pyproject.toml, setup.py) failed. "
                "Tests will likely fail with ModuleNotFoundError. "
                "Check the logs above for specific errors."
            )

        # Create test directories
        (workspace / "tests" / "poc").mkdir(parents=True, exist_ok=True)

        # Ensure workspace is writable
        self._make_writable(workspace)

        if logger:
            logger.debug(
                f"Provisioned {preset.value.upper()} Python workspace: {workspace}"
            )
        return str(workspace)

    def detect_remappings(self, master: Path) -> str:
        """Python doesn't use remappings - return empty string."""
        return ""

    def infer_src_path(self, master: Path) -> Path:
        """
        Infer the source directory for a Python project.

        Checks pyproject.toml and common patterns.
        """
        # Check pyproject.toml for package directory
        pyproject = master / "pyproject.toml"
        if pyproject.exists():
            try:
                import tomllib

                data = tomllib.loads(pyproject.read_text())

                # Check [tool.setuptools] packages
                setuptools = data.get("tool", {}).get("setuptools", {})
                package_dir = setuptools.get("package-dir", {})
                if "" in package_dir:
                    return master / package_dir[""]

                # Check packages list
                packages = setuptools.get("packages", [])
                if packages and isinstance(packages, list):
                    return master / packages[0].replace(".", "/")

            except Exception:
                pass

        # Common Python source patterns
        for src_dir in ["src", "lib", "app", master.name]:
            if (master / src_dir).is_dir():
                return master / src_dir

        return master

    def get_runtime_writable_paths(
        self, project_root: Path, master_context: MasterContext
    ) -> List[Path]:
        """
        Python projects need .venv, __pycache__, and build dirs writable.
        """
        writable = []
        for rel in [".venv", "__pycache__", ".pytest_cache", "build", "dist"]:
            path = project_root / rel
            writable.append(path)
        return writable

    def _create_venv(self, workspace: Path, logger: Optional[Any] = None) -> None:
        """Create a virtual environment in the workspace using uv."""
        venv_path = workspace / ".venv"
        if venv_path.exists():
            return

        try:
            subprocess.run(
                [self._get_uv_bin(), "venv", str(venv_path)],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=60,
            )
            if logger:
                logger.debug(f"Created venv with uv at {venv_path}")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to create venv: {e}")

    def _setup_source_symlinks(
        self,
        workspace: Path,
        master: Path,
        master_context: MasterContext,
    ) -> None:
        """Symlink source directories from master to workspace."""
        # Determine source directory
        src_path = self.infer_src_path(master)

        # Common source directories to symlink
        src_dirs = ["src", "lib", "app"]

        # Add the inferred src path name
        if src_path != master:
            src_dirs.insert(0, src_path.name)

        for src_dir in src_dirs:
            master_dir = master / src_dir
            workspace_dir = workspace / src_dir

            if (
                master_dir.exists()
                and master_dir.is_dir()
                and not workspace_dir.exists()
            ):
                try:
                    rel_path = os.path.relpath(master_dir, workspace)
                    workspace_dir.symlink_to(rel_path)
                except OSError:
                    # Symlink might fail on Windows, copy instead
                    shutil.copytree(master_dir, workspace_dir)

        # Also symlink the package directory if it's named after the project
        project_name = master.name.replace("-", "_").replace(".", "_")
        if (master / project_name).is_dir() and not (workspace / project_name).exists():
            try:
                rel_path = os.path.relpath(master / project_name, workspace)
                (workspace / project_name).symlink_to(rel_path)
            except OSError:
                pass

    def _copy_config_files(self, workspace: Path, master: Path) -> None:
        """Copy Python config files to workspace."""
        config_files = [
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
            "requirements-dev.txt",
            "pytest.ini",
            "conftest.py",
            ".python-version",
            "tox.ini",
        ]

        for config_file in config_files:
            src = master / config_file
            if src.exists() and src.is_file():
                shutil.copy2(src, workspace / config_file)

    def _run_uv_install(
        self,
        cmd: List[str],
        workspace: Path,
        source_name: str,
        timeout: int = 300,
        logger: Optional[Any] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Run a uv pip install command and handle errors uniformly.

        Args:
            cmd: The command to run
            workspace: Workspace directory
            source_name: Name of the dependency source (for error messages)
            timeout: Command timeout in seconds
            logger: Optional logger

        Returns:
            Tuple of (success, error_message). error_message is None on success.
        """
        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                if logger:
                    logger.info(f"Installed dependencies from {source_name} with uv")
                return True, None
            else:
                error_snippet = (result.stderr or result.stdout)[:200]
                if logger:
                    logger.warning(f"Failed to install from {source_name}: {error_snippet}")
                return False, f"{source_name}: {error_snippet}"
        except subprocess.TimeoutExpired:
            if logger:
                logger.warning(f"{source_name} installation timed out")
            return False, f"{source_name}: Installation timed out"
        except Exception as e:
            if logger:
                logger.warning(f"Failed to install from {source_name}: {e}")
            return False, f"{source_name}: {str(e)}"

    def _install_dependencies(
        self, workspace: Path, logger: Optional[Any] = None
    ) -> bool:
        """Install dependencies into the workspace venv using uv.

        Uses uv pip install for all dependency installation. For pyproject.toml,
        parses dependencies directly to avoid invoking build systems (e.g., setuptools-scm)
        that may fail without git metadata.

        Returns:
            True if dependencies were installed successfully from at least one source,
            True if no dependency files exist (not a failure - project may not need deps),
            False if dependency files exist but ALL installation attempts failed.
        """
        venv_path = workspace / ".venv"
        if not venv_path.exists():
            if logger:
                logger.debug("venv not found - skipping dependency installation")
            return True

        venv_python = str(venv_path / "bin" / "python")
        uv_bin = self._get_uv_bin()

        # Track installation status
        any_dep_file_found = False
        installed_successfully = False
        failed_sources: List[str] = []

        # Install from requirements.txt first
        if (workspace / "requirements.txt").exists():
            any_dep_file_found = True
            cmd = [uv_bin, "pip", "install", "-r", "requirements.txt", "--python", venv_python]
            success, error = self._run_uv_install(cmd, workspace, "requirements.txt", logger=logger)
            if success:
                installed_successfully = True
            elif error:
                failed_sources.append(error)

        # Parse pyproject.toml directly to extract dependencies
        # This avoids invoking build systems like setuptools-scm
        pyproject_path = workspace / "pyproject.toml"
        if pyproject_path.exists() and not installed_successfully:
            any_dep_file_found = True
            deps = self._parse_pyproject_dependencies(pyproject_path, logger)
            if deps:
                cmd = [uv_bin, "pip", "install", "--python", venv_python] + deps
                success, error = self._run_uv_install(cmd, workspace, "pyproject.toml", logger=logger)
                if success:
                    installed_successfully = True
                elif error:
                    failed_sources.append(error)
            else:
                # No dependencies found in pyproject.toml, that's OK
                if logger:
                    logger.debug("No dependencies found in pyproject.toml")
                installed_successfully = True

        # For setup.py, we still need pip install -e . but use uv pip
        if (workspace / "setup.py").exists() and not installed_successfully:
            any_dep_file_found = True
            cmd = [uv_bin, "pip", "install", "-e", ".", "--python", venv_python]
            success, error = self._run_uv_install(cmd, workspace, "setup.py", logger=logger)
            if success:
                installed_successfully = True
            elif error:
                failed_sources.append(error)

        # Always install pytest for testing
        self._run_uv_install(
            [uv_bin, "pip", "install", "pytest", "--python", venv_python],
            workspace,
            "pytest",
            timeout=60,
        )

        # Determine return value and log summary
        if any_dep_file_found and not installed_successfully:
            if logger:
                logger.error(
                    f"DEPENDENCY INSTALLATION FAILED: All sources failed. "
                    f"Tests may fail with ModuleNotFoundError. Failed sources: {failed_sources}"
                )
            return False
        elif not any_dep_file_found:
            if logger:
                logger.info("No dependency files found (requirements.txt, pyproject.toml, setup.py)")
            return True

        return installed_successfully

    def _parse_pyproject_dependencies(
        self, pyproject_path: Path, logger: Optional[Any] = None
    ) -> List[str]:
        """
        Parse pyproject.toml to extract project dependencies.

        Extracts from [project.dependencies] and optionally [project.optional-dependencies].
        This bypasses build systems like setuptools-scm that require git metadata.

        Returns:
            List of dependency strings suitable for uv pip install
        """
        try:
            import tomllib
        except ImportError:
            # Python < 3.11
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                if logger:
                    logger.warning("tomllib/tomli not available, cannot parse pyproject.toml")
                return []

        try:
            data = tomllib.loads(pyproject_path.read_text())
        except Exception as e:
            if logger:
                logger.warning(f"Failed to parse pyproject.toml: {e}")
            return []

        deps: List[str] = []

        # Get [project.dependencies]
        project = data.get("project", {})
        project_deps = project.get("dependencies", [])
        if isinstance(project_deps, list):
            deps.extend(project_deps)

        # Get [project.optional-dependencies] - install all optional deps for testing
        optional = project.get("optional-dependencies", {})
        if isinstance(optional, dict):
            for group_deps in optional.values():
                if isinstance(group_deps, list):
                    deps.extend(group_deps)

        if logger and deps:
            logger.debug(f"Parsed {len(deps)} dependencies from pyproject.toml")

        return deps

    def _create_conftest(
        self,
        workspace: Path,
        master: Path,
        master_context: MasterContext,
    ) -> None:
        """Create conftest.py to help pytest find modules."""
        conftest_path = workspace / "tests" / "poc" / "conftest.py"

        src_path = self.infer_src_path(master)
        rel_src = os.path.relpath(src_path, workspace / "tests" / "poc")

        conftest_content = f'''"""
Pytest configuration for PoC tests.
Auto-generated by Kai workspace provisioner.
"""
import sys
from pathlib import Path

# Add source directories to path for imports
_workspace = Path(__file__).parent.parent.parent
_master_src = _workspace / "{rel_src}"

if _master_src.exists():
    sys.path.insert(0, str(_master_src))

# Also add the workspace root
sys.path.insert(0, str(_workspace))
'''
        conftest_path.parent.mkdir(parents=True, exist_ok=True)
        conftest_path.write_text(conftest_content)

    def _copy_with_excludes(self, src: Path, dst: Path, excludes: set) -> None:
        """Copy directory tree excluding certain patterns."""

        def should_exclude(path: Path) -> bool:
            for exc in excludes:
                if exc.startswith("*"):
                    if path.name.endswith(exc[1:]):
                        return True
                elif path.name == exc:
                    return True
            return False

        dst.mkdir(parents=True, exist_ok=True)

        for item in src.iterdir():
            if should_exclude(item):
                continue

            dest_path = dst / item.name

            if item.is_dir():
                if not item.name.startswith(".") or item.name == ".github":
                    self._copy_with_excludes(item, dest_path, excludes)
            else:
                shutil.copy2(item, dest_path)

    def _make_writable(self, root: Path) -> None:
        """Ensure workspace directories/files are writable."""
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
