import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Optional

from kai.agents.agent_types import SetupAgent
from kai.processes.base import BaseProcess
from kai.schemas import (
    AgentResponse,
    EnvironmentSetupInput,
    EnvironmentSetupOutput,
    MasterContext,
)
from kai.utils.workspace import get_supported_frameworks, get_workspace_adapter


class EnvironmentSetupProcess(
    BaseProcess[EnvironmentSetupInput, EnvironmentSetupOutput]
):
    """
    Process to setup the target environment for Kai.

    Creates a reproducible "inputs" checkout and a "master" golden copy:
    - If a URL is provided, clone into `testbed/<slug>/inputs/`.
    - If a local repo path is provided, use it as the input source.
    - Copy inputs into `testbed/<slug>/master/` and run SetupAgent on that master.
    - Mark master as read-only after setup to enforce the golden master contract.
    """

    async def execute(
        self, input_data: EnvironmentSetupInput
    ) -> EnvironmentSetupOutput:
        # Derive the slug from the actual source we will use. If the caller provides a
        # materialized repo_path_override, prefer that for slugging so we don't accidentally
        # reuse a slug from an old URL (which can lead to copying the wrong repository).
        repo_url = input_data.repo_url
        slug_source = input_data.repo_path_override or repo_url
        repo_slug = self._repo_slug(str(slug_source))

        inputs_root = self._inputs_root(repo_slug)
        master_root = self._master_root(repo_slug)

        inputs_repo_path = (
            Path(input_data.repo_path_override).resolve()
            if input_data.repo_path_override
            else inputs_root
        )
        master_repo_path = master_root

        if input_data.repo_path_override:
            if not inputs_repo_path.exists():
                raise FileNotFoundError(
                    f"Materialized repo not found at {input_data.repo_path_override}"
                )
        else:
            self._clone_repo(repo_url, inputs_repo_path)

        self._copy_to_master(inputs_repo_path, master_repo_path)

        # Run SetupAgent on the master copy (never on inputs directly)
        agent = SetupAgent(
            repo_path=str(master_repo_path),
            model=input_data.model_name,
            max_tool_turns=input_data.num_turns,
            use_openai=input_data.use_openai,
            execution_id=input_data.execution_id,
        )

        response: Optional[AgentResponse] = None
        exception_occurred = False
        exception_msg = ""

        try:
            self.logger.info("Starting SetupAgent (native tool-calling)...")
            response = await agent.chat_with_tools(
                "You must start setting up the repository now."
            )

            # If the agent terminated without registering a MasterContext, nudge it.
            if response is not None and response.master_context is None:
                retry_prompt = (
                    "FORMAT REQUIREMENT: You must call register_master_context({...}) "
                    "with the final build information. Call it now to finish."
                )
                response = await agent.chat_with_tools(retry_prompt)
        except Exception as e:
            self.logger.error(f"SetupAgent failed: {e}", exc_info=True)
            exception_occurred = True
            exception_msg = str(e)
        finally:
            try:
                await agent.close()
            except Exception:
                pass

        master_context = response.master_context if response else None
        if master_context:
            master_context = self._normalize_master_context_paths(
                master_context, master_repo_path
            )

        # Ensure build/cache directories exist before locking down the golden master.
        # Foundry/CryticCompile may need to write caches under the project root
        # (e.g., cache_path="forge-cache") even if we want the source tree to remain immutable.
        try:
            self._prepare_runtime_dirs(master_context, master_repo_path)
        except Exception:
            # Best-effort only; a failure here should not block setup completion.
            pass

        # Mark master as read-only to enforce golden master contract
        try:
            self._make_read_only(master_repo_path)
        except Exception:
            pass

        # Re-enable writes for runtime build/cache directories (keep the rest of master read-only).
        try:
            self._make_runtime_dirs_writable(master_context, master_repo_path)
        except Exception:
            pass

        setup_successful = (
            not exception_occurred
            and response is not None
            and response.master_context is not None
        )

        if not setup_successful and not exception_msg:
            exception_msg = "Setup agent did not produce a MasterContext"

        return EnvironmentSetupOutput(
            response=response,
            master_context=master_context,
            estimated_cost=agent.estimated_cost,
            total_tokens=agent.total_tokens,
            success=setup_successful,
            error_message=exception_msg if not setup_successful else None,
            master_repo_path=str(master_repo_path),
            repo_slug=repo_slug,
        )

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    def _inputs_root(self, repo_slug: str) -> Path:
        path = self._project_root() / "testbed" / repo_slug / "inputs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _master_root(self, repo_slug: str) -> Path:
        path = self._project_root() / "testbed" / repo_slug / "master"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_repo_commit_hash(self, repo_url: str) -> Optional[str]:
        """Get short commit hash from repo URL or local path."""
        commit_hash = None
        is_local_repo = os.path.isdir(repo_url)
        command = (
            ["git", "-C", repo_url, "rev-parse", "HEAD"]
            if is_local_repo
            else ["git", "ls-remote", repo_url, "HEAD"]
        )

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            output = result.stdout.strip()
            if output:
                commit_hash = output if is_local_repo else output.split()[0]
        except Exception:
            commit_hash = None

        if commit_hash:
            commit_hash = commit_hash[:8]

        return commit_hash

    def _repo_slug(self, repo_url: str) -> str:
        """Generate a slug for the repo (name-hash)."""
        name = Path(re.sub(r"\\.git$", "", repo_url.split("/")[-1])).stem or "repo"
        short_hash = self._get_repo_commit_hash(repo_url) or "unknown"
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "-", name)
        return f"{safe_name}-{short_hash}"

    def _clone_repo(self, repo_url: str, dest: Path) -> Path:
        """Clone a repo from URL. Removes existing if present."""
        if dest.exists():
            self._safe_rmtree(dest)
        subprocess.run(["git", "clone", repo_url, str(dest)], check=True)
        return dest

    def _safe_rmtree(self, path: Path) -> None:
        """
        Remove a filesystem tree even if it contains read-only files/dirs.

        The golden master is intentionally marked read-only. On subsequent runs we must
        be able to replace it. This helper best-effort chmods problematic paths and retries.
        """
        if not path.exists():
            return

        # Do not follow symlinks; just remove the link itself.
        if path.is_symlink() or path.is_file():
            try:
                path.chmod(path.stat().st_mode | stat.S_IWUSR)
            except Exception:
                pass
            try:
                path.unlink()
            except Exception:
                pass
            return

        def _onerror(func, p, _exc_info):  # type: ignore[no-untyped-def]
            try:
                pp = Path(p)

                # Ensure parent dir is writable (often the real blocker for unlink/rmdir).
                try:
                    parent = pp.parent
                    if parent.exists():
                        os.chmod(
                            parent,
                            parent.stat().st_mode
                            | stat.S_IWUSR
                            | stat.S_IXUSR
                            | stat.S_IRUSR,
                        )
                except Exception:
                    pass

                try:
                    if pp.is_dir():
                        os.chmod(pp, 0o700)
                    else:
                        os.chmod(pp, 0o600)
                except Exception:
                    pass

                func(p)
            except Exception:
                # Best-effort only; rmtree will keep going where it can.
                return

        shutil.rmtree(path, onerror=_onerror)

    def _copy_to_master(self, inputs_path: Path, master_path: Path) -> Path:
        if master_path.exists():
            self._safe_rmtree(master_path)
        shutil.copytree(inputs_path, master_path)
        return master_path

    def _make_read_only(self, path: Path) -> None:
        try:
            path.chmod(path.stat().st_mode & ~0o222)
        except Exception:
            pass
        for root, dirs, files in os.walk(path):
            for d in dirs:
                dir_path = Path(root) / d
                try:
                    dir_path.chmod(dir_path.stat().st_mode & ~0o222)
                except Exception:
                    raise Exception(f"Failed to make {dir_path} read-only")
            for f in files:
                file_path = Path(root) / f
                try:
                    file_path.chmod(file_path.stat().st_mode & ~0o222)
                except PermissionError:
                    pass

    def _runtime_root(
        self, master_context: Optional[MasterContext], master_repo_path: Path
    ) -> Path:
        """
        Determine the project root where toolchains will run.

        SetupAgent may set MasterContext.root_path to a subdirectory (e.g. monorepos).
        """
        master_root = master_repo_path.resolve()
        try:
            if master_context and getattr(master_context, "root_path", None):
                candidate = Path(master_context.root_path).resolve()
                # Safety: never chmod outside the golden master tree.
                if candidate == master_root or master_root in candidate.parents:
                    return candidate
        except Exception:
            pass
        return master_root

    def _get_framework(self, master_context: MasterContext) -> str:
        """
        Select a supported workspace framework from MasterContext.frameworks.
        Defaults to 'foundry' if none is present.
        """
        supported = set(get_supported_frameworks())
        frameworks = getattr(master_context, "frameworks", None) or []
        for fw in frameworks:
            fw_lower = str(fw).lower()
            if fw_lower == "forge":
                fw_lower = "foundry"
            if fw_lower in supported:
                return fw_lower
        raise ValueError(
            f"No supported framework found in MasterContext.frameworks: {frameworks}"
        )

    def _runtime_writable_dirs(
        self, project_root: Path, master_context: MasterContext
    ) -> list[Path]:
        """
        Delegate framework-specific runtime dir selection to the WorkspaceAdapter layer.
        """
        adapter = get_workspace_adapter(self._get_framework(master_context))
        try:
            return list(
                adapter.get_runtime_writable_paths(project_root, master_context)
            )
        except Exception:
            return []

    def _prepare_runtime_dirs(
        self, master_context: Optional[MasterContext], master_repo_path: Path
    ) -> None:
        """
        Create runtime build/cache directories before making master read-only.

        This ensures toolchains don't need to create directories under a read-only root.
        """
        if master_context is None:
            return
        root = self._runtime_root(master_context, master_repo_path)
        for d in self._runtime_writable_dirs(root, master_context):
            try:
                # Only touch directories inside the runtime root.
                if root not in d.parents and d != root:
                    continue
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                continue

    def _chmod_tree_add_owner_write(self, root: Path) -> None:
        """Best-effort: add owner write bit recursively for an existing path."""
        if not root.exists():
            return
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

    def _make_runtime_dirs_writable(
        self, master_context: Optional[MasterContext], master_repo_path: Path
    ) -> None:
        """
        After locking down the golden master, re-enable writes for runtime dirs only.
        """
        if master_context is None:
            return
        root = self._runtime_root(master_context, master_repo_path)
        for d in self._runtime_writable_dirs(root, master_context):
            try:
                if root not in d.parents and d != root:
                    continue
                self._chmod_tree_add_owner_write(d)
            except Exception:
                continue

    def _normalize_master_context_paths(
        self, master_context: MasterContext, repo_path: Path
    ) -> MasterContext:
        """Normalize relative paths in MasterContext to absolute paths."""

        def _normalize_path(value: Optional[str]) -> str:
            if not value or value in (".", "./"):
                return str(repo_path)
            p = Path(value)
            if p.is_absolute():
                return str(p)
            return str(repo_path / p)

        master_context.root_path = _normalize_path(master_context.root_path)
        master_context.artifacts_path = _normalize_path(master_context.artifacts_path)
        master_context.src_path = _normalize_path(master_context.src_path)
        master_context.lib_path = _normalize_path(master_context.lib_path)
        master_context.test_path = _normalize_path(master_context.test_path)
        return master_context
