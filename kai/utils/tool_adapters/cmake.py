"""
CMake tool adapter.

Provides CMake-specific implementations for:
- Configuring/building CMake projects
- Running tests with ctest
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from kai.utils.tool_adapters.base import CompileResult, TestResult, ToolAdapter


class CMakeToolAdapter(ToolAdapter):
    """Tool adapter for CMake-based C/C++ projects."""

    @property
    def framework_name(self) -> str:
        return "cmake"

    @property
    def language(self) -> str:
        return "cpp"

    def find_binary(self) -> str:
        cmake_path = shutil.which("cmake")
        if cmake_path:
            return cmake_path
        raise FileNotFoundError("cmake not found")

    def _find_ctest(self) -> str:
        ctest_path = shutil.which("ctest")
        if ctest_path:
            return ctest_path
        # ctest ships with cmake in most installs; if missing, treat as not available
        raise FileNotFoundError("ctest not found (install CMake)")

    def compile(self, workspace_path: Path, timeout: int = 120) -> CompileResult:
        """
        Configure + build using an in-workspace build directory.
        """
        try:
            cmake_bin = self.find_binary()
        except FileNotFoundError as e:
            return CompileResult(success=False, errors=[str(e)], raw_output="")

        build_dir = workspace_path / "build"
        build_dir.mkdir(parents=True, exist_ok=True)

        outputs: List[str] = []
        try:
            # Configure
            cfg = subprocess.run(
                [cmake_bin, "-S", str(workspace_path), "-B", str(build_dir)],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            outputs.append((cfg.stdout or "") + (cfg.stderr or ""))
            if cfg.returncode != 0:
                raw = "\n\n".join([o for o in outputs if o]).strip()
                return CompileResult(
                    success=False,
                    errors=self.parse_compile_errors(raw),
                    warnings=self.parse_compile_warnings(raw),
                    raw_output=raw[:5000] if len(raw) > 5000 else raw,
                )

            # Build
            bld = subprocess.run(
                [cmake_bin, "--build", str(build_dir), "--parallel"],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            outputs.append((bld.stdout or "") + (bld.stderr or ""))
            raw = "\n\n".join([o for o in outputs if o]).strip()
            success = bld.returncode == 0
            errors = self.parse_compile_errors(raw) if not success else []
            warnings = self.parse_compile_warnings(raw)
            return CompileResult(
                success=success,
                errors=errors,
                warnings=warnings,
                raw_output=raw[:5000] if len(raw) > 5000 else raw,
            )
        except subprocess.TimeoutExpired:
            return CompileResult(
                success=False,
                errors=[f"Compilation timed out after {timeout} seconds"],
                raw_output="",
            )
        except Exception as e:
            raw = "\n\n".join([o for o in outputs if o]).strip()
            return CompileResult(
                success=False,
                errors=[str(e)],
                raw_output=raw[:5000] if len(raw) > 5000 else raw,
            )

    def run_test(
        self,
        workspace_path: Path,
        match_contract: Optional[str] = None,
        match_test: Optional[str] = None,
        verbosity: int = 3,
        timeout: int = 300,
        additional_args: Optional[str] = None,
        framework_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TestResult:
        try:
            ctest_bin = self._find_ctest()
        except FileNotFoundError as e:
            return TestResult(success=False, error=str(e))

        fw = dict(framework_kwargs or {})
        build_dir = fw.get("build_dir")
        if isinstance(build_dir, str) and build_dir.strip():
            build_path = Path(build_dir.strip())
            if not build_path.is_absolute():
                build_path = workspace_path / build_path
        else:
            build_path = workspace_path / "build"

        test_regex = fw.get("test_regex") or match_test or match_contract

        cmd = [ctest_bin, "--output-on-failure"]
        if test_regex:
            cmd.extend(["-R", str(test_regex)])
        if verbosity >= 3:
            cmd.append("-V")
        if additional_args:
            cmd.extend(shlex.split(additional_args))

        try:
            result = subprocess.run(
                cmd,
                cwd=str(build_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (result.stdout or "") + (result.stderr or "")
            success = result.returncode == 0

            passed, failed = _parse_ctest_summary(output)
            return TestResult(
                success=success,
                tests_passed=passed,
                tests_failed=failed,
                raw_output=output[:8000] if len(output) > 8000 else output,
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                success=False,
                error=f"Test execution timed out after {timeout} seconds",
                raw_output=f"Test execution timed out after {timeout} seconds",
            )
        except Exception as e:
            return TestResult(success=False, error=str(e))

    def get_test_file_extension(self) -> str:
        return ".cpp"

    def get_source_file_extension(self) -> str:
        return ".cpp"

    def normalize_test_path(self, file_path: str, workspace: Path) -> Path:
        p = Path(file_path)
        if p.is_absolute():
            p = Path(p.name)

        normalized = p.as_posix().lstrip("/")
        if normalized.startswith("tests/"):
            normalized = normalized[len("tests/") :]
        elif normalized.startswith("test/"):
            normalized = normalized[len("test/") :]

        if not normalized.endswith(self.get_test_file_extension()):
            normalized = normalized + self.get_test_file_extension()

        return workspace / "tests" / normalized

    def get_allowed_patch_directories(self) -> List[str]:
        return ["tests/poc", "tests\\poc"]


_CTEST_SUMMARY_RE = re.compile(
    r"(?P<pct>\d+)%\s+tests\s+passed,\s+(?P<failed>\d+)\s+tests\s+failed\s+out\s+of\s+(?P<total>\d+)",
    re.IGNORECASE,
)


def _parse_ctest_summary(output: str) -> tuple[int, int]:
    """
    Parse ctest summary if present; otherwise return (0, 0).
    """
    m = _CTEST_SUMMARY_RE.search(output or "")
    if not m:
        # ctest often prints "No tests were found!!!" and exits 0
        return 0, 0
    try:
        total = int(m.group("total"))
        failed = int(m.group("failed"))
        passed = max(0, total - failed)
        return passed, failed
    except Exception:
        return 0, 0
