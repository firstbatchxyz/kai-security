"""
Cargo tool adapter.

Provides Cargo-specific implementations for:
- Finding cargo binary
- Compiling Rust projects
- Running tests with cargo test
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from kai.utils.tool_adapters.base import CompileResult, TestResult, ToolAdapter


class CargoToolAdapter(ToolAdapter):
    """Tool adapter for Rust/Cargo projects."""

    @property
    def framework_name(self) -> str:
        return "cargo"

    @property
    def language(self) -> str:
        return "rust"

    def find_binary(self) -> str:
        cargo_path = shutil.which("cargo")
        if cargo_path:
            return cargo_path

        # Common installation path (rustup)
        home = Path.home()
        candidate = home / ".cargo" / "bin" / "cargo"
        if candidate.exists() and candidate.is_file():
            return str(candidate)

        raise FileNotFoundError("cargo not found - is Rust installed? (rustup)")

    def compile(self, workspace_path: Path, timeout: int = 120) -> CompileResult:
        """
        Compile the project.

        We prefer `cargo test --no-run` so injected tests (if any) are typechecked.
        """
        try:
            cargo_bin = self.find_binary()
        except FileNotFoundError as e:
            return CompileResult(success=False, errors=[str(e)], raw_output="")

        cmd = [cargo_bin, "test", "--no-run"]
        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            raw_output = (result.stdout or "") + (result.stderr or "")
            success = result.returncode == 0
            errors = self.parse_compile_errors(raw_output) if not success else []
            warnings = self.parse_compile_warnings(raw_output)
            return CompileResult(
                success=success,
                errors=errors,
                warnings=warnings,
                raw_output=raw_output[:5000] if len(raw_output) > 5000 else raw_output,
            )
        except subprocess.TimeoutExpired:
            return CompileResult(
                success=False,
                errors=[f"Compilation timed out after {timeout} seconds"],
                raw_output="",
            )
        except Exception as e:
            return CompileResult(success=False, errors=[str(e)], raw_output="")

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
            cargo_bin = self.find_binary()
        except FileNotFoundError as e:
            return TestResult(success=False, error=str(e))

        fw = dict(framework_kwargs or {})

        package = fw.get("package") or fw.get("crate")
        features = fw.get("features")
        all_features = fw.get("all_features")
        no_default_features = fw.get("no_default_features")
        release = fw.get("release")

        match_path = fw.get("match_path")
        test_target: Optional[str] = None
        if isinstance(match_path, str) and match_path.strip():
            try:
                test_target = Path(match_path.strip()).name
                if test_target.endswith(".rs"):
                    test_target = test_target[: -len(".rs")]
            except Exception:
                test_target = None

        test_filter = match_test or match_contract

        def _build_cmd(*, include_target: bool) -> List[str]:
            cmd = [cargo_bin, "test"]
            if package:
                cmd.extend(["-p", str(package)])
            if release:
                cmd.append("--release")
            if features:
                if isinstance(features, list):
                    cmd.extend(["--features", ",".join(str(x) for x in features)])
                elif isinstance(features, str):
                    cmd.extend(["--features", features])
            if all_features:
                cmd.append("--all-features")
            if no_default_features:
                cmd.append("--no-default-features")
            if include_target and test_target:
                cmd.extend(["--test", test_target])
            if test_filter:
                cmd.append(str(test_filter))
            # If verbosity is high, avoid swallowing output.
            if verbosity >= 2:
                cmd.extend(["--", "--nocapture"])
            if additional_args:
                # Preserve quoted args.
                cmd.extend(shlex.split(additional_args))
            return cmd

        # Try targeted integration test first (if match_path provided), then fall back.
        tried_target = False
        outputs: List[str] = []
        for include_target in [True, False] if test_target else [False]:
            tried_target = tried_target or include_target
            cmd = _build_cmd(include_target=include_target)
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(workspace_path),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                output = (result.stdout or "") + (result.stderr or "")
                outputs.append(output)
                success = result.returncode == 0

                # If targeted run failed due to missing test target, retry without --test.
                if (
                    include_target
                    and not success
                    and re.search(
                        r"no test target named|unknown test target", output, re.I
                    )
                ):
                    continue

                passed, failed = _parse_cargo_test_summary(output)
                combined = "\n\n".join([o for o in outputs if o]).strip()
                return TestResult(
                    success=success,
                    tests_passed=passed,
                    tests_failed=failed,
                    raw_output=combined[:8000] if len(combined) > 8000 else combined,
                )
            except subprocess.TimeoutExpired:
                return TestResult(
                    success=False,
                    error=f"Test execution timed out after {timeout} seconds",
                    raw_output=f"Test execution timed out after {timeout} seconds",
                )
            except Exception as e:
                return TestResult(success=False, error=str(e))

        # If we somehow fell through, return a conservative failure
        combined = "\n\n".join([o for o in outputs if o]).strip()
        return TestResult(
            success=False,
            error="cargo test did not complete successfully",
            raw_output=combined[:8000] if len(combined) > 8000 else combined,
        )

    def get_test_file_extension(self) -> str:
        return ".rs"

    def get_source_file_extension(self) -> str:
        return ".rs"

    def normalize_test_path(self, file_path: str, workspace: Path) -> Path:
        """
        Normalize test path for Cargo projects.

        Cargo integration tests live under `tests/` (plural). We place injected harnesses there.
        """
        p = Path(file_path)
        if p.is_absolute():
            p = Path(p.name)

        normalized = p.as_posix().lstrip("/")
        # Strip leading "tests/" or "test/" if present
        if normalized.startswith("tests/"):
            normalized = normalized[len("tests/") :]
        elif normalized.startswith("test/"):
            normalized = normalized[len("test/") :]

        if not normalized.endswith(self.get_test_file_extension()):
            normalized = normalized + self.get_test_file_extension()

        return workspace / "tests" / normalized

    def get_allowed_patch_directories(self) -> List[str]:
        return ["tests/poc", "tests\\poc"]


_CARGO_TEST_SUMMARY_RE = re.compile(
    r"test result:\s+(?P<status>ok|FAILED)\.\s+"
    r"(?P<passed>\d+)\s+passed;\s+"
    r"(?P<failed>\d+)\s+failed;",
    re.IGNORECASE,
)


def _parse_cargo_test_summary(output: str) -> tuple[int, int]:
    """
    Parse cargo output summaries; sum across all test binaries.
    """
    passed = 0
    failed = 0
    for m in _CARGO_TEST_SUMMARY_RE.finditer(output or ""):
        try:
            passed += int(m.group("passed"))
            failed += int(m.group("failed"))
        except Exception:
            continue
    return passed, failed
