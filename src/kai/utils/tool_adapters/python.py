"""
Python tool adapter.

Provides Python-specific implementations for:
- Finding uv (or Python interpreter as fallback)
- Running syntax checks with py_compile via uv
- Running tests with pytest via uv
- Managing virtual environments with uv
"""

import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Any, Dict

from kai.utils.tool_adapters.base import (
    ToolAdapter,
    CompileResult,
    InstallResult,
    TestResult,
)


class PythonToolAdapter(ToolAdapter):
    """Tool adapter for Python projects using uv."""

    @property
    def framework_name(self) -> str:
        return "python"

    @property
    def language(self) -> str:
        return "python"

    def _find_uv(self) -> Optional[str]:
        """
        Find the uv binary.

        Returns:
            Path to uv binary or None if not found
        """
        uv_path = shutil.which("uv")
        if uv_path:
            return uv_path
        return None

    def find_binary(self, workspace_path: Optional[Path] = None) -> str:
        """
        Find uv, falling back to Python interpreter if uv is not available.

        Args:
            workspace_path: Optional workspace path (unused for uv)

        Returns:
            Path to uv or Python binary

        Raises:
            FileNotFoundError: If neither uv nor Python is found
        """
        # Prefer uv
        uv_bin = self._find_uv()
        if uv_bin:
            return uv_bin

        # Fallback to workspace venv Python
        if workspace_path:
            venv_python = workspace_path / ".venv" / "bin" / "python"
            if venv_python.exists():
                return str(venv_python)
            # Windows path
            venv_python_win = workspace_path / ".venv" / "Scripts" / "python.exe"
            if venv_python_win.exists():
                return str(venv_python_win)

        raise FileNotFoundError(
            "Neither uv nor Python found - install uv (https://docs.astral.sh/uv/) or Python"
        )

    def _is_uv(self, binary: str) -> bool:
        """Check if the binary is uv."""
        return "uv" in Path(binary).name

    def _get_venv_python(self, workspace_path: Path) -> Optional[Path]:
        """
        Get the venv python binary path.

        Args:
            workspace_path: Path to the workspace directory

        Returns:
            Path to the venv python binary, or None if not found
        """
        # Unix path
        venv_python = workspace_path / ".venv" / "bin" / "python"
        if venv_python.exists():
            return venv_python
        # Windows path
        venv_python = workspace_path / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            return venv_python
        return None

    def compile(
        self,
        workspace_path: Path,
        timeout: int = 120,
    ) -> CompileResult:
        """
        Check Python syntax using py_compile on all .py files.

        Uses the workspace venv python directly to avoid uv project sync
        (which triggers editable installs and setuptools-scm issues).

        Args:
            workspace_path: Path to the workspace directory
            timeout: Timeout in seconds

        Returns:
            CompileResult with success status and parsed errors
        """
        # Use workspace venv python directly - avoids uv project sync
        # which would trigger editable install and setuptools-scm
        venv_python = self._get_venv_python(workspace_path)
        if venv_python is None:
            return CompileResult(
                success=False,
                errors=[f"No venv python found at {workspace_path}/.venv - workspace not properly provisioned"],
                raw_output="",
            )

        # Find all .py files
        py_files = list(workspace_path.rglob("*.py"))

        # Skip venv and common non-source directories
        skip_dirs = {
            ".venv",
            "venv",
            "__pycache__",
            ".git",
            "node_modules",
            "build",
            "dist",
        }
        py_files = [
            f for f in py_files if not any(skip in f.parts for skip in skip_dirs)
        ]

        if not py_files:
            return CompileResult(
                success=True,
                errors=[],
                raw_output="No Python files found to check",
            )

        errors = []
        all_output = []

        for py_file in py_files[:50]:  # Limit to first 50 files
            try:
                cmd = [str(venv_python), "-m", "py_compile", str(py_file)]

                result = subprocess.run(
                    cmd,
                    cwd=str(workspace_path),
                    capture_output=True,
                    text=True,
                    timeout=timeout // max(len(py_files), 1),
                )

                if result.returncode != 0:
                    error_msg = result.stderr.strip() or result.stdout.strip()
                    errors.append(f"{py_file.name}: {error_msg}")
                    all_output.append(f"=== {py_file} ===\n{error_msg}")

            except subprocess.TimeoutExpired:
                errors.append(f"{py_file.name}: Syntax check timed out")
            except Exception as e:
                errors.append(f"{py_file.name}: {str(e)}")

        raw_output = (
            "\n".join(all_output) if all_output else "All files passed syntax check"
        )

        return CompileResult(
            success=len(errors) == 0,
            errors=errors[:10],
            warnings=[],
            raw_output=raw_output[:3000] if len(raw_output) > 3000 else raw_output,
        )

    def install_dependencies(
        self,
        workspace_path: Path,
        packages: Optional[List[str]] = None,
        timeout: int = 300,
    ) -> InstallResult:
        """
        Install Python dependencies using uv.

        If packages are specified, installs those.
        Otherwise, detects and uses pyproject.toml (uv sync) or requirements.txt.

        Args:
            workspace_path: Path to the workspace directory
            packages: Optional list of specific packages to install
            timeout: Timeout in seconds

        Returns:
            InstallResult with success status and installed packages
        """
        try:
            binary = self.find_binary(workspace_path)
        except FileNotFoundError as e:
            return InstallResult(
                success=False,
                errors=[str(e)],
                raw_output="",
            )

        is_uv = self._is_uv(binary)

        installed: List[str] = []
        errors: List[str] = []
        all_output: List[str] = []

        if is_uv:
            # Use uv for dependency management
            if packages:
                # Install specific packages with uv add
                for package in packages:
                    try:
                        result = subprocess.run(
                            [binary, "add", package],
                            cwd=str(workspace_path),
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                        output = result.stdout + result.stderr
                        all_output.append(f"=== uv add {package} ===\n{output}")

                        if result.returncode == 0:
                            installed.append(package)
                        else:
                            errors.append(f"{package}: {output[:200]}")
                    except subprocess.TimeoutExpired:
                        errors.append(f"{package}: Installation timed out")
                    except Exception as e:
                        errors.append(f"{package}: {str(e)}")
            else:
                # Auto-detect and sync dependencies
                if (workspace_path / "pyproject.toml").exists():
                    try:
                        # Use uv sync for pyproject.toml projects
                        result = subprocess.run(
                            [binary, "sync"],
                            cwd=str(workspace_path),
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                        output = result.stdout + result.stderr
                        all_output.append(f"=== uv sync ===\n{output}")

                        if result.returncode == 0:
                            installed.append("pyproject.toml (uv sync)")
                        else:
                            errors.append(f"uv sync: {output[:500]}")
                    except subprocess.TimeoutExpired:
                        errors.append("uv sync: Installation timed out")
                    except Exception as e:
                        errors.append(f"uv sync: {str(e)}")

                elif (workspace_path / "requirements.txt").exists():
                    try:
                        # Use uv pip install for requirements.txt
                        result = subprocess.run(
                            [binary, "pip", "install", "-r", "requirements.txt"],
                            cwd=str(workspace_path),
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                        output = result.stdout + result.stderr
                        all_output.append(
                            f"=== uv pip install -r requirements.txt ===\n{output}"
                        )

                        if result.returncode == 0:
                            installed.append("requirements.txt")
                        else:
                            errors.append(f"requirements.txt: {output[:500]}")
                    except subprocess.TimeoutExpired:
                        errors.append("requirements.txt: Installation timed out")
                    except Exception as e:
                        errors.append(f"requirements.txt: {str(e)}")
                else:
                    all_output.append(
                        "No dependency files found (pyproject.toml, requirements.txt)"
                    )
        else:
            # Fallback to pip-based installation
            # Ensure venv exists
            venv_path = workspace_path / ".venv"
            if not venv_path.exists():
                try:
                    result = subprocess.run(
                        [binary, "-m", "venv", str(venv_path)],
                        cwd=str(workspace_path),
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    if result.returncode != 0:
                        return InstallResult(
                            success=False,
                            errors=[f"Failed to create venv: {result.stderr}"],
                            raw_output=result.stderr,
                        )
                except Exception as e:
                    return InstallResult(
                        success=False,
                        errors=[f"Failed to create venv: {str(e)}"],
                        raw_output="",
                    )

            # Use venv pip
            pip_bin = venv_path / "bin" / "pip"
            if not pip_bin.exists():
                pip_bin = venv_path / "Scripts" / "pip.exe"

            if not pip_bin.exists():
                return InstallResult(
                    success=False,
                    errors=["pip not found in venv"],
                    raw_output="",
                )

            if packages:
                # Install specific packages
                for package in packages:
                    try:
                        result = subprocess.run(
                            [str(pip_bin), "install", package],
                            cwd=str(workspace_path),
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                        output = result.stdout + result.stderr
                        all_output.append(f"=== pip install {package} ===\n{output}")

                        if result.returncode == 0:
                            installed.append(package)
                        else:
                            errors.append(f"{package}: {output[:200]}")
                    except subprocess.TimeoutExpired:
                        errors.append(f"{package}: Installation timed out")
                    except Exception as e:
                        errors.append(f"{package}: {str(e)}")
            else:
                # Auto-detect dependencies
                if (workspace_path / "requirements.txt").exists():
                    try:
                        result = subprocess.run(
                            [str(pip_bin), "install", "-r", "requirements.txt"],
                            cwd=str(workspace_path),
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                        output = result.stdout + result.stderr
                        all_output.append(
                            f"=== pip install -r requirements.txt ===\n{output}"
                        )

                        if result.returncode == 0:
                            installed.append("requirements.txt")
                        else:
                            errors.append(f"requirements.txt: {output[:500]}")
                    except subprocess.TimeoutExpired:
                        errors.append("requirements.txt: Installation timed out")
                    except Exception as e:
                        errors.append(f"requirements.txt: {str(e)}")

                elif (workspace_path / "pyproject.toml").exists():
                    try:
                        result = subprocess.run(
                            [str(pip_bin), "install", "-e", "."],
                            cwd=str(workspace_path),
                            capture_output=True,
                            text=True,
                            timeout=timeout,
                        )
                        output = result.stdout + result.stderr
                        all_output.append(f"=== pip install -e . ===\n{output}")

                        if result.returncode == 0:
                            installed.append("pyproject.toml")
                        else:
                            errors.append(f"pyproject.toml: {output[:500]}")
                    except subprocess.TimeoutExpired:
                        errors.append("pyproject.toml: Installation timed out")
                    except Exception as e:
                        errors.append(f"pyproject.toml: {str(e)}")
                else:
                    all_output.append(
                        "No dependency files found (requirements.txt, pyproject.toml)"
                    )

        raw_output = "\n".join(all_output)
        success = len(installed) > 0 or len(errors) == 0

        return InstallResult(
            success=success,
            installed=installed,
            errors=errors,
            raw_output=raw_output[:5000] if len(raw_output) > 5000 else raw_output,
        )

    def run_test(
        self,
        workspace_path: Path,
        match_contract: Optional[str] = None,
        match_test: Optional[str] = None,
        verbosity: int = 2,
        timeout: int = 300,
        additional_args: Optional[str] = None,
        framework_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TestResult:
        """
        Run Python tests using pytest.

        Uses the workspace venv python directly to avoid uv project sync
        (which triggers editable installs and setuptools-scm issues).

        Args:
            workspace_path: Path to the workspace directory
            match_contract: Filter by module/file pattern
            match_test: Filter by test function pattern (-k flag)
            verbosity: Verbosity level (maps to pytest -v flags)
            timeout: Timeout in seconds
            additional_args: Additional pytest arguments
            framework_kwargs: Framework-specific options

        Returns:
            TestResult with parsed test outcomes
        """
        # Use workspace venv python directly - avoids uv project sync
        venv_python = self._get_venv_python(workspace_path)
        if venv_python is None:
            return TestResult(
                success=False,
                error=f"No venv python found at {workspace_path}/.venv - workspace not properly provisioned",
            )

        # Use venv python with pytest module
        cmd = [str(venv_python), "-m", "pytest"]

        # Framework-specific kwargs
        fw = framework_kwargs or {}

        # Add match_path first - this restricts pytest to only collect from this path
        # This prevents pytest from trying to import other test files in the workspace
        if fw.get("match_path"):
            cmd.append(fw["match_path"])
        elif match_contract:
            # Treat as file path pattern
            cmd.append(match_contract)

        # Add match patterns
        if match_test:
            cmd.extend(["-k", match_test])

        # Add verbosity
        if verbosity > 0:
            cmd.append("-" + "v" * min(verbosity, 3))

        # More framework kwargs
        if fw.get("markers"):
            cmd.extend(["-m", fw["markers"]])
        if fw.get("maxfail"):
            cmd.extend(["--maxfail", str(fw["maxfail"])])

        # Additional args
        if additional_args:
            import shlex

            cmd.extend(shlex.split(additional_args))

        try:
            result = subprocess.run(
                cmd,
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output = result.stdout + result.stderr
            success = result.returncode == 0

            # Parse pytest output
            parsed = self._parse_test_output(output)

            # Check if parsing detected an import/module error
            error_msg = parsed.get("error")
            if error_msg:
                success = False

            return TestResult(
                success=success,
                tests_passed=parsed["tests_passed"],
                tests_failed=parsed["tests_failed"],
                assertion_failures=parsed["assertion_failures"],
                reverts=parsed["reverts"],
                parsed_results=parsed["parsed_results"],
                raw_output=output[:5000] if len(output) > 5000 else output,
                error=error_msg,
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
        """Return Python test file extension."""
        return ".py"

    def get_source_file_extension(self) -> str:
        """Return Python source file extension."""
        return ".py"

    def normalize_test_path(self, file_path: str, workspace: Path) -> Path:
        """Normalize test path for Python projects."""
        p = Path(file_path)

        if p.is_absolute():
            p = Path(p.name)

        normalized = p.as_posix().lstrip("/")

        # Strip leading test directories
        for prefix in ["tests/", "test/"]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break

        # Ensure proper extension
        if not normalized.endswith(".py"):
            normalized = normalized + ".py"

        # Ensure test_ prefix for pytest discovery
        name = Path(normalized).name
        if not name.startswith("test_"):
            parent = Path(normalized).parent
            normalized = (
                str(parent / f"test_{name}") if str(parent) != "." else f"test_{name}"
            )

        return workspace / "tests" / "poc" / normalized

    def get_allowed_patch_directories(self) -> List[str]:
        """Return allowed directories for patching."""
        return ["tests/poc", "tests/exploits", "test/poc", "test/exploits"]

    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get Python-specific tool descriptions."""
        descriptions = {
            "write_and_compile": """Write a Python test file to the workspace and check syntax.

Args:
    file_path: Test file name (e.g., "test_exploit.py" or "poc/test_exploit.py")
    content: Python test file content

Returns:
    {"written": bool, "path": str, "compiled": bool, "errors": List[str], "raw_output": str}

Example:
    result = write_and_compile("test_exploit.py", '''
    import pytest
    from vulnerable_module import VulnerableClass

    def test_exploit():
        \"\"\"Test that demonstrates the vulnerability.\"\"\"
        obj = VulnerableClass()
        # Trigger the vulnerability
        result = obj.vulnerable_method(malicious_input)
        # Assert the exploit succeeded
        assert result.balance < 0, "Exploit: achieved negative balance"
    ''')

    if not result["compiled"]:
        # Fix syntax errors in result["errors"]
        pass""",
            "run_test": """Run Python tests with pytest via uv and get parsed results.

Tests are executed using `uv run --with pytest pytest`.

Args:
    match_contract: Filter by file/module pattern
    match_test: Filter by test function pattern (pytest -k flag)
    verbosity: Verbosity level 0-3
    additional_args: Extra pytest arguments (e.g., "--tb=short")
    framework_kwargs: Optional dict for Python-specific options:
        {"markers": "slow", "maxfail": 1}

Returns:
    {
        "success": bool,
        "tests_passed": int,
        "tests_failed": int,
        "assertion_failures": List[str],
        "parsed_results": Dict[str, str],
        "raw_output": str
    }

Example:
    result = run_test(match_test="test_exploit")

    if result["tests_passed"] > 0:
        print("Exploit test passed - vulnerability confirmed!")""",
            "register_exploit": """Register an exploit finding for Python.

Args:
    exploit_found: True if you found a way to exploit the vulnerability
    reasoning: Explanation of your analysis and conclusion
    poc_path: Path to the PoC test file (e.g., "tests/poc/test_exploit.py")
    poc_code: Full Python code of the PoC

Example:
    register_exploit(
        exploit_found=True,
        reasoning="The input validation in parse_user_data() can be bypassed...",
        poc_path="tests/poc/test_injection.py",
        poc_code='''import pytest
from app import parse_user_data

def test_injection_exploit():
    malicious_input = {"__class__": ...}
    # ... exploit code
    assert compromised
'''
    )""",
        }
        return descriptions.get(tool_name)

    def get_poc_guidance(self) -> str:
        """Get Python-specific PoC writing guidance."""
        return """## PoC Format: Python/pytest (via uv)
Write Python test files in tests/poc/.
- Tests run via `uv run --with pytest pytest`
- Use pytest with descriptive test function names (test_*)
- Import REAL modules from the codebase (don't create mocks)
- Use assertions to prove the exploit: assert, pytest.raises()
- A PASSING test with assertions proving vulnerability = valid exploit
- Use pytest fixtures for setup if needed
- Dependencies managed via pyproject.toml with `uv sync`"""

    def _parse_test_output(self, output: str) -> dict:
        """Parse pytest output to extract results."""
        import re

        # Check for ModuleNotFoundError - common when deps weren't installed
        module_error_match = re.search(
            r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", output
        )
        if module_error_match:
            missing_module = module_error_match.group(1)
            return {
                "tests_passed": 0,
                "tests_failed": 1,
                "assertion_failures": [],
                "reverts": [],
                "parsed_results": {},
                "error": f"Missing dependency: '{missing_module}'. "
                f"The workspace may not have installed all required packages. "
                f"Check if the project's dependencies were installed correctly.",
            }

        # Check for ImportError as well (related issue)
        import_error_match = re.search(
            r"ImportError: cannot import name ['\"]([^'\"]+)['\"]", output
        )
        if import_error_match:
            missing_import = import_error_match.group(1)
            return {
                "tests_passed": 0,
                "tests_failed": 1,
                "assertion_failures": [],
                "reverts": [],
                "parsed_results": {},
                "error": f"Import error: cannot import '{missing_import}'. "
                f"This may indicate missing or incompatible dependencies.",
            }

        tests_passed = 0
        tests_failed = 0
        assertion_failures: List[str] = []
        reverts: List[str] = []
        parsed_results: dict = {}

        lines = output.split("\n")

        for line in lines:
            # pytest summary line: "X passed, Y failed"
            if " passed" in line or " failed" in line:
                import re

                passed_match = re.search(r"(\d+) passed", line)
                failed_match = re.search(r"(\d+) failed", line)
                if passed_match:
                    tests_passed = int(passed_match.group(1))
                if failed_match:
                    tests_failed = int(failed_match.group(1))

            # Individual test results: "PASSED test_module.py::test_name"
            if line.strip().startswith("PASSED"):
                test_name = line.split("::")[-1].strip() if "::" in line else "unknown"
                parsed_results[test_name] = "pass"
            elif line.strip().startswith("FAILED"):
                test_name = line.split("::")[-1].strip() if "::" in line else "unknown"
                parsed_results[test_name] = "fail"
                if "AssertionError" in output:
                    assertion_failures.append(test_name)

        return {
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "assertion_failures": assertion_failures,
            "reverts": reverts,
            "parsed_results": parsed_results,
        }
