"""
Base tool adapter interface.

Tool adapters provide framework-specific implementations for:
- Finding compiler/test runner binaries
- Compiling code
- Running tests
- Parsing outputs

All framework-specific adapters (Foundry, Hardhat, Anchor, etc.) inherit from ToolAdapter.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class InstallResult:
    """Result from a dependency installation attempt."""

    success: bool
    installed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "installed": self.installed,
            "errors": self.errors,
            "raw_output": self.raw_output,
        }


@dataclass
class CompileResult:
    """Result from a compilation attempt."""

    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_output: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "compiled": self.success,
            "errors": self.errors,
            "warnings": self.warnings,
            "raw_output": self.raw_output,
        }


@dataclass
class TestResult:
    """Result from running tests."""

    success: bool
    tests_passed: int = 0
    tests_failed: int = 0
    assertion_failures: List[str] = field(default_factory=list)
    reverts: List[str] = field(default_factory=list)
    parsed_results: Dict[str, str] = field(default_factory=dict)
    raw_output: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "assertion_failures": self.assertion_failures,
            "reverts": self.reverts,
            "parsed_results": self.parsed_results,
            "raw_output": self.raw_output,
        }
        if self.error:
            result["error"] = self.error
        return result


class ToolAdapter(ABC):
    """
    Abstract base class for framework-specific tool adapters.

    Each framework (Foundry, Hardhat, Anchor, etc.) implements its own adapter
    to handle framework-specific operations like compilation and testing.
    """

    @property
    @abstractmethod
    def framework_name(self) -> str:
        """Return the framework name (e.g., 'foundry', 'hardhat', 'anchor')."""
        ...

    @property
    @abstractmethod
    def language(self) -> str:
        """Return the primary language (e.g., 'solidity', 'rust')."""
        ...

    @abstractmethod
    def find_binary(self) -> str:
        """
        Find the framework's compiler/test runner binary.

        Returns:
            Path to the binary

        Raises:
            FileNotFoundError: If the binary is not found
        """
        ...

    @abstractmethod
    def compile(
        self,
        workspace_path: Path,
        timeout: int = 120,
    ) -> CompileResult:
        """
        Compile the project in the given workspace.

        Args:
            workspace_path: Path to the workspace directory
            timeout: Timeout in seconds

        Returns:
            CompileResult with success status and parsed errors
        """
        ...

    def install_dependencies(
        self,
        workspace_path: Path,
        packages: Optional[List[str]] = None,
        timeout: int = 300,
    ) -> InstallResult:
        """
        Install project dependencies.

        This method installs framework-specific dependencies. If no packages are
        specified, it attempts to install all dependencies defined in config files
        (e.g., .gitmodules for Foundry, package.json for Hardhat).

        Args:
            workspace_path: Path to the workspace directory
            packages: Optional list of specific packages to install (e.g., ["OpenZeppelin/openzeppelin-contracts"])
            timeout: Timeout in seconds

        Returns:
            InstallResult with success status, installed packages, and any errors
        """
        # Default implementation: no-op (override in subclasses)
        return InstallResult(
            success=True,
            installed=[],
            raw_output="No dependency installation required for this framework",
        )

    @abstractmethod
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
        """
        Run tests in the given workspace.

        Args:
            workspace_path: Path to the workspace directory
            match_contract: Filter by contract/module name pattern
            match_test: Filter by test function name pattern
            verbosity: Verbosity level (framework-specific interpretation)
            timeout: Timeout in seconds
            additional_args: Additional CLI arguments
            framework_kwargs: Framework-specific knobs as a dict (best-effort).
                Adapters should ignore unknown keys and interpret known ones.

        Returns:
            TestResult with parsed test outcomes
        """
        ...

    @abstractmethod
    def get_test_file_extension(self) -> str:
        """
        Return the test file extension for this framework.

        Returns:
            Extension string (e.g., '.t.sol', '_test.rs')
        """
        ...

    @abstractmethod
    def get_source_file_extension(self) -> str:
        """
        Return the source file extension for this framework.

        Returns:
            Extension string (e.g., '.sol', '.rs')
        """
        ...

    def normalize_test_path(self, file_path: str, workspace: Path) -> Path:
        """
        Normalize a test file path for this framework.

        Handles various input formats:
        - "MyTest.t.sol"
        - "poc/MyTest.t.sol"
        - "test/poc/MyTest.t.sol"

        Default implementation: places file under workspace/test/

        Args:
            file_path: Raw file path from user/agent
            workspace: Workspace root path

        Returns:
            Absolute path where the file should be written
        """
        p = Path(file_path)

        # Strip absolute path to just filename
        if p.is_absolute():
            p = Path(p.name)

        # Strip leading "test/" if present (we add it back)
        if p.parts and p.parts[0] == "test":
            p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(p.name)

        # Ensure proper extension
        test_ext = self.get_test_file_extension()
        if not p.name.endswith(test_ext) and not p.name.endswith(
            self.get_source_file_extension()
        ):
            p = p.with_name(p.name + test_ext)

        return workspace / "test" / p

    def get_test_directory(self) -> str:
        """
        Return the default test directory name.

        Returns:
            Directory name (e.g., 'test', 'tests')
        """
        return "test"

    def parse_compile_errors(self, output: str) -> List[str]:
        """
        Parse compilation output to extract error messages.

        Default implementation: look for lines containing 'error'.

        Args:
            output: Raw compiler output

        Returns:
            List of error message strings
        """
        errors = []
        for line in output.split("\n"):
            line_lower = line.lower()
            if "error" in line_lower:
                errors.append(line.strip())
        return errors[:10]  # Limit to most relevant

    def parse_compile_warnings(self, output: str) -> List[str]:
        """
        Parse compilation output to extract warning messages.

        Default implementation: look for lines containing 'warning'.

        Args:
            output: Raw compiler output

        Returns:
            List of warning message strings
        """
        warnings = []
        for line in output.split("\n"):
            if "warning" in line.lower():
                warnings.append(line.strip())
        return warnings[:10]

    def get_allowed_patch_directories(self) -> List[str]:
        """
        Return directory prefixes where patch_file is allowed to operate.

        Default: ["test/poc", "test\\poc"] for Foundry-style projects.
        Override for framework-specific test directory structures.

        Returns:
            List of allowed directory prefixes (relative to workspace)
        """
        return ["test/poc", "test\\poc"]

    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """
        Get framework-specific description for a tool.

        Override this method to provide custom descriptions for tools
        that need framework-specific examples.

        Args:
            tool_name: Name of the tool (e.g., "write_and_compile", "run_test")

        Returns:
            Description string or None to use default docstring
        """
        return None
