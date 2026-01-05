"""
Foundry tool adapter.

Provides Foundry/Forge-specific implementations for:
- Finding forge binary
- Compiling Solidity with forge build
- Running tests with forge test
- Parsing Foundry output formats
"""

import shutil
import subprocess
import shlex
from pathlib import Path
from typing import Optional, List, Any, Dict

from kai.utils.tool_adapters.base import (
    ToolAdapter,
    CompileResult,
    InstallResult,
    TestResult,
)


class FoundryToolAdapter(ToolAdapter):
    """Tool adapter for Foundry/Forge Solidity projects."""

    @property
    def framework_name(self) -> str:
        return "foundry"

    @property
    def language(self) -> str:
        return "solidity"

    def find_binary(self) -> str:
        """
        Find the forge binary, checking common installation paths.

        Returns:
            Path to forge binary

        Raises:
            FileNotFoundError: If forge is not found
        """
        # First try shutil.which (uses PATH)
        forge_path = shutil.which("forge")
        if forge_path:
            return forge_path

        # Common Foundry installation paths
        home = Path.home()
        common_paths = [
            home / ".foundry" / "bin" / "forge",
            home / ".cargo" / "bin" / "forge",
            Path("/usr/local/bin/forge"),
            Path("/opt/homebrew/bin/forge"),
        ]

        for path in common_paths:
            if path.exists() and path.is_file():
                return str(path)

        raise FileNotFoundError(
            "forge not found - is Foundry installed? "
            "Install with: curl -L https://foundry.paradigm.xyz | bash && foundryup"
        )

    def compile(
        self,
        workspace_path: Path,
        timeout: int = 120,
    ) -> CompileResult:
        """
        Compile Solidity project using forge build.

        Args:
            workspace_path: Path to the workspace directory
            timeout: Timeout in seconds

        Returns:
            CompileResult with parsed errors/warnings
        """
        try:
            forge_bin = self.find_binary()
        except FileNotFoundError as e:
            return CompileResult(
                success=False,
                errors=[str(e)],
                raw_output="",
            )

        try:
            result = subprocess.run(
                [forge_bin, "build"],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            raw_output = result.stdout + result.stderr
            success = result.returncode == 0

            errors = []
            warnings = []

            if not success:
                errors = self.parse_compile_errors(raw_output)

            warnings = self.parse_compile_warnings(raw_output)

            return CompileResult(
                success=success,
                errors=errors,
                warnings=warnings,
                raw_output=raw_output[:3000] if len(raw_output) > 3000 else raw_output,
            )

        except subprocess.TimeoutExpired:
            return CompileResult(
                success=False,
                errors=[f"Compilation timed out after {timeout} seconds"],
                raw_output="",
            )
        except Exception as e:
            return CompileResult(
                success=False,
                errors=[str(e)],
                raw_output="",
            )

    def install_dependencies(
        self,
        workspace_path: Path,
        packages: Optional[List[str]] = None,
        timeout: int = 300,
    ) -> InstallResult:
        """
        Install Solidity dependencies using forge install.

        If packages are specified, installs those specific packages.
        Otherwise, parses .gitmodules to find dependencies and installs them.

        Args:
            workspace_path: Path to the workspace directory
            packages: Optional list of packages (e.g., ["OpenZeppelin/openzeppelin-contracts"])
            timeout: Timeout in seconds

        Returns:
            InstallResult with success status and installed packages
        """
        try:
            forge_bin = self.find_binary()
        except FileNotFoundError as e:
            return InstallResult(
                success=False,
                errors=[str(e)],
                raw_output="",
            )

        # If no packages specified, try to parse from .gitmodules
        if not packages:
            packages = self._parse_gitmodules_packages(workspace_path)

        if not packages:
            return InstallResult(
                success=True,
                installed=[],
                raw_output="No packages to install (no .gitmodules or packages specified)",
            )

        installed: List[str] = []
        errors: List[str] = []
        all_output: List[str] = []

        for package in packages:
            try:
                # forge install <package> (default behavior is no commit in newer versions)
                cmd = [forge_bin, "install", package]
                result = subprocess.run(
                    cmd,
                    cwd=str(workspace_path),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                output = result.stdout + result.stderr
                all_output.append(f"=== {package} ===\n{output}")

                if result.returncode == 0:
                    installed.append(package)
                else:
                    # Check if it's already installed (not an error)
                    if (
                        "already exists" in output.lower()
                        or "already installed" in output.lower()
                    ):
                        installed.append(package)
                    else:
                        errors.append(f"{package}: {output.strip()[:200]}")

            except subprocess.TimeoutExpired:
                errors.append(f"{package}: Installation timed out after {timeout}s")
            except Exception as e:
                errors.append(f"{package}: {str(e)}")

        raw_output = "\n".join(all_output)
        success = len(installed) > 0 or len(errors) == 0

        return InstallResult(
            success=success,
            installed=installed,
            errors=errors,
            raw_output=raw_output[:5000] if len(raw_output) > 5000 else raw_output,
        )

    def _parse_gitmodules_packages(self, workspace_path: Path) -> List[str]:
        """
        Parse .gitmodules to extract package URLs for forge install.

        Converts URLs like:
        - https://github.com/OpenZeppelin/openzeppelin-contracts -> OpenZeppelin/openzeppelin-contracts
        - git@github.com:foundry-rs/forge-std.git -> foundry-rs/forge-std
        """
        gitmodules_path = workspace_path / ".gitmodules"
        if not gitmodules_path.exists():
            return []

        packages: List[str] = []
        try:
            content = gitmodules_path.read_text()
            import re

            # Match url = <value> lines
            url_pattern = re.compile(r"url\s*=\s*(.+)")
            for match in url_pattern.finditer(content):
                url = match.group(1).strip()
                package = self._url_to_package(url)
                if package:
                    packages.append(package)

        except Exception:
            pass

        return packages

    def _url_to_package(self, url: str) -> Optional[str]:
        """Convert a git URL to a forge install package identifier."""
        import re

        # Remove .git suffix
        url = re.sub(r"\.git$", "", url)

        # Handle HTTPS URLs: https://github.com/owner/repo
        https_match = re.match(r"https?://github\.com/([^/]+)/([^/]+)", url)
        if https_match:
            return f"{https_match.group(1)}/{https_match.group(2)}"

        # Handle SSH URLs: git@github.com:owner/repo
        ssh_match = re.match(r"git@github\.com:([^/]+)/([^/]+)", url)
        if ssh_match:
            return f"{ssh_match.group(1)}/{ssh_match.group(2)}"

        return None

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
        Run Foundry tests using forge test.

        Args:
            workspace_path: Path to the workspace directory
            match_contract: Filter by contract name pattern (--match-contract)
            match_test: Filter by test function name pattern (--match-test)
            verbosity: -v level (0-5), default 3 for traces on failure
            timeout: Timeout in seconds
            additional_args: Additional CLI arguments

        Returns:
            TestResult with parsed test outcomes
        """
        try:
            forge_bin = self.find_binary()
        except FileNotFoundError as e:
            return TestResult(success=False, error=str(e))

        cmd = [forge_bin, "test"]

        if match_contract:
            cmd.extend(["--match-contract", match_contract])
        if match_test:
            cmd.extend(["--match-test", match_test])

        # Framework-specific knobs (best-effort)
        fw = framework_kwargs or {}
        match_path = fw.get("match_path")
        if isinstance(match_path, str) and match_path.strip():
            cmd.extend(["--match-path", match_path.strip()])

        fuzz_seed = fw.get("fuzz_seed", fw.get("seed"))
        if fuzz_seed is not None:
            try:
                cmd.extend(["--fuzz-seed", str(int(fuzz_seed))])
            except Exception:
                # Ignore unparseable seeds rather than crashing the run
                pass

        # Add verbosity
        if verbosity > 0:
            cmd.append("-" + "v" * verbosity)

        # Add additional args
        if additional_args:
            # Preserve quoted args.
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

            # Parse results
            parsed = self._parse_test_output(output)

            return TestResult(
                success=success,
                tests_passed=parsed["tests_passed"],
                tests_failed=parsed["tests_failed"],
                assertion_failures=parsed["assertion_failures"],
                reverts=parsed["reverts"],
                parsed_results=parsed["parsed_results"],
                raw_output=output[:5000] if len(output) > 5000 else output,
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
        """Return Foundry test file extension."""
        return ".t.sol"

    def get_source_file_extension(self) -> str:
        """Return Solidity source file extension."""
        return ".sol"

    def normalize_test_path(self, file_path: str, workspace: Path) -> Path:
        """
        Normalize test path for Foundry projects.

        Important: some repos configure a non-default test directory in foundry.toml,
        e.g. `test = "test/foundry"`. In those repos, `forge test --match-path ...`
        will only match files under the configured test dir. So we must write the
        smoke test (and PoCs) under that directory.
        """
        test_dir = self._detect_test_dir(workspace) or "test"

        p = Path(file_path)
        if p.is_absolute():
            p = Path(p.name)

        normalized = p.as_posix().lstrip("/")
        # Strip leading configured test_dir or generic "test/"
        if normalized.startswith(test_dir.rstrip("/") + "/"):
            normalized = normalized[len(test_dir.rstrip("/")) + 1 :]
        elif normalized.startswith("test/"):
            normalized = normalized[len("test/") :]

        # Ensure extension
        if not normalized.endswith(
            self.get_test_file_extension()
        ) and not normalized.endswith(self.get_source_file_extension()):
            normalized = normalized + self.get_test_file_extension()

        return workspace / test_dir / normalized

    def get_allowed_patch_directories(self) -> List[str]:
        # Support both default and common custom Foundry test roots.
        return [
            "test/poc",
            "test\\poc",
            "test/foundry/poc",
            "test\\foundry\\poc",
        ]

    def _detect_test_dir(self, workspace: Path) -> Optional[str]:
        """
        Best-effort parse of foundry.toml to determine the configured test directory.
        """
        candidates = [workspace / "foundry.toml", workspace / "forge" / "foundry.toml"]
        for cfg in candidates:
            if not cfg.exists() or not cfg.is_file():
                continue
            try:
                import tomllib  # py3.11+
            except Exception:
                tomllib = None  # type: ignore[assignment]
            if tomllib is None:
                continue
            try:
                data = tomllib.loads(cfg.read_text(encoding="utf-8"))
                profile = (data.get("profile") or {}).get("default") or {}
                test_dir = profile.get("test") or data.get("test")
                if isinstance(test_dir, str) and test_dir.strip():
                    return test_dir.strip().strip('"').strip("'")
            except Exception:
                continue
        return None

    def parse_compile_errors(self, output: str) -> List[str]:
        """
        Parse Foundry compilation output for errors.

        Foundry error format:
        - "Error: ..." lines
        - "error[...]:" lines (solc style)
        """
        errors = []
        for line in output.split("\n"):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Foundry/solc error patterns
            if line_lower.startswith("error"):
                errors.append(line_stripped)
            elif "error[" in line_lower:
                errors.append(line_stripped)
            elif "error:" in line_lower:
                errors.append(line_stripped)

        # Deduplicate while preserving order
        seen = set()
        unique_errors = []
        for e in errors:
            if e not in seen:
                seen.add(e)
                unique_errors.append(e)

        return unique_errors[:10]

    def _parse_test_output(self, output: str) -> dict:
        """
        Parse forge test output to extract results.

        Forge output format:
        - [PASS] test_name() (gas: 1234)
        - [FAIL. Reason: ...] test_name()
        """
        tests_passed = 0
        tests_failed = 0
        assertion_failures: List[str] = []
        reverts: List[str] = []
        parsed_results: dict = {}

        for line in output.split("\n"):
            # [PASS] test_name()
            if "[PASS]" in line:
                tests_passed += 1
                test_name = self._extract_test_name(line)
                if test_name:
                    parsed_results[test_name] = "pass"

            # [FAIL] or [FAIL. Reason: ...]
            elif "[FAIL" in line:
                tests_failed += 1
                test_name = self._extract_test_name(line)
                full_name = test_name or "unknown"

                # Determine failure type
                line_lower = line.lower()
                if "assertion" in line_lower or "assert" in line_lower:
                    assertion_failures.append(full_name)
                    parsed_results[full_name] = "assertion_fail"
                elif "revert" in line_lower:
                    reverts.append(full_name)
                    parsed_results[full_name] = "revert"
                else:
                    # Generic failure
                    parsed_results[full_name] = "fail"

        return {
            "tests_passed": tests_passed,
            "tests_failed": tests_failed,
            "assertion_failures": assertion_failures,
            "reverts": reverts,
            "parsed_results": parsed_results,
        }

    def _extract_test_name(self, line: str) -> Optional[str]:
        """Extract test function name from forge output line."""
        # Look for test_ pattern
        if "test_" not in line:
            return None

        # Extract: test_something() or test_something(args)
        try:
            start = line.index("test_")
            # Find the opening paren
            paren_idx = line.index("(", start)
            return line[start:paren_idx]
        except ValueError:
            return None

    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get Foundry/Solidity-specific tool descriptions."""
        descriptions = {
            "write_and_compile": """Write a Solidity test file to the workspace and compile it with forge.

Args:
    file_path: Test file name (e.g., "MyExploit.t.sol" or "poc/Exploit.t.sol")
    content: Solidity test file content

Returns:
    {"written": bool, "path": str, "compiled": bool, "errors": List[str], "raw_output": str}

Example:
    result = write_and_compile("Exploit.t.sol", '''
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.0;

    import "forge-std/Test.sol";
    import "src/Vault.sol";

    contract ExploitTest is Test {
        Vault vault;

        function setUp() public {
            vault = new Vault();
        }

        function test_exploit() public {
            // Your exploit logic here
            vault.deposit{value: 1 ether}();
            // ... attack sequence
            assertGt(address(this).balance, 1 ether);
        }
    }
    ''')

    if not result["compiled"]:
        # Fix errors in result["errors"]
        pass""",
            "run_test": """Run Foundry tests with forge test and get parsed results.

Args:
    match_contract: Filter by contract name pattern (e.g., "ExploitTest")
    match_test: Filter by test function pattern (e.g., "test_exploit")
    verbosity: Verbosity level 0-5, default 3 shows traces on failure
    additional_args: Extra forge arguments (e.g., "--gas-report")
    framework_kwargs: Optional dict for Foundry-specific knobs, e.g.:
        {"match_path": "test/poc/Exploit.t.sol", "fuzz_seed": 123}

Returns:
    {
        "success": bool,
        "tests_passed": int,
        "tests_failed": int,
        "assertion_failures": List[str],  # Tests that failed assertions
        "reverts": List[str],              # Tests that reverted
        "parsed_results": Dict[str, str],  # test_name -> "pass"|"fail"|"revert"
        "raw_output": str
    }

Example:
    result = run_test(match_contract="ExploitTest", match_test="test_exploit")

    if result["tests_passed"] > 0:
        print("Exploit test passed - vulnerability confirmed!")
    if result["assertion_failures"]:
        print(f"Assertions failed: {result['assertion_failures']}")""",
            "patch_file": """Patch a Solidity file by replacing old_content with new_content, then recompile.

Useful for fixing compilation errors without rewriting the entire file.
Can only patch files in test/poc/ directory (workspace safety).

Args:
    file_path: Path to file in test/poc/ (e.g., "test/poc/Exploit.t.sol")
    old_content: Exact string to find and replace
    new_content: Replacement string

Returns:
    Same as write_and_compile: {written, path, compiled, errors, ...}

Example:
    # Fix an import error
    result = patch_file(
        "test/poc/Exploit.t.sol",
        'import "../src/Token.sol";',
        'import "src/Token.sol";'
    )

    # Fix a syntax error
    result = patch_file(
        "test/poc/Exploit.t.sol",
        'uint x = ;',
        'uint x = 0;'
    )""",
            "register_exploit": """Register an exploit finding (or verification that invariant holds).

Call this tool when you have determined whether the invariant can be violated.
You can call this multiple times if you find multiple distinct exploits.

Args:
    exploit_found: True if you found a way to violate the invariant
    reasoning: Explanation of your analysis and conclusion
    poc_path: Path to the PoC test file (e.g., "test/poc/Exploit.t.sol")
    poc_code: Full Solidity code of the PoC (required if exploit_found or verification test)

Returns:
    Confirmation of registration with exploit count

Example (exploit found):
    register_exploit(
        exploit_found=True,
        reasoning="The mint() function lacks role check when called via delegatecall...",
        poc_path="test/poc/INV_ACCESS_001.t.sol",
        poc_code='''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;
import "forge-std/Test.sol";
// ... full test code
'''
    )

Example (no exploit, with verification test):
    register_exploit(
        exploit_found=False,
        reasoning="All paths to mint() are guarded by onlyRole(MINTER_ROLE)...",
        poc_path="test/poc/INV_ACCESS_001_verify.t.sol",
        poc_code="// Test that unauthorized users cannot mint..."
    )""",
        }
        return descriptions.get(tool_name)
