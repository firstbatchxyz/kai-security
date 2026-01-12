"""
TypeScript tool adapter.

Extends the JavaScript adapter with TypeScript-specific behavior for:
- File extensions (.ts, .tsx)
- TypeScript-specific test patterns
- Language identification
"""

from pathlib import Path
from typing import Optional

from kai.utils.tool_adapters.javascript import JavaScriptToolAdapter


class TypeScriptToolAdapter(JavaScriptToolAdapter):
    """
    Tool adapter for TypeScript projects.

    Extends JavaScriptToolAdapter since TypeScript projects use the same
    tooling (npm/yarn/pnpm, Jest/Vitest/Mocha) but with TypeScript-specific
    file extensions and compilation.
    """

    @property
    def framework_name(self) -> str:
        return "typescript"

    @property
    def language(self) -> str:
        return "typescript"

    def get_test_file_extension(self) -> str:
        """Return TypeScript test file extension."""
        return ".test.ts"

    def get_source_file_extension(self) -> str:
        """Return TypeScript source file extension."""
        return ".ts"

    def normalize_test_path(self, file_path: str, workspace: Path) -> Path:
        """Normalize test path for TypeScript projects."""
        p = Path(file_path)

        if p.is_absolute():
            p = Path(p.name)

        normalized = p.as_posix().lstrip("/")

        # Strip leading test directories
        for prefix in ["tests/", "test/", "__tests__/"]:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break

        # Ensure proper extension (default to .test.ts for TypeScript)
        if not any(
            normalized.endswith(ext)
            for ext in [".test.ts", ".test.tsx", ".spec.ts", ".spec.tsx"]
        ):
            if normalized.endswith(".ts") or normalized.endswith(".tsx"):
                base = normalized.rsplit(".", 1)[0]
                normalized = base + ".test.ts"
            else:
                normalized = normalized + ".test.ts"

        return workspace / "tests" / "poc" / normalized

    def get_tool_description(self, tool_name: str) -> Optional[str]:
        """Get TypeScript-specific tool descriptions."""
        descriptions = {
            "write_and_compile": """Write a TypeScript test file to the workspace and check syntax.

Args:
    file_path: Test file name (e.g., "exploit.test.ts" or "poc/exploit.test.ts")
    content: TypeScript test file content

Returns:
    {"written": bool, "path": str, "compiled": bool, "errors": List[str], "raw_output": str}

Example:
    result = write_and_compile("exploit.test.ts", '''
    import { expect } from 'chai';
    import { VulnerableContract } from '../src/vulnerable';

    describe('Exploit', () => {
        it('should demonstrate the vulnerability', async () => {
            const contract = new VulnerableContract();
            // Trigger the vulnerability
            const result = await contract.vulnerableMethod(maliciousInput);
            // Assert the exploit succeeded
            expect(result.balance).to.be.lessThan(0);
        });
    });
    ''')

    if not result["compiled"]:
        # Fix errors in result["errors"]
        pass""",
            "run_test": """Run TypeScript tests with detected framework (Jest/Mocha/Vitest).

Args:
    match_contract: Filter by file pattern
    match_test: Filter by test name pattern
    verbosity: Verbosity level
    additional_args: Extra test arguments
    framework_kwargs: Optional dict:
        {"coverage": true}

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
    result = run_test(match_test="exploit")

    if result["tests_passed"] > 0:
        print("Exploit test passed - vulnerability confirmed!")""",
            "register_exploit": """Register an exploit finding for TypeScript.

Args:
    exploit_found: True if you found a way to exploit the vulnerability
    reasoning: Explanation of your analysis and conclusion
    poc_path: Path to the PoC test file (e.g., "tests/poc/exploit.test.ts")
    poc_code: Full TypeScript code of the PoC

Example:
    register_exploit(
        exploit_found=True,
        reasoning="The prototype pollution in merge() allows arbitrary property injection...",
        poc_path="tests/poc/prototype_pollution.test.ts",
        poc_code='''
import { expect } from "chai";
import { merge } from "../src/utils";

describe("Prototype Pollution Exploit", () => {
    it("should pollute Object prototype", () => {
        const payload = JSON.parse('{"__proto__": {"admin": true}}');
        merge({}, payload);
        expect(({} as any).admin).to.equal(true);
    });
});
'''
    )""",
        }
        return descriptions.get(tool_name)

    def get_poc_guidance(self) -> str:
        """Get TypeScript-specific PoC writing guidance."""
        return """## PoC Format: TypeScript/Node.js
Write TypeScript test files in tests/poc/.
- Use the project's test framework (Jest/Mocha/Vitest)
- Import REAL modules from the codebase (don't create mocks)
- Use expect/assert to prove the exploit
- A PASSING test with assertions proving vulnerability = valid exploit
- For async code, use async/await or proper promise handling
- TypeScript type errors won't prevent test execution if using ts-jest/vitest"""
