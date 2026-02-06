"""
Tests for tool adapters (Python, JavaScript, C).

Tests the ToolAdapter interface implementations for each language.
"""

from pathlib import Path

import pytest

from kai.utils.tool_adapters import (
    get_tool_adapter,
    get_supported_frameworks,
    PythonToolAdapter,
    JavaScriptToolAdapter,
    CToolAdapter,
    FoundryToolAdapter,
)


class TestToolAdapterRegistry:
    """Tests for the tool adapter registry."""

    def test_get_supported_frameworks(self):
        """Should return list of supported frameworks."""
        supported = get_supported_frameworks()
        assert "foundry" in supported
        assert "python" in supported
        assert "javascript" in supported
        assert "c" in supported

    def test_get_adapter_foundry(self):
        """Should return FoundryToolAdapter for foundry."""
        adapter = get_tool_adapter("foundry")
        assert isinstance(adapter, FoundryToolAdapter)

    def test_get_adapter_python(self):
        """Should return PythonToolAdapter for python."""
        adapter = get_tool_adapter("python")
        assert isinstance(adapter, PythonToolAdapter)

    def test_get_adapter_python_alias(self):
        """Should return PythonToolAdapter for py alias."""
        adapter = get_tool_adapter("py")
        assert isinstance(adapter, PythonToolAdapter)

    def test_get_adapter_javascript(self):
        """Should return JavaScriptToolAdapter for javascript."""
        adapter = get_tool_adapter("javascript")
        assert isinstance(adapter, JavaScriptToolAdapter)

    def test_get_adapter_javascript_aliases(self):
        """Should return JavaScriptToolAdapter for js/node aliases."""
        adapter_js = get_tool_adapter("js")
        adapter_node = get_tool_adapter("node")
        assert isinstance(adapter_js, JavaScriptToolAdapter)
        assert isinstance(adapter_node, JavaScriptToolAdapter)

    def test_get_adapter_c(self):
        """Should return CToolAdapter for c."""
        adapter = get_tool_adapter("c")
        assert isinstance(adapter, CToolAdapter)

    def test_get_adapter_invalid(self):
        """Should raise ValueError for unknown adapter."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            get_tool_adapter("unknown_framework")


class TestPythonToolAdapter:
    """Tests for PythonToolAdapter."""

    @pytest.fixture
    def adapter(self):
        return PythonToolAdapter()

    @pytest.fixture
    def python_project(self, tmp_path: Path):
        """Create a minimal Python project."""
        # Create pyproject.toml
        (tmp_path / "pyproject.toml").write_text("""
[project]
name = "test-project"
version = "0.1.0"
""")
        # Create a simple Python file
        (tmp_path / "app.py").write_text("""
def hello():
    return "Hello, World!"
""")
        # Create tests directory
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "__init__.py").write_text("")
        (tmp_path / "tests" / "test_app.py").write_text("""
def test_hello():
    from app import hello
    assert hello() == "Hello, World!"
""")
        return tmp_path

    def test_framework_name(self, adapter: PythonToolAdapter):
        """Should return python as framework name."""
        assert adapter.framework_name == "python"

    def test_normalize_test_path(self, adapter: PythonToolAdapter, tmp_path: Path):
        """Should normalize test paths correctly."""
        # With .py extension
        result = adapter.normalize_test_path("test_foo.py", tmp_path)
        assert result.suffix == ".py"
        assert "test_foo" in result.stem

        # Without extension
        result = adapter.normalize_test_path("test_bar", tmp_path)
        assert result.suffix == ".py"

        # In tests directory
        result = adapter.normalize_test_path("tests/test_baz", tmp_path)
        assert "tests" in str(result)

    def test_get_poc_guidance(self, adapter: PythonToolAdapter):
        """Should return Python-specific PoC guidance."""
        guidance = adapter.get_poc_guidance()
        assert "pytest" in guidance.lower() or "python" in guidance.lower()
        assert len(guidance) > 0

    def test_compile_syntax_check(
        self, adapter: PythonToolAdapter, python_project: Path
    ):
        """Compile should perform syntax check."""
        result = adapter.compile(python_project)
        # Should succeed for valid Python if uv/Python is available
        if "Neither uv nor Python found" in str(result.errors):
            pytest.skip("Neither uv nor Python found")
        assert result.success is True

    def test_compile_syntax_error(self, adapter: PythonToolAdapter, tmp_path: Path):
        """Compile should fail for invalid Python syntax."""
        (tmp_path / "bad.py").write_text("def foo( return")
        result = adapter.compile(tmp_path)
        # Should fail for syntax error
        assert result.success is False


class TestJavaScriptToolAdapter:
    """Tests for JavaScriptToolAdapter."""

    @pytest.fixture
    def adapter(self):
        return JavaScriptToolAdapter()

    @pytest.fixture
    def js_project(self, tmp_path: Path):
        """Create a minimal JavaScript project."""
        (tmp_path / "package.json").write_text("""{
  "name": "test-project",
  "version": "1.0.0",
  "scripts": {
    "test": "jest"
  }
}""")
        (tmp_path / "index.js").write_text("""
function hello() {
    return "Hello, World!";
}
module.exports = { hello };
""")
        return tmp_path

    def test_framework_name(self, adapter: JavaScriptToolAdapter):
        """Should return javascript as framework name."""
        assert adapter.framework_name == "javascript"

    def test_normalize_test_path(self, adapter: JavaScriptToolAdapter, tmp_path: Path):
        """Should normalize test paths correctly."""
        # With .test.js extension
        result = adapter.normalize_test_path("foo.test.js", tmp_path)
        assert (
            ".test.js" in str(result)
            or ".spec.js" in str(result)
            or ".js" in str(result)
        )

        # Without extension - defaults to .mjs for framework-agnostic PoC files
        result = adapter.normalize_test_path("foo", tmp_path)
        assert result.suffix == ".mjs"

    def test_get_poc_guidance(self, adapter: JavaScriptToolAdapter):
        """Should return JavaScript-specific PoC guidance."""
        guidance = adapter.get_poc_guidance()
        assert (
            "jest" in guidance.lower()
            or "javascript" in guidance.lower()
            or "node" in guidance.lower()
        )
        assert len(guidance) > 0

    def test_detect_package_manager_npm(
        self, adapter: JavaScriptToolAdapter, tmp_path: Path
    ):
        """Should detect npm as package manager."""
        (tmp_path / "package-lock.json").write_text("{}")
        manager = adapter._detect_package_manager(tmp_path)
        assert manager == "npm"

    def test_detect_package_manager_yarn(
        self, adapter: JavaScriptToolAdapter, tmp_path: Path
    ):
        """Should detect yarn as package manager."""
        (tmp_path / "yarn.lock").write_text("")
        manager = adapter._detect_package_manager(tmp_path)
        assert manager == "yarn"

    def test_detect_package_manager_pnpm(
        self, adapter: JavaScriptToolAdapter, tmp_path: Path
    ):
        """Should detect pnpm as package manager."""
        (tmp_path / "pnpm-lock.yaml").write_text("")
        manager = adapter._detect_package_manager(tmp_path)
        assert manager == "pnpm"


class TestCToolAdapter:
    """Tests for CToolAdapter."""

    @pytest.fixture
    def adapter(self):
        return CToolAdapter()

    @pytest.fixture
    def c_project(self, tmp_path: Path):
        """Create a minimal C project."""
        (tmp_path / "Makefile").write_text("""
all: main

main: main.c
\tgcc -o main main.c

test: main
\t./main

clean:
\trm -f main
""")
        (tmp_path / "main.c").write_text("""
#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}
""")
        return tmp_path

    def test_framework_name(self, adapter: CToolAdapter):
        """Should return c as framework name."""
        assert adapter.framework_name == "c"

    def test_normalize_test_path(self, adapter: CToolAdapter, tmp_path: Path):
        """Should normalize test paths correctly."""
        # With .c extension
        result = adapter.normalize_test_path("test_foo.c", tmp_path)
        assert result.suffix == ".c"

        # Without extension
        result = adapter.normalize_test_path("test_bar", tmp_path)
        assert result.suffix == ".c"

    def test_get_poc_guidance(self, adapter: CToolAdapter):
        """Should return C-specific PoC guidance."""
        guidance = adapter.get_poc_guidance()
        assert (
            "c" in guidance.lower()
            or "gcc" in guidance.lower()
            or "make" in guidance.lower()
        )
        assert len(guidance) > 0

    def test_detect_build_system_cmake(self, adapter: CToolAdapter, tmp_path: Path):
        """Should detect CMake build system."""
        (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)")
        system = adapter._detect_build_system(tmp_path)
        assert system == "cmake"

    def test_detect_build_system_make(self, adapter: CToolAdapter, tmp_path: Path):
        """Should detect Make build system."""
        (tmp_path / "Makefile").write_text("all:\n\techo hello")
        system = adapter._detect_build_system(tmp_path)
        assert system == "make"

    def test_detect_build_system_meson(self, adapter: CToolAdapter, tmp_path: Path):
        """Should detect Meson build system."""
        (tmp_path / "meson.build").write_text("project('test', 'c')")
        system = adapter._detect_build_system(tmp_path)
        assert system == "meson"


class TestToolAdapterPoCGuidance:
    """Tests for PoC guidance across all adapters."""

    def test_all_adapters_have_poc_guidance(self):
        """All adapters should provide non-empty PoC guidance."""
        for framework in ["foundry", "python", "javascript", "c", "cargo", "cmake"]:
            adapter = get_tool_adapter(framework)
            guidance = adapter.get_poc_guidance()
            assert isinstance(guidance, str)
            assert len(guidance) > 0, f"{framework} adapter should provide guidance"

    def test_poc_guidance_contains_framework_info(self):
        """PoC guidance should contain framework-specific information."""
        python_adapter = get_tool_adapter("python")
        js_adapter = get_tool_adapter("javascript")
        c_adapter = get_tool_adapter("c")
        foundry_adapter = get_tool_adapter("foundry")
        cargo_adapter = get_tool_adapter("cargo")
        cmake_adapter = get_tool_adapter("cmake")

        # Python guidance should mention pytest or python
        python_guidance = python_adapter.get_poc_guidance().lower()
        assert (
            "python" in python_guidance
            or "pytest" in python_guidance
            or "test" in python_guidance
        )

        # JavaScript guidance should mention npm, node, or test framework
        js_guidance = js_adapter.get_poc_guidance().lower()
        assert (
            "javascript" in js_guidance
            or "node" in js_guidance
            or "jest" in js_guidance
            or "test" in js_guidance
        )

        # C guidance should mention compilation or testing
        c_guidance = c_adapter.get_poc_guidance().lower()
        assert (
            "c" in c_guidance
            or "gcc" in c_guidance
            or "make" in c_guidance
            or "test" in c_guidance
        )

        # Foundry guidance should mention solidity or forge
        foundry_guidance = foundry_adapter.get_poc_guidance().lower()
        assert (
            "solidity" in foundry_guidance
            or "forge" in foundry_guidance
            or "foundry" in foundry_guidance
        )

        # Cargo guidance should mention rust or cargo
        cargo_guidance = cargo_adapter.get_poc_guidance().lower()
        assert (
            "rust" in cargo_guidance
            or "cargo" in cargo_guidance
            or "test" in cargo_guidance
        )

        # CMake guidance should mention cmake or ctest
        cmake_guidance = cmake_adapter.get_poc_guidance().lower()
        assert (
            "cmake" in cmake_guidance
            or "ctest" in cmake_guidance
            or "test" in cmake_guidance
        )
