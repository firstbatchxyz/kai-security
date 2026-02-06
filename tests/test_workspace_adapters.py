"""
Tests for workspace adapters (Python, JavaScript, C).

Tests the WorkspaceAdapter interface implementations for each language.
"""

from pathlib import Path

import pytest

from kai.schemas import MasterContext, WorkspacePreset
from kai.utils.workspace import (
    get_workspace_adapter,
    get_supported_frameworks,
    PythonWorkspaceAdapter,
    JavaScriptWorkspaceAdapter,
    CWorkspaceAdapter,
    FoundryWorkspaceAdapter,
)


class TestWorkspaceAdapterRegistry:
    """Tests for the workspace adapter registry."""

    def test_get_supported_frameworks(self):
        """Should return list of supported frameworks."""
        supported = get_supported_frameworks()
        assert "foundry" in supported
        assert "python" in supported
        assert "javascript" in supported
        assert "c" in supported

    def test_get_adapter_foundry(self):
        """Should return FoundryWorkspaceAdapter for foundry."""
        adapter = get_workspace_adapter("foundry")
        assert isinstance(adapter, FoundryWorkspaceAdapter)

    def test_get_adapter_python(self):
        """Should return PythonWorkspaceAdapter for python."""
        adapter = get_workspace_adapter("python")
        assert isinstance(adapter, PythonWorkspaceAdapter)

    def test_get_adapter_python_alias(self):
        """Should return PythonWorkspaceAdapter for py alias."""
        adapter = get_workspace_adapter("py")
        assert isinstance(adapter, PythonWorkspaceAdapter)

    def test_get_adapter_javascript(self):
        """Should return JavaScriptWorkspaceAdapter for javascript."""
        adapter = get_workspace_adapter("javascript")
        assert isinstance(adapter, JavaScriptWorkspaceAdapter)

    def test_get_adapter_javascript_aliases(self):
        """Should return JavaScriptWorkspaceAdapter for js/node aliases."""
        adapter_js = get_workspace_adapter("js")
        adapter_node = get_workspace_adapter("node")
        assert isinstance(adapter_js, JavaScriptWorkspaceAdapter)
        assert isinstance(adapter_node, JavaScriptWorkspaceAdapter)

    def test_get_adapter_c(self):
        """Should return CWorkspaceAdapter for c."""
        adapter = get_workspace_adapter("c")
        assert isinstance(adapter, CWorkspaceAdapter)

    def test_get_adapter_invalid(self):
        """Should raise ValueError for unknown adapter."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            get_workspace_adapter("unknown_framework")


class TestPythonWorkspaceAdapter:
    """Tests for PythonWorkspaceAdapter."""

    @pytest.fixture
    def adapter(self):
        return PythonWorkspaceAdapter()

    @pytest.fixture
    def python_project(self, tmp_path: Path):
        """Create a minimal Python project."""
        master = tmp_path / "master"
        master.mkdir()

        # Create pyproject.toml
        (master / "pyproject.toml").write_text("""
[project]
name = "test-project"
version = "0.1.0"
dependencies = ["requests"]
""")
        # Create src directory
        src = master / "src"
        src.mkdir()
        (src / "__init__.py").write_text("")
        (src / "app.py").write_text("""
def hello():
    return "Hello, World!"
""")
        # Create tests directory
        tests = master / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("")
        (tests / "test_app.py").write_text("""
def test_hello():
    from src.app import hello
    assert hello() == "Hello, World!"
""")
        return master

    def test_framework_name(self, adapter: PythonWorkspaceAdapter):
        """Should return python as framework name."""
        assert adapter.framework_name == "python"

    def test_infer_src_path(
        self, adapter: PythonWorkspaceAdapter, python_project: Path
    ):
        """Should infer src path correctly."""
        src_path = adapter.infer_src_path(python_project)
        assert src_path.name == "src"

    def test_infer_src_path_no_src(
        self, adapter: PythonWorkspaceAdapter, tmp_path: Path
    ):
        """Should return root if no src directory."""
        (tmp_path / "app.py").write_text("x = 1")
        src_path = adapter.infer_src_path(tmp_path)
        assert src_path == tmp_path

    def test_provision_lightweight(
        self, adapter: PythonWorkspaceAdapter, python_project: Path, tmp_path: Path
    ):
        """Should provision lightweight workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        mc = MasterContext(
            root_path=str(python_project),
            compile_success=True,
            src_path=str(python_project / "src"),
        )

        result = adapter.provision_lightweight(workspace, python_project, mc)
        assert Path(result).exists()
        # Should create tests directory
        assert (workspace / "tests").exists()

    def test_provision_full_sandbox(
        self, adapter: PythonWorkspaceAdapter, python_project: Path, tmp_path: Path
    ):
        """Should provision full sandbox workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        mc = MasterContext(
            root_path=str(python_project),
            compile_success=True,
            src_path=str(python_project / "src"),
        )

        result = adapter.provision_full(
            workspace, python_project, mc, WorkspacePreset.SANDBOX
        )
        assert Path(result).exists()
        # Should copy source files
        assert (
            (workspace / "src").exists()
            or (workspace / "app.py").exists()
            or (workspace / "pyproject.toml").exists()
        )


class TestJavaScriptWorkspaceAdapter:
    """Tests for JavaScriptWorkspaceAdapter."""

    @pytest.fixture
    def adapter(self):
        return JavaScriptWorkspaceAdapter()

    @pytest.fixture
    def js_project(self, tmp_path: Path):
        """Create a minimal JavaScript project."""
        master = tmp_path / "master"
        master.mkdir()

        (master / "package.json").write_text("""{
  "name": "test-project",
  "version": "1.0.0",
  "scripts": {
    "test": "jest"
  },
  "dependencies": {}
}""")
        (master / "index.js").write_text("""
function hello() {
    return "Hello, World!";
}
module.exports = { hello };
""")
        # Create src directory
        src = master / "src"
        src.mkdir()
        (src / "app.js").write_text("""
module.exports.foo = function() { return 42; };
""")
        return master

    def test_framework_name(self, adapter: JavaScriptWorkspaceAdapter):
        """Should return javascript as framework name."""
        assert adapter.framework_name == "javascript"

    def test_infer_src_path(
        self, adapter: JavaScriptWorkspaceAdapter, js_project: Path
    ):
        """Should infer src path correctly."""
        src_path = adapter.infer_src_path(js_project)
        assert src_path.name == "src"

    def test_detect_package_manager_npm(
        self, adapter: JavaScriptWorkspaceAdapter, tmp_path: Path
    ):
        """Should detect npm as package manager."""
        (tmp_path / "package-lock.json").write_text("{}")
        manager = adapter._detect_package_manager(tmp_path)
        assert manager == "npm"

    def test_detect_package_manager_yarn(
        self, adapter: JavaScriptWorkspaceAdapter, tmp_path: Path
    ):
        """Should detect yarn as package manager."""
        (tmp_path / "yarn.lock").write_text("")
        manager = adapter._detect_package_manager(tmp_path)
        assert manager == "yarn"

    def test_provision_lightweight(
        self, adapter: JavaScriptWorkspaceAdapter, js_project: Path, tmp_path: Path
    ):
        """Should provision lightweight workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        mc = MasterContext(
            root_path=str(js_project),
            compile_success=True,
            src_path=str(js_project / "src"),
        )

        result = adapter.provision_lightweight(workspace, js_project, mc)
        assert Path(result).exists()

    def test_provision_full_sandbox(
        self, adapter: JavaScriptWorkspaceAdapter, js_project: Path, tmp_path: Path
    ):
        """Should provision full sandbox workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        mc = MasterContext(
            root_path=str(js_project),
            compile_success=True,
            src_path=str(js_project / "src"),
        )

        result = adapter.provision_full(
            workspace, js_project, mc, WorkspacePreset.SANDBOX
        )
        assert Path(result).exists()


class TestCWorkspaceAdapter:
    """Tests for CWorkspaceAdapter."""

    @pytest.fixture
    def adapter(self):
        return CWorkspaceAdapter()

    @pytest.fixture
    def c_project(self, tmp_path: Path):
        """Create a minimal C project."""
        master = tmp_path / "master"
        master.mkdir()

        (master / "Makefile").write_text("""
all: main

main: main.c
\tgcc -o main main.c

test: main
\t./main

clean:
\trm -f main
""")
        (master / "main.c").write_text("""
#include <stdio.h>

int main() {
    printf("Hello, World!\\n");
    return 0;
}
""")
        # Create src directory
        src = master / "src"
        src.mkdir()
        (src / "lib.c").write_text("""
int add(int a, int b) { return a + b; }
""")
        (src / "lib.h").write_text("""
int add(int a, int b);
""")
        return master

    def test_framework_name(self, adapter: CWorkspaceAdapter):
        """Should return c as framework name."""
        assert adapter.framework_name == "c"

    def test_infer_src_path(self, adapter: CWorkspaceAdapter, c_project: Path):
        """Should infer src path correctly."""
        src_path = adapter.infer_src_path(c_project)
        assert src_path.name == "src"

    def test_detect_build_system_cmake(
        self, adapter: CWorkspaceAdapter, tmp_path: Path
    ):
        """Should detect CMake build system."""
        (tmp_path / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.10)")
        system = adapter._detect_build_system(tmp_path)
        assert system == "cmake"

    def test_detect_build_system_make(self, adapter: CWorkspaceAdapter, tmp_path: Path):
        """Should detect Make build system."""
        (tmp_path / "Makefile").write_text("all:\n\techo hello")
        system = adapter._detect_build_system(tmp_path)
        assert system == "make"

    def test_provision_lightweight(
        self, adapter: CWorkspaceAdapter, c_project: Path, tmp_path: Path
    ):
        """Should provision lightweight workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        mc = MasterContext(
            root_path=str(c_project),
            compile_success=True,
            src_path=str(c_project / "src"),
        )

        result = adapter.provision_lightweight(workspace, c_project, mc)
        assert Path(result).exists()

    def test_provision_full_sandbox(
        self, adapter: CWorkspaceAdapter, c_project: Path, tmp_path: Path
    ):
        """Should provision full sandbox workspace."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        mc = MasterContext(
            root_path=str(c_project),
            compile_success=True,
            src_path=str(c_project / "src"),
        )

        result = adapter.provision_full(
            workspace, c_project, mc, WorkspacePreset.SANDBOX
        )
        assert Path(result).exists()


class TestWorkspaceAdapterRuntimePaths:
    """Tests for runtime writable paths across adapters."""

    def test_foundry_runtime_paths(self, tmp_path: Path):
        """Foundry should return out/cache paths."""
        adapter = get_workspace_adapter("foundry")
        mc = MasterContext(root_path=str(tmp_path), compile_success=True)
        paths = adapter.get_runtime_writable_paths(tmp_path, mc)
        assert isinstance(paths, list)
        # Should include out and cache directories
        path_names = [p.name for p in paths]
        assert "out" in path_names
        assert "cache" in path_names

    def test_python_runtime_paths(self, tmp_path: Path):
        """Python should return venv/__pycache__ paths."""
        adapter = get_workspace_adapter("python")
        mc = MasterContext(root_path=str(tmp_path), compile_success=True)
        paths = adapter.get_runtime_writable_paths(tmp_path, mc)
        assert isinstance(paths, list)
        assert len(paths) > 0
        # Should include common Python writable directories
        path_names = [p.name for p in paths]
        assert ".venv" in path_names
        assert "__pycache__" in path_names
        assert ".pytest_cache" in path_names

    def test_javascript_runtime_paths(self, tmp_path: Path):
        """JavaScript should return node_modules paths."""
        adapter = get_workspace_adapter("javascript")
        mc = MasterContext(root_path=str(tmp_path), compile_success=True)
        paths = adapter.get_runtime_writable_paths(tmp_path, mc)
        assert isinstance(paths, list)
        assert len(paths) > 0
        # Should include node_modules and build directories
        path_names = [p.name for p in paths]
        assert "node_modules" in path_names
        assert "dist" in path_names or "build" in path_names

    def test_c_runtime_paths(self, tmp_path: Path):
        """C should return build paths."""
        adapter = get_workspace_adapter("c")
        mc = MasterContext(root_path=str(tmp_path), compile_success=True)
        paths = adapter.get_runtime_writable_paths(tmp_path, mc)
        assert isinstance(paths, list)
        assert len(paths) > 0
        # Should include build directories
        path_names = [p.name for p in paths]
        assert "build" in path_names or "out" in path_names
