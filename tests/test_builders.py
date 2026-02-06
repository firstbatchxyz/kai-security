"""
Tests for tree-sitter builders (Python, JavaScript, C).

Tests the Builder interface implementations for each language.
"""

from pathlib import Path
import pytest

from kai.utils.dependency.builders import (
    get_builder,
    PythonBuilder,
    JavaScriptBuilder,
    CBuilder,
)
from kai.utils.dependency.models import NodeKind


# Helper functions to check tree-sitter availability (must be defined before use in decorators)
def _has_tree_sitter_python() -> bool:
    """Check if tree-sitter-python is installed."""
    import importlib.util

    if importlib.util.find_spec("tree_sitter_python") is not None:
        return True
    if importlib.util.find_spec("tree_sitter_languages") is not None:
        return True
    return False


def _has_tree_sitter_javascript() -> bool:
    """Check if tree-sitter-javascript is installed."""
    import importlib.util

    if importlib.util.find_spec("tree_sitter_javascript") is not None:
        return True
    if importlib.util.find_spec("tree_sitter_languages") is not None:
        return True
    return False


def _has_tree_sitter_c() -> bool:
    """Check if tree-sitter-c is installed."""
    import importlib.util

    if importlib.util.find_spec("tree_sitter_c") is not None:
        return True
    if importlib.util.find_spec("tree_sitter_languages") is not None:
        return True
    return False


class TestBuilderRegistry:
    """Tests for the builder registry."""

    def test_get_builder_python(self):
        """Should return PythonBuilder for python."""
        builder = get_builder("python")
        assert isinstance(builder, PythonBuilder)

    def test_get_builder_python_alias(self):
        """Should return PythonBuilder for py alias."""
        builder = get_builder("py")
        assert isinstance(builder, PythonBuilder)

    def test_get_builder_javascript(self):
        """Should return JavaScriptBuilder for javascript."""
        builder = get_builder("javascript")
        assert isinstance(builder, JavaScriptBuilder)

    def test_get_builder_javascript_alias(self):
        """Should return JavaScriptBuilder for js alias."""
        builder = get_builder("js")
        assert isinstance(builder, JavaScriptBuilder)

    def test_get_builder_c(self):
        """Should return CBuilder for c."""
        builder = get_builder("c")
        assert isinstance(builder, CBuilder)

    def test_get_builder_invalid(self):
        """Should raise ValueError for unknown builder."""
        with pytest.raises(ValueError, match="No builder for language"):
            get_builder("unknown_language")


class TestPythonBuilder:
    """Tests for PythonBuilder."""

    @pytest.fixture
    def builder(self):
        return PythonBuilder()

    @pytest.fixture
    def python_project(self, tmp_path: Path):
        """Create a minimal Python project."""
        (tmp_path / "app.py").write_text("""
class UserService:
    def __init__(self):
        self.users = []

    def add_user(self, name: str) -> None:
        self.users.append(name)

    def get_user(self, index: int) -> str:
        return self.users[index]

@decorator
def helper_function():
    pass

global_var = 42
""")
        (tmp_path / "utils.py").write_text("""
def format_name(name: str) -> str:
    return name.strip().title()

class Config:
    DEBUG = True
""")
        return tmp_path

    def test_language_property(self, builder: PythonBuilder):
        """Should return python as language."""
        assert builder.language == "python"

    def test_file_extensions(self, builder: PythonBuilder):
        """Should return .py extension."""
        assert ".py" in builder.file_extensions

    @pytest.mark.skipif(
        not _has_tree_sitter_python(), reason="tree-sitter-python not installed"
    )
    def test_build_extracts_classes(self, builder: PythonBuilder, python_project: Path):
        """Should extract class definitions."""
        graph = builder.build(python_project)

        # Find containers (classes)
        containers = list(graph.nodes(NodeKind.CONTAINER))
        container_names = [graph.node(c).name for c in containers]

        assert "UserService" in container_names
        assert "Config" in container_names

    @pytest.mark.skipif(
        not _has_tree_sitter_python(), reason="tree-sitter-python not installed"
    )
    def test_build_extracts_functions(
        self, builder: PythonBuilder, python_project: Path
    ):
        """Should extract function definitions."""
        graph = builder.build(python_project)

        # Find units (functions)
        units = list(graph.nodes(NodeKind.UNIT))
        unit_names = [graph.node(u).name for u in units]

        assert "helper_function" in unit_names or "format_name" in unit_names

    @pytest.mark.skipif(
        not _has_tree_sitter_python(), reason="tree-sitter-python not installed"
    )
    def test_build_extracts_methods(self, builder: PythonBuilder, python_project: Path):
        """Should extract method definitions."""
        graph = builder.build(python_project)

        # Find units (methods)
        units = list(graph.nodes(NodeKind.UNIT))
        unit_names = [graph.node(u).name for u in units]

        # Should have methods from UserService
        assert "add_user" in unit_names or "get_user" in unit_names


class TestJavaScriptBuilder:
    """Tests for JavaScriptBuilder."""

    @pytest.fixture
    def builder(self):
        return JavaScriptBuilder()

    @pytest.fixture
    def js_project(self, tmp_path: Path):
        """Create a minimal JavaScript project."""
        (tmp_path / "app.js").write_text("""
class UserService {
    constructor() {
        this.users = [];
    }

    addUser(name) {
        this.users.push(name);
    }

    getUser(index) {
        return this.users[index];
    }
}

function helperFunction() {
    return 42;
}

const arrowFunc = (x) => x * 2;

module.exports = { UserService, helperFunction };
""")
        (tmp_path / "utils.js").write_text("""
export function formatName(name) {
    return name.trim();
}

export class Config {
    static DEBUG = true;
}
""")
        return tmp_path

    def test_language_property(self, builder: JavaScriptBuilder):
        """Should return javascript as language."""
        assert builder.language == "javascript"

    def test_file_extensions(self, builder: JavaScriptBuilder):
        """Should return .js and .ts extensions."""
        assert ".js" in builder.file_extensions

    @pytest.mark.skipif(
        not _has_tree_sitter_javascript(), reason="tree-sitter-javascript not installed"
    )
    def test_build_extracts_classes(self, builder: JavaScriptBuilder, js_project: Path):
        """Should extract class definitions."""
        graph = builder.build(js_project)

        # Find containers (classes)
        containers = list(graph.nodes(NodeKind.CONTAINER))
        container_names = [graph.node(c).name for c in containers]

        assert "UserService" in container_names or "Config" in container_names

    @pytest.mark.skipif(
        not _has_tree_sitter_javascript(), reason="tree-sitter-javascript not installed"
    )
    def test_build_extracts_functions(
        self, builder: JavaScriptBuilder, js_project: Path
    ):
        """Should extract function definitions."""
        graph = builder.build(js_project)

        # Find units (functions)
        units = list(graph.nodes(NodeKind.UNIT))
        unit_names = [graph.node(u).name for u in units]

        assert "helperFunction" in unit_names or "formatName" in unit_names


class TestCBuilder:
    """Tests for CBuilder."""

    @pytest.fixture
    def builder(self):
        return CBuilder()

    @pytest.fixture
    def c_project(self, tmp_path: Path):
        """Create a minimal C project."""
        (tmp_path / "main.c").write_text("""
#include <stdio.h>

struct User {
    char name[100];
    int age;
};

enum Status {
    ACTIVE,
    INACTIVE,
    PENDING
};

int global_counter = 0;

void add_user(struct User* user) {
    global_counter++;
    printf("Added user: %s\\n", user->name);
}

int get_counter() {
    return global_counter;
}

int main() {
    struct User user = {"Alice", 30};
    add_user(&user);
    return 0;
}
""")
        (tmp_path / "utils.h").write_text("""
#ifndef UTILS_H
#define UTILS_H

typedef struct {
    int x;
    int y;
} Point;

int add(int a, int b);

#endif
""")
        (tmp_path / "utils.c").write_text("""
#include "utils.h"

int add(int a, int b) {
    return a + b;
}

static int helper(int x) {
    return x * 2;
}
""")
        return tmp_path

    def test_language_property(self, builder: CBuilder):
        """Should return c as language."""
        assert builder.language == "c"

    def test_file_extensions(self, builder: CBuilder):
        """Should return .c and .h extensions."""
        assert ".c" in builder.file_extensions
        assert ".h" in builder.file_extensions

    @pytest.mark.skipif(not _has_tree_sitter_c(), reason="tree-sitter-c not installed")
    def test_build_extracts_functions(self, builder: CBuilder, c_project: Path):
        """Should extract function definitions."""
        graph = builder.build(c_project)

        # Find units (functions)
        units = list(graph.nodes(NodeKind.UNIT))
        unit_names = [graph.node(u).name for u in units]

        assert "main" in unit_names or "add_user" in unit_names or "add" in unit_names

    @pytest.mark.skipif(not _has_tree_sitter_c(), reason="tree-sitter-c not installed")
    def test_build_extracts_structs(self, builder: CBuilder, c_project: Path):
        """Should extract struct definitions."""
        graph = builder.build(c_project)

        # Find containers (structs)
        containers = list(graph.nodes(NodeKind.CONTAINER))
        container_names = [graph.node(c).name for c in containers]

        assert "User" in container_names or "Point" in container_names

    @pytest.mark.skipif(not _has_tree_sitter_c(), reason="tree-sitter-c not installed")
    def test_build_extracts_enums(self, builder: CBuilder, c_project: Path):
        """Should extract enum definitions."""
        graph = builder.build(c_project)

        # Find type_defs (enums)
        type_defs = list(graph.nodes(NodeKind.TYPE_DEF))
        type_def_names = [graph.node(t).name for t in type_defs]

        assert "Status" in type_def_names

    @pytest.mark.skipif(not _has_tree_sitter_c(), reason="tree-sitter-c not installed")
    def test_build_extracts_global_variables(self, builder: CBuilder, c_project: Path):
        """Should extract global variable definitions."""
        graph = builder.build(c_project)

        # Find variables
        variables = list(graph.nodes(NodeKind.VARIABLE))
        var_names = [graph.node(v).name for v in variables]

        assert "global_counter" in var_names


class TestBuilderInterface:
    """Tests for Builder interface compliance."""

    @pytest.mark.parametrize("language", ["python", "javascript", "c"])
    def test_builder_implements_interface(self, language: str):
        """All builders should implement the BaseBuilder interface."""
        builder = get_builder(language)

        # Check required properties exist
        assert hasattr(builder, "language")
        assert hasattr(builder, "file_extensions")

        # Check required methods exist
        assert hasattr(builder, "build")
        assert callable(builder.build)

    @pytest.mark.parametrize("language", ["python", "javascript", "c"])
    def test_builder_returns_correct_types(self, language: str, tmp_path: Path):
        """All builders should return correct types."""
        builder = get_builder(language)

        # language should return string
        assert isinstance(builder.language, str)
        assert len(builder.language) > 0

        # file_extensions should return list of strings
        assert isinstance(builder.file_extensions, list)
        assert all(isinstance(ext, str) for ext in builder.file_extensions)
        assert all(ext.startswith(".") for ext in builder.file_extensions)
