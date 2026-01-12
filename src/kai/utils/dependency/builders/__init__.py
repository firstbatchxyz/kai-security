"""
Language-specific graph builders for Kai.

Each builder maps language constructs to the generic NodeKind/EdgeKind model:

    NodeKind Mapping:
        FILE       - Source files (all languages)
        CONTAINER  - Contract (Sol), Module (Rust), Class (Py)
        UNIT       - Function (Sol), Method (Py), Instruction (Anchor)
        INTERFACE  - Modifier (Sol), Attribute (Rust), Decorator (Py)
        VARIABLE   - StateVar (Sol), Account (Rust), Global (Py)
        TYPE_DEF   - Struct, Enum, Typedef (all languages)
        EVENT      - Events (Sol), Logs (all languages)
        EXTERNAL   - Unresolved external references

    EdgeKind Mapping:
        DEFINES    - Container defines children
        IMPORTS    - File imports file
        INHERITS   - Container extends container
        CALLS      - Unit calls unit
        ACCEPTS    - Unit uses interface/guard
        READS      - Unit reads variable
        WRITES     - Unit writes variable
        EMITS      - Unit emits event
        USES_TYPE  - Unit references type definition

Usage:
    from kai.utils.dependency.builders import SolidityBuilder

    builder = SolidityBuilder()
    graph = builder.build("/path/to/project")
"""

from pathlib import Path

from .base import BaseBuilder
from .solidity import SolidityBuilder
from .treesitter_base import TreeSitterBuilder
from .python import PythonBuilder
from .javascript import JavaScriptBuilder
from .typescript import TypeScriptBuilder
from .c import CBuilder
from ..graph import DependencyGraph

__all__ = [
    "BaseBuilder",
    "SolidityBuilder",
    "TreeSitterBuilder",
    "PythonBuilder",
    "JavaScriptBuilder",
    "TypeScriptBuilder",
    "CBuilder",
    "build_from_slither",
    "get_builder",
]


def get_builder(language: str) -> BaseBuilder:
    """Get a builder for the specified language."""
    builders = {
        "solidity": SolidityBuilder,
        "sol": SolidityBuilder,
        # BountyBench language builders (tree-sitter based)
        "python": PythonBuilder,
        "py": PythonBuilder,
        "javascript": JavaScriptBuilder,
        "js": JavaScriptBuilder,
        "typescript": TypeScriptBuilder,
        "ts": TypeScriptBuilder,
        "c": CBuilder,
    }

    builder_cls = builders.get(language.lower())
    if builder_cls is None:
        raise ValueError(
            f"No builder for language: {language}. Available: {list(builders.keys())}"
        )

    return builder_cls()


def build_from_slither(project_root: str | Path, **kwargs) -> DependencyGraph:
    """
    Convenience helper to build a dependency graph for a Solidity project using Slither.
    Falls back to an empty graph if Slither is unavailable.
    """
    try:
        builder = SolidityBuilder()
        return builder.build(project_root, **kwargs)
    except ImportError:
        # Slither not installed; return an empty graph so callers can proceed gracefully
        return DependencyGraph(Path(project_root).resolve())
