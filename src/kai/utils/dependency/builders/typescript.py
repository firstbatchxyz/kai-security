"""
TypeScript dependency graph builder using tree-sitter.

Extends the JavaScript builder to handle TypeScript-specific syntax
including type annotations, interfaces, enums, and .ts/.tsx files.
"""

from __future__ import annotations

from typing import List

from .javascript import JavaScriptBuilder


class TypeScriptBuilder(JavaScriptBuilder):
    """
    Tree-sitter based builder for TypeScript projects.

    Extends JavaScriptBuilder to:
    - Use tree-sitter-typescript parser
    - Handle .ts, .tsx, .mts, .cts file extensions
    - Parse TypeScript-specific constructs
    """

    @property
    def language(self) -> str:
        return "typescript"

    @property
    def file_extensions(self) -> List[str]:
        # Include both TypeScript and JavaScript extensions for mixed projects
        # TypeScript parser can handle JavaScript syntax
        return [".ts", ".tsx", ".mts", ".cts", ".js", ".mjs", ".cjs"]

    def _create_parser(self):
        """Create a tree-sitter parser for TypeScript."""
        try:
            from tree_sitter_language_pack import get_parser

            return get_parser("typescript")
        except ImportError:
            pass

        try:
            import tree_sitter
            import tree_sitter_typescript

            parser = tree_sitter.Parser()
            parser.language = tree_sitter.Language(
                tree_sitter_typescript.language_typescript()
            )
            return parser
        except ImportError:
            pass

        raise ImportError(
            "No tree-sitter parser available for TypeScript. "
            "Install tree-sitter-language-pack or tree-sitter-typescript"
        )
