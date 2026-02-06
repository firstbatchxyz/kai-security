"""
Tree-sitter based builder base class.

Provides common functionality for building dependency graphs from
source code using tree-sitter parsers.
"""

from __future__ import annotations
from abc import abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from .base import BaseBuilder
from ..models import Node, SourceSpan, NodeKind, EdgeKind
from ..graph import DependencyGraph

if TYPE_CHECKING:
    pass


class TreeSitterBuilder(BaseBuilder):
    """
    Abstract base class for tree-sitter based builders.

    Provides common functionality for:
    - Lazy-loading tree-sitter parsers
    - Walking directory trees to find source files
    - Building dependency graphs from parsed ASTs
    """

    def __init__(self, skip_patterns: Optional[List[str]] = None):
        """
        Initialize the tree-sitter builder.

        Args:
            skip_patterns: Directory patterns to skip when walking
        """
        self._parser = None
        self._skip_patterns = skip_patterns or [
            "test",
            "tests",
            "__test__",
            "__tests__",
            "node_modules",
            "venv",
            ".venv",
            "__pycache__",
            ".git",
            "dist",
            "build",
            "vendor",
            "third_party",
        ]

    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language name (e.g., 'python', 'javascript', 'c')."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Return list of file extensions (e.g., ['.py'], ['.js', '.mjs'])."""
        pass

    @abstractmethod
    def _extract_from_tree(
        self, tree: Any, file_path: Path, source_bytes: bytes
    ) -> Tuple[List[Node], List[Tuple[str, str, EdgeKind]]]:
        """
        Extract nodes and edges from a parsed syntax tree.

        Args:
            tree: tree-sitter Tree object
            file_path: Path to the source file
            source_bytes: Raw file content as bytes

        Returns:
            Tuple of (nodes, edges) where edges are (from_id, to_id, kind)
        """
        pass

    @property
    def parser(self):
        """Lazy-load the tree-sitter parser."""
        if self._parser is None:
            self._parser = self._create_parser()
        return self._parser

    def _create_parser(self):
        """
        Create a tree-sitter parser for this language.

        Tries tree-sitter-language-pack first, then individual packages.
        """
        try:
            # Try tree-sitter-language-pack (preferred)
            from tree_sitter_language_pack import get_parser

            # ignore type here, as `language` is dynamic but get_parser expects a literal
            return get_parser(self.language)  # type: ignore
        except ImportError:
            pass

        try:
            # Try tree-sitter with individual language package
            import tree_sitter

            parser = tree_sitter.Parser()

            # Try to get language from individual packages
            lang_module = None
            if self.language == "python":
                try:
                    import tree_sitter_python

                    lang_module = tree_sitter_python
                except ImportError:
                    pass
            elif self.language == "javascript":
                try:
                    import tree_sitter_javascript

                    lang_module = tree_sitter_javascript
                except ImportError:
                    pass
            elif self.language == "c":
                try:
                    import tree_sitter_c

                    lang_module = tree_sitter_c
                except ImportError:
                    pass
            elif self.language == "rust":
                try:
                    import tree_sitter_rust

                    lang_module = tree_sitter_rust
                except ImportError:
                    pass

            if lang_module:
                parser.language = tree_sitter.Language(lang_module.language())
                return parser

        except ImportError:
            pass

        raise ImportError(
            f"No tree-sitter parser available for {self.language}. "
            f"Install tree-sitter-language-pack or tree-sitter-{self.language}"
        )

    def build(self, source: Any, **kwargs) -> DependencyGraph:
        """
        Build a dependency graph from the project.

        Args:
            source: Path to project root directory
            **kwargs: Builder-specific options

        Returns:
            Populated DependencyGraph
        """
        root = Path(source).resolve()
        graph = DependencyGraph(root)

        # Find all source files
        source_files = self._find_source_files(root)

        # Parse each file and extract nodes/edges
        for file_path in source_files:
            try:
                nodes, edges = self._parse_file(file_path)

                # Add file node
                file_id = str(file_path.relative_to(root))
                file_node = Node(
                    id=file_id,
                    kind=NodeKind.FILE,
                    name=file_path.name,
                    span=SourceSpan(file=file_id, start_line=1, end_line=1),
                    meta={"path": str(file_path)},
                )
                graph.add_node(file_node)

                # Add extracted nodes
                for node in nodes:
                    graph.add_node(node)
                    # Add DEFINES edge from file to top-level containers
                    if node.kind == NodeKind.CONTAINER and node.parent_id is None:
                        graph.add_edge(file_id, node.id, EdgeKind.DEFINES)

                # Add extracted edges
                for from_id, to_id, kind in edges:
                    graph.add_edge(from_id, to_id, kind)

            except Exception as e:
                # Log error but continue with other files
                import logging

                logging.debug(f"Failed to parse {file_path}: {e}")

        return graph

    def _find_source_files(self, root: Path) -> List[Path]:
        """Find all source files in the project."""
        files = []

        for ext in self.file_extensions:
            for file_path in root.rglob(f"*{ext}"):
                # Skip files in excluded directories
                if self._should_skip(file_path):
                    continue
                files.append(file_path)

        return files

    def _should_skip(self, file_path: Path) -> bool:
        """Check if a file should be skipped."""
        parts = file_path.parts
        for pattern in self._skip_patterns:
            if pattern in parts:
                return True
        return False

    def _parse_file(
        self, file_path: Path
    ) -> Tuple[List[Node], List[Tuple[str, str, EdgeKind]]]:
        """Parse a single file and extract nodes/edges."""
        try:
            source_bytes = file_path.read_bytes()
            tree = self.parser.parse(source_bytes)
            return self._extract_from_tree(tree, file_path, source_bytes)
        except Exception as e:
            # Return empty on failure
            import logging

            logging.debug(f"Parse error for {file_path}: {e}")
            return [], []

    def extract_span(self, obj: Any) -> Optional[SourceSpan]:
        """
        Extract source span from a tree-sitter node.

        Args:
            obj: tree-sitter Node object with start_point and end_point

        Returns:
            SourceSpan or None
        """
        if obj is None:
            return None

        try:
            start = obj.start_point
            end = obj.end_point
            return SourceSpan(
                file="",  # Caller should fill in
                start_line=start[0] + 1,  # tree-sitter uses 0-indexed lines
                end_line=end[0] + 1,
                start_col=start[1],
                end_col=end[1],
            )
        except Exception:
            return None

    def _get_node_text(self, node: Any, source_bytes: bytes) -> str:
        """Get the text content of a tree-sitter node."""
        try:
            return source_bytes[node.start_byte : node.end_byte].decode(
                "utf-8", errors="replace"
            )
        except Exception:
            return ""

    def _find_children_by_type(self, node: Any, type_name: str) -> List[Any]:
        """Find all direct children of a node with a given type."""
        return [child for child in node.children if child.type == type_name]

    def _find_child_by_type(self, node: Any, type_name: str) -> Optional[Any]:
        """Find first direct child of a node with a given type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _find_descendants_by_type(self, node: Any, type_name: str) -> List[Any]:
        """Find all descendants of a node with a given type (recursive)."""
        results = []
        for child in node.children:
            if child.type == type_name:
                results.append(child)
            results.extend(self._find_descendants_by_type(child, type_name))
        return results
