"""
Python dependency graph builder using tree-sitter.

Extracts:
- Classes (CONTAINER)
- Functions and methods (UNIT)
- Decorators (INTERFACE)
- Global variables and class attributes (VARIABLE)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .treesitter_base import TreeSitterBuilder
from ..models import Node, SourceSpan, NodeKind, EdgeKind


class PythonBuilder(TreeSitterBuilder):
    """
    Tree-sitter based builder for Python projects.

    Maps Python constructs to NodeKind:
    - class -> CONTAINER
    - function/async_function -> UNIT
    - decorated_definition -> extracts decorator as INTERFACE
    - assignment (module-level) -> VARIABLE
    """

    @property
    def language(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> List[str]:
        return [".py"]

    def _extract_from_tree(
        self, tree: Any, file_path: Path, source_bytes: bytes
    ) -> Tuple[List[Node], List[Tuple[str, str, EdgeKind]]]:
        """
        Extract Python nodes and edges from the AST.
        """
        nodes: List[Node] = []
        edges: List[Tuple[str, str, EdgeKind]] = []

        file_id = str(file_path)
        root = tree.root_node

        # Track current container for nesting
        self._extract_module_level(root, file_path, file_id, source_bytes, nodes, edges)

        return nodes, edges

    def _extract_module_level(
        self,
        root: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
        parent_id: Optional[str] = None,
    ) -> None:
        """Extract module-level constructs."""
        for child in root.children:
            if child.type == "class_definition":
                self._extract_class(
                    child, file_path, file_id, source_bytes, nodes, edges, parent_id
                )
            elif child.type in ("function_definition", "async_function_definition"):
                self._extract_function(
                    child, file_path, file_id, source_bytes, nodes, edges, parent_id
                )
            elif child.type == "decorated_definition":
                self._extract_decorated(
                    child, file_path, file_id, source_bytes, nodes, edges, parent_id
                )
            elif child.type == "expression_statement":
                # Check for module-level assignments
                assignment = self._find_child_by_type(child, "assignment")
                if assignment and parent_id is None:
                    self._extract_assignment(
                        assignment, file_path, file_id, source_bytes, nodes, parent_id
                    )
            elif (
                child.type == "import_statement"
                or child.type == "import_from_statement"
            ):
                # Track imports for edges
                pass  # Could add IMPORTS edges here

    def _extract_class(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
        parent_id: Optional[str],
    ) -> None:
        """Extract a class definition."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        class_name = self._get_node_text(name_node, source_bytes)
        class_id = f"{file_id}:{class_name}"

        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
                start_col=span.start_col,
                end_col=span.end_col,
            )

        # Check for base classes
        bases = []
        arg_list = self._find_child_by_type(node, "argument_list")
        if arg_list:
            for arg in arg_list.children:
                if arg.type == "identifier":
                    bases.append(self._get_node_text(arg, source_bytes))

        class_node = Node(
            id=class_id,
            kind=NodeKind.CONTAINER,
            name=class_name,
            span=span,
            parent_id=parent_id,
            meta={
                "type": "class",
                "bases": bases,
            },
        )
        nodes.append(class_node)

        # Add INHERITS edges
        for base in bases:
            edges.append((class_id, base, EdgeKind.INHERITS))

        # Extract class body
        body = self._find_child_by_type(node, "block")
        if body:
            self._extract_class_body(
                body, file_path, file_id, source_bytes, nodes, edges, class_id
            )

    def _extract_class_body(
        self,
        body: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
        class_id: str,
    ) -> None:
        """Extract members from a class body."""
        for child in body.children:
            if child.type in ("function_definition", "async_function_definition"):
                self._extract_function(
                    child, file_path, file_id, source_bytes, nodes, edges, class_id
                )
            elif child.type == "decorated_definition":
                self._extract_decorated(
                    child, file_path, file_id, source_bytes, nodes, edges, class_id
                )
            elif child.type == "expression_statement":
                # Class attributes
                assignment = self._find_child_by_type(child, "assignment")
                if assignment:
                    self._extract_assignment(
                        assignment, file_path, file_id, source_bytes, nodes, class_id
                    )

    def _extract_function(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
        parent_id: Optional[str],
        decorators: Optional[List[str]] = None,
    ) -> None:
        """Extract a function definition."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        func_name = self._get_node_text(name_node, source_bytes)

        # Build function ID
        if parent_id:
            func_id = f"{parent_id}.{func_name}"
        else:
            func_id = f"{file_id}:{func_name}"

        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
                start_col=span.start_col,
                end_col=span.end_col,
            )

        # Extract parameters
        params = []
        param_node = self._find_child_by_type(node, "parameters")
        if param_node:
            for param in param_node.children:
                if param.type == "identifier":
                    params.append(self._get_node_text(param, source_bytes))
                elif param.type == "typed_parameter":
                    name = self._find_child_by_type(param, "identifier")
                    if name:
                        params.append(self._get_node_text(name, source_bytes))

        # Determine visibility
        is_private = func_name.startswith("_") and not func_name.startswith("__")
        is_dunder = func_name.startswith("__") and func_name.endswith("__")
        visibility = "private" if is_private else ("dunder" if is_dunder else "public")

        is_async = node.type == "async_function_definition"

        func_node = Node(
            id=func_id,
            kind=NodeKind.UNIT,
            name=func_name,
            span=span,
            parent_id=parent_id,
            meta={
                "type": "method" if parent_id else "function",
                "visibility": visibility,
                "is_async": is_async,
                "parameters": params,
                "decorators": decorators or [],
            },
        )
        nodes.append(func_node)

        # Add DEFINES edge
        if parent_id:
            edges.append((parent_id, func_id, EdgeKind.DEFINES))

        # Add ACCEPTS edges for decorators
        if decorators:
            for dec in decorators:
                # Construct full decorator ID to match node created in _extract_decorated
                dec_id = f"{file_id}:{dec}"
                edges.append((func_id, dec_id, EdgeKind.ACCEPTS))

        # Extract function calls and variable accesses within body
        body = self._find_child_by_type(node, "block")
        if body:
            self._extract_calls(body, func_id, source_bytes, edges)
            self._extract_variable_accesses(
                body, func_id, file_id, parent_id, source_bytes, edges
            )

    def _extract_decorated(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
        parent_id: Optional[str],
    ) -> None:
        """Extract a decorated definition."""
        decorators = []

        # Collect decorators
        for child in node.children:
            if child.type == "decorator":
                dec_text = self._get_node_text(child, source_bytes)
                # Strip @ and any arguments
                dec_name = dec_text.lstrip("@").split("(")[0].strip()
                decorators.append(dec_name)

                # Add decorator as INTERFACE node
                dec_id = f"{file_id}:{dec_name}"
                dec_node = Node(
                    id=dec_id,
                    kind=NodeKind.INTERFACE,
                    name=dec_name,
                    span=self.extract_span(child),
                    meta={"type": "decorator"},
                )
                nodes.append(dec_node)

        # Find the decorated item
        for child in node.children:
            if child.type == "class_definition":
                self._extract_class(
                    child, file_path, file_id, source_bytes, nodes, edges, parent_id
                )
            elif child.type in ("function_definition", "async_function_definition"):
                self._extract_function(
                    child,
                    file_path,
                    file_id,
                    source_bytes,
                    nodes,
                    edges,
                    parent_id,
                    decorators,
                )

    def _extract_assignment(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        parent_id: Optional[str],
    ) -> None:
        """Extract a variable assignment."""
        left = node.children[0] if node.children else None
        if not left:
            return

        # Get variable name
        if left.type == "identifier":
            var_name = self._get_node_text(left, source_bytes)
        elif left.type == "pattern_list":
            # Multiple assignment
            return
        else:
            return

        # Build variable ID
        if parent_id:
            var_id = f"{parent_id}.{var_name}"
        else:
            var_id = f"{file_id}:{var_name}"

        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
            )

        is_private = var_name.startswith("_")

        var_node = Node(
            id=var_id,
            kind=NodeKind.VARIABLE,
            name=var_name,
            span=span,
            parent_id=parent_id,
            meta={
                "type": "class_attribute" if parent_id else "global",
                "visibility": "private" if is_private else "public",
            },
        )
        nodes.append(var_node)

    def _extract_calls(
        self,
        node: Any,
        caller_id: str,
        source_bytes: bytes,
        edges: List[Tuple[str, str, EdgeKind]],
    ) -> None:
        """Extract function calls from a node."""
        for child in node.children:
            if child.type == "call":
                func = child.children[0] if child.children else None
                if func:
                    if func.type == "identifier":
                        callee = self._get_node_text(func, source_bytes)
                        edges.append((caller_id, callee, EdgeKind.CALLS))
                    elif func.type == "attribute":
                        # method.call()
                        callee = self._get_node_text(func, source_bytes)
                        edges.append((caller_id, callee, EdgeKind.CALLS))

            # Recurse into children
            self._extract_calls(child, caller_id, source_bytes, edges)

    def _extract_variable_accesses(
        self,
        node: Any,
        func_id: str,
        file_id: str,
        container_id: Optional[str],
        source_bytes: bytes,
        edges: List[Tuple[str, str, EdgeKind]],
    ) -> None:
        """
        Extract READS and WRITES edges from AST nodes.

        Args:
            node: Tree-sitter node to analyze
            func_id: ID of the containing function
            file_id: Path to the source file
            container_id: ID of the containing class (or None for module-level)
            source_bytes: Raw file content
            edges: List to append edges to
        """
        for child in node.children:
            # WRITES: Only track attribute assignments (self.x = ...) as "state writes"
            # Skip local variable assignments - they're not persistent state
            if child.type == "assignment":
                # Only track attribute writes (self.x, cls.x)
                for i, part in enumerate(child.children):
                    if part.type == "=" or part.type == "type":
                        break
                    if part.type == "attribute":
                        var_id = self._resolve_attribute_id(
                            part, file_id, container_id, source_bytes
                        )
                        if var_id:
                            edges.append((func_id, var_id, EdgeKind.WRITES))

                # Process right side for reads
                found_eq = False
                for part in child.children:
                    if part.type == "=":
                        found_eq = True
                        continue
                    if found_eq:
                        self._extract_variable_accesses(
                            part, func_id, file_id, container_id, source_bytes, edges
                        )
                continue

            # WRITES: Augmented assignment - only for attributes (self.x += ...)
            if child.type == "augmented_assignment":
                left = child.children[0] if child.children else None
                if left and left.type == "attribute":
                    var_id = self._resolve_attribute_id(
                        left, file_id, container_id, source_bytes
                    )
                    if var_id:
                        edges.append((func_id, var_id, EdgeKind.WRITES))
                        edges.append((func_id, var_id, EdgeKind.READS))

                # Process right side for reads
                for part in child.children[2:]:
                    self._extract_variable_accesses(
                        part, func_id, file_id, container_id, source_bytes, edges
                    )
                continue

            # For loops - just recurse, don't track loop variable as WRITES
            if child.type == "for_statement":
                # Recurse into for statement body
                self._extract_variable_accesses(
                    child, func_id, file_id, container_id, source_bytes, edges
                )
                continue

            # READS: Identifiers (variable references)
            if child.type == "identifier":
                var_name = self._get_node_text(child, source_bytes)
                # Skip special names and common builtins
                if var_name not in (
                    "self", "cls", "True", "False", "None",
                    "print", "len", "range", "str", "int", "float", "list", "dict",
                    "set", "tuple", "bool", "type", "object", "super", "isinstance",
                    "hasattr", "getattr", "setattr", "delattr", "callable", "iter",
                    "next", "enumerate", "zip", "map", "filter", "sorted", "reversed",
                    "min", "max", "sum", "abs", "round", "open", "input", "format",
                    "repr", "ord", "chr", "hex", "oct", "bin", "id", "hash", "dir",
                    "vars", "globals", "locals", "eval", "exec", "compile",
                    "Exception", "BaseException", "ValueError", "TypeError",
                    "KeyError", "IndexError", "AttributeError", "RuntimeError",
                    "StopIteration", "AssertionError", "ImportError", "OSError",
                ):
                    var_id = self._make_var_id(var_name, file_id, container_id)
                    edges.append((func_id, var_id, EdgeKind.READS))
                continue

            # READS: Attribute access (self.x, obj.x)
            if child.type == "attribute":
                var_id = self._resolve_attribute_id(
                    child, file_id, container_id, source_bytes
                )
                if var_id:
                    edges.append((func_id, var_id, EdgeKind.READS))
                continue

            # Skip function/method names in calls (don't treat as variable read)
            if child.type == "call":
                # Process arguments only, not the function name
                args = self._find_child_by_type(child, "argument_list")
                if args:
                    self._extract_variable_accesses(
                        args, func_id, file_id, container_id, source_bytes, edges
                    )
                continue

            # Recurse into other children
            self._extract_variable_accesses(
                child, func_id, file_id, container_id, source_bytes, edges
            )

    def _make_var_id(
        self, var_name: str, file_id: str, container_id: Optional[str]
    ) -> str:
        """
        Create variable ID matching the format used in _extract_assignment.

        Args:
            var_name: Name of the variable
            file_id: Path to the source file
            container_id: ID of the containing class (or None for module-level)

        Returns:
            Variable ID string
        """
        if container_id:
            return f"{container_id}.{var_name}"
        return f"{file_id}:{var_name}"

    def _resolve_attribute_id(
        self,
        attr_node: Any,
        file_id: str,
        container_id: Optional[str],
        source_bytes: bytes,
    ) -> Optional[str]:
        """
        Resolve attribute access (self.x, cls.x) to a variable ID.

        Maps self.x -> ClassName.x to match class attribute VARIABLE nodes.

        Args:
            attr_node: Tree-sitter attribute node
            file_id: Path to the source file
            container_id: ID of the containing class (or None for module-level)
            source_bytes: Raw file content

        Returns:
            Variable ID string or None if cannot resolve
        """
        if len(attr_node.children) < 2:
            return None

        obj = attr_node.children[0]
        attr_name_node = attr_node.children[-1]  # Last child is the attribute name

        if attr_name_node.type != "identifier":
            return None

        attr_name = self._get_node_text(attr_name_node, source_bytes)
        obj_name = (
            self._get_node_text(obj, source_bytes) if obj.type == "identifier" else None
        )

        # self.x or cls.x -> ClassName.x
        if obj_name in ("self", "cls") and container_id:
            return f"{container_id}.{attr_name}"

        # Other attribute access - use full form
        if obj_name:
            return f"{file_id}:{obj_name}.{attr_name}"

        return None
