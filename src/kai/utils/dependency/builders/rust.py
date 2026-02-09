"""
Rust dependency graph builder using tree-sitter.

Extracts:
- Functions and methods (UNIT)
- Structs, enums (CONTAINER / TYPE_DEF)
- Impl blocks (CONTAINER)
- Traits (INTERFACE)
- Static/const variables (VARIABLE)
- Modules (tracked via file structure)
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .treesitter_base import TreeSitterBuilder
from ..models import Node, SourceSpan, NodeKind, EdgeKind


class RustBuilder(TreeSitterBuilder):
    """
    Tree-sitter based builder for Rust projects.

    Maps Rust constructs to NodeKind:
    - struct_item, enum_item -> CONTAINER / TYPE_DEF
    - impl_item -> CONTAINER
    - trait_item -> INTERFACE
    - function_item, function_signature_item -> UNIT
    - static_item, const_item -> VARIABLE
    """

    def __init__(self, skip_patterns: Optional[List[str]] = None):
        super().__init__(
            skip_patterns=skip_patterns
            or [
                "target",
                "tests",
                "test",
                "benches",
                "examples",
                "node_modules",
                ".git",
                "vendor",
                "third_party",
            ]
        )

    @property
    def language(self) -> str:
        return "rust"

    @property
    def file_extensions(self) -> List[str]:
        return [".rs"]

    def _extract_from_tree(
        self, tree: Any, file_path: Path, source_bytes: bytes
    ) -> Tuple[List[Node], List[Tuple[str, str, EdgeKind]]]:
        """Extract Rust nodes and edges from the AST."""
        nodes: List[Node] = []
        edges: List[Tuple[str, str, EdgeKind]] = []

        file_id = self._make_file_id(file_path)
        root = tree.root_node

        self._extract_source_file(root, file_path, file_id, source_bytes, nodes, edges)

        return nodes, edges

    def _make_file_id(self, file_path: Path) -> str:
        """
        Compute a file identifier that matches the FILE node id used by the base TreeSitterBuilder.

        Prefer a repo-relative path when a repository root attribute is available
        on the builder; otherwise, fall back to a normalized path string.
        """
        # Try common attribute names that might store the repository root.
        for attr_name in ("repo_root", "root_dir", "project_root", "root"):
            root = getattr(self, attr_name, None)
            if isinstance(root, Path):
                try:
                    # If file_path is absolute, relativize it to the root.
                    if file_path.is_absolute():
                        return file_path.relative_to(root).as_posix()
                    # If file_path is already relative, normalize it with respect to root.
                    return (root / file_path).relative_to(root).as_posix()
                except ValueError:
                    # file_path is not under this root; try the next candidate.
                    continue
        # Fallback: use a normalized path string.
        return file_path.as_posix()

    def _extract_source_file(
        self,
        root: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
    ) -> None:
        """Extract top-level items from a Rust source file."""
        for child in root.children:
            self._extract_item(
                child, file_path, file_id, source_bytes, nodes, edges, None
            )

    def _extract_item(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
        parent_id: Optional[str],
    ) -> None:
        """Dispatch extraction based on item type."""
        if node.type == "function_item":
            self._extract_function(
                node, file_path, file_id, source_bytes, nodes, edges, parent_id
            )
        elif node.type == "struct_item":
            self._extract_struct(node, file_path, file_id, source_bytes, nodes, edges)
        elif node.type == "enum_item":
            self._extract_enum(node, file_path, file_id, source_bytes, nodes, edges)
        elif node.type == "impl_item":
            self._extract_impl(node, file_path, file_id, source_bytes, nodes, edges)
        elif node.type == "trait_item":
            self._extract_trait(node, file_path, file_id, source_bytes, nodes, edges)
        elif node.type == "static_item":
            self._extract_static(node, file_path, file_id, source_bytes, nodes)
        elif node.type == "const_item":
            self._extract_const(node, file_path, file_id, source_bytes, nodes)
        elif node.type == "type_item":
            self._extract_type_alias(node, file_path, file_id, source_bytes, nodes)
        elif node.type == "mod_item":
            self._extract_mod(node, file_path, file_id, source_bytes, nodes, edges)
        elif node.type == "use_declaration":
            self._extract_use(node, file_id, source_bytes, edges)

    def _get_visibility(self, node: Any, source_bytes: bytes) -> str:
        """Extract visibility modifier from a node."""
        vis = self._find_child_by_type(node, "visibility_modifier")
        if vis:
            return self._get_node_text(vis, source_bytes).strip()
        return ""

    def _get_attributes(self, node: Any, source_bytes: bytes) -> List[str]:
        """Extract attribute names from preceding attribute items."""
        attrs: List[str] = []
        # In tree-sitter-rust, attributes are children of the item
        for child in node.children:
            if child.type == "attribute_item":
                attr_text = self._get_node_text(child, source_bytes)
                # Strip #[ and ]
                inner = attr_text.strip().lstrip("#").strip("[]")
                attrs.append(inner)
        return attrs

    def _extract_function(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
        parent_id: Optional[str],
    ) -> None:
        """Extract a function item."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        func_name = self._get_node_text(name_node, source_bytes)
        if parent_id:
            func_id = f"{parent_id}::{func_name}"
        else:
            func_id = f"{file_id}:{func_name}"

        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
            )

        visibility = self._get_visibility(node, source_bytes)
        attributes = self._get_attributes(node, source_bytes)

        # Check for async/unsafe keywords
        is_async = any(
            c.type == "async"
            or (
                c.type == "identifier"
                and self._get_node_text(c, source_bytes) == "async"
            )
            for c in node.children
        )
        is_unsafe = any(
            c.type == "unsafe"
            or (
                c.type == "identifier"
                and self._get_node_text(c, source_bytes) == "unsafe"
            )
            for c in node.children
        )

        # Extract parameters
        params = self._extract_parameters(node, source_bytes)

        # Extract return type
        return_type = ""
        ret_type_node = self._find_child_by_type(node, "return_type")
        if ret_type_node:
            return_type = (
                self._get_node_text(ret_type_node, source_bytes).lstrip("->").strip()
            )

        # Check for &mut self
        takes_mut_self = any("& mut self" in p or "&mut self" in p for p in params)

        # Get source code for the function body
        source_code = self._get_node_text(node, source_bytes)

        func_node = Node(
            id=func_id,
            kind=NodeKind.UNIT,
            name=func_name,
            span=span,
            parent_id=parent_id,
            meta={
                "type": "function",
                "visibility": visibility,
                "is_async": is_async,
                "is_unsafe": is_unsafe,
                "parameters": params,
                "return_type": return_type,
                "attributes": attributes,
                "takes_mut_self": takes_mut_self,
                "source_code": source_code[:5000]
                if len(source_code) > 5000
                else source_code,
            },
        )
        nodes.append(func_node)

        # Add DEFINES edge from parent
        if parent_id:
            edges.append((parent_id, func_id, EdgeKind.DEFINES))

        # Extract calls and variable accesses within function body
        body = self._find_child_by_type(node, "block")
        if body:
            self._extract_calls(body, func_id, source_bytes, edges)
            self._extract_variable_accesses(body, func_id, source_bytes, edges)

    def _extract_parameters(self, func_node: Any, source_bytes: bytes) -> List[str]:
        """Extract parameter strings from a function."""
        params: List[str] = []
        param_list = self._find_child_by_type(func_node, "parameters")
        if not param_list:
            return params

        for child in param_list.children:
            if child.type == "parameter":
                params.append(self._get_node_text(child, source_bytes).strip())
            elif child.type == "self_parameter":
                params.append(self._get_node_text(child, source_bytes).strip())
        return params

    def _extract_struct(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
    ) -> None:
        """Extract a struct item."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return

        struct_name = self._get_node_text(name_node, source_bytes)
        struct_id = f"{file_id}:struct_{struct_name}"

        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
            )

        visibility = self._get_visibility(node, source_bytes)
        attributes = self._get_attributes(node, source_bytes)

        # Extract fields
        fields: List[str] = []
        field_list = self._find_child_by_type(node, "field_declaration_list")
        if field_list:
            for field in field_list.children:
                if field.type == "field_declaration":
                    field_name = self._find_child_by_type(field, "field_identifier")
                    if field_name:
                        fields.append(self._get_node_text(field_name, source_bytes))

        struct_node = Node(
            id=struct_id,
            kind=NodeKind.CONTAINER,
            name=struct_name,
            span=span,
            parent_id=None,
            meta={
                "type": "struct",
                "visibility": visibility,
                "fields": fields,
                "attributes": attributes,
            },
        )
        nodes.append(struct_node)

    def _extract_enum(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
    ) -> None:
        """Extract an enum item."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return

        enum_name = self._get_node_text(name_node, source_bytes)
        enum_id = f"{file_id}:enum_{enum_name}"

        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
            )

        visibility = self._get_visibility(node, source_bytes)
        attributes = self._get_attributes(node, source_bytes)

        # Extract variants
        variants: List[str] = []
        variant_list = self._find_child_by_type(node, "enum_variant_list")
        if variant_list:
            for variant in variant_list.children:
                if variant.type == "enum_variant":
                    variant_name = self._find_child_by_type(variant, "identifier")
                    if variant_name:
                        variants.append(self._get_node_text(variant_name, source_bytes))

        enum_node = Node(
            id=enum_id,
            kind=NodeKind.TYPE_DEF,
            name=enum_name,
            span=span,
            parent_id=None,
            meta={
                "type": "enum",
                "visibility": visibility,
                "variants": variants,
                "attributes": attributes,
            },
        )
        nodes.append(enum_node)

    def _extract_impl(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
    ) -> None:
        """Extract an impl block."""
        # Get the type being implemented
        type_node = self._find_child_by_type(node, "type_identifier")
        if not type_node:
            # Could be a generic type or path type
            generic = self._find_child_by_type(node, "generic_type")
            if generic:
                type_node = self._find_child_by_type(generic, "type_identifier")
            if not type_node:
                scoped = self._find_child_by_type(node, "scoped_type_identifier")
                if scoped:
                    type_node = scoped
        if not type_node:
            return

        impl_type_name = self._get_node_text(type_node, source_bytes)

        # Check for trait impl (impl Trait for Type)
        trait_name = ""
        if "for" in [
            self._get_node_text(c, source_bytes)
            for c in node.children
            if c.type == "identifier"
        ]:
            # This is a trait impl - the first type_identifier is the trait
            type_ids = [c for c in node.children if c.type == "type_identifier"]
            if len(type_ids) >= 2:
                trait_name = self._get_node_text(type_ids[0], source_bytes)
                impl_type_name = self._get_node_text(type_ids[1], source_bytes)

        if trait_name:
            impl_id = f"{file_id}:impl_{trait_name}_for_{impl_type_name}"
        else:
            impl_id = f"{file_id}:impl_{impl_type_name}"

        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
            )

        impl_node = Node(
            id=impl_id,
            kind=NodeKind.CONTAINER,
            name=f"impl {trait_name + ' for ' if trait_name else ''}{impl_type_name}",
            span=span,
            parent_id=None,
            meta={
                "type": "impl",
                "target_type": impl_type_name,
                "trait": trait_name,
            },
        )
        nodes.append(impl_node)

        # Link impl to struct if it exists
        struct_id = f"{file_id}:struct_{impl_type_name}"
        edges.append((struct_id, impl_id, EdgeKind.DEFINES))

        # If it's a trait impl, add INHERITS edge
        if trait_name:
            edges.append((impl_id, trait_name, EdgeKind.INHERITS))

        # Extract methods in the impl body
        body = self._find_child_by_type(node, "declaration_list")
        if body:
            for child in body.children:
                self._extract_item(
                    child, file_path, file_id, source_bytes, nodes, edges, impl_id
                )

    def _extract_trait(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
    ) -> None:
        """Extract a trait item."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return

        trait_name = self._get_node_text(name_node, source_bytes)
        trait_id = f"{file_id}:trait_{trait_name}"

        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
            )

        visibility = self._get_visibility(node, source_bytes)
        attributes = self._get_attributes(node, source_bytes)

        # Check for super-traits
        super_traits: List[str] = []
        trait_bounds = self._find_child_by_type(node, "trait_bounds")
        if trait_bounds:
            for bound in trait_bounds.children:
                if bound.type == "type_identifier":
                    super_traits.append(self._get_node_text(bound, source_bytes))

        trait_node = Node(
            id=trait_id,
            kind=NodeKind.INTERFACE,
            name=trait_name,
            span=span,
            parent_id=None,
            meta={
                "type": "trait",
                "visibility": visibility,
                "super_traits": super_traits,
                "attributes": attributes,
            },
        )
        nodes.append(trait_node)

        # Add INHERITS edges for super-traits
        for st in super_traits:
            edges.append((trait_id, st, EdgeKind.INHERITS))

        # Extract trait methods (signatures)
        body = self._find_child_by_type(node, "declaration_list")
        if body:
            for child in body.children:
                if child.type in ("function_item", "function_signature_item"):
                    self._extract_function(
                        child, file_path, file_id, source_bytes, nodes, edges, trait_id
                    )

    def _extract_static(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
    ) -> None:
        """Extract a static item."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        var_name = self._get_node_text(name_node, source_bytes)
        var_id = f"{file_id}:{var_name}"

        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
            )

        visibility = self._get_visibility(node, source_bytes)

        # Check for mut
        is_mutable = any(
            self._get_node_text(c, source_bytes) == "mut"
            for c in node.children
            if c.type == "mutable_specifier"
        )

        var_node = Node(
            id=var_id,
            kind=NodeKind.VARIABLE,
            name=var_name,
            span=span,
            parent_id=None,
            meta={
                "type": "static_mut" if is_mutable else "static",
                "visibility": visibility,
                "is_mutable": is_mutable,
            },
        )
        nodes.append(var_node)

    def _extract_const(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
    ) -> None:
        """Extract a const item."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        var_name = self._get_node_text(name_node, source_bytes)
        var_id = f"{file_id}:const_{var_name}"

        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
            )

        visibility = self._get_visibility(node, source_bytes)

        const_node = Node(
            id=var_id,
            kind=NodeKind.VARIABLE,
            name=var_name,
            span=span,
            parent_id=None,
            meta={
                "type": "const",
                "visibility": visibility,
            },
        )
        nodes.append(const_node)

    def _extract_type_alias(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
    ) -> None:
        """Extract a type alias."""
        name_node = self._find_child_by_type(node, "type_identifier")
        if not name_node:
            return

        type_name = self._get_node_text(name_node, source_bytes)
        type_id = f"{file_id}:type_{type_name}"

        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
            )

        type_node = Node(
            id=type_id,
            kind=NodeKind.TYPE_DEF,
            name=type_name,
            span=span,
            parent_id=None,
            meta={"type": "type_alias"},
        )
        nodes.append(type_node)

    def _extract_mod(
        self,
        node: Any,
        file_path: Path,
        file_id: str,
        source_bytes: bytes,
        nodes: List[Node],
        edges: List[Tuple[str, str, EdgeKind]],
    ) -> None:
        """Extract a mod item (inline module)."""
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return

        mod_name = self._get_node_text(name_node, source_bytes)

        # Only process inline modules (those with a body)
        body = self._find_child_by_type(node, "declaration_list")
        if not body:
            return

        mod_id = f"{file_id}:mod_{mod_name}"
        span = self.extract_span(node)
        if span:
            span = SourceSpan(
                file=file_id,
                start_line=span.start_line,
                end_line=span.end_line,
            )

        mod_node = Node(
            id=mod_id,
            kind=NodeKind.CONTAINER,
            name=mod_name,
            span=span,
            parent_id=None,
            meta={"type": "module"},
        )
        nodes.append(mod_node)

        # Extract items inside the module
        for child in body.children:
            self._extract_item(
                child, file_path, file_id, source_bytes, nodes, edges, mod_id
            )

    def _extract_use(
        self,
        node: Any,
        file_id: str,
        source_bytes: bytes,
        edges: List[Tuple[str, str, EdgeKind]],
    ) -> None:
        """Extract a use declaration as an IMPORTS edge."""
        use_text = self._get_node_text(node, source_bytes).strip()
        # Strip "use " prefix and trailing ";"
        path = use_text.lstrip("use ").rstrip(";").strip()
        if path:
            edges.append((file_id, path, EdgeKind.IMPORTS))

    def _extract_calls(
        self,
        node: Any,
        caller_id: str,
        source_bytes: bytes,
        edges: List[Tuple[str, str, EdgeKind]],
    ) -> None:
        """Extract function/method calls from a node."""
        for child in node.children:
            if child.type == "call_expression":
                func = child.children[0] if child.children else None
                if func:
                    if func.type == "identifier":
                        callee = self._get_node_text(func, source_bytes)
                        edges.append((caller_id, callee, EdgeKind.CALLS))
                    elif func.type == "scoped_identifier":
                        callee = self._get_node_text(func, source_bytes)
                        edges.append((caller_id, callee, EdgeKind.CALLS))
                    elif func.type == "field_expression":
                        # method call: obj.method(...)
                        method = self._find_child_by_type(func, "field_identifier")
                        if method:
                            callee = self._get_node_text(method, source_bytes)
                            edges.append((caller_id, callee, EdgeKind.CALLS))

            elif child.type == "macro_invocation":
                # macro!(...)
                macro_name = child.children[0] if child.children else None
                if macro_name:
                    callee = self._get_node_text(macro_name, source_bytes)
                    edges.append((caller_id, callee, EdgeKind.CALLS))

            # Recurse into children
            self._extract_calls(child, caller_id, source_bytes, edges)

    def _extract_variable_accesses(
        self,
        node: Any,
        func_id: str,
        source_bytes: bytes,
        edges: List[Tuple[str, str, EdgeKind]],
    ) -> None:
        """Extract variable reads and writes."""
        for child in node.children:
            if child.type == "assignment_expression":
                # Left side is written
                left = child.children[0] if child.children else None
                if left and left.type == "identifier":
                    var_name = self._get_node_text(left, source_bytes)
                    edges.append((func_id, var_name, EdgeKind.WRITES))
            elif child.type == "compound_assignment_expr":
                # +=, -=, etc.
                left = child.children[0] if child.children else None
                if left and left.type == "identifier":
                    var_name = self._get_node_text(left, source_bytes)
                    edges.append((func_id, var_name, EdgeKind.WRITES))
                    edges.append((func_id, var_name, EdgeKind.READS))

            # Recurse
            self._extract_variable_accesses(child, func_id, source_bytes, edges)
