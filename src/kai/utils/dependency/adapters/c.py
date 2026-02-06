"""
C-specific domain adapter for the GraphQueryEngine.

Provides domain knowledge for C security analysis:
- Entrypoint detection (main, non-static functions, header declarations)
- Library file identification (system headers, third-party)
- Trust level patterns
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from .base import DomainAdapter, LensDefinition
from ..models import Node, NodeKind

if TYPE_CHECKING:
    from ..graph import DependencyGraph


class CAdapter(DomainAdapter):
    """
    C-specific implementation of DomainAdapter.

    Provides domain knowledge for:
    - Symbol resolution (functions, structs, globals)
    - Entrypoint detection (main, exported functions)
    - Test file identification
    - Library detection
    """

    @property
    def name(self) -> str:
        return "c"

    def get_domain_mapping(self) -> Dict[str, str]:
        """Return C-specific NodeKind mappings."""
        return {
            "struct": "CONTAINER",
            "union": "CONTAINER",
            "function": "UNIT",
            "macro": "INTERFACE",
            "global": "VARIABLE",
            "enum": "TYPE_DEF",
            "typedef": "TYPE_DEF",
        }

    def is_public_entrypoint(self, node: Node) -> bool:
        """
        Check if a node is a public entrypoint.

        Entrypoints in C include:
        - main() function
        - Non-static functions (visible to other translation units)
        - Functions declared in header files
        """
        if node.kind != NodeKind.UNIT:
            return False

        name = node.name
        meta = node.meta

        # main() is always an entrypoint
        if name == "main":
            return True

        # Static functions are not public
        if meta.get("is_static", False):
            return False

        # Functions starting with _ are typically internal
        if name.startswith("_"):
            return False

        # Check visibility from meta
        visibility = meta.get("visibility", "public")
        if visibility == "private":
            return False

        # Non-static functions are public entrypoints
        return True

    def is_state_variable(self, node: Node) -> bool:
        """Check if a node is a state variable."""
        if node.kind != NodeKind.VARIABLE:
            return False

        var_type = node.meta.get("type", "")
        # Global and static variables are state
        return var_type == "global" or node.meta.get("is_static", False)

    def is_test_file(self, file_path: str) -> bool:
        """Check if a file path looks like a test file."""
        p = file_path.lower()
        return (
            "/test/" in p
            or "/tests/" in p
            or "test_" in p.split("/")[-1]
            or "_test." in p
            or "/t/" in p  # Common in some projects
            or "/check_" in p
            or "check_" in p.split("/")[-1]
        )

    def is_library_file(self, file_path: str) -> bool:
        """
        Check if a file path is from an external library.

        Identifies common C library/dependency patterns.
        """
        p = file_path.lower()
        library_indicators = [
            # System headers
            "/usr/include/",
            "/usr/local/include/",
            # Common library directories
            "/lib/",
            "/libs/",
            "/third_party/",
            "/third-party/",
            "/vendor/",
            "/external/",
            "/deps/",
            # Build directories
            "/build/",
            "/cmake-build",
            # Package managers
            "/vcpkg/",
            "/conan/",
        ]
        return any(indicator in p for indicator in library_indicators)

    def resolve_symbol(
        self,
        name: str,
        context_graph: "DependencyGraph",
        scope: Optional[str] = None,
    ) -> List[str]:
        """
        Resolve a symbol name to node IDs in the dependency graph.

        Args:
            name: Symbol name to resolve
            context_graph: The dependency graph to search
            scope: Optional scope to limit search

        Returns:
            List of matching node IDs
        """
        candidate_ids: List[str] = []

        # Check if symbol is already a node ID
        if name in context_graph._nodes:
            return [name]

        # Search by name across all nodes
        for nid, node in context_graph._nodes.items():
            if node.name == name:
                candidate_ids.append(nid)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for cid in candidate_ids:
            if cid not in seen:
                seen.add(cid)
                unique.append(cid)

        return unique

    def is_non_auth_guard(self, modifier_name: str) -> bool:
        """
        Check if this is a non-auth guard pattern.

        In C, this would be checking for non-auth macros/attributes.
        """
        non_auth_patterns = {
            # Compiler attributes
            "unused",
            "deprecated",
            "warn_unused_result",
            "nonnull",
            "noreturn",
            # Common macros
            "likely",
            "unlikely",
            "inline",
            "always_inline",
            "noinline",
            # Assert-like
            "assert",
            "static_assert",
            # Annotations
            "const",
            "restrict",
            "volatile",
        }
        return modifier_name.lower() in non_auth_patterns

    def get_trust_for_modifiers(self, modifier_names: List[str]) -> str:
        """
        Determine trust level based on patterns.

        In C, this is less applicable but can check for things like
        static (file-local) or specific naming conventions.

        Returns: "high", "medium", "low", "none", "review_required"
        """
        if not modifier_names:
            return "none"

        # High trust patterns
        high_trust = {
            "root_only",
            "privileged",
            "secure",
        }

        # Medium trust patterns
        medium_trust = {
            "authenticated",
            "checked",
        }

        for mod in modifier_names:
            mod_lower = mod.lower()
            if mod_lower in high_trust:
                return "high"

        for mod in modifier_names:
            mod_lower = mod.lower()
            if mod_lower in medium_trust:
                return "medium"

        return "review_required"

    def get_entrypoint_visibility(self) -> List[str]:
        """Return visibility levels that indicate public entrypoints."""
        return ["public"]

    def get_role_patterns(self) -> Dict[str, List[str]]:
        """Return patterns that indicate roles (limited in C)."""
        return {
            "Root": ["root_only", "privileged"],
            "User": ["user_check", "authenticated"],
        }

    def get_guard_patterns(self) -> List[str]:
        """Return common guard patterns."""
        return [
            "check_permissions",
            "validate_input",
            "sanitize",
            "verify",
        ]

    def get_dangerous_functions(self) -> List[str]:
        """Return list of dangerous C functions (security-relevant)."""
        return [
            # Memory
            "strcpy",
            "strcat",
            "sprintf",
            "gets",
            "scanf",
            # Format string
            "printf",
            "fprintf",
            "snprintf",  # can still be dangerous
            # Memory allocation
            "malloc",
            "calloc",
            "realloc",
            "free",
            # File operations
            "fopen",
            "fread",
            "fwrite",
            # System
            "system",
            "popen",
            "exec",
            "execve",
            "fork",
        ]

    def get_safe_alternatives(self) -> Dict[str, str]:
        """Return safer alternatives to dangerous functions."""
        return {
            "strcpy": "strncpy or strlcpy",
            "strcat": "strncat or strlcat",
            "sprintf": "snprintf",
            "gets": "fgets",
            "scanf": "fgets + sscanf with limits",
        }

    # FIXME: this was added so that tests pass
    def get_lens_definitions(self) -> List[LensDefinition]:
        """C-specific lens definitions for security analysis."""
        return [
            LensDefinition(
                name="memory",
                description="Buffer overflows, use-after-free, double-free, format strings",
                invariant_types=["ACCESS", "OTHER"],
                prompt_template="""
## MEMORY LENS - C

Focus on memory safety vulnerabilities.

### Buffer Overflows
For EACH function using string/memory operations:
- Check for unsafe functions (strcpy, strcat, sprintf, gets)
- Verify buffer sizes in memcpy/memmove
- Check for off-by-one errors
- Generate ACCESS invariant: "Buffer X has bounds checking"

### Use-After-Free / Double-Free
For EACH free() call:
- Verify pointer is not used after free
- Check for double-free paths
- Verify pointer is set to NULL after free

### Format String Vulnerabilities
For EACH printf-family call:
- Verify format string is not user-controlled
- Check for missing format specifiers
""",
                checklist=[
                    "No unsafe string functions with unchecked bounds",
                    "No use-after-free patterns",
                    "No user-controlled format strings",
                    "Buffer sizes validated before copy",
                ],
            ),
            LensDefinition(
                name="input_validation",
                description="Integer overflow, null pointer, input sanitization",
                invariant_types=["VALUE_FLOW", "OTHER"],
                prompt_template="""
## INPUT VALIDATION LENS - C

Focus on input validation and integer safety.

### Integer Overflow/Underflow
For EACH arithmetic operation on external input:
- Check for overflow in size calculations
- Verify signed/unsigned conversions
- Check for integer truncation

### Null Pointer Dereference
For EACH pointer parameter:
- Verify null check before dereference
- Check for null returns from malloc/calloc

### Command Injection
For EACH system()/popen()/exec() call:
- Verify input is sanitized
- Check for shell metacharacter filtering
""",
                checklist=[
                    "Arithmetic on sizes checked for overflow",
                    "Pointers checked for NULL before use",
                    "External input sanitized before system calls",
                ],
            ),
        ]

    # FIXME: this was added so that tests pass
    def get_function_metadata_extractors(self) -> Dict[str, Callable]:
        """C-specific metadata extractors."""

        def extract_is_static(node: Node, graph: "DependencyGraph") -> bool:
            return node.meta.get("is_static", False)

        def extract_calls_dangerous(node: Node, graph: "DependencyGraph") -> bool:
            """Check if function calls dangerous C functions."""
            dangerous = set(self.get_dangerous_functions())
            calls = node.meta.get("calls", [])
            return any(c in dangerous for c in calls if isinstance(c, str))

        return {
            "is_static": extract_is_static,
            "calls_dangerous": extract_calls_dangerous,
        }
