"""
Rust-specific domain adapter for the GraphQueryEngine.

Provides domain knowledge for Rust security analysis:
- Entrypoint detection (pub fn, #[no_mangle], instruction handlers)
- Library file identification (target/, .cargo/registry/)
- Trust level patterns (Solana/Anchor program patterns)
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

from .base import DomainAdapter, LensDefinition
from ..models import Node, NodeKind, EdgeKind

if TYPE_CHECKING:
    from ..graph import DependencyGraph


class RustAdapter(DomainAdapter):
    """
    Rust-specific implementation of DomainAdapter.

    Provides domain knowledge for:
    - Symbol resolution (modules, structs, impls, traits)
    - Entrypoint detection (pub fn, instruction handlers)
    - Test file identification
    - Library detection
    """

    @property
    def name(self) -> str:
        return "rust"

    def get_domain_mapping(self) -> Dict[str, str]:
        """Return Rust-specific NodeKind mappings."""
        return {
            "module": "FILE",

            "impl": "CONTAINER",
            "struct": "CONTAINER",

            "function": "UNIT",
            "method": "UNIT",

            "enum": "TYPE_DEF",
            "type_alias": "TYPE_DEF",

            # static & const can be MUCH more than just state variables, but this is a starting point for bucketing
            # e.g. we can have const functions
            "static": "VARIABLE",
            "const": "VARIABLE",

            "trait": "INTERFACE",
            "macro": "INTERFACE",
        }

    def is_public_entrypoint(self, node: Node) -> bool:
        """
        Check if a node is a public entrypoint.

        Entrypoints in Rust include:
        - pub fn (public functions)
        - #[no_mangle] functions
        - main() function
        - Solana/Anchor instruction handlers (process_instruction, #[program])
        """
        if node.kind != NodeKind.UNIT:
            return False

        name, meta = node.name, node.meta

        # main() is always an entrypoint
        if name == "main":
            return True

        # Check visibility
        visibility = meta.get("visibility", "")
        if visibility == "pub" or visibility == "pub(crate)":
            return True

        # #[no_mangle] marks FFI entrypoints
        attributes = meta.get("attributes", [])
        if "no_mangle" in attributes:
            return True

        # Solana instruction handlers
        solana_patterns = {"process_instruction", "process", "instruction"}
        if name in solana_patterns:
            return True

        # Private by default in Rust
        return False

    def is_state_variable(self, node: Node) -> bool:
        """Check if a node is a state variable."""
        if node.kind != NodeKind.VARIABLE:
            return False

        var_type = node.meta.get("type", "")
        return var_type in ("static", "const", "static_mut")

    def is_test_file(self, file_path: str) -> bool:
        """Check if a file path looks like a test file."""
        p = file_path.lower()
        return (
            "/tests/" in p
            or "/test/" in p
            or "test_" in p.split("/")[-1]
            or "_test.rs" in p
            or "/benches/" in p
        )

    def is_library_file(self, file_path: str) -> bool:
        """
        Check if a file path is from an external library.

        Identifies common Rust dependency patterns.
        """
        p = file_path.lower()
        library_indicators = [
            # Cargo build artifacts
            "/target/",
            # Cargo registry
            "/.cargo/registry/",
            "/.cargo/git/",
            # Common vendored deps
            "/vendor/",
            "/third_party/",
            "/third-party/",
            # Rustup toolchains
            "/.rustup/",
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

        Handles Rust module paths (e.g. crate::foo::bar).
        """
        candidate_ids: List[str] = []

        # Check if symbol is already a node ID
        if name in context_graph._nodes:
            return [name]

        # Strip common path prefixes for matching
        clean_name = name
        for prefix in ("crate::", "self::", "super::"):
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]

        # If scope specified, limit search
        valid_parents: Optional[set] = None
        if scope:
            valid_parents = {scope}

        # Search by name across all nodes
        for nid, node in context_graph._nodes.items():
            if node.name == clean_name or nid.endswith(f"::{clean_name}") or nid.endswith(f":{clean_name}"):
                if valid_parents is not None:
                    if node.parent_id not in valid_parents:
                        continue
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
        Check if this is a non-auth guard attribute.

        In Rust, these are proc-macro attributes and derive macros
        that don't relate to authorization.
        """
        # FIXME: this is just a best-effort heuristic list of common non-auth attributes.
        # would rather prefer an agent to do this
        non_auth_patterns = {
            # Derive macros
            "derive",
            "clone",
            "copy",
            "debug",
            "default",
            "eq",
            "hash",
            "ord",
            "partialeq",
            "partialord",
            # Compiler attributes
            "inline",
            "cold",
            "allow",
            "deny",
            "warn",
            "deprecated",
            "must_use",
            "cfg",
            "cfg_attr",
            "test",
            "bench",
            "doc",
            "repr",
            # Serde
            "serde",
            "serialize",
            "deserialize",
            # Common procedural macros
            "tokio::main",
            "tokio::test",
            "async_trait",
        }
        return modifier_name.lower().strip("#[]") in non_auth_patterns

    def get_trust_for_modifiers(self, modifier_names: List[str]) -> str:
        # NOTE: rust does not have modifiers, so just returning none here
        return "none"

    def get_entrypoint_visibility(self) -> List[str]:
        """Return visibility levels that indicate public entrypoints."""
        return ["pub", "pub(crate)"]

    def get_role_patterns(self) -> Dict[str, List[str]]:
        """Return patterns that indicate roles."""
        # FIXME: not sure what these mean
        return {
            "Admin": ["admin", "authority", "owner", "governance"],
            "Signer": ["signer", "payer", "fee_payer"],
            "User": ["user", "caller", "sender"],
        }

    def get_guard_patterns(self) -> List[str]:
        """Return common guard patterns."""
        return [
            "assert!",
            "assert_eq!",
            "debug_assert!",
            "debug_assert_eq!",
        ]

    def get_time_source_patterns(self) -> List[str]:
        """Rust-specific time source patterns."""
        return [
            "systemtime::now",
            "instant::now",
            "clock::get",
            "clock.unix_timestamp",
            "clock.slot",
            "solana_program::clock",
            "chrono::utc::now",
            "chrono::local::now",
            "std::time::",
            "tokio::time::",
        ]

    def get_reset_patterns(self) -> List[str]:
        """Rust-specific reset patterns."""
        return [
            "= false",
            "= true",
            "= 0",
            "= None",
            "Default::default()",
            ".clear()",
            "reset",
            "= Vec::new()",
            "= HashMap::new()",
        ]

    # =========================================================================
    # Lens-based invariant generation
    # =========================================================================

    def get_lens_definitions(self) -> List[LensDefinition]:
        """Rust-specific lens definitions for security analysis."""
        return [
            LensDefinition(
                name="memory_safety",
                description="Unsafe blocks, raw pointers, use-after-free, buffer overflows",
                invariant_types=["ACCESS", "OTHER"],
                prompt_template="""
## MEMORY SAFETY LENS - Rust

Focus on unsafe code and memory safety issues.

### Unsafe Blocks - CRITICAL
For EACH `unsafe` block:
- Check for raw pointer dereferences
- Check for FFI boundary safety
- Verify invariants maintained across unsafe boundaries
- Generate ACCESS invariant: "Unsafe block in X maintains Y invariant"

### Integer Overflow/Underflow
For EACH arithmetic operation:
- Check for unchecked arithmetic in critical paths
- Verify checked_add/checked_sub/checked_mul usage where needed
- Check for wrapping behavior assumptions
- Generate invariant: "Arithmetic in X cannot overflow"

### Buffer/Slice Safety
For EACH slice/array access:
- Check for unchecked indexing ([] vs .get())
- Verify bounds checking before access
- Check for panic paths in production code
""",
                checklist=[
                    "All unsafe blocks justified and sound",
                    "No unchecked arithmetic in value calculations",
                    "Slice access uses bounds checking",
                    "FFI boundaries validate inputs",
                ],
            ),
            LensDefinition(
                name="access_control",
                description="Authorization checks, signer verification, account validation",
                invariant_types=["ACCESS", "VALUE_FLOW"],
                prompt_template="""
## ACCESS CONTROL LENS - Rust

Focus on authorization and access control patterns.

### Signer/Authority Verification
For EACH public function that modifies state:
- Verify caller/signer is checked
- Check for missing authority validation
- Verify account ownership checks
- Generate ACCESS invariant: "Function X requires signer Y"

### Account Validation (Solana/Anchor)
For EACH account parameter:
- Verify account owner is checked
- Check for account type confusion
- Verify seeds/PDA derivation
- Check for reinitialization attacks

### Privilege Escalation
Check for patterns where:
- User can set themselves as admin/authority
- Missing checks allow unauthorized state changes
- Token/asset transfers lack proper authorization
""",
                checklist=[
                    "All state-modifying functions check authority",
                    "Account ownership validated",
                    "No privilege escalation paths",
                    "PDA seeds are deterministic and validated",
                ],
            ),
            LensDefinition(
                name="concurrency",
                description="Data races, deadlocks, async safety, shared state",
                invariant_types=["ORDERING", "LIVENESS"],
                prompt_template="""
## CONCURRENCY LENS - Rust

Focus on concurrency and async safety issues.

### Data Races
For EACH shared mutable state:
- Verify proper synchronization (Mutex, RwLock, Atomic)
- Check for lock ordering to prevent deadlocks
- Verify Arc usage for shared ownership
- Generate ORDERING invariant: "Shared state X properly synchronized"

### Async Safety
For EACH async function:
- Check for holding locks across .await points
- Verify cancellation safety
- Check for async deadlock patterns
- Generate LIVENESS invariant: "Async function X is cancellation-safe"

### Unsafe Sync/Send
For EACH manual Sync/Send impl:
- Verify soundness of implementation
- Check for interior mutability issues
""",
                checklist=[
                    "No data races on shared state",
                    "No locks held across await points",
                    "Lock ordering prevents deadlocks",
                    "Manual Sync/Send impls are sound",
                ],
            ),
            LensDefinition(
                name="resource",
                description="Resource leaks, panic safety, error handling",
                invariant_types=["LIVENESS", "OTHER"],
                prompt_template="""
## RESOURCE LENS - Rust

Focus on resource management and error handling.

### Error Handling
For EACH Result/Option usage:
- Check for unwrap() in production code
- Verify error propagation is correct
- Check for swallowed errors (let _ = ...)
- Generate invariant: "Errors in X are properly handled"

### Panic Safety
For EACH function that can panic:
- Check for panic paths in critical code
- Verify catch_unwind usage if needed
- Check for panic in Drop implementations
- Generate LIVENESS invariant: "Function X cannot panic"

### Resource Cleanup
For EACH resource acquisition:
- Verify Drop implementation if needed
- Check for resource leaks on error paths
- Verify file/socket/connection cleanup
""",
                checklist=[
                    "No unwrap() in production paths",
                    "Errors propagated or handled explicitly",
                    "No panics in Drop implementations",
                    "Resources cleaned up on all paths",
                ],
            ),
        ]

    def get_function_metadata_extractors(self) -> Dict[str, Callable]:
        """
        Rust-specific metadata extractors.

        These are called for each function to populate metadata used for bucketing.
        """

        def extract_is_pub(node: Node, graph: "DependencyGraph") -> bool:
            """Check if function is public."""
            visibility = node.meta.get("visibility", "")
            return visibility.startswith("pub")

        def extract_is_async(node: Node, graph: "DependencyGraph") -> bool:
            """Check if function is async."""
            return node.meta.get("is_async", False)

        def extract_is_unsafe(node: Node, graph: "DependencyGraph") -> bool:
            """Check if function is unsafe."""
            return node.meta.get("is_unsafe", False)

        def extract_has_unsafe_block(node: Node, graph: "DependencyGraph") -> bool:
            """Check if function contains unsafe blocks."""
            code = node.meta.get("source_code", "")
            return "unsafe {" in code or "unsafe{" in code

        def extract_returns_result(node: Node, graph: "DependencyGraph") -> bool:
            """Check if function returns Result type."""
            return_type = node.meta.get("return_type", "")
            return "Result" in return_type

        def extract_uses_unwrap(node: Node, graph: "DependencyGraph") -> bool:
            """Check if function uses unwrap()."""
            code = node.meta.get("source_code", "")
            return ".unwrap()" in code or ".expect(" in code

        def extract_modifies_state(node: Node, graph: "DependencyGraph") -> bool:
            """Check if function takes &mut self or &mut references."""
            params = node.meta.get("parameters", [])
            return any("&mut" in str(p) for p in params) or node.meta.get("takes_mut_self", False)

        def extract_is_test(node: Node, graph: "DependencyGraph") -> bool:
            """Check if function is a test."""
            attributes = node.meta.get("attributes", [])
            return "test" in attributes or "tokio::test" in attributes

        def extract_has_arithmetic(node: Node, graph: "DependencyGraph") -> bool:
            """Check if function contains arithmetic operations."""
            code = node.meta.get("source_code", "")
            return any(op in code for op in [" + ", " - ", " * ", " / ", " % "])

        def _get_var_names_from_edges(
            node: Node, graph: "DependencyGraph", edge_kind: EdgeKind,
        ) -> List[str]:
            """Get variable names that a function reads/writes via graph edges."""
            var_names: List[str] = []
            try:
                var_ids = list(
                    graph.neighbors(node.id, edge_kinds={edge_kind}, direction="out")
                )
                for var_id in var_ids:
                    var_node = graph.node(var_id)
                    if var_node and var_node.name:
                        var_names.append(var_node.name)
            except (KeyError, AttributeError):
                pass
            return var_names

        def _get_source_code(node: Node) -> str:
            return node.meta.get("source_code", "")

        # Lifecycle/temporal extractors using base classifiers
        def extract_timerish_score(node: Node, graph: "DependencyGraph") -> int:
            """Score indicating how likely function is timer-related (0-10)."""
            read_vars = _get_var_names_from_edges(node, graph, EdgeKind.READS)
            write_vars = _get_var_names_from_edges(node, graph, EdgeKind.WRITES)
            code = _get_source_code(node)
            is_view = node.meta.get("visibility", "") == "" or "&self" in str(
                node.meta.get("parameters", [])
            )
            score, _ = self.classify_timerish(
                node.name, code, read_vars, write_vars, is_view
            )
            return score

        def extract_timerish_roles(node: Node, graph: "DependencyGraph") -> List[str]:
            """List of timer-related roles for this function."""
            read_vars = _get_var_names_from_edges(node, graph, EdgeKind.READS)
            write_vars = _get_var_names_from_edges(node, graph, EdgeKind.WRITES)
            code = _get_source_code(node)
            is_view = node.meta.get("visibility", "") == "" or "&self" in str(
                node.meta.get("parameters", [])
            )
            _, roles = self.classify_timerish(
                node.name, code, read_vars, write_vars, is_view
            )
            return roles

        def extract_economic_score(node: Node, graph: "DependencyGraph") -> int:
            """Score indicating how likely function is economic/value-flow related (0-10)."""
            read_vars = _get_var_names_from_edges(node, graph, EdgeKind.READS)
            write_vars = _get_var_names_from_edges(node, graph, EdgeKind.WRITES)
            code = _get_source_code(node)
            is_payable = "lamports" in code.lower() or "transfer" in code.lower()
            score, _ = self.classify_economic(
                node.name, code, read_vars, write_vars, is_payable
            )
            return score

        def extract_economic_roles(node: Node, graph: "DependencyGraph") -> List[str]:
            """List of economic-related roles for this function."""
            read_vars = _get_var_names_from_edges(node, graph, EdgeKind.READS)
            write_vars = _get_var_names_from_edges(node, graph, EdgeKind.WRITES)
            code = _get_source_code(node)
            is_payable = "lamports" in code.lower() or "transfer" in code.lower()
            _, roles = self.classify_economic(
                node.name, code, read_vars, write_vars, is_payable
            )
            return roles

        return {
            "is_pub": extract_is_pub,
            "is_async": extract_is_async,
            "is_unsafe": extract_is_unsafe,
            "has_unsafe_block": extract_has_unsafe_block,
            "returns_result": extract_returns_result,
            "uses_unwrap": extract_uses_unwrap,
            "modifies_state": extract_modifies_state,
            "is_test": extract_is_test,
            "has_arithmetic": extract_has_arithmetic,
            # Lifecycle/temporal extractors
            "timerish_score": extract_timerish_score,
            "timerish_roles": extract_timerish_roles,
            # Economic extractors
            "economic_score": extract_economic_score,
            "economic_roles": extract_economic_roles,
        }
