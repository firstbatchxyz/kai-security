"""
Tests for domain adapters (Python, JavaScript, C, Rust).

Tests the DomainAdapter interface implementations for each language.
"""

import pytest

from kai.utils.dependency.adapters import (
    get_adapter,
    ADAPTER_REGISTRY,
    SolidityAdapter,
    PythonAdapter,
    JavaScriptAdapter,
    CAdapter,
    RustAdapter,
)


class TestDomainAdapterRegistry:
    """Tests for the domain adapter registry."""

    def test_registry_contains_all_adapters(self):
        """Registry should contain all adapters."""
        assert "solidity" in ADAPTER_REGISTRY
        assert "python" in ADAPTER_REGISTRY
        assert "javascript" in ADAPTER_REGISTRY
        assert "c" in ADAPTER_REGISTRY
        assert "rust" in ADAPTER_REGISTRY

    def test_get_adapter_solidity(self):
        """Should return SolidityAdapter for solidity."""
        adapter = get_adapter("solidity")
        assert isinstance(adapter, SolidityAdapter)

    def test_get_adapter_python(self):
        """Should return PythonAdapter for python."""
        adapter = get_adapter("python")
        assert isinstance(adapter, PythonAdapter)

    def test_get_adapter_javascript(self):
        """Should return JavaScriptAdapter for javascript."""
        adapter = get_adapter("javascript")
        assert isinstance(adapter, JavaScriptAdapter)

    def test_get_adapter_c(self):
        """Should return CAdapter for c."""
        adapter = get_adapter("c")
        assert isinstance(adapter, CAdapter)

    def test_get_adapter_rust(self):
        """Should return RustAdapter for rust."""
        adapter = get_adapter("rust")
        assert isinstance(adapter, RustAdapter)

    def test_get_adapter_rust_aliases(self):
        """Should return RustAdapter for rs and cargo aliases."""
        assert isinstance(get_adapter("rs"), RustAdapter)
        assert isinstance(get_adapter("cargo"), RustAdapter)

    def test_get_adapter_case_insensitive(self):
        """Should be case insensitive."""
        adapter_lower = get_adapter("python")
        adapter_upper = get_adapter("PYTHON")
        adapter_mixed = get_adapter("PyThOn")
        assert type(adapter_lower) is type(adapter_upper) is type(adapter_mixed)

    def test_get_adapter_invalid(self):
        """Should raise ValueError for unknown adapter."""
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter("unknown_adapter")


class TestPythonAdapter:
    """Tests for PythonAdapter."""

    @pytest.fixture
    def adapter(self):
        return PythonAdapter()

    def test_is_library_file(self, adapter: PythonAdapter):
        """Should identify library files."""
        # Full path with site-packages
        assert (
            adapter.is_library_file("/some/path/site-packages/requests/api.py") is True
        )
        assert (
            adapter.is_library_file("/venv/lib/python3.9/site-packages/flask/app.py")
            is True
        )
        # Non-library files
        assert adapter.is_library_file("myapp/views.py") is False
        assert adapter.is_library_file("src/app.py") is False


class TestJavaScriptAdapter:
    """Tests for JavaScriptAdapter."""

    @pytest.fixture
    def adapter(self):
        return JavaScriptAdapter()

    def test_is_library_file(self, adapter: JavaScriptAdapter):
        """Should identify library files."""
        assert adapter.is_library_file("node_modules/express/index.js") is True
        assert adapter.is_library_file("node_modules/@types/node/index.d.ts") is True
        assert adapter.is_library_file("src/app.js") is False


class TestCAdapter:
    """Tests for CAdapter."""

    @pytest.fixture
    def adapter(self):
        return CAdapter()

    def test_is_library_file(self, adapter: CAdapter):
        """Should identify library files."""
        assert adapter.is_library_file("/usr/include/stdio.h") is True
        assert adapter.is_library_file("/usr/local/include/openssl/ssl.h") is True
        assert adapter.is_library_file("src/main.c") is False

    def test_dangerous_functions(self, adapter: CAdapter):
        """Should provide list of dangerous functions."""
        dangerous = adapter.get_dangerous_functions()
        assert isinstance(dangerous, list)
        assert len(dangerous) > 0
        # Should include common dangerous functions
        has_dangerous = any(
            f in dangerous for f in ["strcpy", "gets", "sprintf", "strcat"]
        )
        assert has_dangerous, (
            f"Should have common dangerous functions, got: {dangerous}"
        )

    def test_safe_alternatives(self, adapter: CAdapter):
        """Should provide safe alternatives mapping."""
        alternatives = adapter.get_safe_alternatives()
        assert isinstance(alternatives, dict)
        assert len(alternatives) > 0
        # Should map dangerous functions to safer alternatives
        assert "strcpy" in alternatives
        assert (
            "strncpy" in alternatives["strcpy"] or "strlcpy" in alternatives["strcpy"]
        )
        assert "sprintf" in alternatives
        assert "snprintf" in alternatives["sprintf"]
        assert "gets" in alternatives
        assert "fgets" in alternatives["gets"]


class TestRustAdapter:
    """Tests for RustAdapter."""

    @pytest.fixture
    def adapter(self):
        return RustAdapter()

    def test_is_library_file(self, adapter: RustAdapter):
        """Should identify library files."""
        assert adapter.is_library_file("/project/target/debug/deps/foo.rs") is True
        assert adapter.is_library_file("/home/user/.cargo/registry/src/crate/lib.rs") is True
        assert adapter.is_library_file("/home/user/.cargo/git/checkouts/foo/lib.rs") is True
        assert adapter.is_library_file("/project/vendor/some_dep/lib.rs") is True
        assert adapter.is_library_file("/home/user/.rustup/toolchains/stable/lib.rs") is True
        # Non-library files
        assert adapter.is_library_file("src/main.rs") is False
        assert adapter.is_library_file("src/lib.rs") is False

    def test_is_test_file(self, adapter: RustAdapter):
        """Should identify test files."""
        assert adapter.is_test_file("tests/integration_test.rs") is True
        assert adapter.is_test_file("src/test/helpers.rs") is True
        assert adapter.is_test_file("test_utils.rs") is True
        assert adapter.is_test_file("src/foo_test.rs") is True
        assert adapter.is_test_file("src/benches/bench_main.rs") is True
        # Non-test files
        assert adapter.is_test_file("src/main.rs") is False
        assert adapter.is_test_file("src/lib.rs") is False

    def test_is_public_entrypoint(self, adapter: RustAdapter):
        """Should identify public entrypoints."""
        from kai.utils.dependency.models import Node, NodeKind

        # main() is always an entrypoint
        main_node = Node(id="main", kind=NodeKind.UNIT, name="main", meta={})
        assert adapter.is_public_entrypoint(main_node) is True

        # pub fn
        pub_node = Node(id="pub_fn", kind=NodeKind.UNIT, name="do_thing", meta={"visibility": "pub"})
        assert adapter.is_public_entrypoint(pub_node) is True

        # pub(crate) fn
        pub_crate_node = Node(id="pub_crate", kind=NodeKind.UNIT, name="internal", meta={"visibility": "pub(crate)"})
        assert adapter.is_public_entrypoint(pub_crate_node) is True

        # #[no_mangle]
        ffi_node = Node(id="ffi", kind=NodeKind.UNIT, name="ffi_func", meta={"attributes": ["no_mangle"]})
        assert adapter.is_public_entrypoint(ffi_node) is True

        # Solana handler
        solana_node = Node(id="solana", kind=NodeKind.UNIT, name="process_instruction", meta={})
        assert adapter.is_public_entrypoint(solana_node) is True

        # Private function (no visibility, no special name)
        private_node = Node(id="priv", kind=NodeKind.UNIT, name="helper", meta={})
        assert adapter.is_public_entrypoint(private_node) is False

        # Non-UNIT node should never be entrypoint
        struct_node = Node(id="s", kind=NodeKind.CONTAINER, name="MyStruct", meta={"visibility": "pub"})
        assert adapter.is_public_entrypoint(struct_node) is False


class TestDomainAdapterInterface:
    """Tests for DomainAdapter interface compliance."""

    @pytest.mark.parametrize("adapter_name", ["solidity", "python", "javascript", "c", "rust"])
    def test_adapter_has_is_library_file(self, adapter_name: str):
        """All adapters should have is_library_file method."""
        adapter = get_adapter(adapter_name)
        assert hasattr(adapter, "is_library_file")
        assert callable(adapter.is_library_file)

    @pytest.mark.parametrize("adapter_name", ["solidity", "python", "javascript", "c", "rust"])
    def test_is_library_file_returns_bool(self, adapter_name: str):
        """is_library_file should return bool."""
        adapter = get_adapter(adapter_name)
        result = adapter.is_library_file("test.py")
        assert isinstance(result, bool)
