# Kai Adapter System

Kai uses a three-layer adapter pattern to support multiple smart contract frameworks and programming languages. Each adapter type handles a different concern:

| Adapter Type          | Location                         | Purpose                                                          |
| --------------------- | -------------------------------- | ---------------------------------------------------------------- |
| **Tool Adapter**      | `kai/utils/tool_adapters/`       | Compile code, run tests, parse outputs                           |
| **Workspace Adapter** | `kai/utils/workspace/`           | Provision agent workspaces with correct project structure        |
| **Domain Adapter**    | `kai/utils/dependency/adapters/` | Semantic understanding (entrypoints, state vars, access control) |
| **Builder**           | `kai/utils/dependency/builders/` | Parse source code into dependency graphs (tree-sitter based)     |

```
┌─────────────────────────────────────────────────────────────────┐
│                         StateAgent                               │
├─────────────────────────────────────────────────────────────────┤
│  Tools (write_and_compile, run_test)  ←── ToolAdapter           │
│  Workspace provisioning               ←── WorkspaceAdapter       │
│  Graph queries (is_entrypoint, etc.)  ←── DomainAdapter          │
│  Code parsing                         ←── Builder (tree-sitter)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Supported Frameworks & Languages

| Framework      | Language   | Tool Adapter | Workspace Adapter | Domain Adapter | Builder          |
| -------------- | ---------- | ------------ | ----------------- | -------------- | ---------------- |
| **Foundry**    | Solidity   | ✅           | ✅                | ✅ (Solidity)  | ✅ (Slither)     |
| **Cargo**      | Rust       | ✅           | ✅                | ❌             | ❌               |
| **CMake**      | C/C++      | ✅           | ✅                | ❌             | ❌               |
| **Python**     | Python     | ✅ (uv)      | ✅                | ✅             | ✅ (tree-sitter) |
| **JavaScript** | JavaScript | ✅           | ✅                | ✅             | ✅ (tree-sitter) |
| **C**          | C          | ✅           | ✅                | ✅             | ✅ (tree-sitter) |

---

## 1. Tool Adapters

**Purpose**: Framework-specific compilation and test execution.

**Location**: `kai/utils/tool_adapters/`

### Base Class

```python
# kai/utils/tool_adapters/base.py
class ToolAdapter(ABC):
    @property
    def framework_name(self) -> str: ...      # "foundry", "python", "cargo"
    @property
    def language(self) -> str: ...            # "solidity", "python", "rust"

    def find_binary() -> str: ...             # Find compiler/test runner
    def compile(workspace_path, timeout) -> CompileResult: ...
    def run_test(workspace_path, match_contract, match_test, ..., framework_kwargs=None) -> TestResult: ...

    def get_test_file_extension(self) -> str: ...    # ".t.sol", ".py", ".rs"
    def get_source_file_extension(self) -> str: ...  # ".sol", ".py", ".rs"
    def normalize_test_path(file_path, workspace) -> Path: ...
    def get_allowed_patch_directories(self) -> List[str]: ...
    def get_tool_description(tool_name) -> Optional[str]: ...  # LLM tool descriptions
    def get_poc_guidance(self) -> str: ...    # Framework-specific PoC writing guidance
```

### Current Implementations

| Adapter                 | File            | Test Runner          | Key Features                       |
| ----------------------- | --------------- | -------------------- | ---------------------------------- |
| `FoundryToolAdapter`    | `foundry.py`    | `forge test`         | Solidity, fuzz seeds, gas tracking |
| `PythonToolAdapter`     | `python.py`     | `uv run pytest`      | uv integration, venv fallback      |
| `JavaScriptToolAdapter` | `javascript.py` | `npm/yarn/pnpm test` | Package manager detection          |
| `CToolAdapter`          | `c.py`          | `make test`          | CMake/Make/Meson detection         |
| `CargoToolAdapter`      | `cargo.py`      | `cargo test`         | Rust, feature flags                |
| `CMakeToolAdapter`      | `cmake.py`      | `ctest`              | C/C++, build dir management        |

### Python Tool Adapter (uv Integration)

The Python adapter uses `uv` as the primary tool when available:

```python
# Commands used:
# Syntax check: uv run python -m py_compile <file>
# Run tests:    uv run --with pytest pytest
# Install deps: uv sync (pyproject.toml) or uv pip install -r requirements.txt

# Falls back to venv/pip if uv not available
```

### Adding a New Tool Adapter

```python
# kai/utils/tool_adapters/myframework.py
from kai.utils.tool_adapters.base import ToolAdapter, CompileResult, TestResult

class MyFrameworkToolAdapter(ToolAdapter):
    @property
    def framework_name(self) -> str:
        return "myframework"

    @property
    def language(self) -> str:
        return "mylang"

    def find_binary(self) -> str:
        # Find the framework binary
        ...

    def compile(self, workspace_path: Path, timeout: int = 120) -> CompileResult:
        # Run compilation
        ...

    def run_test(self, workspace_path: Path, ...) -> TestResult:
        # Run tests
        ...

    def get_poc_guidance(self) -> str:
        return """## PoC Format: MyFramework
Write test files in tests/poc/.
- Use your framework's test conventions
- Import REAL modules from the codebase
- Use assertions to prove the exploit
- A PASSING test = valid exploit"""

    # ... implement other required methods
```

**Register in `__init__.py`:**

```python
# kai/utils/tool_adapters/__init__.py
from kai.utils.tool_adapters.myframework import MyFrameworkToolAdapter

_ADAPTERS = {
    "foundry": FoundryToolAdapter,
    "myframework": MyFrameworkToolAdapter,  # Add here
    # ...
}
```

---

## 2. Workspace Adapters

**Purpose**: Provision isolated workspaces for agents to write and compile PoC tests.

**Location**: `kai/utils/workspace/`

### Base Class

```python
# kai/utils/workspace/base.py
class WorkspaceAdapter(ABC):
    @property
    def framework_name(self) -> str: ...

    def provision_lightweight(workspace, master, master_context, logger) -> str:
        """Create workspace with remappings to master (no file copy)."""
        ...

    def provision_full(workspace, master, master_context, preset, logger) -> str:
        """Create workspace by copying files from master."""
        ...

    def detect_remappings(master: Path) -> str:
        """Generate import remappings pointing to master."""
        ...

    def infer_src_path(master: Path) -> Path:
        """Infer source directory (src/, contracts/, etc.)."""
        ...

    def get_runtime_writable_paths(project_root, master_context) -> List[Path]:
        """Return directories that must remain writable (build caches, etc.)."""
        ...
```

### Current Implementations

| Adapter                      | Framework  | Key Directories                            |
| ---------------------------- | ---------- | ------------------------------------------ |
| `FoundryWorkspaceAdapter`    | Foundry    | `out/`, `cache/`, `lib/`                   |
| `PythonWorkspaceAdapter`     | Python     | `.venv/`, `__pycache__/`, `.pytest_cache/` |
| `JavaScriptWorkspaceAdapter` | JavaScript | `node_modules/`, `dist/`, `.cache/`        |
| `CWorkspaceAdapter`          | C          | `build/`, `cmake-build-*/`, `out/`         |
| `CargoWorkspaceAdapter`      | Cargo      | `target/`                                  |
| `CMakeWorkspaceAdapter`      | CMake      | `build/`, `cmake-build-*`                  |

---

## 3. Domain Adapters

**Purpose**: Language-specific semantic understanding for security analysis.

**Location**: `kai/utils/dependency/adapters/`

### Base Class

```python
# kai/utils/dependency/adapters/base.py
class DomainAdapter(ABC):
    @property
    def name(self) -> str: ...

    def get_domain_mapping(self) -> Dict[str, str]:
        """Map generic domain terms to NodeKind's ('Contract' -> CONTAINER)."""
        ...

    def is_public_entrypoint(self, node: Node) -> bool:
        """Is this callable by external attackers?"""
        ...

    def is_state_variable(self, node: Node) -> bool:
        """Does this represent persistent state?"""
        ...

    def is_test_file(self, file_path: str) -> bool: ...
    def is_library_file(self, file_path: str) -> bool: ...

    def resolve_symbol(self, name, context_graph, scope=None) -> List[str]:
        """Fuzzy-match user input to node IDs."""
        ...
```

### Current Implementations

| Adapter             | Language   | Library Detection                      | Entrypoint Detection             |
| ------------------- | ---------- | -------------------------------------- | -------------------------------- |
| `SolidityAdapter`   | Solidity   | `lib/`, `node_modules/`, `forge-std/`  | `public`/`external` functions    |
| `PythonAdapter`     | Python     | `site-packages/`, `venv/`              | Public functions (no `_` prefix) |
| `JavaScriptAdapter` | JavaScript | `node_modules/`                        | `export` functions               |
| `CAdapter`          | C          | `/usr/include/`, `/usr/local/include/` | Non-static functions             |

### C Adapter Special Features

The C adapter includes security-focused features:

```python
# Get dangerous functions that may have vulnerabilities
dangerous = adapter.get_dangerous_functions()
# Returns: ["strcpy", "strcat", "sprintf", "gets", "scanf", ...]

# Get safer alternatives
alternatives = adapter.get_safe_alternatives()
# Returns: {"strcpy": "strncpy or strlcpy", "sprintf": "snprintf", ...}
```

---

## 4. Builders (Tree-sitter)

**Purpose**: Parse source code into dependency graphs for analysis.

**Location**: `kai/utils/dependency/builders/`

### Base Class

```python
# kai/utils/dependency/builders/treesitter_base.py
class TreeSitterBuilder(BaseBuilder):
    @property
    def language(self) -> str: ...

    @property
    def file_extensions(self) -> List[str]: ...

    def build(self, project_root: Path) -> DependencyGraph:
        """Parse all source files and build a dependency graph."""
        ...
```

### Current Implementations

| Builder             | Language   | Extensions   | Extracts                                       |
| ------------------- | ---------- | ------------ | ---------------------------------------------- |
| `SolidityBuilder`   | Solidity   | `.sol`       | Contracts, functions, state vars (via Slither) |
| `PythonBuilder`     | Python     | `.py`        | Classes, functions, methods, globals           |
| `JavaScriptBuilder` | JavaScript | `.js`, `.ts` | Classes, functions, arrow functions            |
| `CBuilder`          | C          | `.c`, `.h`   | Structs, enums, functions, globals             |

### NodeKind Mapping

```python
NodeKind Mapping:
    FILE       - Source files (all languages)
    CONTAINER  - Contract (Sol), Class (Py/JS), Struct (C)
    UNIT       - Function (all languages), Method (Py/JS)
    INTERFACE  - Modifier (Sol), Decorator (Py)
    VARIABLE   - StateVar (Sol), Global (Py/C)
    TYPE_DEF   - Struct, Enum, Typedef (all languages)
    EVENT      - Events (Sol), Logs (all languages)
    EXTERNAL   - Unresolved external references
```

---

## Framework Detection

The system auto-detects frameworks via config files:

```python
# kai/dispatcher/workspace.py
def _detect_framework(self, master: Path, master_context=None) -> str:
    if master_context and master_context.frameworks:
        return master_context.frameworks[0].lower()

    # Smart contract frameworks
    if (master / "foundry.toml").exists():
        return "foundry"

    # Build system detection
    if (master / "Cargo.toml").exists():
        return "cargo"
    if (master / "CMakeLists.txt").exists():
        return "cmake"

    # Language detection
    if (master / "pyproject.toml").exists() or (master / "setup.py").exists():
        return "python"
    if (master / "package.json").exists():
        return "javascript"
    if (master / "Makefile").exists():
        return "c"

    return "foundry"  # Default
```

---

## PoC Guidance

Each tool adapter provides framework-specific guidance for writing proof-of-concept tests:

```python
# Example: Get guidance for Python
adapter = get_tool_adapter("python")
guidance = adapter.get_poc_guidance()
# Returns:
# ## PoC Format: Python/pytest (via uv)
# Write Python test files in tests/poc/.
# - Tests run via `uv run --with pytest pytest`
# - Use pytest with descriptive test function names (test_*)
# - Import REAL modules from the codebase (don't create mocks)
# - Use assertions to prove the exploit: assert, pytest.raises()
# - A PASSING test with assertions proving vulnerability = valid exploit
# ...
```

This guidance is injected into agent prompts via `{{poc_guidance}}` template variable.

---

## Checklist for Adding a New Framework

### Minimum Viable Support

- [ ] **Tool Adapter**: `kai/utils/tool_adapters/<framework>.py`
  - [ ] `compile()` - Run compiler
  - [ ] `run_test()` - Run tests
  - [ ] `get_poc_guidance()` - PoC writing guidance
  - [ ] `get_tool_description()` - LLM-friendly descriptions
  - [ ] Register in `__init__.py`

- [ ] **Workspace Adapter**: `kai/utils/workspace/<framework>.py`
  - [ ] `provision_lightweight()` - Minimal workspace
  - [ ] `provision_full()` - Full workspace with file copying
  - [ ] `get_runtime_writable_paths()` - Build/cache directories
  - [ ] Register in `__init__.py`

### Full Support (recommended)

- [ ] **Domain Adapter**: `kai/utils/dependency/adapters/<domain>.py`
  - [ ] `is_public_entrypoint()` - Attack surface detection
  - [ ] `is_library_file()` - Filter non-protocol code
  - [ ] `is_test_file()` - Filter test files
  - [ ] Register in `__init__.py`

- [ ] **Builder**: `kai/utils/dependency/builders/<language>.py`
  - [ ] Extend `TreeSitterBuilder`
  - [ ] Implement language-specific parsing
  - [ ] Register in `__init__.py`

- [ ] **Tests**:
  - [ ] Unit tests for each adapter in `tests/test_<adapter_type>_adapters.py`
  - [ ] Integration test with sample project

---

## Common Patterns

### Getting Adapters from Agent Context

```python
# In agent code, get adapters from the tool adapter registry
from kai.utils.tool_adapters import get_tool_adapter
from kai.utils.workspace import get_workspace_adapter
from kai.utils.dependency.adapters import get_adapter
from kai.utils.dependency.builders import get_builder

# Get by framework name
tool_adapter = get_tool_adapter("python")
workspace_adapter = get_workspace_adapter("python")

# Get by language (for domain adapters and builders)
domain_adapter = get_adapter("python")
builder = get_builder("python")
```

### Singleton Pattern

Adapters are instantiated fresh on each call. If you need caching:

```python
from functools import lru_cache
from kai.utils.tool_adapters import get_tool_adapter

@lru_cache(maxsize=16)
def get_cached_adapter(framework: str):
    return get_tool_adapter(framework)
```

### Framework-Agnostic Code

```python
def compile_project(workspace: Path, framework: str) -> CompileResult:
    """Compile any supported project type."""
    adapter = get_tool_adapter(framework)
    return adapter.compile(workspace)

def run_tests(workspace: Path, framework: str, test_filter: str = None) -> TestResult:
    """Run tests for any supported project type."""
    adapter = get_tool_adapter(framework)
    return adapter.run_test(workspace, match_test=test_filter)
```

---

## Caveats & Tips

### Tool Adapters

- **Binary detection**: Always handle `FileNotFoundError` from `find_binary()` - the tool may not be installed
- **Timeouts**: Use reasonable timeouts; compilation can be slow for large projects
- **Output truncation**: `raw_output` is truncated to prevent memory issues; check `errors` list for parsed errors
- **Framework kwargs**: Use `framework_kwargs` dict for framework-specific options (e.g., `fuzz_seed` for Foundry)

### Workspace Adapters

- **Lightweight vs Full**: Prefer `provision_lightweight()` when possible - it's faster and uses less disk
- **Writable paths**: Always check `get_runtime_writable_paths()` before making directories read-only
- **Symlinks**: Lightweight provisioning uses symlinks; ensure the master directory remains accessible

### Domain Adapters

- **Library detection**: `is_library_file()` uses path heuristics; may need tuning for non-standard layouts
- **Entrypoint detection**: Based on visibility modifiers; doesn't account for indirect calls
- **Symbol resolution**: `resolve_symbol()` returns multiple matches; caller must disambiguate

### Builders

- **Tree-sitter dependencies**: Requires `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-c` packages
- **Large projects**: Building graphs for large codebases can be memory-intensive
- **Slither for Solidity**: SolidityBuilder requires Slither; falls back to empty graph if unavailable

---

## Real-World Example: Adding Hardhat Support

Here's a complete example of adding Hardhat (an alternative Solidity framework):

### Tool Adapter

```python
# kai/utils/tool_adapters/hardhat.py
from kai.utils.tool_adapters.base import ToolAdapter, CompileResult, TestResult

class HardhatToolAdapter(ToolAdapter):
    @property
    def framework_name(self) -> str:
        return "hardhat"

    @property
    def language(self) -> str:
        return "solidity"

    def find_binary(self) -> str:
        # Check for npx (Hardhat is typically run via npx)
        npx = shutil.which("npx")
        if npx:
            return npx
        raise FileNotFoundError("npx not found - install Node.js")

    def compile(self, workspace_path: Path, timeout: int = 120) -> CompileResult:
        try:
            result = subprocess.run(
                ["npx", "hardhat", "compile"],
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return CompileResult(
                success=result.returncode == 0,
                errors=self.parse_compile_errors(result.stderr) if result.returncode != 0 else [],
                raw_output=result.stdout + result.stderr,
            )
        except subprocess.TimeoutExpired:
            return CompileResult(success=False, errors=["Compilation timed out"])

    def run_test(self, workspace_path: Path, match_test: str = None, **kwargs) -> TestResult:
        cmd = ["npx", "hardhat", "test"]
        if match_test:
            cmd.extend(["--grep", match_test])

        result = subprocess.run(cmd, cwd=str(workspace_path), capture_output=True, text=True)
        return TestResult(
            success=result.returncode == 0,
            raw_output=result.stdout + result.stderr,
        )

    def get_poc_guidance(self) -> str:
        return """## PoC Format: Hardhat/Solidity
Write Solidity test files in test/poc/.
- Tests run via `npx hardhat test`
- Use Mocha/Chai style: describe(), it(), expect()
- Import contracts with: const Contract = await ethers.getContractFactory("Name")
- A PASSING test with assertions proving vulnerability = valid exploit"""
```

### Workspace Adapter

```python
# kai/utils/workspace/hardhat.py
from kai.utils.workspace.base import WorkspaceAdapter

class HardhatWorkspaceAdapter(WorkspaceAdapter):
    @property
    def framework_name(self) -> str:
        return "hardhat"

    def get_runtime_writable_paths(self, project_root: Path, master_context=None) -> List[Path]:
        return [
            project_root / "artifacts",
            project_root / "cache",
            project_root / "node_modules",
            project_root / "typechain-types",
        ]

    def provision_lightweight(self, workspace: Path, master: Path, master_context, logger) -> str:
        # Symlink contracts and config from master
        (workspace / "contracts").symlink_to(master / "contracts")
        (workspace / "hardhat.config.js").symlink_to(master / "hardhat.config.js")
        (workspace / "node_modules").symlink_to(master / "node_modules")

        # Create test directory for PoCs
        (workspace / "test" / "poc").mkdir(parents=True, exist_ok=True)

        return f"Hardhat workspace provisioned at {workspace}"
```

---

## Testing Adapters

```python
# tests/test_tool_adapters.py
def test_python_tool_adapter():
    adapter = get_tool_adapter("python")
    assert adapter.framework_name == "python"
    assert adapter.language == "python"
    assert adapter.get_test_file_extension() == ".py"
    assert "pytest" in adapter.get_poc_guidance().lower()

def test_workspace_adapter():
    adapter = get_workspace_adapter("python")
    workspace = Path("/tmp/test_workspace")
    master = Path("/path/to/project")

    result = adapter.provision_lightweight(workspace, master, mock_context)
    assert (workspace / "tests").exists()

def test_domain_adapter():
    adapter = get_adapter("python")

    # Test library file detection
    assert adapter.is_library_file("/path/site-packages/requests/api.py") == True
    assert adapter.is_library_file("src/app.py") == False

def test_builder():
    builder = get_builder("python")
    assert builder.language == "python"
    assert ".py" in builder.file_extensions

    # Build graph from test project
    graph = builder.build(test_project_path)
    containers = list(graph.nodes(NodeKind.CONTAINER))
    assert len(containers) > 0
```

Run all adapter tests:

```bash
uv run --with pytest pytest tests/test_tool_adapters.py tests/test_workspace_adapters.py tests/test_domain_adapters.py tests/test_builders.py -v
```
