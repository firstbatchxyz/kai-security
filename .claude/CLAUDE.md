# Kai - Exploit Agent

## Project Overview

**Kai** (v0.1.0) is an automated smart contract vulnerability discovery and fix generation system by First Batch XYZ. It uses AI agents with a dispatcher-driven architecture to systematically explore codebases, identify security exploits, verify findings, and generate patches.

- **Language:** Python 3.12+
- **Build System:** uv (fast Python package installer/builder)
- **LLM Integration:** OpenAI API + OpenRouter (API-agnostic)
- **Code Analysis:** Slither (Solidity), Tree-sitter (multi-language)
- **Smart Contract Framework:** Foundry (primary), plus Cargo, CMake, Node
- **Testing:** pytest, pytest-asyncio

## Architecture & Structure

### Pipeline (3 Phases)

```
BOOT (Preprocessing):  Setup → DependencyGraph → Profiler → Actors → Invariants
RUN LOOP (Missions):   State/Quant/Blackbox/Http agents → Verifier
POST-LOOP (Fixes):     FixerAgent → Patches
```

### Directory Layout

```
src/kai/
├── agents/                 # 12 specialized agent types + tools
│   ├── agent_types/        # SetupAgent, ProfilerAgent, StateAgent, QuantAgent,
│   │                       # VerifierAgent, FixerAgent, BlackboxAgent, HttpAgent,
│   │                       # GamifiedAgent, InvariantSynthesizerAgent, BucketingAgent,
│   │                       # WorkspaceValidationAgent
│   ├── base.py             # BaseAgent abstract class
│   ├── engine.py           # Sandboxed code execution engine
│   ├── settings.py         # Agent configuration & defaults
│   └── tools/              # Modular tool definitions (20 files)
│       ├── shared.py       # Context management helpers
│       ├── graph_tools.py  # Dependency graph queries
│       ├── file_tools.py   # File reading/listing
│       ├── build_tools.py  # Compilation, test execution
│       ├── workspace_tools.py # Workspace management
│       ├── http_tools.py   # HTTP request handling
│       └── *_tools.py      # Per-agent specialized tools
├── dispatcher/             # Mission control & orchestration
│   ├── core.py             # Main Dispatcher class
│   ├── planner.py          # Mission planning from invariants
│   ├── workspace.py        # Workspace provisioning
│   └── agent_factories.py  # Agent creation + scope_paths derivation
├── processes/              # Preprocessing pipeline stages
│   ├── envsetup.py         # Clone/compile repository
│   ├── profiler.py         # Extract protocol understanding
│   ├── invariants.py       # Generate security invariants
│   ├── actors.py           # Actor role analysis
│   ├── blackbox.py         # Blackbox exploration process
│   ├── verifier.py         # Exploit verification process
│   ├── adapter_chooser.py  # Framework auto-detection
│   ├── invariant_synthesizer.py # Observations → Invariants
│   └── workspace_validation.py  # Import recipe discovery
├── utils/
│   ├── dependency/         # Dependency graph analysis
│   │   ├── adapters/       # Language-specific semantics (Solidity, Python, JS, C)
│   │   ├── builders/       # Tree-sitter-based code parsers
│   │   └── analysis/       # Graph query engine
│   ├── tool_adapters/      # Framework-specific tool execution (Foundry, Python, JS/TS+Bun, Cargo, CMake)
│   ├── workspace/          # Workspace provisioning adapters
│   └── state_managers/     # Persistence backends (local JSON)
├── schemas.py              # Pydantic models for all data structures
├── inference.py            # LLM API integration
├── exceptions.py           # Custom exception types
├── evaluation/             # Benchmark evaluation suite
│   ├── evaluator.py        # Benchmark evaluation
│   ├── runner.py           # Test execution pipeline
│   ├── post_hoc_deduplicator.py # Exploit deduplication
│   ├── metrics_collector.py # Metrics tracking
│   └── cli.py              # Evaluation CLI
└── prompts/                # Agent prompt templates (15 files)

tests/                      # 19 test files
├── test_dependency.py      # DependencyGraph loading, queries
├── test_domain_adapters.py # Language adapter tests
├── test_tool_adapters.py   # Tool execution tests
├── test_builders.py        # Code parser tests
├── evaluation/             # Benchmark evaluation suite
└── fixtures/               # Test data (JSON)
```

## Key Components

### Agents (12 types)
| Agent | Purpose |
|-------|---------|
| SetupAgent | Clone, detect framework, compile |
| ProfilerAgent | Extract protocol understanding |
| StateAgent | Find call sequences violating state invariants |
| QuantAgent | Find inputs violating math/solvency invariants |
| VerifierAgent | Validate PoC, assess severity |
| FixerAgent | Generate unified diffs fixing exploits |
| BlackboxAgent | Unguided exploration, emit Observations |
| HttpAgent | HTTP protocol testing |
| GamifiedAgent | Game-theory-based exploitation |
| InvariantSynthesizerAgent | Convert Observations → Invariants |
| BucketingAgent | Function categorization |
| WorkspaceValidationAgent | Import path discovery |

### 4-Layer Adapter System
1. **Tool Adapters** (9) - Compile, run tests, parse output (Foundry, Python, JS, TS+Bun, Cargo, CMake)
2. **Workspace Adapters** (8) - Provision isolated work directories
3. **Domain Adapters** (4) - Language-specific semantic understanding (Solidity, Python, JS, C)
4. **Builders** (5) - Parse source code into DependencyGraphs

### Core Data Models (schemas.py)
`MasterContext`, `Invariant`, `ExploitCandidate`, `Verdict`, `Fix`, `Observation`, `ActorMatrix`, `ProtocolManifesto`, `Mission`, `CampaignBrief`

### Target Anchoring System
- `scope_paths` on agents restricts file access to invariant-relevant files
- `principle` field on Invariants captures abstract vulnerability patterns
- `blocked_by_root_cause` / `blocking_invariant_id` on Verdicts for dependency tracking

## Development Notes

### Setup
```bash
make install          # Install all deps + Foundry
# OR manually:
uv sync --all-groups
```

### Environment Variables
```bash
OPENROUTER_API_KEY=sk-or-v1-...  # OR OPENAI_API_KEY
MONGO_URI=mongodb://...          # Optional
```

### Common Commands
```bash
make test             # Run tests
make typecheck        # Type checking (ruff/ty)
make format           # Format code (ruff)
pytest tests/ -v      # Run tests directly
python scripts/playground_dispatcher.py --repo-path ./path  # Run dispatcher
```

### Key Settings (src/kai/agents/settings.py)
- `DEFAULT_MAX_TURNS`: 32
- `VERIFIER_MAX_TURNS`: 16
- `VALIDATION_MAX_TURNS`: 8
- `INVARIANT_SYNTH_MAX_TURNS`: 8
- `MAX_CONCURRENT_AGENTS`: 2
- `TOOL_OUTPUT_MAX_LENGTH`: 50,000 chars
- `MAX_REQUESTS_PER_RUN`: 50 (HTTP tools rate limiting)
- Sandbox timeout: 3600s
- Role-specific model assignments per agent type

## Testing Strategy

- **Unit tests:** Graph operations, adapters, schema validation, utility functions
- **Integration tests:** Repo setup, agent E2E, process pipelines (require API keys)
- **Evaluation tests:** Benchmarking against known vulnerable repos
- **Framework:** pytest + pytest-asyncio (`anyio_backend = "asyncio"`)
- **Fixtures:** JSON files in `tests/fixtures/` (Ethena BBP public assets)
- **Config:** `pytest.ini` excludes `testbed/`, `output/`, `node_modules/`
