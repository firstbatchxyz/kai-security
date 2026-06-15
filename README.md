![kai-security banner](kai-security.png)

# kai-security

Automated vulnerability discovery, verification, and patching using recursive language models.

Kai runs a multi-stage pipeline: a **setup agent** prepares and builds the target project, then an **exploit agent** orchestrates sub-agents (analysis, verification, critique, research, patching) to find and confirm vulnerabilities with working proof-of-concept exploits.

Built on [ra](src/ra/), a recursive language model framework where LLMs write code that launches other LLMs.

## Quickstart

```bash
git clone https://github.com/firstbatchxyz/kai-security.git
cd kai-security
uv sync
cp .env.example .env          # add OPENROUTER_API_KEY (or OPENAI_API_KEY)

# Audit the bundled, intentionally-vulnerable example target
uv run kai-security audit --repo-path examples/vulnerable-vault --verbose

# Explore the findings + the agent's reasoning in your browser...
uv run kai-security view output/state/<run_id> --open
# ...or print a Markdown report (or a styled HTML one)
uv run kai-security report output/state/<run_id>
```

`<run_id>` is printed during the run (the directory created under
`output/state/`). Point `--repo-path` at any local checkout you're authorized
to test. See [`examples/`](examples/) for more, and the [full CLI](#command-line-interface)
and [Usage](#usage) below for every option.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone kai-security
git clone https://github.com/firstbatchxyz/kai-security.git
cd kai-security

# Install dependencies
uv sync

# Copy and fill in API keys
cp .env.example .env
```

`uv sync` installs the `kai-security` command (the distribution is published as
`kai-security`; the command and import package are `kai`).

Common developer commands are available through `make`:

```bash
make test
make lint
make typecheck
make run REPO_PATH=/path/to/target
```

## Command-line interface

```bash
# Audit a repository you're authorized to test (setup → exploit pipeline)
uv run kai-security audit --repo-path /path/to/target --verbose

# Open a finished run as an interactive HTML report (findings + agent trace)
uv run kai-security view output/state/<run_id> --open

# Render a run's findings — Markdown to stdout, or a styled HTML document
uv run kai-security report output/state/<run_id>
uv run kai-security report output/state/<run_id> --format html -o report.html
```

`kai-security audit` is the friendly alias for the full pipeline; `kai-security pipeline` and
`kai-security agent` expose the complete interface documented under [Usage](#usage)
(equivalently `uv run python -m kai.main ...`). Run `kai-security <command> -h` for
per-command options.

## Examples

The [`examples/`](examples/) directory has small, self-contained,
intentionally-vulnerable targets you can audit end to end without a private
repo or large spend — start with
[`vulnerable-vault`](examples/vulnerable-vault/) (a Solidity vault with a
reentrancy and an unchecked-transfer bug, plus a ready-made threat context).

### API keys

| Key | Required | Used by |
|---|---|---|
| `OPENROUTER_API_KEY` | Required when `KAI_BACKEND=openrouter` | LLM calls through OpenRouter |
| `OPENAI_API_KEY` | Required when `KAI_BACKEND=openai` | Direct OpenAI LLM calls (`OPEN_AI_API_KEY` is also accepted as an alias) |
| `JINA_API_KEY` | Optional | Web search and URL reading (researcher agent) |

When using OpenRouter, Kai sends attribution headers by default:
`HTTP-Referer=https://kai.dria.co/` and
`X-OpenRouter-Title=kai-security`, plus categories
`cli-agent,programming-app`. OpenRouter derives the app icon from the favicon
of `OPENROUTER_APP_URL`; point that variable at a public project page with a
favicon if you want a custom image. Override categories with
`OPENROUTER_APP_CATEGORIES`.

### Model configuration

OpenRouter is the default backend. To use direct OpenAI instead, set:

```bash
KAI_BACKEND=openai
OPENAI_API_KEY=...
```

You can also override one agent at a time with `KAI_<AGENT>_BACKEND`, for example `KAI_VERIFIER_BACKEND=openai`.

Each agent's model can be overridden via environment variables. For OpenRouter, use provider-prefixed IDs such as `anthropic/claude-opus-4.5`; for direct OpenAI, use OpenAI model IDs such as `gpt-5.5`.

| Variable | OpenRouter default | OpenAI default | Agent |
|---|---|---|---|
| `KAI_ROOT_MODEL` | `anthropic/claude-opus-4.5` | `gpt-5.5` | Root exploit orchestrator |
| `KAI_ANALYZER_MODEL` | `minimax/minimax-m2.5` | `gpt-5.5` | Vulnerability analysis |
| `KAI_VERIFIER_MODEL` | `openai/gpt-5.2` | `gpt-5.5` | Proof-of-concept verification |
| `KAI_FIXER_MODEL` | `openai/gpt-5.2` | `gpt-5.4` | Patch generation |
| `KAI_CRITIC_MODEL` | `anthropic/claude-opus-4.5` | `gpt-5.4` | Adversarial viability assessment |
| `KAI_RESEARCHER_MODEL` | `minimax/minimax-m2.5` | `gpt-5.4` | Web research |
| `KAI_SETUP_MODEL` | `minimax/minimax-m2.5` | `gpt-5.4` | Project setup |
| `KAI_POC_AUDITOR_MODEL` | `openai/gpt-4.1-mini` | `gpt-5.4` | PoC soundness audit |
| `KAI_CHAIN_MODEL` | `anthropic/claude-opus-4.5` | `gpt-5.4` | Multi-step chain assembly |
| `KAI_PATCH_ASSEMBLER_MODEL` | `openai/gpt-4.1` | `gpt-5.4` | Iterative patch assembly |

Before deploying a new model, run the REPL compliance test to verify it can follow the interaction pattern:

```bash
uv run --with pytest --with python-dotenv -- pytest tests/test_model_repl.py -v
```

### Iteration budgets

Each agent's iteration limit can be tuned independently:

| Variable | Default | Agent |
|---|---|---|
| `KAI_ROOT_ITERS` | `45` | Root exploit orchestrator |
| `KAI_ANALYZER_ITERS` | `30` | Vulnerability analysis |
| `KAI_VERIFIER_ITERS` | `30` | PoC verification |
| `KAI_FIXER_ITERS` | `25` | Patch generation |
| `KAI_CRITIC_ITERS` | `10` | Adversarial viability assessment |
| `KAI_RESEARCHER_ITERS` | `15` | Web research |
| `KAI_SETUP_ITERS` | `30` | Project setup |
| `KAI_POC_AUDITOR_ITERS` | `5` | PoC soundness audit |
| `KAI_CHAIN_ITERS` | `20` | Multi-step chain assembly |
| `KAI_PATCH_ASSEMBLER_ITERS` | `15` | Iterative patch assembly |

### Timeouts

| Variable | Default | Controls |
|---|---|---|
| `KAI_EXEC_TIMEOUT` | `600` | REPL code execution (seconds) |
| `KAI_SOCKET_TIMEOUT` | `300` | LLM request socket (seconds) |
| `KAI_SHELL_TIMEOUT` | `300` | Shell command subprocess (seconds) |

## Usage

### Analyze A Cloned Repository

Kai works against a local checkout of the target you are authorized to test. Keep the target checkout separate from the `kai-security` checkout:

```bash
# Clone the target project somewhere outside kai-security
git clone https://github.com/example/target-project.git /tmp/target-project

# Run kai-security from this repo
cd /path/to/kai-security
uv run python -m kai.main pipeline --repo-path /tmp/target-project --verbose
```

The setup agent copies the target into an internal workspace, installs dependencies, builds it when it can infer the build command, and writes `output/recipe.json`. The exploit agent then uses that recipe to analyze, verify, and optionally patch findings. The original target checkout is the input; generated state and results stay under `output/` unless you override the paths.

### Full pipeline (setup + exploit)

Point Kai at a local repository checkout. The setup agent prepares an internal workspace, installs dependencies, builds it when possible, and produces a workspace recipe. The exploit agent then analyzes the codebase for vulnerabilities.

```bash
# From a local repo path
uv run python -m kai.main pipeline --repo-path /path/to/target --verbose

# Equivalent Makefile entry point
make run REPO_PATH=/path/to/target ARGS="--verbose"

# With logging to file
uv run python -m kai.main pipeline --repo-path /path/to/target --verbose --log-file run.log
```

### Verbose Mode And Rollout Export

`--verbose` turns on a Rich console trace for the agent loop. It shows each iteration, emitted REPL code, execution output, sub-agent spawn calls, nested `llm_query()` calls, final answers, timing, and usage summaries. It does not change the analysis; it only makes the run observable.

Use `--log-file PATH` with `--verbose` to save the same human-readable trace:

```bash
uv run python -m kai.main pipeline --repo-path /path/to/target --verbose --log-file output/run.log
```

Use `--save-rollouts` when you want machine-readable rollout histories for later inspection or evaluation:

```bash
uv run python -m kai.main pipeline --repo-path /path/to/target --save-rollouts
```

Rollouts are JSONL files under `output/state/<run_id>/rollouts/<agent>.jsonl` by default. Each line is one `metadata`, `iteration`, or `result` entry. To enable rollout export without a CLI flag, set `KAI_SAVE_ROLLOUTS=1`; to limit export to specific agents, set `KAI_ROLLOUT_AGENTS=exploit,analyzer,verifier` using agent names such as `setup`, `exploit`, `analyzer`, `verifier`, `critic`, `researcher`, `fixer`, `chain_assembler`, `patch_assembler`, or `poc_auditor`.

Rollout export uses the state manager, so leave state tracking enabled. If you pass `--no-state`, Kai still writes the final result JSON but does not write rollout files.

### Output

Results are always saved to disk as JSON. By default they go to `output/run_<timestamp>.json`. Use `--output` / `-o` to choose a custom path:

```bash
# Default: writes to output/run_20250101T120000Z.json
uv run python -m kai.main pipeline --repo-path /path/to/target

# Custom path
uv run python -m kai.main pipeline --repo-path /path/to/target -o results/my_run.json
```

The JSON file contains:

```json
{
  "model": "...",
  "execution_time": 123.4,
  "usage": { "model_usage_summaries": { "...": { } } },
  "result": [ ]
}
```

### Iterative re-verification

When the exploit agent finds and patches showstopper bugs, candidates that were rejected as "unreachable" or part of a multi-bug chain are automatically re-verified against the patched codebase. This happens within a single run — no manual re-launch is needed. Disable with `--no-iterative`.

### Extra instructions

Pass free-text guidance to steer the exploit agent:

```bash
uv run python -m kai.main pipeline --recipe recipe.json --instructions "Focus on economic invariants and fee arithmetic"
```

### Threat context

A threat context file tells Kai **who** can interact with the target, **what** trust boundaries exist, and **what** operational constraints apply. This dramatically improves finding quality — without it, the agent may flag admin-only operations as exploits or report economically infeasible attacks as critical.

```bash
uv run python -m kai.main pipeline --repo-path /path/to/target --threat-context threat_context.yaml
```

The file is YAML or JSON with these fields:

```yaml
# Required — what kind of project is this?
deployment_type: smart-contract  # smart-contract | web-app | cli-tool | library | ...
environment: on-chain            # on-chain | server | local | cloud | ...

# Who can interact with the system and how trusted are they?
access_roles:
  - name: anyone
    trust: none          # none | low | medium | high
    description: "Permissionless caller — any EOA or contract"
  - name: admin
    trust: high
    description: "Multisig owner with upgrade authority"

# Where are the trust boundaries?
boundaries:
  - "User input → contract storage (validation boundary)"
  - "Cross-chain message relay (L1 ↔ L2)"
  - "Proxy upgrade boundary (owner-only)"

# Operational constraints the agent should respect
known_constraints:
  - "Max realistic single-tx value: 10000 ETH (total supply ~120M)"
  - "Deployment-only functions: initialize, setup"
  - "Rate limiters cap per-period volume to 1000 tokens"
  - "Upgradeable proxies — owner can pause within 1 hour"
```

#### Field reference

| Field | Type | Required | Description |
|---|---|---|---|
| `deployment_type` | string | Yes | Project kind (`smart-contract`, `web-app`, `cli-tool`, `library`, etc.) |
| `environment` | string | No | Runtime environment (`on-chain`, `server`, `local`, `cloud`) |
| `access_roles` | list | No | Actors that interact with the system. Each has `name`, `trust` level (`none`/`low`/`medium`/`high`), and `description` |
| `boundaries` | list | No | Trust boundaries where privilege transitions occur |
| `known_constraints` | list | No | Operational limits, lifecycle constraints, economic bounds |

#### How it affects analysis

- **access_roles**: The verifier checks whether an exploit requires a high-trust actor. If so, it downgrades the finding from `active_exploit` to `trust_assumption_violation` — a real issue, but not exploitable by an unprivileged attacker.
- **boundaries**: Guides the analyzer toward the most security-critical code paths.
- **known_constraints**: The verifier checks PoC preconditions against these. For example, if a constraint says "max realistic single-tx value: 10000 ETH" and a PoC requires 79M ETH, the finding is reclassified as `theoretical_bounds`. Functions listed as deployment-only are reclassified as `deployment_hazard`.

#### Finding categories

| Category | Meaning | Gets fix? |
|---|---|---|
| `active_exploit` | Exploitable by an unprivileged attacker at runtime | Yes |
| `trust_assumption_violation` | Requires a trusted actor to misbehave | No (verify-only) |
| `deployment_hazard` | Only exploitable during initial deployment | No (verify-only) |
| `theoretical_bounds` | Economically or physically infeasible | No (verify-only) |
| `admin_misconfiguration` | Requires admin to misconfigure the system | No (verify-only) |
| `upgrade_hygiene` | Related to upgrade/migration safety | No (verify-only) |

### Skip setup (use a saved recipe)

If you already have a workspace recipe from a previous setup run:

```bash
uv run python -m kai.main pipeline --recipe recipe.json --verbose
```

### Run a single agent

```bash
# Run the setup agent alone
uv run python -m kai.main agent setup --input '{"repo_path": "/path/to/target", "master_dir": "/tmp/master"}'

# Run the exploit agent alone (needs a recipe injected separately)
uv run python -m kai.main agent exploit --input '{"master_path": "/tmp/master"}' --verbose
```

### CLI options

| Flag | Mode | Description |
|---|---|---|
| `--output PATH`, `-o` | both | Save result JSON to PATH (default: `output/run_<timestamp>.json`) |
| `--verbose` | both | Rich console output showing each iteration, REPL block, sub-call, timing, and usage summary |
| `--log-file PATH` | both | Save verbose output to a file |
| `--log-structured` | both | Emit JSON logs for log aggregation |
| `--save-rollouts` | both | Save per-agent JSONL rollout histories under the state directory |
| `--threat-context PATH` | both | Load a YAML/JSON threat context file |
| `--instructions TEXT` | pipeline | Extra instructions for the exploit agent |
| `--state-dir PATH` | pipeline | Directory for state storage and rollout export (default: `output/state`) |
| `--no-state` | pipeline | Disable state tracking and rollout export |
| `--skip-fixer` | pipeline | Analyze and verify findings without running fixer agents |
| `--no-iterative` | pipeline | Disable iterative re-verification of unreachable rejects |
| `--backend NAME` | agent | Override LLM backend |
| `--model NAME` | agent | Override model name |
| `--max-iterations N` | agent | Override iteration budget |

### Execution Environments

Kai currently ships two REPL environments:

| Environment | Status | Notes |
|---|---|---|
| `local` | Default, actively used | Runs agent-generated Python in a local workspace. This is the path currently exercised most heavily. |
| `docker` | Available, secondary | Runs agent-generated Python inside a Docker container with a host HTTP proxy for LLM/tool calls. Requires Docker and may pull `python:3.11-slim` on first use. |

The registered environment surface is intentionally limited to these two choices.

## Responsible use

Kai is intended for authorized security research on repositories and systems
you are allowed to test. Do not use it to attack, degrade, or exploit systems
without permission.

The default local REPL is a developer convenience, not a hard security
boundary. Run untrusted targets in disposable containers, virtual machines, or
isolated CI workers.

See [SECURITY.md](SECURITY.md) for vulnerability reporting and disclosure
guidance.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md)
before opening a pull request, and run:

```bash
make check
```

## Supported Languages

The dependency graph indexes source files using [tree-sitter](https://tree-sitter.github.io/) grammars. The following languages are supported:

| Language | Extensions |
|---|---|
| Solidity | `.sol` |
| Rust | `.rs` |
| Python | `.py` |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` |
| TypeScript | `.ts`, `.tsx`, `.mts`, `.cts` |
| Go | `.go` |
| C | `.c`, `.h` |

## Architecture

```
pipeline --repo-path /path/to/target
    |
    v
[Setup Agent]  — prepares workspace, installs deps, builds, produces recipe
    |
    v
[Exploit Agent] (root RLM, depth 0)
    |-- dep_* tools (dependency graph navigation)
    |-- llm_query (single-shot LLM calls)
    |-- spawn_analyzer(...)    -> [Analyzer Agent]    (depth 1, workspace)
    |-- spawn_verifier(...)    -> [Verifier Agent]    (depth 1, workspace)
    |-- spawn_critic(...)      -> [Critic Agent]      (depth 1)
    |-- spawn_researcher(...)  -> [Researcher Agent]  (depth 1, web tools)
    |-- spawn_fixer(...)       -> [Fixer Agent]       (depth 1, workspace)
```

Each sub-agent is a full RLM with its own REPL, iteration budget, and `llm_query` access. The root orchestrator decides how to partition work and which agents to spawn — there is no fixed pipeline.

## Development

```bash
# Run tests
make test

# Format
uv run ruff format src tests scripts

# Lint
make lint

# Type check
make typecheck
```

## Benchmarking

Kai ships an optional harness for scoring it against external security
benchmarks (CyberGym, BountyBench, EVMBench) and for running fleets of audits
in parallel. It drives `kai` as a subprocess and lives entirely in
[`evaluation/`](evaluation/) — see [`evaluation/README.md`](evaluation/README.md).
Most users don't need it; it's for measuring and improving Kai itself.

## Related Work

Kai uses ideas from the Recursive Language Models paper. To cite that
underlying research, use:

```bibtex
@misc{zhang2026recursivelanguagemodels,
      title={Recursive Language Models},
      author={Alex L. Zhang and Tim Kraska and Omar Khattab},
      year={2026},
      eprint={2512.24601},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24601},
}
```

## License

Kai is available under the [MIT License](LICENSE).
