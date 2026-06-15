# Examples

Runnable targets for trying `kai` end to end — no private repos, minimal API
spend.

| Example | What it is | Highlights |
|---------|------------|------------|
| [`vulnerable-vault/`](vulnerable-vault/) | A tiny Solidity vault with two planted bugs | reentrancy + unchecked ERC-20 return; ships a `threat_context.yaml` |

Each example is **intentionally vulnerable** and is for authorized
demonstration only — do not deploy them.

Quick run (see each example's README for details):

```bash
uv run kai audit --repo-path examples/vulnerable-vault --verbose
uv run kai view output/state/<run_id> --open
```

Running an audit makes real LLM calls, so it needs an API key configured (see
the project [README](../README.md#api-keys)) and incurs some cost.
