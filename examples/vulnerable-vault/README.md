# vulnerable-vault

A tiny, self-contained, **intentionally vulnerable** Solidity project — a
target you can point `kai audit` at to see the whole pipeline run end to end
without a private repo or a large API spend.

> ⚠️ Intentionally insecure. Do not deploy. For authorized demonstration only.

## Planted bugs

| # | Bug | Location |
|---|-----|----------|
| 1 | **Reentrancy** — the caller's balance is zeroed *after* the external call, no guard (a re-entrant caller drains the contract) | `src/Vault.sol` · `withdraw()` |
| 2 | **Unchecked ERC-20 return** — `transfer()`'s boolean result is ignored | `src/Vault.sol` · `sweepToken()` |

## Run it

```bash
# From the kai-security repo root
uv run kai audit --repo-path examples/vulnerable-vault \
  --threat-context examples/vulnerable-vault/threat_context.yaml --verbose
```

Then look at the results:

```bash
# Interactive HTML (findings + the agent's reasoning trace)
uv run kai view output/state/<run_id> --open

# Or a Markdown report (stdout), or a styled HTML document
uv run kai report output/state/<run_id>
uv run kai report output/state/<run_id> --format html -o report.html
```

`<run_id>` is printed during the run and is the directory name under
`output/state/`.

## What a real run produced

This isn't hypothetical — here's an actual result. With the reentrancy bug,
Kai built a Foundry proof-of-concept, confirmed the drain, and proposed a fix:

```
| CVSS | Severity | Finding                                   | Location            | Status              |
| 9.8  | critical | Reentrancy in withdraw() (CEI violation)  | Vault.sol:withdraw  | verified_and_fixed ✓ |
```

with the correct Check-Effects-Interaction patch (move the balance update
*before* the external call):

```diff
 function withdraw() external {
     uint256 amount = balances[msg.sender];
     require(amount > 0, "nothing to withdraw");
+    balances[msg.sender] = 0;
     (bool ok, ) = msg.sender.call{value: amount}("");
     require(ok, "transfer failed");
-    balances[msg.sender] = 0;
 }
```

> **Kai is an agentic system, so runs are not deterministic.** Which bugs get
> confirmed, their CVSS scores, and the exact wording vary by run and by the
> models you configure. In one run Kai confirmed the reentrancy as Critical
> (above); in another it confirmed the unchecked-return in `sweepToken()` as
> Medium instead. It also reasons about *exploitability*, not just patterns —
> given a `withdraw()` that used a checked `-= amount`, it correctly **disproved**
> a textbook-looking reentrancy because the subtraction underflows and reverts
> under Solidity 0.8.x. Treat the output as a strong signal to investigate, not
> a fixed checklist.
