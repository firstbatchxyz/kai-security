# Help command (default)
.PHONY: help
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  1. help                - Show this help message"
	@echo "  2. install             - Install ALL dependencies (uv + Python packages + Foundry)"
	@echo "  3. run                 - Run all agents (finder → setup → generator)"
	@echo "  4. test                - Run pytest tests"
	@echo ""
	@echo "Agent Execution Targets:"
	@echo "  4. run-finder-only     - Run only the finder agent"
	@echo "  5. run-setup-only      - Run only the setup agent"
	@echo "  6. run-generator-only  - Run only the generator agent"
	@echo "  7. run-skip-setup      - Run finder and generator (skip setup)"
	@echo ""
	@echo "Benchmark Targets:"
	@echo "  8. extract-metrics     - Extract and analyze metrics from benchmark results"
	@echo "  9. analyse-costs       - Comprehensive cost analysis with OpenRouter data"
	@echo ""
	@echo "Exploit Analysis Targets:"
	@echo "  10. combine-exploits   - Combine all exploits from conversation files (requires REPO_SLUG)"
	@echo "  11. extract-verified   - Extract verified exploits from exploit validation conversations (optional REPO_SLUG)"
	@echo "  12. generator-report   - Generate comprehensive report for generator agent (optional REPO_SLUG)"
	@echo ""
	@echo "Quick Start:"
	@echo "  make install"
	@echo ""
	@echo "For more options, run: uv run run_scaffold.py --help"

# Complete installation: uv, Python dependencies, and Foundry
.PHONY: install
install:
	@bash scripts/install.sh
	@if [ -f "$$HOME/.foundry/bin/forge" ] && ! command -v forge > /dev/null 2>&1; then \
		export PATH="$$HOME/.foundry/bin:$$PATH"; \
		echo "════════════════════════════════════════════════════════════════"; \
		echo "  🚀 Ready! To use Foundry in THIS shell, run:"; \
		echo "════════════════════════════════════════════════════════════════"; \
		echo ""; \
		echo "  export PATH=\"\$$HOME/.foundry/bin:\$$PATH\""; \
		echo ""; \
	fi

.PHONY: run
run:
	uv run run_scaffold.py

.PHONY: test
test:
	uv run pytest -v

.PHONY: typecheck
typecheck:
	uv run ty check

.PHONY: run-finder-only run-setup-only run-generator-only run-skip-setup
run-finder-only:
	uv run run_scaffold.py --finder-only
run-setup-only:
	uv run run_scaffold.py --setup-only
run-generator-only:
	uv run run_scaffold.py --generator-only
run-skip-setup:
	uv run run_scaffold.py --skip-setup

.PHONY: extract-metrics
extract-metrics:
	cd benchmark && uv run extract_metrics.py

.PHONY: analyse-costs
analyse-costs:
	cd benchmark && uv run analyse_costs.py

.PHONY: combine-exploits
combine-exploits:
	@if [ -z "$(REPO_SLUG)" ]; then \
		echo "Error: REPO_SLUG is not set"; \
		echo "Usage: make combine-exploits REPO_SLUG=2025-09-monad-60078b9e [COPY_TO_REPO=1]"; \
		echo "Example: make combine-exploits REPO_SLUG=2025-09-monad-60078b9e"; \
		echo "         make combine-exploits REPO_SLUG=2025-09-monad-60078b9e COPY_TO_REPO=1"; \
		exit 1; \
	fi
	@if [ -n "$(COPY_TO_REPO)" ]; then \
		uv run combine_exploits.py $(REPO_SLUG) --copy-to-repo; \
	else \
		uv run combine_exploits.py $(REPO_SLUG); \
	fi

.PHONY: extract-verified
extract-verified:
	@if [ -n "$(REPO_SLUG)" ]; then \
		uv run scripts/extract_verified_exploits.py $(REPO_SLUG); \
	else \
		uv run scripts/extract_verified_exploits.py; \
	fi

.PHONY: generator-report
generator-report:
	@if [ -n "$(REPO_SLUG)" ]; then \
		uv run scripts/generate_generator_report.py $(REPO_SLUG); \
	else \
		uv run scripts/generate_generator_report.py; \
	fi