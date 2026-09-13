# UniqKache developer entry points.
#
# Every target is intentionally thin: it shells out to a tool that is also
# documented in CONTRIBUTING.md, so CI and a human contributor run the exact
# same command.

PYTHON ?= python
MODEL  ?= synthetic-tiny
CTX    ?= 4096
POLICY ?= full_cache
BATCH  ?= 1
OUTDIR ?= experiments/results

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install the package in editable mode
	$(PYTHON) -m pip install -e .

.PHONY: install-dev
install-dev: ## Install with dev + hf extras
	$(PYTHON) -m pip install -e ".[dev,hf]"

.PHONY: test
test: ## Run the full test suite
	$(PYTHON) -m pytest

.PHONY: test-fast
test-fast: ## Run unit + regression tests only (no GPU, no downloads)
	$(PYTHON) -m pytest tests/unit tests/regression -q

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest --cov --cov-report=term-missing

.PHONY: lint
lint: ## Lint with ruff
	$(PYTHON) -m ruff check src tests benchmarks examples

.PHONY: format
format: ## Auto-format with ruff
	$(PYTHON) -m ruff format src tests benchmarks examples
	$(PYTHON) -m ruff check --fix src tests benchmarks examples

.PHONY: typecheck
typecheck: ## Static type check (requires mypy)
	$(PYTHON) -m mypy src/uniqkache

.PHONY: benchmark
benchmark: ## Run the reference benchmark (full cache vs sliding window)
	$(PYTHON) -m uniqkache.bench \
		--model $(MODEL) \
		--context-length $(CTX) \
		--policy full_cache \
		--batch-size $(BATCH) \
		--output-dir $(OUTDIR)
	$(PYTHON) -m uniqkache.bench \
		--model $(MODEL) \
		--context-length $(CTX) \
		--policy sliding_window \
		--batch-size $(BATCH) \
		--output-dir $(OUTDIR)

.PHONY: bench-sweep
bench-sweep: ## Run the configured policy x context sweep
	$(PYTHON) -m uniqkache.bench --config experiments/configs/sweep_policies.json

.PHONY: bench-correctness
bench-correctness: ## Verify eviction policies preserve full-cache output where expected
	$(PYTHON) -m uniqkache.bench --config experiments/configs/correctness.json

.PHONY: results-table
results-table: ## Summarise collected results into a Markdown table
	$(PYTHON) -m uniqkache.metrics.report --input $(OUTDIR) --format markdown

.PHONY: clean
clean: ## Remove build artefacts and caches
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
