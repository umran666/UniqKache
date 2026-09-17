# UniqKache developer entry points.
#
# Every target is intentionally thin: it shells out to a tool that is also
# documented in CONTRIBUTING.md, so CI and a human contributor run the exact
# same command.

PYTHON ?= python
# Note the colon: the synthetic presets are namespaced (`synthetic:tiny`), so
# `synthetic-tiny` is treated as a Hugging Face repository id and fails with a
# download error rather than running the built-in model.
MODEL  ?= synthetic:tiny
# 512 rather than a long-context default: quality is measured one token at a
# time, so cost scales with the context and a 4k+ default is minutes of compute
# per run on a laptop GPU. Override for a real long-context run on a real model.
CTX    ?= 512
POLICY ?= full_cache
BATCH  ?= 1
# An evicting policy must be given a budget: RunSpec refuses one without, because
# a bounded cache with no budget has nothing to decide what to drop.
KEEP   ?= 0.25
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
typecheck: ## Static type check with mypy
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
		--keep-ratio $(KEEP) \
		--batch-size $(BATCH) \
		--output-dir $(OUTDIR)

.PHONY: bench-sweep
bench-sweep: ## Run the configured policy x context sweep
	$(PYTHON) -m uniqkache.bench \
		--config experiments/configs/sweep_policies.json \
		--output-dir $(OUTDIR)

.PHONY: bench-correctness
bench-correctness: ## Verify eviction policies preserve full-cache output where expected
	$(PYTHON) -m uniqkache.bench \
		--config experiments/configs/correctness.json \
		--output-dir $(OUTDIR)

.PHONY: results-table
results-table: ## Summarise collected results into a Markdown table
	$(PYTHON) -m uniqkache.metrics.report --input $(OUTDIR) --format markdown

.PHONY: clean
clean: ## Remove build artefacts and caches
	rm -rf build dist .pytest_cache .ruff_cache .coverage htmlcov
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +
