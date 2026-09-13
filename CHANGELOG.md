# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Two project-specific conventions:

- **Result changes are called out explicitly.** A change that moves a measured number is a
  research event, not a refactor, and is listed under `Changed` with the old and new value.
- **Withdrawn claims are recorded, not deleted.** If a previously reported result stops being
  supported, it stays in the changelog as a withdrawal.

---

## [Unreleased]

### Added

- Repository scaffolding: packaging (`pyproject.toml`), `Makefile`, pre-commit hooks, Apache-2.0
  licence, security policy, code of conduct, citation metadata.
- `uniqkache.cache` — policy-agnostic K/V storage with a `KVCache` facade exposing
  `append` / `get` / `evict` / `clear` / `compress` / `decompress` / `offload` / `prefetch` /
  `stats`, plus `CacheConfig` byte accounting and `PolicyState` signal plumbing.
- `uniqkache.policies` — `full_cache`, `sliding_window`, `lru`, `attention_based`,
  `token_importance`, `adaptive`, with a registry, aliases, and shared signal helpers.
- `uniqkache.compression` — asymmetric int8 KV quantisation with KIVI's granularity
  (keys per-channel, values per-token).
- `uniqkache.offload` and `uniqkache.prefetch` — tier planning, accounting, and prefetch
  policies. Bandwidth is deliberately left unset rather than guessed.
- `uniqkache.controllers` — an action-selection prototype with a documented rule chain.
- `uniqkache.models` — a zero-download synthetic dense-attention transformer (`tiny`, `small`,
  `medium`) and a Hugging Face backend.
- `uniqkache.runtime` — an explicit prefill + decode loop with synchronised per-step timing.
- `uniqkache.metrics` — memory, latency and quality measurement; the `BenchmarkRecord` schema;
  `validate_record` integrity enforcement; JSONL/CSV/Markdown reporting.
- `uniqkache.bench` — `RunSpec`, `ExperimentConfig`, the runner, and the `uniqkache.bench` CLI
  with `--sweep` retention sweeps.
- Tests: unit, integration (incremental decode equals a single-shot forward pass) and
  regression suites. 291 tests, no GPU and no downloads required.
- Documentation: `README.md`, `CONTRIBUTING.md`, `docs/architecture.md`, `docs/research.md`
  (including a Failed Experiments section), `docs/benchmarks.md`, `benchmarks/README.md`,
  `baselines/README.md`.
- Contributor infrastructure: six issue templates encoding the
  Issue → Hypothesis → Implementation → Baseline → Experiment → Benchmark → PR → Review →
  Merge → Research result pipeline, a pull request template, and a CI workflow.
- `experiments/configs/correctness.json` — non-evicting budgets must reproduce the full-cache
  result exactly.
- `experiments/configs/sweep_policies.json` — cross-policy comparison at a fixed 25% budget.
- `experiments/scripts/check_ordering_effect.py` — tests whether a retention sweep's timing
  trend follows the token budget or the run position.
- `examples/quickstart.py` — runnable demonstration including the warmup the methodology
  requires.

### Changed

- **`bytes_for_tokens` parameter renamed `num_tokens` → `tokens_per_layer`.** The method takes
  a per-layer token count, while `CacheStats.total_tokens` is summed across layers; passing one
  where the other was meant over-counted by `num_layers`. The unit is now in the parameter name
  and pinned by a test.
- **`BenchmarkRecord` gained `generated_tokens`.** `tpot_ms` and `tokens_per_second` are both
  derived from the decode step count, so a record reporting them without the count could not be
  checked. `validate_record` now flags TPOT with a missing or zero token count.

### Withdrawn

- **Retention-sweep latency results.** The initial sweep reported a monotonic TTFT decrease as
  retention fell (43.2 → 33.9 ms) and it was briefly presented as a possible speedup. It did
  not reproduce: repeated runs varied more than the effect, and one fixed budget measured
  88.8 ms in one pass and 58.5 ms in another. The latency columns were withdrawn from the
  README, `docs/benchmarks.md` and `benchmarks/README.md`. Memory accounting and quality were
  unaffected — both are deterministic. See `docs/research.md`, F10.

### Fixed

Recorded in `docs/research.md` under Failed Experiments, each with a regression test.

- Quantisation gather confused the reduction axis with the granularity it produces, raising
  `IndexError` when evicting a compressed layer. Key/value axes were also transposed.
- Asymmetric quantisation stored a clamped zero-point but dequantised with the unclamped value,
  silently degrading reconstruction error by 7×.
- Perplexity saturated at `inf` because `nn.Embedding`'s default initialisation produced
  logits too large for float32 `exp`.
- The benchmark warmup did not exercise the eviction path, so the first bounded run absorbed
  kernel-compilation cost and reported a TTFT around 8× too high.
- `ExperimentConfig.from_dict` rejected `resolved_capacity`, which its own `to_dict` emits.
- The prefetch registry did not register its default policy, and registering it in place would
  have created an import cycle.
- `--sweep` validated a budget-less single spec before expanding, rejecting valid sweeps.
- `.gitignore`'s bare `models/` pattern matched `src/uniqkache/models/`, silently excluding the
  model backends from the repository. Ignore patterns for cache directories are now anchored to
  the repository root.
- `metrics.report` globbed `*.json`, which matched the `.config.json` files the runner writes
  beside every result, so `make results-table` failed on any directory the runner had populated.

---

## [0.0.1] — unreleased

Initial pre-alpha. The framework, its tests and its documentation. **No adaptive method in this
repository has been validated on a real model**, and no result in it should be quoted as a
finding about long-context inference.

[Unreleased]: https://github.com/uniqkache/UniqKache/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/uniqkache/UniqKache/releases/tag/v0.0.1
