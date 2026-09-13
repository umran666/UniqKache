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
  regression suites. 306 tests, no GPU and no downloads required.
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

- **Quality results for attention-based policies.** The quality pass did not record attention,
  so `attention_based`, `token_importance` and `adaptive` were scored on a degenerate selection
  rather than on the policy they implement. The bug is fixed; no committed result depended on
  the affected numbers, but any number copied out of a pre-fix run must be discarded. See
  `docs/research.md`, F13.
- **Retention-sweep latency results.** The initial sweep reported a monotonic TTFT decrease as
  retention fell (43.2 → 33.9 ms) and it was briefly presented as a possible speedup. It did
  not reproduce: repeated runs varied more than the effect, and one fixed budget measured
  88.8 ms in one pass and 58.5 ms in another. The latency columns were withdrawn from the
  README, `docs/benchmarks.md` and `benchmarks/README.md`. Memory accounting and quality were
  unaffected — both are deterministic. See `docs/research.md`, F10.

### Fixed

Recorded in `docs/research.md` under Failed Experiments. The defects that can recur silently
carry a regression test; the two that cannot — a documentation error (F14) and a warning-noise
fix — are covered by the checks they name instead.

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
- `make benchmark` used `synthetic-tiny`, which is parsed as a Hugging Face repository id rather
  than the built-in preset, and passed `sliding_window` with no budget, which `RunSpec` correctly
  refuses. Both commands are fixed and the target now runs.
- **The quality pass never recorded attention.** `metrics.quality.perplexity` discarded the
  attention weights returned by the model, so `cum_attention` stayed all-zero for the whole
  evaluation and every attention-based policy silently scored every token equally.
  `token_importance` collapsed onto its recency terms and `attention_based` degenerated to
  keeping the *oldest* tokens. Any quality number previously reported for `attention_based`,
  `token_importance` or `adaptive` is invalid — see Withdrawn. `perplexity` now mirrors the
  generation engine's signal plumbing, and a regression test asserts the diagnostic actually
  discriminates between retention rules.
- **A flaky timing assertion.** `test_timer_measures_elapsed_time` slept for 10 ms and asserted
  the timer reported at least 10 ms. `time.sleep` is a minimum hint, not a promise, and the test
  was measured failing at 8.39 ms. The body now busy-waits on `perf_counter`, so the asserted
  interval is guaranteed rather than hoped for. The first attempt at the fix still failed — at
  9.91 ms — because the deadline was taken before entering the timer, charging
  `Timer.__enter__`'s synchronisation against the wait. Same lesson as the withdrawn latency
  results, one level down: a timing assertion is only meaningful when the measurement is
  reproducible at the scale being asserted.
- `docs/**` in `per-file-ignores` was inert — `ruff check` does not read Markdown, so the ignore
  could never fire. Removed rather than left to imply a check that is not running. `ruff format`
  does rewrite Python inside `.md` code blocks, and was collapsing the intentional comment
  alignment in `docs/architecture.md`; Markdown is now excluded from the formatter, since
  reformatting prose that is never linted buys no correctness.
- **Result artifacts were written with the platform's line endings.** `.gitattributes` normalises
  them to LF, so on Windows every generated file disagreed with the index and git warned on each
  add — recurring noise on every result commit, which is how warnings stop being read. All three
  writers now specify `"\n"`, and the config file gained the trailing newline `json.dumps` omits.
  `write_results` had no test coverage at all before this, despite producing every committed
  artifact; it now has three.
- **The documented sweep command did not reproduce the reported sweep.** The committed run had
  been produced with `--max-new-tokens 2`; the CLI default is `8`, and the flag was not written
  down. The `full_cache` reference row moved from 193 tokens / 395,264 bytes to 199 / 407,552,
  and because the `vs full` column is measured against that row, the four bounded ratios shifted
  silently — `144/193 = 0.746` became `144/199 = 0.724` — while the bounded rows themselves
  stayed byte-identical. The decode length is now part of the stated workload and written into
  the command. See `docs/research.md`, F14.
- **`KVCache.evict()` counted an eviction when nothing was dropped.** The increment was
  unconditional, whereas `enforce_capacity` increments only when something was evicted. A no-op
  evict on an empty cache, or one whose selection kept every token, bumped `stats().evictions`
  and `state_dict()["evictions"]` — so a record could claim an eviction that removed zero tokens.
  Token accounting was unaffected; `evicted_tokens` was always correct. `evict()` now counts an
  eviction only when `dropped > 0`, matching `enforce_capacity`, with a regression test pinning
  the agreement between the two paths.

### Results

The committed artifacts in `experiments/results/` were regenerated. The first generation
predated both the git history and the `generated_tokens` field, so no record in it could name
the code that produced it; a number that cannot be tied to a commit cannot be checked, so it is
not a result. They move to `superseded/` for that reason alone.

**No value changed.** The correctness experiment reproduces 524 final tokens and perplexity
`502.059692` for all five policies, and the retention sweep reproduces every cache byte and
every perplexity value digit for digit. Only the records changed: they now carry `git_commit`,
`git_dirty` and `generated_tokens`, and the only integrity warning left is the intended one
about randomly-initialised weights.

The intermediate generations are kept in `superseded/`, which is what makes the reproducibility
claim checkable rather than asserted: three generations of the same sweep agree on every
perplexity value to all printed digits and disagree on every latency value.


---

## [0.0.1] — unreleased

Initial pre-alpha. The framework, its tests and its documentation. **No adaptive method in this
repository has been validated on a real model**, and no result in it should be quoted as a
finding about long-context inference.

[Unreleased]: https://github.com/umran666/UniqKache/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/umran666/UniqKache/releases/tag/v0.0.1
