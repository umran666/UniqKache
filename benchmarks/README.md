# Benchmark suite

This directory holds the benchmark *catalogue*: what each benchmark measures, what it is
compared against, and what it does not establish. The runnable definitions live in
[`experiments/configs/`](../experiments/configs/) and the harness lives in
[`src/uniqkache/bench/`](../src/uniqkache/bench/).

Methodology and the record schema: [`docs/benchmarks.md`](../docs/benchmarks.md).

---

## Where things live

| Concern | Location |
| --- | --- |
| Benchmark harness, runner, CLI | `src/uniqkache/bench/` |
| Measurement primitives (memory, latency, quality) | `src/uniqkache/metrics/` |
| Declared experiments | `experiments/configs/*.json` |
| Raw results | `experiments/results/` (JSONL, CSV, config) |
| Reported findings | `docs/research.md` |
| Record schema | `docs/benchmarks.md#record-schema` |

A benchmark is a **config plus a description**. This directory contains only this README, on
purpose: there is deliberately no `benchmarks/latency/run.py`-style tree of one-off scripts
(an earlier draft of this document described subdirectories that were never populated and were
never tracked by git). A benchmark that exists only as a script cannot be reviewed, diffed, or
re-run from its own description. If you want a new benchmark, add a config and an entry below.
The same applies to `baselines/`, which likewise contains only its README.

---

## Catalogue

### Correctness

| Benchmark | Config | Measures | Pass condition |
| --- | --- | --- | --- |
| Non-evicting budget | `correctness.json` | Whether an evicting policy reproduces the full-cache result when it cannot evict | `cache_final_tokens` **and** `quality_value` both equal the `full_cache` reference row |
| Incremental vs single-shot | `tests/integration/test_incremental_matches_single_shot.py` | Whether the cache path is equivalent to a plain forward pass | max absolute difference ≤ 1e-4 (measured: 4.6e-5) |

The correctness benchmarks are gates, not measurements. They are the reason a later
performance number means anything: if incremental decode did not equal a single-shot forward
pass, every eviction result would be measuring a broken cache rather than a policy.

### Memory

| Benchmark | Measures | Notes |
| --- | --- | --- |
| Cache footprint vs budget | `cache_bytes_total` against `layers × tokens × kv_heads × head_dim × 2 × dtype_bytes` | Verified exact to the byte at every retention ratio. See `docs/research.md`, R3. |
| Peak device memory | `peak_memory_bytes` | `null` on CPU, never `0`. |
| Tier split | `cache_bytes_on_device` vs `cache_bytes_offloaded` | Reported separately; an offload frees nothing and costs bandwidth. |

### Latency

> **No latency result in this repository is currently supported.** On the reference device,
> run-to-run variation exceeded the between-policy differences, so the timing columns of the
> retention sweep were withdrawn. Re-measure and repeat before quoting any latency figure, and
> use `experiments/scripts/check_ordering_effect.py` to check whether a sweep's trend follows
> the budget or the measurement order. See `docs/research.md`, F10.

| Benchmark | Measures | Notes |
| --- | --- | --- |
| TTFT | Synchronised prefill wall-clock | Requires warmup covering **both** the plain and the eviction path; two measurement bugs came from this (F1, F2). |
| TPOT | Mean synchronised per-step decode time | `null` when nothing was decoded. |
| Percentiles | `latency_p50_ms`, `latency_p90_ms` over per-step samples | `null` when there are no samples. |
| Throughput | `tokens_per_second` over **generated** tokens | Prefill excluded so a long prompt cannot flatter the number. |

**Latency comparisons are only valid between policies under an identical store.** `append` is
O(T) per step (`torch.cat`), so absolute numbers are not throughput figures for any real
serving configuration.

### Quality

| Benchmark | Metric | Status |
| --- | --- | --- |
| Language modelling | `perplexity` | Implemented, wired into the runner, measured through a cache |
| Needle retrieval | `needle_retrieval` | Implemented and wired in: `--quality-metric needle_retrieval` (see `experiments/configs/needle_comparison.json`) |
| Exact match | `exact_match` | Implemented in `metrics/quality.py`, **not yet wired into a config** (the needle task uses exact match internally) |
| Task accuracy | — | **Not yet supported** |
| Summarisation / reasoning | — | **Not yet supported** |

Quality is mandatory. `validate_record` returns a problem for any record that reports latency
or memory without a quality metric, and the runner prints those problems to the console. A
record with such a warning may not be quoted as a result.

### Scaling

| Benchmark | Status |
| --- | --- |
| Context-length scaling | **Planned** — needs a real model |
| Batch-size scaling | **Planned** |
| Cross-device comparison | **Not supported** — no cross-device claim has been validated |

---

## Running

```bash
# the whole catalogue of declared experiments
make bench-sweep          # experiments/configs/sweep_policies.json
make bench-correctness    # experiments/configs/correctness.json

# the standard retention sweep for one policy
# `--max-new-tokens 2` is required to match the reference numbers below; see F14
python -m uniqkache.bench --model synthetic:tiny --context-length 192 \
    --policy sliding_window --sweep --max-new-tokens 2

# summarise whatever is in experiments/results
make results-table
```

---

## Reference numbers

`synthetic:tiny`, context 192, 2 generated tokens, batch 1, float32, CUDA, greedy, seed 0, on an
RTX 3050 Laptop (4 GiB, compute 8.6, CUDA 12.4, torch 2.6.0+cu124):

| Policy | Budget | Cache bytes | vs full | Perplexity |
| --- | --- | --- | --- | --- |
| `full_cache` | — | 395,264 | 1.000 | 515.57421875 |
| `sliding_window` | 144 | 294,912 | 0.746 | 514.56475830 |
| `sliding_window` | 96 | 196,608 | 0.497 | 514.48822021 |
| `sliding_window` | 48 | 98,304 | 0.249 | 515.14343262 |
| `sliding_window` | 19 | 38,912 | 0.098 | 527.70678711 |

Use these to check that your environment produces comparable numbers. They are **not findings**:

- The model has randomly-initialised weights, so perplexity is a diagnostic of cache behaviour
  and not a language-modelling result.
- The 25% row's perplexity (515.14) is marginally *lower* than the full cache's (515.57). On a
  random model that is noise, not an improvement.
- **No latency column is shown, deliberately.** Repeated runs of one fixed configuration varied
  more than the between-policy differences (the same budget measured 88.8 ms in one pass and
  58.5 ms in another), so the timing columns were withdrawn. Test your own hardware with
  `experiments/scripts/check_ordering_effect.py` before quoting a latency sweep.

See `docs/research.md`, R4 and F10, for the full reading.

---

## Adding a benchmark

1. Add a config to `experiments/configs/` with a `description` stating the purpose and the
   pass condition.
2. Include a `full_cache` reference row at the same context length, precision and seed.
3. Keep quality measurement on, or state in the description why quality does not apply.
4. Add a row to the catalogue above, including what the benchmark does **not** establish.
5. Run it on a clean git tree so `git_commit` is recorded and `git_dirty` is `false`.
