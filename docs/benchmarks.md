# Benchmarks

How to run a benchmark, what gets measured, what gets written, and how to read the output
without drawing a conclusion the data does not support.

The governing rule:

> **A performance number without a quality number is not a result.** The runner enforces this
> mechanically and prints a warning when a record violates it.

---

## Running a benchmark

### A single declared run

```bash
python -m uniqkache.bench \
    --model synthetic:tiny \
    --context-length 192 \
    --policy full_cache \
    --batch-size 1
```

### The same workload under an evicting policy

```bash
python -m uniqkache.bench \
    --model synthetic:tiny \
    --context-length 192 \
    --policy sliding_window \
    --keep-ratio 0.25 \
    --batch-size 1
```

### The standard retention sweep

```bash
python -m uniqkache.bench \
    --model synthetic:tiny \
    --context-length 192 \
    --policy sliding_window \
    --sweep
```

`--sweep` expands to five runs at 100 / 75 / 50 / 25 / 10 % retention. The 100 % run is a
`full_cache` reference, and it records `attention_sinks = 0`, because a full cache has nothing
to protect and recording a non-zero sink count there would be a meaningless number.

### From a config file

```bash
python -m uniqkache.bench --config experiments/configs/sweep_policies.json
python -m uniqkache.bench --config experiments/configs/correctness.json
```

A config is the preferred form for anything you intend to report. It is diffable, reviewable,
and re-runnable without reconstructing the command line from memory.

### Inspecting instead of writing

```bash
python -m uniqkache.bench --markdown ...     # print a Markdown table to stdout
python -m uniqkache.bench --print-records ...# print full records as JSON
python -m uniqkache.bench --list-policies    # what is registered, and the aliases
```

---

## CLI reference

| Flag | Default | Meaning |
| --- | --- | --- |
| `--model` | `synthetic:tiny` | `synthetic:<preset>` uses the built-in model; anything else is an HF repo id. |
| `--context-length` | `1024` | Prompt length in tokens. |
| `--policy` | `full_cache` | Registered policy name or alias. |
| `--batch-size` | `1` | Batch size. |
| `--capacity` | — | Explicit per-layer token budget. Mutually exclusive with `--keep-ratio`. |
| `--keep-ratio` | — | Fraction of the context to retain, in `(0, 1]`. |
| `--attention-sinks` | `4` | Leading tokens protected from eviction. |
| `--precision` | `float32` | `float32`, `float16` or `bfloat16`. |
| `--device` | `auto` | `auto`, `cpu` or `cuda`. |
| `--seed` | `0` | Random seed. |
| `--max-new-tokens` | `8` | Decode steps. |
| `--compressor` | off | `int8` compresses the cache before generation. |
| `--no-quality` | off | Skip quality. Strongly discouraged; the record will be flagged. |
| `--quality-chunk-size` | `1` | Tokens per forward pass during quality evaluation. `1` reproduces true streaming decode. |
| `--output-dir` | `experiments/results` | Where results go. |
| `--config` | — | Run an experiment config instead of a single ad-hoc run. |
| `--sweep` | off | Expand into the 100/75/50/25/10 % retention sweep. |
| `--repo-path` | — | Repository root for git metadata. |

Exit codes: `0` success, `2` configuration error, `3` a run failed.

Policies: `full_cache`, `sliding_window`, `lru`, `attention_based`, `token_importance`,
`adaptive`. Aliases: `streamingllm`/`streaming_llm` → `sliding_window`, `h2o`/`heavy_hitter` →
`attention_based`, `full`/`none` → `full_cache`, `sliding` → `sliding_window`, `recency` → `lru`.

---

## What is measured

### Memory

| Field | Meaning |
| --- | --- |
| `cache_bytes_total` | Total KV footprint across all layers. |
| `cache_bytes_on_device` | The part resident on the compute device. |
| `cache_bytes_offloaded` | The part moved to another tier. |
| `cache_compression_ratio` | Representation change. `1.0` means uncompressed. |
| `cache_final_tokens` | Tokens held, **summed across layers**. Divide by the layer count for a per-layer figure. |
| `peak_memory_bytes` | Peak device memory during the run. `null` on CPU — never `0`. |

**`cache_bytes_on_device` and `cache_bytes_offloaded` are separate on purpose.** Reporting a
single "memory saved" number would let an offload, which frees nothing and costs bandwidth,
look identical to an eviction, which actually destroys information.

The accounting identity, verified exact against measurement (see `docs/research.md`, R3):

```
bytes = num_layers × tokens × num_kv_heads × head_dim × 2 (K and V) × element_size
```

For `synthetic:tiny`: 4 × tokens × 2 × 32 × 2 × 4 = **2048 bytes per token**.

### Latency

| Field | Meaning |
| --- | --- |
| `ttft_ms` | Time to first token. Wall-clock time of the prefill call, synchronised. |
| `tpot_ms` | Mean per-token decode time, synchronised per step. `null` when nothing was decoded. |
| `prefill_ms` / `decode_ms` / `total_ms` | Phase breakdown. |
| `latency_p50_ms` / `latency_p90_ms` | Percentiles over the per-step decode samples. |
| `tokens_per_second` | Decode-phase throughput over **generated** tokens. Prefill is excluded so a long prompt cannot flatter the number. |

`tpot_ms` is `null`, not `0`, when no decode step occurred. `0 ms/token` would read as
infinitely fast rather than as "not measured".

### Quality

| Field | Meaning |
| --- | --- |
| `quality_metric` | `perplexity`, `exact_match`, `needle_retrieval`, … |
| `quality_value` | The measurement. |
| `quality_reference` | The full-cache reference on the same workload. |
| `quality_delta` | Change versus the reference, **sign-normalised**: positive always means better. For perplexity, lower is better, so the sign is inverted. A caller that forgets this reports a perplexity increase as an improvement. |

Quality is evaluated **through a cache**, using the same code path as the measured run. A
quality number produced by a different path is not evidence about that path.

That includes the policy's *signals*: if a policy reads attention, the quality pass must record
attention exactly as generation does. It did not, once — `cum_attention` stayed all-zero, and
every attention-based policy silently scored every token equally, so four different retention
rules returned bit-identical perplexity. The performance half of each result ran the real
policy while the quality half ran a degenerate one. See `docs/research.md`, F13.

---

## Record schema

Every run produces one `BenchmarkRecord`. The field list is not decoration: a memory figure
without a precision, a context length and a git commit is not reproducible, and an
irreproducible result is not evidence.

**Identity** — `run_id`, `schema_version`, `timestamp`, `git_commit`, `git_dirty`

**Model** — `model`, `model_revision`, `model_num_parameters`, `model_config`,
`weights_are_random`, `tokenizer`

**Workload** — `task`, `dataset`, `context_length`, `generated_tokens`, `batch_size`,
`precision`

**Cache** — `policy`, `policy_config`, `capacity`, `attention_sinks`, `compressor`

**Hardware** — `device`, `gpu_name`, `gpu_total_memory_bytes`, `gpu_compute_capability`

**Memory** — `peak_memory_bytes`, `cache_bytes_total`, `cache_bytes_on_device`,
`cache_bytes_offloaded`, `cache_compression_ratio`, `cache_final_tokens`

**Latency** — `ttft_ms`, `tpot_ms`, `prefill_ms`, `decode_ms`, `total_ms`, `latency_p50_ms`,
`latency_p90_ms`, `tokens_per_second`

**Quality** — `quality_metric`, `quality_value`, `quality_reference`

**Environment** — `seed`, `torch_version`, `cuda_version`, `platform`, `python_version`,
`environment` (device name, total memory, compute capability, cudnn, quality settings)

**Provenance** — `notes`, `status`

`generated_tokens` is present because `tpot_ms` and `tokens_per_second` are both derived from
the decode step count. The same tok/s is a different claim at 4 tokens than at 400, so a
record that reports throughput without the count cannot be checked.

### Output files

| File | Contents |
| --- | --- |
| `<name>-<timestamp>.jsonl` | One JSON record per run. The canonical form. |
| `<name>-<timestamp>.csv` | The same records flattened. Nested keys are joined with `.` so no information is dropped. |
| `<name>-<timestamp>.config.json` | The experiment config that produced them, plus any integrity problems. |

The CSV path flattens rather than selects. A CSV row that dropped the policy config would be
less reproducible than the JSONL row it came from.

---

## Reading a result

### Integrity warnings

The runner calls `validate_record` on every record and prints the problems it finds:

```
5 run(s) produced records with integrity warnings:
  full_cache-synthetic:tiny-128-0d4ca4b0: git_commit is missing; the result cannot be tied to a code version
  full_cache-synthetic:tiny-128-0d4ca4b0: quality was measured on a randomly-initialised model...
```

A warning is not noise. Each one names a reason the result cannot be used as evidence yet:

| Warning | Means |
| --- | --- |
| `git_commit is missing` | The result cannot be tied to a code version. Commit, then re-run. |
| `working tree was dirty` | The recorded commit does not describe the code that ran. |
| `performance metrics are present but no quality metric` | The central reporting rule. A memory or latency gain might be a quality loss. |
| `policy '...' has no capacity recorded` | A bounded policy without its budget is not reproducible. |
| `tpot_ms is reported but generated_tokens is missing or zero` | Per-token latency with no step count to interpret it. |
| `quality was measured on a randomly-initialised model` | Valid as a diagnostic of cache behaviour; **not** a language-modelling result. |

Quote these warnings when reporting a result. Hiding them is the failure mode this
infrastructure exists to prevent.

### Reference numbers

`synthetic:tiny`, context 192, 2 generated tokens, batch 1, float32, CUDA, greedy decoding,
seed 0, on an RTX 3050 Laptop (4 GiB, compute 8.6, CUDA 12.4, torch 2.6.0+cu124):

| Policy | Budget | Cache bytes | vs full | Perplexity |
| --- | --- | --- | --- | --- |
| `full_cache` | — | 395,264 | 1.000 | 515.57421875 |
| `sliding_window` | 144 | 294,912 | 0.746 | 514.56475830 |
| `sliding_window` | 96 | 196,608 | 0.497 | 514.48822021 |
| `sliding_window` | 48 | 98,304 | 0.249 | 515.14343262 |
| `sliding_window` | 19 | 38,912 | 0.098 | 527.70678711 |

These are **reference points for checking that your environment produces comparable memory and
quality numbers**, not findings.

**There is deliberately no latency column.** On this hardware, repeated runs of the same
configuration varied by more than the between-policy difference — one fixed budget measured
88.8 ms in one pass and 58.5 ms in another — so the timing columns were withdrawn rather than
reported. If you quote a latency sweep from this repository, re-measure it and repeat it first;
`experiments/scripts/check_ordering_effect.py` exists to test whether a sweep's trend follows
the budget or the measurement order. See `docs/research.md`, F10.

Memory is arithmetic and exact to the byte. Quality is bit-reproducible: the perplexity values
above were identical to all printed digits across two independent sweeps. See
`docs/research.md`, R4.

### Summarising collected results

```bash
python -m uniqkache.metrics.report --input experiments/results --format markdown
```

The Markdown report **always emits a quality column**. A table that shows memory and latency
without quality makes a trade-off look like a win, so the format does not allow it.

---

## Methodology

Applied uniformly, because a measurement artefact that affects one policy and not another is
indistinguishable from a real difference.

**Warmup runs two phases.** First a plain forward pass, then a forward pass with a deliberately
tiny capacity that forces eviction to fire several times. Both code paths — normal decode and
eviction — are compiled before any timing begins. Two separate measurement bugs came from
getting this wrong; see `docs/research.md`, F1 and F2.

**Device synchronisation before every timing boundary.** CUDA kernel launches are asynchronous,
so timing without a barrier measures launch overhead rather than work. `Timer` synchronises in
both `__enter__` and `__exit__`.

**Greedy decoding with a fixed seed.** Two runs on identical inputs must produce identical
tokens, or a difference cannot be attributed to the cache rather than to sampling.

**One policy per run, identical everything else.** Same model, context, precision, device and
seed. When budgets differ, that is stated — a comparison across different budgets compares
budgets, not policies.

**Quality measured on every run.** Including the reference, so `quality_delta` is always
computed against a same-workload baseline rather than an assumed one.

### Known measurement limitations

- **Latency is not currently measurable at the development model scale.** Run-to-run variation
  exceeded the between-policy differences, so no latency result in this repository is
  supported yet. `experiments/scripts/check_ordering_effect.py` tests whether a sweep's trend
  follows the budget or the run position. See `docs/research.md`, F10.
- **`append` is O(T) per step, O(T²) per sequence** (`torch.cat`). Latency comparisons are
  valid *between policies under an identical store*; they are not absolute throughput numbers.
  This affects all policies equally, which is what makes the comparison fair.
- **Small models are overhead-dominated.** At 0.66M parameters, kernel-launch and Python
  overheads dominate. Timing differences at this scale do not extrapolate.
- **One device, one process.** No sharding. No cross-device comparison has been validated.
- **No cross-machine comparison is supported.** Numbers are only comparable on identical
  hardware, precision and software versions — which is why all of those are recorded.

---

## Adding a benchmark

1. Declare it as a config in `experiments/configs/`, with a `description` that states the
   purpose and the pass condition.
2. Include a `full_cache` reference row at the same context length, precision and seed.
3. Keep quality measurement on.
4. Run it on a clean tree, so `git_commit` is recorded and `git_dirty` is `false`.
5. Report the raw table and any integrity warnings, including the inconvenient rows.

See [CONTRIBUTING.md](../CONTRIBUTING.md#benchmark-requirements) for what a reviewer will ask
for.
