# Results

Raw output from benchmark runs. These files are what the numbers in
[`docs/research.md`](../../docs/research.md) and the README were read from.

## Layout

| Path | Status |
| --- | --- |
| `*.jsonl` / `*.csv` / `*.config.json` | **Current.** Backs a claim that is still made. |
| `superseded/` | **Not current.** Kept as evidence. Do not quote. |
| `runs/`, `raw/` | Git-ignored scratch output from exploratory runs. |

Each run produces three files:

- `.jsonl` — one `BenchmarkRecord` per run, the canonical form;
- `.csv` — the same records flattened (nested keys joined with `.`);
- `.config.json` — the experiment config that produced them, plus any integrity problems
  `validate_record` raised.

The schema is documented in [`docs/benchmarks.md`](../../docs/benchmarks.md#record-schema).

## A result must be able to name the code that produced it

A record is only evidence if it carries `git_commit` and `git_dirty`. Everything currently in
this directory does. The first generation of results did not — they were produced before the
repository had a history — and they were moved to `superseded/` for that reason alone, even
though their memory and quality values were correct. A number that cannot be tied to a commit
cannot be checked, so it is not a result.

`git_dirty` counts *any* uncommitted change, including untracked files, and a run writes its own
output. Records are therefore built before the files are written, so a run started from a clean
tree reports `git_dirty: false` — but only if the tree was clean when it started. Regenerate one
experiment per commit, or run into the git-ignored `runs/` directory, if you want the flag to
mean anything.

## Current results

### `correctness-non-evicting-20260913-133304.*`

**Backs:** R2 in `docs/research.md`, and the "Correctness first" table in the README.

`experiments/configs/correctness.json` — `synthetic:tiny`, context 128, 4 generated tokens,
capacity 256 for the bounded policies. Because 256 exceeds the whole 132-token workload, no
policy can evict, so all of them must reproduce the full-cache result exactly.

Pass condition: `cache_final_tokens` **and** `quality_value` both equal the `full_cache` row.

| Policy | Final tokens | Perplexity | Difference from reference |
| --- | --- | --- | --- |
| `full_cache` (reference) | 524 | 502.059692 | — |
| `sliding_window` | 524 | 502.059692 | `+0.00e+00` |
| `lru` | 524 | 502.059692 | `+0.00e+00` |
| `attention_based` | 524 | 502.059692 | `+0.00e+00` |
| `token_importance` | 524 | 502.059692 | `+0.00e+00` |

**Result: PASS.** All four evicting policies reproduce the reference bit-for-bit. Recorded
`git_commit = 6973a8e1`, `git_dirty = false`, `generated_tokens = 4`.

The one integrity warning is the intended one — the model weights are random, so the quality
column is a diagnostic of cache behaviour and not a claim about model quality.

### `sliding_window-retention-sweep-20260913-133813.*`

**Backs:** R3 and R4 in `docs/research.md`, and the retention table in the README.

`synthetic:tiny`, context 192, **2 generated tokens**, batch 1, float32, CUDA, greedy, seed 0,
`sliding_window` with 4 attention sinks, plus a `full_cache` reference.

| Budget | Cache bytes | vs full | Perplexity |
| --- | --- | --- | --- |
| full (193) | 395,264 | 1.000 | 515.57421875 |
| 144 (75%) | 294,912 | 0.746 | 514.56475830 |
| 96 (50%) | 196,608 | 0.497 | 514.48822021 |
| 48 (25%) | 98,304 | 0.249 | 515.14343262 |
| 19 (10%) | 38,912 | 0.098 | 527.70678711 |

**The decode length is part of the result, not a detail.** It sets the `full_cache` reference
occupancy (`context + generated - 1` = 193), while the bounded budgets are computed from the
*context length* (192 × 75/50/25/10% = 144/96/48/19). Drop `--max-new-tokens 2` and the
reference becomes 199 tokens and 407,552 bytes, so the bounded rows' "vs full" ratios silently
drift from the percentages they are labelled with. The bounded rows themselves do not move —
they are capacity-limited — which is what makes the drift easy to miss. See `docs/research.md`,
F14.

**Read the memory and quality columns only.** This run also recorded TTFT, TPOT and throughput,
but those columns are not reproducible: repeating the sweep gave run-to-run variation larger
than the between-policy difference, and one fixed budget measured 88.8 ms in one pass and
58.5 ms in another. The latency columns are present in the file because the schema records what
was measured; they are **not** reported as results. See `docs/research.md`, F10.

Memory is arithmetic and exact. Quality is bit-reproducible — every perplexity value here is
identical, digit for digit, to the two superseded sweeps below.

## Superseded results

Kept because each one documents why it is not current. No number from any of them should be
quoted.

| File | Why it is not current | Recorded as |
| --- | --- | --- |
| `single-run-20260913-121622.*` | Reported `ttft_ms = 1313.79` (CUDA context initialisation inside the first timed prefill) and `quality_value = Infinity` (perplexity saturated because the default `nn.Embedding` initialisation produced logits too large for float32 `exp`). | F1, F3 |
| `sliding_window-retention-sweep-20260913-121905.*` | The warmup did not exercise the eviction path, so the first bounded run reported `ttft_ms = 292.0` against ~37 ms for its neighbours. | F2 |
| `correctness-non-evicting-20260913-123031.*` | Built on a wrong premise: `keep_ratio: 1.0` gives a budget of exactly 128, which is *less* than prefill (128) plus decode (4), so the bounded policies did evict and their token counts legitimately differed from the reference. Quality matched exactly; the pass condition did not. | F7 |
| `correctness-non-evicting-20260913-123122.*` | Values were correct — all four evicting policies reproduced the reference at 524 final tokens and perplexity `502.059692` — but the records carry no `git_commit`, no `git_dirty` and no `generated_tokens`: they predate the repository history and that field. Superseded by the regeneration, not by a contradiction. | Provenance gap |
| `sliding_window-retention-sweep-20260913-122018.*` | Same provenance gap. Its memory and quality columns back R3 and R4 and were reproduced digit-for-digit by the regeneration, which is the point: the values were right, the records were unverifiable. | Provenance gap |
| `correctness-non-evicting-20260913-132530.*` | Identical values and full provenance, but written before the artifact writer specified LF endings: the files on disk carried CRLF against an LF index, and the config file had no trailing newline. Superseded by `133304` for formatting alone. | Line endings |

The wrong-workload sweep described under F14 is *not* listed here. It was an exploratory run,
never believed and never committed, so it lives in the git-ignored `runs/` directory with the
other scratch output. `superseded/` is for results that were once reported.

### Why keep the two pre-provenance files

They are the evidence for a claim that is easy to make and hard to support: that quality is
reproducible across independent runs while timing is not. Three generations of the same sweep —
`121905` (broken warmup), `122018` (pre-provenance) and the current one — agree on every
perplexity value to all printed digits, and disagree on every latency value. Deleting the
intermediate one would weaken that evidence to a single comparison.

## Regenerating

```bash
python -m uniqkache.bench --config experiments/configs/correctness.json

python -m uniqkache.bench --model synthetic:tiny --context-length 192 \
    --policy sliding_window --sweep --max-new-tokens 2
```

`--max-new-tokens 2` is not optional for the sweep: the default is 8, which moves the
`full_cache` reference row and breaks the comparison against the table above. See F14.

Results are written with a timestamped name, so a re-run never overwrites an earlier one.
Compare the new files against these before replacing them; if a number moved, that is a
research event, and `CHANGELOG.md` asks for it to be recorded rather than quietly absorbed.

Commit each experiment separately. `git_dirty` is read before the output is written, so
regenerating both in one dirty tree would mark the second one dirty for no reason.

## Verifying results against drift (`make verify-results`)

To mechanically verify that the committed results reproduce without numerical drift:

```bash
make verify-results
# or directly:
python experiments/scripts/verify_results.py --results-dir experiments/results
```

Why this exists (F14): documented commands can succeed while producing drifting numbers (e.g.
omitted `--max-new-tokens 2` silently shifted the reference row). Re-running the configs and
diffing against committed artifacts is the mechanical guard against that drift.

- **Quality fields must match exactly:** `quality_value`, `quality_reference`, `policy`,
  `capacity`, `context_length`, and `cache_final_tokens` are strictly checked.
- **Timing fields are excluded by design:** `ttft_ms`, `tpot_ms`, `tokens_per_second`, and
  `peak_memory_bytes` vary with hardware clock and power state (F10), so they are excluded
  from the diff with that exclusion and its rationale stated in the output.

CI runs `make verify-results` on every PR, and `CONTRIBUTING.md` requires it before any
results-changing PR merges.
