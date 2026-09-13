# Results

Raw output from benchmark runs. These files are what the numbers in
[`docs/research.md`](../../docs/research.md) and the README were read from.

## Layout

| Path | Status |
| --- | --- |
| `*.jsonl` / `*.csv` / `*.config.json` | **Current.** Backs a claim that is still made. |
| `superseded/` | **Withdrawn.** Kept as evidence of a failed experiment. Do not quote. |
| `runs/`, `raw/` | Git-ignored scratch output from exploratory runs. |

Each run produces three files:

- `.jsonl` — one `BenchmarkRecord` per run, the canonical form;
- `.csv` — the same records flattened (nested keys joined with `.`);
- `.config.json` — the experiment config that produced them, plus any integrity problems
  `validate_record` raised.

The schema is documented in [`docs/benchmarks.md`](../../docs/benchmarks.md#record-schema).

## Current results

### `correctness-non-evicting-20260913-123122.*`

**Backs:** R2 in `docs/research.md`, and the "Correctness first" table in the README.

`experiments/configs/correctness.json` — `synthetic:tiny`, context 128, 4 generated tokens,
capacity 256 for the bounded policies. Because 256 exceeds the whole 132-token workload, no
policy can evict, so all of them must reproduce the full-cache result exactly.

Pass condition: `cache_final_tokens` **and** `quality_value` both equal the `full_cache` row.

Result: all four evicting policies reproduce the reference — 524 final tokens and perplexity
`502.059692` (difference `0.00e+00`).

### `sliding_window-retention-sweep-20260913-122018.*`

**Backs:** R3 and R4 in `docs/research.md`, and the retention table in the README.

`synthetic:tiny`, context 192, batch 1, float32, CUDA, greedy, seed 0, `sliding_window` with 4
attention sinks, plus a `full_cache` reference.

| Budget | Cache bytes | Perplexity |
| --- | --- | --- |
| full (193) | 395,264 | 515.57421875 |
| 144 (75%) | 294,912 | 514.56475830 |
| 96 (50%) | 196,608 | 514.48822021 |
| 48 (25%) | 98,304 | 515.14343262 |
| 19 (10%) | 38,912 | 527.70678711 |

**Read the memory and quality columns only.** This run also recorded TTFT, TPOT and
throughput, but those columns are not reproducible: repeating the sweep gave run-to-run
variation larger than the between-policy difference, and one fixed budget measured 88.8 ms in
one pass and 58.5 ms in another. The latency columns are present in the file because the schema
records what was measured; they are **not** reported as results. See
`docs/research.md`, F10.

Memory is arithmetic and exact. Quality is bit-reproducible — these perplexity values were
identical to all printed digits in the superseded sweep below.

## Superseded results

Kept because each one is the evidence for a documented failure. They are not current, and no
number from them should be quoted.

| File | Why it is superseded | Recorded as |
| --- | --- | --- |
| `single-run-20260913-121622.*` | Reported `ttft_ms = 1313.79` (CUDA context initialisation inside the first timed prefill) and `quality_value = Infinity` (perplexity saturated because the default `nn.Embedding` initialisation produced logits too large for float32 `exp`). | F1, F3 |
| `sliding_window-retention-sweep-20260913-121905.*` | The warmup did not exercise the eviction path, so the first bounded run reported `ttft_ms = 292.0` against ~37 ms for its neighbours. | F2 |
| `correctness-non-evicting-20260913-123031.*` | Built on a wrong premise: `keep_ratio: 1.0` gives a budget of exactly 128, which is *less* than prefill (128) plus decode (4), so the bounded policies did evict and their token counts legitimately differed from the reference. Quality matched exactly; the pass condition did not. | F7 |

Note that the superseded sweep and the current sweep agree on every quality value, digit for
digit. That agreement is the evidence for the claim that quality is reproducible while timing
is not.

## Regenerating

```bash
python -m uniqkache.bench --config experiments/configs/correctness.json
python -m uniqkache.bench --model synthetic:tiny --context-length 192 \
    --policy sliding_window --sweep
```

Results are written with a timestamped name, so a re-run never overwrites an earlier one.
Compare the new files against these before replacing them; if a number moved, that is a
research event, and `CHANGELOG.md` asks for it to be recorded rather than quietly absorbed.
