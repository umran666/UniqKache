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

_Empty pending regeneration._ Run the commands under **Regenerating** below; they take seconds
on the synthetic model and need no downloads.

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
    --policy sliding_window --sweep
```

Results are written with a timestamped name, so a re-run never overwrites an earlier one.
Compare the new files against these before replacing them; if a number moved, that is a
research event, and `CHANGELOG.md` asks for it to be recorded rather than quietly absorbed.

Commit each experiment separately. `git_dirty` is read before the output is written, so
regenerating both in one dirty tree would mark the second one dirty for no reason.
