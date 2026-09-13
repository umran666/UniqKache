# UniqKache — Research Notes

This file is the project's memory of *what we tried and what happened*. It is deliberately
not a highlight reel. Failed experiments stay in it permanently: a negative result that is
deleted has to be rediscovered, and a hypothesis that was quietly dropped is indistinguishable
from one that was quietly abandoned.

**Status: pre-alpha.** No adaptive method in this repository has been validated on a real
model. Read the Results section as "what the harness currently measures", not as findings
about long-context inference.

---

## Research Question

> Can an LLM dynamically decide what inference information should be **retained**,
> **compressed**, **offloaded**, **recomputed** or **prefetched**, according to context, model
> behaviour, memory pressure, latency requirements and quality constraints?

A full answer requires all five decisions to be made *and evaluated against a strong
baseline under a stated objective*. The objective is always stated explicitly, never implied:

- **Quality-constrained:** maximise quality subject to `memory ≤ budget` and `latency ≤ budget`.
- **Cost-constrained:** minimise memory or latency subject to `quality ≥ floor`.

A method that reduces memory while reducing quality has not won. It has moved along the
Pareto frontier, and must be reported as a trade-off. This is enforced in
`controllers.Objective` and in the report tables, where quality is a mandatory column.

---

## Hypotheses

Each hypothesis is stated so that it can be **falsified**, and each carries its failure
criteria. Unfalsifiable statements are not listed here.

| # | Hypothesis | Failure criteria | State |
| --- | --- | --- | --- |
| H1 | The cache storage layer is correct: incremental decode with a cache equals a single-shot forward pass. | Any output difference above float32 tolerance. | **Supported** (max abs diff 4.6e-5) |
| H2 | An evicting policy given a non-evicting budget reproduces the full-cache result exactly. | Any difference in final token count or quality. | **Supported** (diff `0.00e+00`) |
| H3 | Cache memory scales linearly with the token budget, matching the closed-form accounting. | Measured bytes deviate from `layers × tokens × kv_heads × head_dim × 2 × dtype_bytes`. | **Supported** (exact, to the byte) |
| H4 | Information loss from eviction is visible in perplexity at aggressive retention. | Perplexity unchanged at 10% retention. | **Weakly supported** at 10% on a random model (+2.4%); not established on a real model |
| H5 | Eviction must worsen perplexity. | Perplexity unchanged or improved. | **Falsified** on a randomly-initialised model. See Failed Experiments. |
| H6 | Attention-based eviction (H2O-style) beats pure recency (LRU) at equal budget on a real model. | No significant difference at equal budget and quality. | **Not yet tested** — needs a real model |
| H7 | Pressure-aware adaptive weighting beats the best fixed-weight policy at equal budget. | No significant difference, or a quality loss at equal budget. | **Not yet tested** — `AdaptivePolicy` is a prototype |
| H8 | Offloading is only worth it when the transfer cost is below the recompute cost for the same tokens. | Offload wins where transfer cost exceeds recompute cost. | **Not yet tested** — needs a real transfer tier |
| H9 | Latency differences between policies at equal budget are measurable above run-to-run noise on this hardware. | Run-to-run variation exceeds the between-policy difference. | **Falsified** at this scale. See F10. |

H5 is the most important entry here, because it was believed before it was tested and the
test contradicted it.

---

## Related Work

UniqKache does not claim novelty for any of the following. They are reproduced (or scheduled
for reproduction) as **baselines**, and the original papers are the citations to use. Fidelity
notes are in [baselines/README.md](../baselines/README.md).

### KV-cache eviction and retention

| Method | Source | Reproduced as | Fidelity |
| --- | --- | --- | --- |
| Attention sinks / StreamingLLM | Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, ICLR 2024 (arXiv:2309.17453) | `SlidingWindowPolicy` + `attention_sinks` | Faithful in mechanism: a recency window plus permanently protected leading tokens. No per-layer window variation. |
| H2O heavy hitters | Zhang et al., *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs*, NeurIPS 2023 (arXiv:2306.14048) | `AttentionBasedPolicy` | Cumulative attention score with a protected sink prefix. The published dynamic-budget variant is not reproduced. |
| Scissorhands | Liu et al., *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time*, NeurIPS 2023 (arXiv:2305.17118) | Scheduled | The persistence-of-importance assumption is the hypothesis behind `TokenImportancePolicy`'s cumulative-attention term. |
| SnapKV | Li et al., *SnapKV: LLM Knows What You Are Looking For Before Generation*, NeurIPS 2024 (arXiv:2404.14469) | Not yet | Requires observation-window attention pooling, which the current runtime does not produce. |
| LRU / recency | Classic cache replacement | `LRUPolicy` | Not a paper reproduction; included as the simplest signal-free baseline. |

### Quantisation and representation

| Method | Source | Reproduced as | Fidelity |
| --- | --- | --- | --- |
| KIVI asymmetric 2-bit KV quantisation | Liu et al., *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*, ICML 2024 (arXiv:2402.02750) | `Int8KVCompressor` | Mechanism reproduced at 8-bit: **keys per-channel** (reduce over sequence), **values per-token** (reduce over head dimension). 2-bit and the residual-window scheme are not implemented. |

### Memory tiers and systems

| Method | Source | Reproduced as | Fidelity |
| --- | --- | --- | --- |
| PagedAttention / vLLM | Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023 (arXiv:2309.06180) | Not yet | UniqKache's store is contiguous, not paged. This is a known gap, and the reason `append` is O(T). |
| FlexGen offloading | Sheng et al., *FlexGen: High-Throughput Generative Inference of LLMs with a Single GPU*, ICML 2023 (arXiv:2303.06865) | `offload.TierManager` (planning only) | Plans and accounts for tier movement; does **not** perform a real transfer or overlap it with compute. |

### Architecture context

The synthetic models follow the Llama-family design: grouped-query attention (Ainslie et al.,
arXiv:2305.13245), RoPE (Su et al., arXiv:2104.09864), RMSNorm, SwiGLU. RoPE is applied with
**absolute** positions, which matters here: after eviction, positions must remain the original
ones or the attention pattern is silently wrong.

---

## Baselines

A baseline is only useful if it runs in the *same harness* as the thing it is compared
against. All of the following are registered policies in `policies/` and reachable from the
benchmark CLI:

| Baseline | `--policy` | What it isolates |
| --- | --- | --- |
| Full cache | `full_cache` | The reference. No eviction, no compression, no offload. |
| Sliding window + sinks | `sliding_window` (alias `streamingllm`) | Pure recency retention. |
| LRU | `lru` | Recency by access time, attention deliberately unused. |
| Attention-based (H2O-style) | `attention_based` (alias `h2o`) | Whether attention mass identifies what to keep. |
| Weighted token importance | `token_importance` | Attention + recency + frequency + position, blended. |
| Adaptive (prototype) | `adaptive` | Pressure-dependent weighting. **Unvalidated.** |

The baseline every claim must be measured against is `full_cache`, at the same context
length, precision, seed and device. A comparison across different budgets is not a comparison
of policies; it is a comparison of budgets.

---

## Experiments

Declared, re-runnable configurations live in `experiments/configs/`. Nothing below was
produced by an ad-hoc command whose arguments are lost.

| Config | Purpose | Status |
| --- | --- | --- |
| `experiments/configs/correctness.json` | Non-evicting budget must reproduce full cache exactly | Runs; passes |
| `experiments/configs/sweep_policies.json` | Cross-policy comparison at a fixed 25% budget | Runs |
| `--sweep` (CLI) | Retention sweep at 100/75/50/25/10% | Runs |

Measurement protocol, applied uniformly:

- **Warmup before timing.** Two phases: a plain forward pass, then a forced-eviction pass.
  Without the second phase, the first bounded run carries CUDA kernel-compilation cost for the
  eviction path and reports a fake TTFT. This was a real bug; see Failed Experiments.
- **Synchronise before every timing boundary.** CUDA launches are asynchronous; timing without
  a barrier measures launch overhead rather than work.
- **Greedy decoding, fixed seed.** Two runs on identical inputs must produce identical tokens,
  or a difference cannot be attributed to the cache.
- **Quality on every run.** `validate_record` flags a record that reports performance without
  quality, and the runner surfaces that warning on the console.

---

## Results

### R1 — The cache storage layer is correct (H1)

Incremental decode with a cache must equal a single-shot forward pass over the same tokens.
Measured maximum absolute difference: **4.6e-5**, within float32 tolerance. This is the
load-bearing correctness property of the whole project: if it failed, every eviction result
would be measuring a broken cache rather than a policy.

Covered by `tests/integration/test_incremental_matches_single_shot.py`, including a GQA
variant and a prefill-chunking-invariance check.

### R2 — Non-evicting budgets reproduce the reference exactly (H2)

`experiments/configs/correctness.json` — `synthetic:tiny`, context 128, 4 generated tokens,
capacity 256 for the bounded policies (larger than the whole 132-token workload, so no
eviction can occur):

| Policy | Final tokens | Perplexity | vs reference |
| --- | --- | --- | --- |
| `full_cache` | 524 | 502.059692 | reference |
| `sliding_window` | 524 | 502.059692 | `0.00e+00` |
| `lru` | 524 | 502.059692 | `0.00e+00` |
| `attention_based` | 524 | 502.059692 | `0.00e+00` |
| `token_importance` | 524 | 502.059692 | `0.00e+00` |

All four evicting policies reproduce the reference bit-for-bit. Note that 524 = 131 tokens
per layer × 4 layers, and 131 = 128 prompt + 3, not 132: the *final* generated token is never
fed back through the model, because there is no next token to produce from it. This is correct
and is worth stating, since it looks like an off-by-one at a glance.

### R3 — Memory accounting is exact (H3)

`synthetic:tiny`, context 192, `sliding_window`, 4 attention sinks. Predicted cost is
`layers × tokens × kv_heads × head_dim × 2 (K and V) × dtype_bytes` = 4 × tokens × 2 × 32 × 2
× 4 = 2048 bytes per token.

| Budget | Tokens/layer | Measured bytes | Predicted bytes | vs full cache |
| --- | --- | --- | --- | --- |
| full (193) | 193 | 395,264 | 395,264 | 1.000 |
| 144 (75%) | 144 | 294,912 | 294,912 | 0.746 |
| 96 (50%) | 96 | 196,608 | 196,608 | 0.497 |
| 48 (25%) | 48 | 98,304 | 98,304 | 0.249 |
| 19 (10%) | 19 | 38,912 | 38,912 | 0.098 |

Exact to the byte, and the reduction tracks the budget exactly. This validates the accounting,
not any policy.

### R4 — Retention sweep: memory and quality (H4, provisional)

`synthetic:tiny`, context 192, batch 1, float32, CUDA, greedy, seed 0. The reference device is
an RTX 3050 Laptop (4 GiB, compute 8.6, CUDA 12.4, torch 2.6.0+cu124).

**Only the deterministic columns are shown.** Memory is arithmetic and quality is
bit-reproducible (verified identical across two independent sweeps); latency at this scale is
not reproducible, and is excluded — see F10.

| Policy | Budget | Cache bytes | vs full | Perplexity |
| --- | --- | --- | --- | --- |
| `full_cache` | — | 395,264 | 1.000 | 515.57421875 |
| `sliding_window` | 144 | 294,912 | 0.746 | 514.56475830 |
| `sliding_window` | 96 | 196,608 | 0.497 | 514.48822021 |
| `sliding_window` | 48 | 98,304 | 0.249 | 515.14343262 |
| `sliding_window` | 19 | 38,912 | 0.098 | 527.70678711 |

What this supports, and what it does not:

- **Quality degradation appears at 10% retention.** 515.57 → 527.71, a 2.4% rise. The first
  clear sign of information loss, and evidence that the perplexity diagnostic is sensitive
  enough to detect it.
- **The 25% row is not an improvement.** 515.14 is marginally *below* the full cache's 515.57.
  On a randomly-initialised model that is noise. There is no learned long-range dependency for
  eviction to destroy, so there is nothing for the full cache to be better *at*. This row is
  reported because omitting an inconvenient number is precisely the failure mode this project
  exists to prevent.
- **No latency claim is made.** An earlier version of this section reported a clean monotonic
  TTFT curve (43.2 → 33.9 ms as retention fell) and described it as a possible speedup. That
  curve was not reproducible: repeating the sweep gave run-to-run variation larger than the
  effect, and a fixed budget measured 88.8 ms in one pass and 58.5 ms in another. The trend was
  coincidence. See F10.
- **Nothing here transfers to a real model.** The model is randomly initialised; see
  Limitations.

---

## Failed Experiments

Kept permanently. Each entry records what was expected, what happened, why, and what changed.
Every one of these was found by a measurement or a test that disagreed with an assumption.

### F1 — TTFT reported as 1313 ms, an order of magnitude too slow

**Expected:** prefill latency around tens of milliseconds for a 256-token prompt on a tiny
model.

**Observed:** `ttft_ms = 1313.79`, while TPOT was a plausible 45.9 ms.

**Cause:** CUDA context and kernel initialisation happened *inside* the first timed prefill.
The measurement was real; it was measuring one-time initialisation, not prefill.

**Fix:** added a warmup pass before timing. Result: 37.9 ms.

**Lesson:** a warmup must exercise the code path being measured, or the first sample is not
comparable to the rest.

### F2 — The warmup did not cover the eviction path

**Expected:** after F1's fix, every run should report a comparable TTFT.

**Observed:** in the first sweep, the first *bounded* run reported TTFT 292 ms while its
neighbours reported ~37 ms. Every full-cache run was fine.

**Cause:** the warmup ran a plain forward pass. The eviction path — scoring, sorting,
selecting, gathering, re-materialising — was compiled on its first real use, which happened to
be inside the first timed bounded run.

**Fix:** `_warmup` now runs a second phase with a deliberately tiny capacity, forcing eviction
to fire several times before any measurement. After the fix every run in the sweep reported
33.9–43.2 ms.

**Lesson:** "we warmed up" is not a property of the code, it is a property of the *path*.
Regression-tested.

### F3 — Perplexity saturated at `inf`, making the quality diagnostic useless

**Expected:** a finite perplexity that moves when the cache loses information.

**Observed:** `quality_value: Infinity` on every run. The diagnostic could not distinguish a
good cache from a destroyed one.

**Cause:** `nn.Embedding`'s default initialisation is `N(0, 1)`. With a 128-dimensional hidden
size that gives logits with an absolute maximum around 137 and a cross-entropy loss around
118, so `exp(loss)` overflows float32.

**Fix:** added `SyntheticCausalLM._init_weights` with `std=0.02`, the GPT-2 / Llama
convention. Perplexity became finite and sensitive (515–528 across the sweep, moving by
+12.13 at 10% retention).

**Lesson:** a quality metric that saturates is worse than no metric, because it looks like a
result. Regression-tested.

### F4 — Assumed that eviction must worsen perplexity (H5 falsified)

**Expected:** evicting tokens must raise perplexity. This was written as a test assertion.

**Observed:** the test failed. Measured 65.06 under eviction versus 65.20 for the full cache —
eviction made perplexity *marginally better*.

**Cause:** the model's weights are random. There is no learned long-range dependency, so there
is no information whose removal can hurt. With random weights, eviction removes noise as often
as it removes signal, and the direction of the change is not determined.

**Fix:** the assertion was replaced with one that tests *sensitivity* — a relative change
above 1e-3 — rather than direction. The honest limitation is recorded in both the test and
this document.

**Lesson:** this is the clearest illustration in the project of why a baseline plus a quality
measurement is mandatory, and why "quality is random-model perplexity" must never be quoted as
a language-modelling result. It is also why H4 is only *weakly* supported: the +2.4% at 10%
retention is a real signal about information loss, but its magnitude on a random model says
nothing about a trained one.

### F5 — Quantisation axes for keys and values were swapped

**Expected:** compressing the cache, then evicting from it, should work.

**Observed:** `IndexError: index 1 is out of bounds for dimension 0 with size 1` when evicting
a compressed layer.

**Cause:** two distinct concepts were conflated. The *reduction axis* (which axis the scale
and zero-point are computed over) is not the same as the *granularity* it produces (one
parameter per channel, or one per token). The gather logic compared `axis == 2` instead of
inspecting the resulting parameter extent, and `Int8KVCompressor` had the defaults backwards:
`key_axis=3, value_axis=2`.

**Fix:** the gather now checks whether the parameter's sequence extent is 1 or `T`, which is
what actually determines the gather shape. The defaults were corrected to **`key_axis=2`**
(keys per-channel, reducing over the sequence) and **`value_axis=3`** (values per-token,
reducing over the head dimension), matching KIVI. Regression-tested.

**Lesson:** naming a parameter after the axis hides which granularity it produces. The
regression test asserts the *granularity*, not the axis number.

### F6 — Asymmetric quantisation silently degraded accuracy

**Expected:** asymmetric int8 quantisation should be at least as accurate as symmetric.

**Observed:** maximum reconstruction error 0.115, against 0.016 for the symmetric path — a
7× degradation, with no error raised.

**Cause:** for an all-positive slice, the zero-point `qmin - xmin/scale` fell below `qmin`. The
stored zero-point was clamped to `qmin`, but the dequantisation arithmetic used the *unclamped*
value, so encode and decode disagreed.

**Fix:** clamp the observed range to include zero (`xmin.clamp_max(0.0)`, `xmax.clamp_min(0.0)`)
so the zero-point is representable by construction. Regression-tested.

**Lesson:** a silently-wrong quantiser produces plausible-looking output. The test asserts
reconstruction error, not that a tensor came back with the right shape.

### F7 — The correctness experiment's premise was wrong, and it "failed" for the right reason

**Expected:** with `keep_ratio: 1.0` the budget equals the context length, so a bounded policy
should never evict and should reproduce the full cache exactly.

**Observed:** the quality values matched exactly (`0.00e+00`) but the final token counts did
not: 524 for `full_cache` versus 512 for every bounded policy.

**Cause:** the premise was wrong, not the code. `keep_ratio: 1.0` at context 128 gives a
budget of exactly 128 — but the workload is 128 prompt tokens **plus** 4 generated tokens, so
the bounded policies evicted 4 times and correctly capped at 128. `full_cache` grew to 131.
Comparing token counts across a bounded and an unbounded cache is not a correctness check at
all; they are *expected* to differ.

**Fix:** the config now uses an explicit `capacity: 256`, which exceeds the whole 132-token
workload, so non-eviction is genuinely true. All policies then reproduce the reference exactly
(524 tokens, `0.00e+00`). The pass condition — token count *and* quality must both match — is
now meaningful.

**Lesson:** a test whose premise is wrong can fail while the code is correct. The first version
of this experiment was measuring the budget, not the policy.

### F8 — `ExperimentConfig` could not round-trip its own output

**Expected:** `RunSpec.to_dict()` → `ExperimentConfig.from_dict()` should be lossless.

**Observed:** `ConfigError: runs[0] has unknown key(s) ['resolved_capacity']`.

**Cause:** `to_dict()` emits the derived `resolved_capacity` for the convenience of records,
but the parser treated it as an unknown input key.

**Fix:** the parser accepts and ignores a small `derived` set. Regression-tested.

**Lesson:** a serialiser and its parser must be tested against each other, not only against
hand-written fixtures.

### F9 — Prefetch registry missing its default

**Expected:** `NoPrefetch` available as `"none"`.

**Observed:** the registry lookup failed; the class lives in `prefetch/base.py`, and
registering it there would have created an import cycle with the registry.

**Fix:** `prefetch/__init__.py` explicitly registers it after the module graph is complete.

**Lesson:** registration order is part of the public behaviour of a plugin registry.

### F10 — The retention sweep's latency curve was not reproducible (H9 falsified)

**Expected:** a retention sweep should show TTFT and TPOT falling as the budget falls, because
there are fewer tokens to attend over. The first sweep measured exactly that:

```
full_cache  43.2 ms
75%         37.3 ms
50%         36.6 ms
25%         36.0 ms
10%         33.9 ms
```

A clean monotonic decrease, which reads as a latency benefit of eviction.

**Observed:** repeating the sweep did not reproduce it. Two consecutive orders of the *same*
five configurations:

```
forward (100 -> 10%)   51.96  43.65  49.27  88.79  64.77 ms
reverse ( 10 -> 100%)  47.56  58.48  56.63  47.38  29.84 ms
```

Neither is monotonic, and they do not share a shape. Worse, the **same budget** measured very
differently depending on the pass: budget 32 gave 88.79 ms forward and 58.48 ms reverse;
budget 12 gave 64.77 ms and 47.56 ms. A five-run repeat of one fixed configuration gave a
1.10–1.20× spread (TTFT 53.2–64.0 ms), so the variation is not only between passes.

**Cause:** at 0.66M parameters a 128-token prefill is a handful of tiny kernels, so the wall
clock is dominated by Python overhead, kernel-launch cost and the host machine's power and
clock state — not by the cache. This is a 4 GiB laptop GPU, and its clocks move. The magnitude
of the real effect (fewer tokens to attend over) is below the noise floor of the measurement.

**Fix:** the sweep's latency columns were removed from the reported results. The
`check_ordering_effect.py` script was added to `experiments/scripts/` so this is checkable on
other hardware rather than assumed: it runs the sweep in both orders and reports whether the
trend follows the budget or the run position.

**Lesson:** a monotonic curve is not evidence of a mechanism. With one sample per point, a
five-point sweep that happens to be ordered monotonically will *look* like a dose-response
relationship. The only defence is repetition, and the willingness to discard a result that
looked good. This is the most consequential failure recorded here, because the first version of
it was written into the README as a finding.

**What this does not affect:** memory accounting (arithmetic, exact to the byte) and quality
(bit-reproducible — perplexity was identical to all printed digits across two independent
sweeps). Only the timing columns were withdrawn.

### F11 — The results reporter could not read the runner's own output

**Expected:** `python -m uniqkache.metrics.report --input experiments/results --format
markdown` — the command behind `make results-table` — summarises the collected results.

**Observed:**

```
WARNING ignoring unknown field(s) ['name', 'problems', 'runs'] ...
TypeError: BenchmarkRecord.__init__() missing 1 required positional argument: 'run_id'
```

**Cause:** the reporter globbed `*.json`, which matches the
`<name>-<timestamp>.config.json` file that `write_results` writes *beside* every result. A
config file is not a record: it carries `name`, `runs` and `problems`, and no `run_id`. The
unknown-field warning was the only clue before the crash.

The failure was embarrassing in a specific way: the command was broken on precisely the
directory it exists to read — one the runner had populated itself. It survived the test suite
because the existing tests built directories containing only `.jsonl` files, so the config file
was never present.

**Fix:** the reporter excludes `.config.json` and stays non-recursive, so withdrawn results
under `superseded/` are not folded into a current summary. Covered by
`TestResultsReporterRegression`, which builds a directory shaped exactly as the runner writes
it — record file *and* config file — rather than a convenient one.

**Lesson:** a test fixture that is tidier than reality tests the fixture. The directory the
tool will actually be pointed at has the runner's own by-products in it.

---

## Open Questions

1. **Does any of this survive a real model?** Every result above is from a
   randomly-initialised 0.66M-parameter model. The immediate next step is a dense-attention
   Hugging Face model — the blocker is that evicting policies are unsupported on the HF
   backend, which is the highest-priority gap in the framework.
2. **Does attention mass identify what to keep?** H6 is untested. If `attention_based` does not
   beat `lru` at equal budget on a real model, a large part of the eviction literature does not
   transfer to this harness, and that is itself a finding worth publishing.
3. **Is the retention threshold sharp or gradual?** On the random model, degradation appears
   between 25% and 10%. On a real model, is there a knee? At what context length does it move?
4. **What is the right quality metric?** Perplexity is sensitive but indirect. Needle-in-a-
   haystack and task accuracy are implemented in `metrics/quality.py` but not yet wired into a
   real-model experiment.
5. **When is offloading worth it?** H8 is untested. It needs a real transfer tier with a
   *measured* bandwidth — not an assumed one. `Tier.bandwidth_gbps` deliberately defaults to
   `None` rather than guessing a PCIe number.
6. **Does pressure-aware weighting help, or is a fixed weight enough?** H7 is untested. If
   `adaptive` does not beat the best fixed-weight `token_importance` configuration at equal
   budget, the adaptive machinery is unjustified complexity, and the honest move is to say so.
7. **What does recomputation actually cost?** `RECOMPUTE` is in the action vocabulary and the
   controller refuses to apply it without a `recompute_fn`, but no recompute strategy has been
   measured against eviction or offloading.
8. **Is `append` being O(T) the dominant cost at real context lengths?** If it is, policy
   comparisons at long context will be swamped by storage cost, and paging has to come before
   any further policy work.

---

## How to add a result here

1. State the hypothesis and its failure criteria **before** running anything.
2. Declare the experiment in `experiments/configs/`.
3. Run it with a clean git tree, and record the commit.
4. Report quality alongside performance, or the result is not accepted.
5. Add it under Results if it worked, under **Failed Experiments** if it did not.
6. State the limitation. Every result has one; a result without a stated limitation has not
   been examined closely enough.
