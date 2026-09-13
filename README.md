# UniqKache

**Research and implementation framework for adaptive KV-cache management in long-context LLM inference.**

UniqKache is an open-source research framework, not a product. Its purpose is to make a
specific question *answerable with evidence*, and to make it hard to answer it with
marketing. Nothing in this repository is claimed to be faster, smaller or better than any
existing method until a benchmark in this repository shows it — and the benchmark must
include a quality measurement, or it is not accepted.

> **Status: pre-alpha.** The cache abstraction, the policies and the benchmark runner are
> implemented and tested. No adaptive method in this repository has been validated on a
> real model. See [Current limitations](#current-limitations) before reading any number.

---

## Research question

> Can an LLM dynamically decide what inference information should be **retained**,
> **compressed**, **offloaded**, **recomputed** or **prefetched**, according to context,
> model behaviour, memory pressure, latency requirements and quality constraints?

The framework exists to make that question falsifiable. Concretely, it is built so that the
following *cannot* happen:

| Failure mode | How this repository prevents it |
| --- | --- |
| A memory saving is reported without checking accuracy | `validate_record` flags any record with performance metrics and no quality metric |
| A reproduction is presented as an invention | Reproduced methods are named after their papers and marked `Experimental`; see [baselines/](baselines/README.md) |
| A method is called "better" when it only trades accuracy away | Quality is a mandatory column in every report table; `quality_delta` normalises direction across metrics |
| An unmeasurable metric silently becomes a good-looking `0` | Missing metrics are `None`/`null`, never `0` |
| Offloaded bytes are counted as freed bytes | `bytes_on_device` and `bytes_offloaded` are separate fields |
| A result cannot be traced to the code that produced it | Every record carries git commit, dirty flag, seed, precision, hardware and policy config |
| Failed experiments quietly disappear | They are written down in [docs/research.md](docs/research.md#failed-experiments) and kept |

---

## Architecture

UniqKache separates *what to keep* from *where and how it is stored*. That separation is the
whole design: it is what makes a policy comparable against another policy, and what makes an
ablation meaningful.

```
                    ┌──────────────────────────────┐
                    │   controllers/               │  decides WHICH ACTION to take
                    │   AdaptiveController         │  (research prototype)
                    └───────────────┬──────────────┘
                                    │ chooses an action
        ┌───────────────┬───────────┼───────────┬────────────────┐
        ▼               ▼           ▼           ▼                ▼
   ┌─────────┐   ┌───────────┐ ┌────────┐ ┌──────────┐   ┌────────────┐
   │policies/│   │compression│ │offload/│ │prefetch/ │   │ recompute  │
   │ score & │   │  change   │ │ memory │ │  stage   │   │  (runtime) │
   │ select  │   │represent- │ │ tiers  │ │  ahead   │   │            │
   │         │   │  ation    │ │        │ │          │   │            │
   └────┬────┘   └─────┬─────┘ └───┬────┘ └────┬─────┘   └─────┬──────┘
        │  scores      │ bytes       │          │               │
        └──────────────┴─────────────┴──────────┴───────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │   cache/                     │  owns all mutation
                    │   KVCache facade             │  append / get / evict /
                    │   KVStore  (per-layer)       │  clear / compress /
                    │   LayerMetadata (signals)    │  offload / prefetch / stats
                    └───────────────┬──────────────┘
                                    │ read by
                                    ▼
                    ┌──────────────────────────────┐
                    │   runtime/ + models/         │  explicit prefill + decode
                    │   GenerationEngine           │  loop, synchronised timing
                    └───────────────┬──────────────┘
                                    │ measured by
                                    ▼
                    ┌──────────────────────────────┐
                    │   metrics/ + bench/          │  records, validation,
                    │   BenchmarkRecord            │  JSONL / CSV / Markdown
                    └──────────────────────────────┘
```

Three distinctions the architecture enforces, because collapsing them produces misleading
results:

1. **Occupancy vs representation.** Eviction destroys information (occupancy falls).
   Compression and offload do not (occupancy is unchanged). Reporting them as one
   "memory reduction" number hides the fact that only one of them is lossy in the same way.
2. **Policy vs storage.** A policy is a pure function from signals to scores and a
   selection. It never mutates storage. This is what makes `full_cache` vs `sliding_window`
   an ablation rather than a rewrite.
3. **Measured vs unmeasurable.** On CPU, peak GPU memory is `None`. A policy that could not
   be scored because no attention was recorded says so, instead of scoring `0`.

Full module map, invariants and extension points: [docs/architecture.md](docs/architecture.md).

---

## Installation

Requires Python 3.10+. `torch` is the only hard dependency — the cache, policies and runtime
are importable and testable without `transformers`.

```bash
git clone https://github.com/umran666/UniqKache.git
cd UniqKache

python -m pip install -e .            # core only
python -m pip install -e ".[dev]"     # + pytest, ruff
python -m pip install -e ".[hf]"      # + transformers, for real models
```

Verify the install:

```bash
python -m uniqkache.bench --list-policies
pytest -q                             # 306 tests, no GPU and no downloads required
```

> **Note on multiple Python installs.** If `python` on your `PATH` is a different interpreter
> than the one you installed into, you will see `ModuleNotFoundError: No module named 'torch'`.
> Check with `python -c "import torch, sys; print(sys.executable, torch.__version__)"` and
> invoke that interpreter explicitly. This bites on Windows in particular.

---

## Quick start

Run a full-cache reference and an evicting policy at the same context length:

```bash
# reference: keep everything
python -m uniqkache.bench \
    --model synthetic:tiny \
    --context-length 192 \
    --policy full_cache \
    --batch-size 1

# the same workload, retaining ~25% of the context
python -m uniqkache.bench \
    --model synthetic:tiny \
    --context-length 192 \
    --policy sliding_window \
    --keep-ratio 0.25 \
    --batch-size 1
```

Or the standard retention sweep (100 / 75 / 50 / 25 / 10 %) in one command:

```bash
python -m uniqkache.bench \
    --model synthetic:tiny \
    --context-length 192 \
    --policy sliding_window \
    --sweep \
    --max-new-tokens 2      # required to reproduce the retention table below; see F14
```

Every run writes three files to `experiments/results/`: a `.jsonl` of records, a `.csv` of the
same records flattened, and a `.config.json` describing the experiment. Print a table instead
of files with `--markdown`, or see the integrity warnings a record raised with the default
console output.

Programmatic use:

```python
import torch
from uniqkache import CacheConfig, KVCache, build_policy

config = CacheConfig(
    num_layers=4, num_kv_heads=2, head_dim=32,
    dtype=torch.float32, device="cuda",
    capacity=64, attention_sinks=4,
)
# The policy reads the budget and the sink count from the cache it is asked to
# score, so it needs no configuration of its own. Pass `window=` to override.
cache = KVCache(config, policy=build_policy("sliding_window"))

cache.append(layer_idx=0, keys=torch.randn(1, 2, 8, 32), values=torch.randn(1, 2, 8, 32))
print(cache.stats().summary())
# tokens=8 util=12.5% mem=0.00MiB (device=0.00MiB, offloaded=0.00MiB) ...
```

More: [examples/](examples/).

---

## Supported models

| Backend | Status | Notes |
| --- | --- | --- |
| `synthetic:tiny` (0.66M params) | **Stable** | Zero download, deterministic. The development and CI target. |
| `synthetic:small`, `synthetic:medium` | **Stable** | Same code path, larger presets for scaling checks. |
| Hugging Face dense-attention models (`--model <repo-id>`) | **Experimental** | Forward pass and full-cache path work. Requires `.[hf]` and enough RAM/VRAM. |
| HF models under an **evicting** policy | **Not yet supported** | Raises `HF_EVICTION_NOT_SUPPORTED` rather than silently running a full cache and reporting it as eviction. `transformers` 5.x requires a `Cache` *layer* implementation; see [docs/architecture.md](docs/architecture.md#hugging-face-backend). |
| Hybrid / conv-attention models (e.g. LFM2) | **Not yet supported** | Most layers have no KV cache, so cache-management conclusions would not transfer. |

The synthetic models are real dense-attention transformers — GQA, RoPE with absolute
positions, RMSNorm, SwiGLU, causal masking from absolute positions — with **randomly
initialised weights**. Random weights are deliberate: they make the development target
zero-download and byte-for-byte reproducible. They also mean quality numbers from them are a
*diagnostic of cache behaviour*, never a language-modelling result. `weights_are_random` is
recorded on every run and `validate_record` flags any quality claim built on it.

---

## Supported hardware

| Configuration | Status | Notes |
| --- | --- | --- |
| CUDA GPU | **Experimental** | Reference numbers below are from one device: RTX 3050 Laptop, 4 GiB, compute 8.6, CUDA 12.4, torch 2.6.0+cu124. |
| CPU | **Stable** | Correct, but slow. `peak_memory_bytes` is `None` — not `0`. |
| Multi-GPU / tensor parallel | **Not yet supported** | No sharding. |
| Real CPU↔GPU offload | **Research prototype** | `offload.TierManager` plans and accounts for tier movement; a real transfer requires a CUDA source device. On a CPU cache, offloading to CPU is correctly a no-op. |
| Fabricated bandwidth constants | **Not present** | `Tier.bandwidth_gbps` defaults to `None`. Guessing a PCIe bandwidth would make every offload claim unverifiable. |

---

## Benchmarks

Full methodology, the record schema and how to read the tables:
[docs/benchmarks.md](docs/benchmarks.md).

### Correctness first

Before any performance number means anything, the eviction machinery has to be shown to be
correct. `experiments/configs/correctness.json` gives each evicting policy a budget *larger*
than the whole workload, so it must never evict — and therefore must reproduce the full-cache
result **exactly**.

```bash
python -m uniqkache.bench --config experiments/configs/correctness.json
```

Measured (`synthetic:tiny`, context 128, 4 generated tokens, capacity 256, CUDA):

| Policy | Final tokens | Perplexity | vs reference |
| --- | --- | --- | --- |
| `full_cache` | 524 | 502.059692 | reference |
| `sliding_window` | 524 | 502.059692 | `0.00e+00` |
| `lru` | 524 | 502.059692 | `0.00e+00` |
| `attention_based` | 524 | 502.059692 | `0.00e+00` |
| `token_importance` | 524 | 502.059692 | `0.00e+00` |

All four evicting policies reproduce the reference bit-for-bit under a non-evicting budget.
That is a correctness result, not a performance one.

### Retention sweep

`synthetic:tiny`, context 192, 2 generated tokens, batch 1, float32, CUDA, greedy decoding,
seed 0. `sliding_window` with 4 attention sinks. Cache bytes are the summed KV footprint across
all 4 layers.

The decode length is stated because it is load-bearing, not incidental: it sets the `full_cache`
reference occupancy (`context + generated - 1` = 193 tokens), and the bounded rows' budgets are
computed from the *context length* (192 × 75/50/25/10% = 144/96/48/19). A longer decode would
push the reference further above 192 and make the labelled ratios drift from the ratios they
name. See [docs/research.md, F14](docs/research.md#failed-experiments).

Only the deterministic columns are reported. Memory is arithmetic; quality is bit-reproducible
(identical to all printed digits across two independent sweeps). **Latency is excluded** —
repeated runs varied more than the effect being measured. See
[docs/research.md, F10](docs/research.md#failed-experiments).

| Policy | Budget | Cache bytes | vs full | Perplexity |
| --- | --- | --- | --- | --- |
| `full_cache` | — | 395,264 | 1.000 | 515.57421875 |
| `sliding_window` | 144 (75%) | 294,912 | 0.746 | 514.56475830 |
| `sliding_window` | 96 (50%) | 196,608 | 0.497 | 514.48822021 |
| `sliding_window` | 48 (25%) | 98,304 | 0.249 | 515.14343262 |
| `sliding_window` | 19 (10%) | 38,912 | 0.098 | 527.70678711 |

What this does and does not support:

- **Supported:** cache memory scales linearly and exactly with the token budget. The measured
  footprint matches `layers × tokens × kv_heads × head_dim × 2 (K and V) × dtype_bytes` to the
  byte. This is an accounting check that passes.
- **Supported:** at 10% retention, perplexity rises from 515.57 to 527.71 (+2.4%) — the first
  clear sign of information loss. Degradation is visible and the diagnostic is sensitive.
- **NOT supported — there is no latency result here.** An earlier version of this README
  reported a clean monotonic TTFT curve (43.2 → 33.9 ms as retention fell) and described it as
  a possible speedup. That curve did not reproduce. Repeating the sweep gave run-to-run
  variation *larger* than the effect, and one fixed budget measured 88.8 ms in one pass and
  58.5 ms in another. At 0.66M parameters the wall clock is dominated by Python overhead,
  kernel-launch cost and the host machine's clock state, not by the cache. The trend was
  coincidence, and the latency columns have been withdrawn. See
  [docs/research.md, F10](docs/research.md#failed-experiments).
- **NOT supported — the 25% row is not an improvement.** At 25% retention perplexity is
  515.14, slightly *lower* than the full cache's 515.57. On a randomly-initialised model that
  is noise, not a win: there is no learned long-range dependency for eviction to destroy. It
  is reported here precisely because hiding an inconvenient number would be the failure mode
  this project is built to avoid.

---

## Current limitations

Stated plainly, because a limitation discovered by a reader is a defect in the documentation.

- **No validated result exists yet.** Every number above is from a randomly-initialised
  0.66M-parameter model on one 4 GiB laptop GPU. The framework is the deliverable so far; the
  findings are not.
- **Eviction is not supported on Hugging Face models.** `HF_EVICTION_NOT_SUPPORTED` is raised
  rather than running a full cache and calling it eviction. This is the largest gap between
  the framework and a real long-context result.
- **`AdaptivePolicy` and `AdaptiveController` are unvalidated.** Both report
  `validated: False` in `state_dict()`. Their rule chains are documented, not shown to help.
  A result from them is a hypothesis.
- **Latency is not measurable at the current model scale.** On the reference device,
  run-to-run variation in TTFT and TPOT exceeded the between-policy differences, so the
  timing columns of the retention sweep were withdrawn rather than reported. Memory and
  quality are deterministic and reproducible; timing is not. See
  [docs/research.md, F10](docs/research.md#failed-experiments).
- **Append is O(T) per step, O(T²) per sequence.** `LayerStorage.append` uses `torch.cat`, so
  a long context copies the whole cache every step. Latency comparisons are therefore valid
  *between policies under an identical store*, and not as absolute throughput numbers.
- **Quality is perplexity on a random model.** Needle retrieval, exact match and task accuracy
  are implemented in `metrics/quality.py` but not yet wired into a real-model experiment.
- **Single device, single process.** No sharding, no paged allocation, no serving integration.
- **Only one compression method (`int8`)** and no prefetch policy that has been shown to help.
- **No CI has run yet.** The workflow exists at `.github/workflows/ci.yml`; it has not
  executed because the repository has not been pushed.

---

## Research roadmap

Phases are ordered so that each is buildable on the last. Later phases are not started early.

| Phase | Content | State |
| --- | --- | --- |
| 0 | Clean repository, packaging, governance | **Done** |
| 1 | KV-cache abstraction, policy protocol, storage separation | **Done** |
| 2 | Full-cache baseline + correctness proof (incremental == single-shot) | **Done** |
| 3 | Reproduction baselines: sliding window, LRU, H2O-style, int8 quantisation, offload planning | **Done** |
| 4 | Reproducible benchmark runner + record schema + integrity validation | **Done** |
| 5 | Quality harness (perplexity, needle retrieval, exact match) | **Done** (harness), not yet on a real model |
| 6 | Real-model validation on a dense-attention HF model | **Next** |
| 7 | Characterise baselines: retention sweeps on a real model, Pareto frontier | Planned |
| 8 | Ablations for every component; publish negative results | Planned |
| 9 | Adaptive policy evaluated against the best baseline, under a stated objective | Planned |
| 10 | Controller-level decisions (offload / prefetch / recompute) evaluated end-to-end | Planned |

The optimisation objective is stated explicitly and per experiment, never implied:

- **Quality-constrained:** maximise quality subject to `memory ≤ budget` and `latency ≤ budget`.
- **Cost-constrained:** minimise memory or latency subject to `quality ≥ floor`.

A method that reduces memory while reducing quality has not "won" — it has moved along the
frontier, and must be reported as such. See `controllers.Objective`.

---

## Contributing

The contribution pipeline is:

```
Issue → Hypothesis → Implementation → Baseline → Experiment → Benchmark → PR → Review → Merge → Research result
```

Every step is required, and the order matters: the baseline comes before the experiment, and
the experiment comes before the benchmark. A pull request that reports a win without a
baseline and a quality measurement will be asked for them.

- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup, coding standards, branch naming, commit
  conventions, test and benchmark requirements.
- [docs/research.md](docs/research.md) — the research process, hypotheses, what has been
  established, **what has failed**, and open questions.
- [docs/architecture.md](docs/architecture.md) — how to add a policy, compressor, model
  backend, metric or hardware tier.
- Issue templates: [bug](.github/ISSUE_TEMPLATE/bug_report.yml) ·
  [feature](.github/ISSUE_TEMPLATE/feature_request.yml) ·
  [research proposal](.github/ISSUE_TEMPLATE/research_proposal.yml) ·
  [benchmark proposal](.github/ISSUE_TEMPLATE/benchmark_proposal.yml) ·
  [documentation](.github/ISSUE_TEMPLATE/documentation.yml) ·
  [performance investigation](.github/ISSUE_TEMPLATE/performance_investigation.yml)

## Citation

If you use this framework, cite it as described in [CITATION.cff](CITATION.cff). If you use a
reproduced method, cite **the original paper** — see [baselines/](baselines/README.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
