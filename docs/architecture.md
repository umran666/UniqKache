# Architecture

This document explains how UniqKache is put together, which invariants hold it together, and
exactly where to add something new. It is written for someone about to change the code, so it
records *decisions and their reasons*, not a tour of the file tree.

The design has one organising idea:

> **Separate what to keep from where and how it is stored.**

A policy decides *what* survives. Storage decides *how* it is held. Nothing does both. This is
what makes `full_cache` versus `sliding_window` a genuine ablation — the storage layer is
byte-for-byte identical between them, so any difference is attributable to the policy.

---

## Module map

| Module | Responsibility | Stability |
| --- | --- | --- |
| `uniqkache.cache` | Policy-agnostic K/V storage, metadata, and the `KVCache` facade. Owns all mutation. | Stable |
| `uniqkache.policies` | Retention policies: score tokens, select survivors. Pure functions of signals. | Stable |
| `uniqkache.compression` | Representation changes that preserve occupancy (quantisation). | Experimental |
| `uniqkache.offload` | Memory-tier planning and accounting. | Research prototype |
| `uniqkache.prefetch` | Staging decisions ahead of use. | Research prototype |
| `uniqkache.controllers` | Chooses *which action* to take under constraints. | Research prototype |
| `uniqkache.models` | Model backends: a zero-download synthetic transformer, and Hugging Face. | Synthetic: Stable · HF: Experimental |
| `uniqkache.runtime` | Explicit prefill + decode loop with synchronised timing. | Experimental |
| `uniqkache.metrics` | Memory, latency and quality measurement; the record schema and its validation. | Stable |
| `uniqkache.bench` | Experiment config, runner, CLI, result writing. | Experimental |
| `uniqkache.utils` | Device, seeding, logging, error types. | Stable |

---

## Data flow of one benchmark run

```
RunSpec (bench/config.py)
  │  validated: policy name resolved, budget present if the policy evicts
  ▼
build_model_for_spec ──► model + tokenizer + CacheConfig
  │
  ├─► _warmup: phase 1 plain forward, phase 2 forced-eviction forward
  │            (both code paths compiled before any timing)
  ▼
GenerationEngine.generate
  │
  ├─ prefill:  forward(prompt, start_pos=0)  ──► cache.append(...)   [timed → TTFT]
  ├─ decode:   for step in 1..max_new_tokens [timed per step → TPOT]
  │              forward(token, start_pos)   ──► cache.append(...)
  │              cache.enforce_capacity()    ──► policy.score / policy.select
  │              note_attention(...)             ──► signals for the next decision
  ▼
GenerationResult (timings, per-step ms, peak memory, cache stats)
  │
  ├─► metrics.quality: perplexity over the same token stream, through a cache
  ▼
BenchmarkRecord ──► validate_record ──► problems[] surfaced to console and log
  │
  ▼
write_results: .jsonl + .csv + .config.json
```

The quality pass is deliberately run **through a cache**, not by calling the model directly.
A quality number produced by a different code path than the one being measured would not be
evidence about that path.

---

## Invariants

These are properties the code is written to preserve. Breaking one is a bug, and most of them
are pinned by a test.

1. **A policy never mutates storage.** `score()` and `select()` read signals and return
   indices. Only `KVCache` mutates. If a policy needs to change the cache, it returns a
   selection and the facade applies it.

2. **Missing is not zero.** An unmeasurable metric is `None`/`null`. Peak GPU memory on CPU is
   `None`. A policy that could not be scored reports that it could not be scored.

3. **Occupancy and representation are reported separately.** `bytes_on_device` and
   `bytes_offloaded` are distinct fields. `compression_ratio` describes representation, not
   occupancy. Collapsing them would let an offload be described as a memory saving.

4. **Positions are absolute, forever.** RoPE uses the *original* position of each token. After
   eviction, a token's position must not be renumbered, or attention silently computes the
   wrong thing. `PolicyState.positions` carries absolute positions through eviction.

5. **Attention sinks are derived, not stored as a flag.** `LayerMetadata.is_sink()` is
   `positions < num_sinks`. A boolean flag would have to be kept in sync through every gather;
   a derived predicate cannot drift.

6. **Every record carries its own provenance.** Model, revision, precision, device, seed, git
   commit, dirty flag, and policy configuration. A record that cannot be reproduced from itself
   is not evidence.

7. **Performance cannot be reported without quality.** `validate_record` returns a problem for
   any record with latency or memory metrics and no quality metric. The runner prints those
   problems; it does not swallow them.

8. **Unsupported is refused, not approximated.** The HF backend raises
   `HF_EVICTION_NOT_SUPPORTED` rather than running a full cache and reporting it as eviction.

9. **No fabricated constants.** `Tier.bandwidth_gbps` defaults to `None`. A guessed PCIe
   bandwidth would make every offload claim unverifiable.

10. **Timing is synchronised.** `synchronize(device)` is called before every timing boundary,
    because CUDA launches are asynchronous.

---

## The cache layer

### `KVCache` — the facade

The only object that mutates storage. Its interface is deliberately small:

```python
cache.append(layer_idx, keys, values)     # grow
cache.get(layer_idx)                      # read back K/V
cache.evict(indices=...)                  # or keep=..., or neither → enforce_capacity()
cache.clear()
cache.compress(method="int8")             # representation
cache.offload(target="cpu")               # tier
cache.prefetch(device="cuda")
cache.stats()                             # CacheStats
```

`evict()` has three modes and they are not interchangeable:

- `evict(indices=[...])` — drop exactly these.
- `evict(keep=[...])` — keep exactly these. Expressed as `keep` rather than `indices` because
  that is what a policy actually decides, and inverting it at every call site is a source of
  off-by-one errors.
- `evict()` — enforce the configured capacity, asking the policy to choose.

A bounded cache constructed **without** a policy is rejected in the constructor: there would be
nothing to decide what to drop, and defaulting to "drop the oldest" would silently make every
bounded run a sliding window.

### `KVStore` / `LayerStorage` — the storage

`KVStore` holds one `LayerStorage` per layer. Each `LayerStorage` holds `keys`, `values`, an
optional `_compressed` representation, and `LayerMetadata`.

Two behaviours worth knowing:

- **`keys` / `values` dequantise on demand.** Reading a compressed layer materialises it.
  `apply_compression` stores the compressed form; `materialize` reverses it.
- **`append` materialises a compressed layer first.** Compressing and then appending is
  therefore a decompress-append-recompress cycle. This is correct but not cheap; it is
  documented on the method rather than hidden.

**Known limitation — `append` is O(T) per step, O(T²) per sequence.** It uses `torch.cat`, which
copies the whole cache on every append. A paged store would fix this; UniqKache does not have
one. The consequence is stated where it matters: **latency comparisons are valid between
policies under an identical store, and are not absolute throughput numbers.**

### `CacheConfig` — the accounting

```python
bytes_per_token_per_layer = num_kv_heads × head_dim × 2 (K and V) × element_size
bytes_for_tokens(T)       = num_layers × T × bytes_per_token_per_layer
```

Verified exact against measurement — see `docs/research.md`, R3.

---

## The policy layer

### The interface

```python
class CachePolicy(Protocol):
    name: str
    uses_attention: bool

    def reset(self) -> None: ...
    def score(self, state: PolicyState) -> torch.Tensor: ...      # one score per cached token
    def select(self, scores, budget, protect=None) -> torch.Tensor: ...  # indices to keep
```

`score` returns a tensor with one entry per cached token, where **higher means more worth
keeping**. `select` receives the budget and a set of protected indices and returns what to
keep. `BaseCachePolicy.select` implements the contract once, deterministically, so a policy
author writes only `score`.

`PolicyState` is the read-only view a policy gets: per-layer token counts, capacity,
absolute positions, last-access times, cumulative attention, hit counts, sink flags, memory
pressure, and optionally query relevance. Every per-token tensor is validated to share a
length, so a policy cannot silently score against a misaligned signal.

### Registry

```python
from uniqkache.policies import register_policy, register_alias, build_policy

register_policy(MyPolicy)                 # by class
register_alias("mymethod", "my_policy")   # CLI convenience
policy = build_policy("my_policy", config=cache_config)
```

Aliases exist so that a user typing the name from a paper (`h2o`, `streamingllm`) gets the
reproduction, and the record then shows the canonical name that actually ran.

### Signal helpers

`policies/signals.py` provides `minmax`, `rank_normalize` and `weighted_sum`.
`minmax` returns zeros for a constant input rather than `NaN` — a constant signal carries no
information, and propagating `NaN` into a selection would turn a no-op into a crash.

---

## Extension points

### 1. A new eviction policy

```python
from uniqkache.policies import BaseCachePolicy, register_policy

@register_policy
class MyPolicy(BaseCachePolicy):
    """One line on the mechanism, and the paper it reproduces, if any."""

    name = "my_policy"
    uses_attention = True   # set True only if score() actually reads attention

    def score(self, state):
        return state.cum_attention + 0.5 * state.last_access
```

Then: a unit test in `tests/unit/test_policies.py`, an entry in `baselines/README.md` with its
citation and fidelity notes, and an experiment config if you are proposing a finding.

Set `uses_attention = False` if you do not read attention. It is not a hint — the engine uses
it to decide whether to materialise attention tensors at all, and materialising them costs
memory proportional to the context length, which would distort the measurement.

### 2. A new compression method

Implement `Compressor` from `compression/base.py`:

```python
class Compressor(Protocol):
    name: str
    lossy: bool

    def compress(self, tensor, axis: int) -> CompressionResult: ...
    def decompress(self, result) -> torch.Tensor: ...
```

Return a `CompressionResult` carrying the dequantisation parameters, and make sure
dequantisation reproduces the encode arithmetic exactly. `F6` in `docs/research.md` is what
happens when it does not.

**Do not confuse the reduction axis with the granularity it produces.** Keys are per-channel
(reduce over the sequence); values are per-token (reduce over the head dimension). If your
gather logic branches on an axis number, it is wrong — branch on the resulting parameter
extent.

### 3. A new model backend

Satisfy `models/base.py`:

```python
class LanguageModel(Protocol):
    def forward(self, input_ids, *, cache, start_pos, return_attention=False): ...
```

Return `(logits, attention_weights_or_None)`. The backend must:

- write K/V into the cache and **read it back** rather than keeping its own copy;
- apply RoPE with absolute positions;
- return `logits` with one position per input token during prefill.

`models/synthetic.py` is the reference implementation and is short enough to read in full.

### 4. A new metric

Add to `metrics/quality.py` and return a `QualityResult` carrying `metric`, `value`,
`num_tokens`, and — critically — `is_interpretable` and `caveat`. A metric computed on a
randomly-initialised model sets `is_interpretable=False` and explains why. A metric that
cannot say whether it is interpretable is not finished.

### 5. A new hardware tier

`offload/tiers.py`. Add a `Tier` with a **measured** bandwidth. If you have not measured it,
leave `bandwidth_gbps=None`; a guessed constant is worse than an absent one, because it will
be used in a calculation and believed.

### 6. A new benchmark or experiment

Two options:

- **Config file** (preferred): add JSON to `experiments/configs/` and run
  `python -m uniqkache.bench --config ...`. Declared, diffable, re-runnable.
- **Programmatic**: build an `ExperimentConfig` and call `run_config`.

A config that adds a field to `RunSpec` needs a validator in `RunSpec.__post_init__` if the
field can be invalid, and a `derived` entry if it should not be parsed back in.

### 7. A new controller action

Add to `ActionKind` in `controllers/types.py`, map it to a `QualityRisk` in `quality_risk_of`,
and add a rule to the `AdaptiveController` chain. Every `Decision` must carry a `reason`
string and record the `alternatives` it rejected — an action chosen without a recorded
alternative cannot be audited.

---

## Hugging Face backend

The HF backend is **Experimental**, and evicting policies on it are **Not yet supported**.

Why: `transformers` 5.x requires a `Cache` *layer* implementation
(`layer_class_to_replicate`), and subclassing `update` for bespoke behaviour is no longer
supported. UniqKache's eviction model — evict from a populated store and read back — does not
map onto that interface without a real `CacheLayer` implementation.

Rather than half-working, `build_hf_model` raises `HF_EVICTION_NOT_SUPPORTED`. Running a full
cache under `--policy sliding_window` and reporting the result would be exactly the kind of
silent misreporting this project is built to prevent. Closing this gap is the highest-priority
item on the roadmap, because it is what stands between the framework and a real-model result.

---

## Testing strategy

| Layer | What it protects |
| --- | --- |
| `tests/unit/` | One behaviour each: append validation, eviction modes, policy contracts, signal helpers, quantisation primitives, config validation, record validation. |
| `tests/integration/` | The property that matters: **incremental decode equals a single-shot forward pass** (max diff 4.6e-5), plus GQA and prefill-chunking invariance. If this fails, no eviction result means anything. |
| `tests/regression/` | One test per fixed bug, named after the symptom. `F1`–`F9` in `docs/research.md` correspond to tests here. Never delete one. |

The suite runs without a GPU and without any download. That is a deliberate constraint: it
means a contributor can verify a change before pushing, and it means CI does not need a GPU to
protect the correctness properties.
