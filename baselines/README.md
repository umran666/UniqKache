# Baselines

Every method in this directory is either a **reproduction** of published work or a
**deliberately simple reference point**. None of it is claimed as an invention of this project.

If you use one of these in your own work, cite the original paper. The citation here is for
orientation and to record fidelity; the authority is the paper.

> **Fidelity matters.** A simplified reproduction that underperforms may be underperforming
> because of the simplification, not because the published method is weak. Each entry below
> states what was reproduced faithfully and what was not. When a result looks surprising,
> check the fidelity note before drawing a conclusion about the original method.

---

## Registered baselines

All of these are reachable from the benchmark CLI via `--policy <name>`.

### `full_cache` — the reference

The conventional KV cache: retain everything, evict nothing. No paper.

This is the baseline every claim must be measured against, at the same context length,
precision, seed and device. A comparison across different budgets is a comparison of budgets,
not of policies.

### `sliding_window` — recency window with attention sinks

**Reproduces:** Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models with
Attention Sinks.* ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453)

**Mechanism.** Score by absolute position, keep the most recent `budget` tokens, and
permanently protect the first `attention_sinks` tokens. The protected prefix is what makes
streaming stable: without it, evicting the first tokens destroys the attention sink that the
model relies on.

**Faithful:** the recency window, the protected sink prefix, and the observation that
attention sinks must survive eviction.

**Not reproduced:** per-layer window variation; the original's positional-shift handling for
the cache after eviction (UniqKache keeps absolute positions, which is a deliberate design
choice — see `docs/architecture.md`, invariant 4); any KV-cache-aware fine-tuning.

**Aliases:** `streamingllm`, `streaming_llm`, `sliding`.

### `lru` — least recently used

**Source:** classic cache replacement, not a paper reproduction.

**Mechanism.** Score by `(last_access, position)` packed lexicographically into a float64, so
ties are broken deterministically by position rather than by tensor order.

**Why it is here.** It is the simplest signal-free baseline: it uses recency of *use* rather
than recency of *arrival*. If an attention-based policy cannot beat this at equal budget, the
attention machinery is not earning its cost. `uses_attention = False` — attention is
deliberately not consulted.

**Aliases:** `recency`.

### `attention_based` — heavy hitters

**Reproduces:** Zhang, Sheng, Zhou, Chen, Zheng, Cai, Song, Tian, Ré, Barrett, Wang, Chen.
*H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.*
NeurIPS 2023. [arXiv:2306.14048](https://arxiv.org/abs/2306.14048)

**Mechanism.** Accumulate attention received per token, keep the highest-scoring tokens under
the budget, protect the sink prefix.

**Faithful:** cumulative attention as an importance signal; the protected sink prefix; greedy
retention of the top-scoring tokens.

**Not reproduced:** the published dynamic-budget variant, where the number of retained tokens
is derived from the attention distribution rather than fixed. UniqKache uses a fixed budget so
that comparisons across policies are at equal memory — which is a precondition for attributing
a difference to the policy.

**Caveat.** Requires attention weights, so the engine materialises them
(`uses_attention = True`). That costs memory proportional to the context length and is part of
this policy's measured footprint, not free. Any comparison against `full_cache` must account
for that.

**Aliases:** `h2o`, `heavy_hitter`.

### `token_importance` — weighted blend

**Related to:** Liu et al. *Scissorhands: Exploiting the Persistence of Importance Hypothesis
for LLM KV Cache Compression at Test Time.* NeurIPS 2023.
[arXiv:2305.17118](https://arxiv.org/abs/2305.17118) — specifically the
persistence-of-importance hypothesis, which is what justifies including a cumulative-attention
term at all.

**Mechanism.** A weighted sum of four normalised signals — attention, recency, frequency and
position — with defaults `{attention: 1.0, recency: 0.5, frequency: 0.25, position: 0.0}`.
Rank normalisation by default, so signals on different scales do not need hand-tuned
coefficients.

**Status: not a faithful reproduction of any single published method.** It is a composable
framework for testing which signals carry information. The default weights are a starting
point, not a tuned configuration, and no claim is made that they are good. Reporting a result
from this policy requires reporting the weights used.

**Not reproduced:** Scissorhands' actual selection procedure.

### `int8` compressor — quantisation

**Reproduces:** Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu. *KIVI: A Tuning-Free
Asymmetric 2bit Quantization for KV Cache.* ICML 2024.
[arXiv:2402.02750](https://arxiv.org/abs/2402.02750)

**Mechanism.** Asymmetric int8 quantisation with the granularity the paper specifies:
**keys per-channel** (reduce over the sequence axis, `key_axis=2`) and **values per-token**
(reduce over the head dimension, `value_axis=3`). Enabling it via `--compressor int8`.

**Faithful:** the quantisation granularity and the asymmetric scheme with a zero-point. The
observed range is clamped to include zero so the zero-point is representable — an earlier
version did not, and silently degraded reconstruction error by 7× (see `docs/research.md`, F6).

**Not reproduced:** the 2-bit setting, the residual window, and the group-wise scheme for
keys. This is an 8-bit implementation of the paper's mechanism, not of its full configuration.

**Important.** Compression changes *representation*, not *occupancy*. It does not free tokens
and it does not, by itself, reduce the number of tokens the model attends over. It is reported
via `cache_compression_ratio` and must never be presented as an eviction-equivalent memory
saving.

**Alias:** `quantize`.

### `adaptive` — **research prototype, not a baseline**

**No paper. No validated result.** `AdaptivePolicy` interpolates its recency weight from
memory pressure, and reports `validated: False` in `state_dict()`.

It is listed here only so that it is not mistaken for a reproduction. A result from this
policy is a hypothesis, and the honest framing of any positive result is "this needs to beat
`token_importance` at equal budget before it means anything" — see `docs/research.md`, H7.

---

## Planned baselines

| Method | Source | Blocker |
| --- | --- | --- |
| SnapKV | Li et al., NeurIPS 2024. [arXiv:2404.14469](https://arxiv.org/abs/2404.14469) | Needs observation-window attention pooling, which the current runtime does not produce. |
| KIVI 2-bit + residual window | [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) | Needs group-wise quantisation. |
| PagedAttention allocation | Kwon et al., SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) | `append` is currently O(T); a paged store is the fix and is not implemented. |
| FlexGen-style offload scheduling | Sheng et al., ICML 2023. [arXiv:2303.06865](https://arxiv.org/abs/2303.06865) | Needs a real transfer tier with measured bandwidth. |

---

## Adding a baseline

1. Implement the policy in `src/uniqkache/policies/`, registering it with `@register_policy`.
2. Add a docstring naming the paper, the mechanism, and what is **not** reproduced.
3. Add an entry to this file with the same three things, plus the citation.
4. Add an alias if the paper's name differs from the registered name.
5. Add a unit test asserting the *contract* (scores align with cached tokens; protection is
   honoured), not just that it runs.
6. Add an entry to `docs/research.md` under Related Work.

Do not describe a reproduction as an improvement, and do not describe a simplification as a
reproduction.
