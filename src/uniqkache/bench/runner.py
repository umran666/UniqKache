"""The benchmark runner.

Ties together a model, a cache policy, a workload and the measurement layer, and
emits one :class:`~uniqkache.metrics.record.BenchmarkRecord` per run.

Two invariants this module maintains
------------------------------------
**A run is reproducible from its spec.** Everything that could change a number —
seed, precision, device, context length, capacity, sink count, chunk size, model
revision, git commit — is either in the spec or captured in the record. Nothing
that affects the result is left implicit.

**Quality is measured alongside performance.** When ``measure_quality`` is set,
each run also evaluates perplexity through a cache configured identically to the
generation cache. A run whose performance improves while its quality collapses
is visible in a single record rather than requiring a reader to join two tables.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from uniqkache.bench.config import ExperimentConfig, RunSpec
from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.metrics.latency import LatencyStats
from uniqkache.metrics.quality import (
    QualityResult,
    needle_retrieval,
    perplexity,
    random_token_ids,
)
from uniqkache.metrics.record import BenchmarkRecord, git_commit, git_is_dirty, validate_record
from uniqkache.models.synthetic import build_model, get_preset
from uniqkache.policies import build_policy
from uniqkache.runtime.generation import GenerationConfig, GenerationEngine, GenerationResult
from uniqkache.utils.device import describe_hardware, resolve_device
from uniqkache.utils.errors import BackendError, ConfigError
from uniqkache.utils.logging import get_logger
from uniqkache.utils.seed import set_seed

_log = get_logger(__name__)

PRECISION_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(precision: str) -> torch.dtype:
    """Map a precision label to a torch dtype."""
    if precision not in PRECISION_DTYPES:
        raise ConfigError(
            f"unknown precision {precision!r}. Supported: {', '.join(sorted(PRECISION_DTYPES))}"
        )
    return PRECISION_DTYPES[precision]


@dataclass
class RunOutcome:
    """Everything one run produced."""

    record: BenchmarkRecord
    generation: GenerationResult | None
    quality: QualityResult | None
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class _BuiltModel:
    """A model plus the metadata needed to describe it in a record."""

    model: Any
    identifier: str
    revision: str | None
    num_parameters: int
    weights_are_random: bool
    vocab_size: int
    tokenizer: str | None
    config: dict[str, Any]
    cache_config_factory: Any  # Callable[[capacity, sinks, dtype, device], CacheConfig]
    # True when the model came from uniqkache.models.hf_backend, whose K/V mirror
    # overstates the recorded cache memory and must be annotated on the record.
    is_hf_backend: bool = False
    # Extra K/V bytes the run holder keeps on top of the model's own cache
    # (the HF adapter's mirror duplication). A callable resolved after the run,
    # so a live counter can be read at record time; None when not measurable.
    mirror_overhead_bytes_fn: Any = None


def _build_synthetic(spec: RunSpec, dtype: torch.dtype, device: str) -> _BuiltModel:
    """Build the zero-download synthetic model."""
    preset = spec.model.split(":", 1)[1] if ":" in spec.model else spec.model
    config = get_preset(preset)
    if spec.model_config:
        # Allow a config file to reshape the model for a targeted experiment,
        # while keeping the preset as the named default.
        from dataclasses import replace as _replace

        allowed = set(config.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(spec.model_config) - allowed
        if unknown:
            raise ConfigError(
                f"model_config has unknown key(s) {sorted(unknown)}. Valid keys: {sorted(allowed)}"
            )
        config = _replace(config, **spec.model_config)

    model = build_model(config=config, seed=spec.seed, device=device, dtype=dtype)

    def cache_config_factory(
        capacity: int | None, sinks: int, cache_dtype: torch.dtype, cache_device: str
    ) -> CacheConfig:
        return config.cache_config(
            capacity=capacity,
            attention_sinks=sinks,
            dtype=cache_dtype,
            device=cache_device,
            batch_size=spec.batch_size,
        )

    return _BuiltModel(
        model=model,
        identifier=f"synthetic:{preset}",
        revision=None,
        num_parameters=model.num_parameters,
        weights_are_random=True,
        vocab_size=config.vocab_size,
        tokenizer=None,
        config=config.to_dict(),
        cache_config_factory=cache_config_factory,
    )


def build_model_for_spec(spec: RunSpec, dtype: torch.dtype, device: str) -> _BuiltModel:
    """Resolve a spec's model identifier into a built model.

    ``synthetic:<preset>`` builds the built-in model. Any other identifier is
    treated as a Hugging Face repository id and routed to
    :mod:`uniqkache.models.hf_backend`.
    """
    if spec.model.startswith("synthetic:") or spec.model in {"tiny", "small", "medium"}:
        return _build_synthetic(spec, dtype, device)

    from uniqkache.models.hf_backend import build_hf_model

    return build_hf_model(spec, dtype=dtype, device=device)


def _make_cache(
    spec: RunSpec,
    built: _BuiltModel,
    *,
    capacity: int | None,
    dtype: torch.dtype,
    device: str,
) -> KVCache:
    """Construct a cache matching the spec, with the spec's policy."""
    config = built.cache_config_factory(capacity, spec.attention_sinks, dtype, device)
    if capacity is None and spec.policy == "full_cache":
        # An unbounded cache needs no decisions, so no policy is required.
        return KVCache(config, policy=None)
    policy = build_policy(spec.policy)
    return KVCache(config, policy=policy)


def _quality_reference(
    spec: RunSpec,
    built: _BuiltModel,
    prompt: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: str,
    memo: dict[tuple[Any, ...], float],
) -> float | None:
    """Full-cache quality in the spec's metric for this run, memoised.

    The reference is the no-information-loss value *in the same metric as the
    run*: a perplexity reference against a needle value would make
    `quality_delta` meaningless. Without the reference, a bounded run's quality
    is a bare number with nothing to compare against, and the reader cannot
    tell whether the policy cost anything.
    """
    key = (
        spec.model,
        spec.seed,
        int(prompt.shape[1]),
        spec.quality_chunk_size,
        str(dtype),
        spec.quality_metric,
        spec.needle_length,
        spec.needle_depth,
    )
    if key in memo:
        return memo[key]

    unbounded = _make_cache(spec, built, capacity=None, dtype=dtype, device=device)
    if spec.quality_metric == "needle_retrieval":
        needle = random_token_ids(built.vocab_size, spec.needle_length, seed=spec.seed + 1)
        needle = needle.to(device)
        result = needle_retrieval(
            built.model,
            haystack_length=max(spec.context_length, spec.needle_length + 2),
            needle=needle,
            vocab_size=built.vocab_size,
            depth=spec.needle_depth,
            seed=spec.seed,
            cache=unbounded,
        )
    else:
        result = perplexity(
            built.model, prompt, cache=unbounded, chunk_size=spec.quality_chunk_size, device=device
        )
    if result.value is not None:
        memo[key] = result.value
    return result.value


def _warmup(
    spec: RunSpec,
    built: _BuiltModel,
    *,
    dtype: torch.dtype,
    device: str,
) -> None:
    """Pay one-off device costs before any timed work.

    The first CUDA operation in a process initialises the CUDA context and cuDNN
    handles, which can cost hundreds of milliseconds. If that cost lands inside
    the first timed prefill, whichever policy happens to run first is recorded as
    dramatically slower than the others — a pure measurement artefact that would
    be read as a real difference between policies.

    Warmup must exercise **every code path the measurement will take**, not just
    the model forward pass. In particular, the eviction path (policy scoring, a
    descending argsort, and a gather on the K/V tensors) has its own first-call
    cost on an accelerator. A warmup that never evicts leaves that cost to be
    paid inside the first *bounded* run, which then looks anomalous relative to
    its neighbours. That is exactly what happened during development: the first
    evicting run measured ~292ms TTFT against ~40ms for the others. Hence the
    second warmup phase below, which forces eviction regardless of the spec's own
    capacity.

    Warmup cost is not recorded, because it is not part of the measured workload.
    """
    tokens = random_token_ids(built.vocab_size, 8, seed=spec.seed).to(device)

    # Phase 1: the plain forward path.
    plain_cache = _make_cache(spec, built, capacity=None, dtype=dtype, device=device)
    with torch.no_grad():
        built.model.forward(tokens, cache=plain_cache, start_pos=0)
    plain_cache.clear()

    # Phase 2: the eviction path, if this run can evict. A deliberately tiny
    # warmup capacity guarantees eviction fires several times.
    warm_budget: int | None = spec.resolved_capacity
    if spec.memory_budget_mb is not None:
        # A byte budget is not yet resolved to tokens at warmup time (that needs
        # the built model in the caller's scope); warm with a fixed tiny budget.
        warm_budget = 8
    if warm_budget is not None:
        warm_capacity = max(2, min(warm_budget, 8))
        warm_spec = RunSpec(
            model=spec.model,
            policy=spec.policy,
            context_length=spec.context_length,
            batch_size=spec.batch_size,
            capacity=warm_capacity,
            keep_ratio=None,
            memory_budget_mb=None,  # capacity is already resolved for the warmup
            attention_sinks=min(spec.attention_sinks, warm_capacity - 1),
            precision=spec.precision,
            device=spec.device,
            seed=spec.seed,
            max_new_tokens=spec.max_new_tokens,
            measure_quality=False,
        )
        evict_cache = _make_cache(
            warm_spec, built, capacity=warm_capacity, dtype=dtype, device=device
        )
        long_tokens = random_token_ids(built.vocab_size, 24, seed=spec.seed).to(device)
        with torch.no_grad():
            built.model.forward(long_tokens, cache=evict_cache, start_pos=0)
        evict_cache.clear()

    if device == "cuda":
        torch.cuda.synchronize()
    _log.debug("warmup complete on %s", device)


def run_spec(
    spec: RunSpec,
    *,
    repo_path: str | Path | None = None,
    reference_memo: dict[tuple[Any, ...], float] | None = None,
) -> RunOutcome:
    """Execute one benchmark run and produce its record.

    Parameters
    ----------
    spec:
        What to run.
    repo_path:
        Repository root, used to read the git commit for the record.
    reference_memo:
        Shared cache of full-cache reference perplexities, so a sweep computes
        the reference once rather than once per run.

    Returns
    -------
    RunOutcome
        The record, the raw generation and quality results, and any integrity
        problems detected.
    """
    memo = reference_memo if reference_memo is not None else {}
    device = str(resolve_device(spec.device))
    dtype = resolve_dtype(spec.precision)
    set_seed(spec.seed)

    run_id = f"{spec.policy}-{spec.model}-{spec.context_length}-{uuid.uuid4().hex[:8]}"
    _log.info(
        "run %s: model=%s policy=%s ctx=%d capacity=%s sinks=%d device=%s",
        run_id,
        spec.model,
        spec.policy,
        spec.context_length,
        spec.resolved_capacity if spec.memory_budget_mb is None else "<from budget>",
        spec.attention_sinks,
        device,
    )

    built = build_model_for_spec(spec, dtype, device)

    # A byte budget is resolved here, in the runner, because the translation
    # needs the built model's exact cache shape (layers, heads, head_dim,
    # dtype, batch). `tokens_for_bytes` floors, so the derived capacity never
    # exceeds the requested budget. The requested budget is recorded alongside
    # the derived capacity: `capacity` alone cannot distinguish "I asked for
    # 4096 tokens" from "I asked for 512 MiB and got 4096 tokens".
    capacity = spec.resolved_capacity
    if spec.memory_budget_mb is not None:
        byte_budget = int(spec.memory_budget_mb * 1024 * 1024)
        probe_config = built.cache_config_factory(None, spec.attention_sinks, dtype, device)
        capacity = probe_config.tokens_for_bytes(byte_budget)
        if capacity < 1:
            raise ConfigError(
                f"memory budget {spec.memory_budget_mb} MiB fits 0 tokens at this "
                f"model's cache shape ({probe_config.bytes_per_token()} bytes/token "
                "across all layers); the budget is too small to be meaningful"
            )
        if spec.attention_sinks > capacity:
            raise ConfigError(
                f"attention_sinks {spec.attention_sinks} exceeds the capacity {capacity} "
                f"derived from the {spec.memory_budget_mb} MiB budget"
            )
        if spec.policy == "full_cache":
            raise ConfigError(
                "memory_budget_mb sets a token budget, but policy 'full_cache' never "
                "evicts; the budget would be recorded but never enforced. Use an "
                "evicting policy."
            )

    if device == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        # Half precision on CPU is slow and, for some ops, unsupported. The
        # result would be a latency number that says more about the CPU backend
        # than about the cache, so we refuse rather than emit it.
        raise ConfigError(
            f"precision {spec.precision!r} on CPU would produce latency figures that "
            "reflect the CPU half-precision path rather than the cache. Use float32 "
            "on CPU, or run on a CUDA device."
        )

    prompt = random_token_ids(built.vocab_size, spec.context_length, seed=spec.seed)
    prompt = prompt.to(device)

    # Pay device initialisation costs outside the timed region.
    _warmup(spec, built, dtype=dtype, device=device)

    # ---- generation ------------------------------------------------------
    gen_cache = _make_cache(spec, built, capacity=capacity, dtype=dtype, device=device)
    if spec.compressor is not None:
        from uniqkache.compression.quantize import Int8KVCompressor

        if spec.compressor != "int8":
            raise ConfigError(f"unknown compressor {spec.compressor!r}; only 'int8' is implemented")
        gen_cache.compressor = Int8KVCompressor()

    engine = GenerationEngine(
        built.model,
        gen_cache,
        GenerationConfig(max_new_tokens=spec.max_new_tokens, seed=spec.seed),
    )
    generation = engine.generate(prompt)
    latencies = LatencyStats.from_samples(generation.per_step_ms)

    if spec.compressor is not None:
        # Compression is applied here, after the generation phase, and is
        # applied to the *generation* cache only: the quality pass runs through
        # its own uncompressed cache, so `quality_value` stays comparable to
        # the no-compressor run and the record does not mix two cache states.
        # The recorded bytes/ratio are therefore the end-of-run compressed
        # state, while TTFT/TPOT were measured uncompressed. Both facts are
        # stated in docs/benchmarks.md so the record cannot be misread.
        compressed_layers = gen_cache.compress(method=spec.compressor)
        if compressed_layers == 0:
            raise BackendError(
                f"--compressor {spec.compressor} compressed nothing; the record would "
                "claim a mechanism that never ran"
            )
        generation.cache_stats = gen_cache.stats().to_dict()

    # ---- quality ---------------------------------------------------------
    quality: QualityResult | None = None
    reference: float | None = None
    if spec.measure_quality:
        quality_cache = _make_cache(spec, built, capacity=capacity, dtype=dtype, device=device)
        if spec.quality_metric == "needle_retrieval":
            # The needle task builds its own haystack from the same length and
            # seed knobs, so the run stays reproducible from its spec. The
            # reference is evaluated in the same metric, not in perplexity:
            # mixing the two would make quality_delta meaningless.
            needle = random_token_ids(built.vocab_size, spec.needle_length, seed=spec.seed + 1)
            needle = needle.to(device)
            quality = needle_retrieval(
                built.model,
                haystack_length=max(spec.context_length, spec.needle_length + 2),
                needle=needle,
                vocab_size=built.vocab_size,
                depth=spec.needle_depth,
                seed=spec.seed,
                cache=quality_cache,
            )
        else:
            quality = perplexity(
                built.model,
                prompt,
                cache=quality_cache,
                chunk_size=spec.quality_chunk_size,
                device=device,
            )
        if spec.policy == "full_cache":
            reference = quality.value
        else:
            reference = _quality_reference(
                spec, built, prompt, dtype=dtype, device=device, memo=memo
            )

    # ---- record ----------------------------------------------------------
    hardware = describe_hardware(resolve_device(device))
    cache_stats = generation.cache_stats

    record = BenchmarkRecord(
        run_id=run_id,
        git_commit=git_commit(str(repo_path) if repo_path else None),
        git_dirty=git_is_dirty(str(repo_path) if repo_path else None),
        model=built.identifier,
        model_revision=built.revision,
        model_num_parameters=built.num_parameters,
        model_config=built.config,
        weights_are_random=built.weights_are_random,
        tokenizer=built.tokenizer,
        model_loaded_offline=spec.offline if built.is_hf_backend else None,
        mirror_overhead_bytes=(
            int(built.mirror_overhead_bytes_fn())
            if built.mirror_overhead_bytes_fn is not None
            else None
        ),
        task=spec.task,
        dataset=spec.dataset,
        context_length=spec.context_length,
        generated_tokens=generation.generated_tokens,
        batch_size=spec.batch_size,
        precision=spec.precision,
        policy=spec.policy,
        policy_config=_policy_config(gen_cache),
        capacity=capacity,
        attention_sinks=spec.attention_sinks,
        memory_budget_mb=spec.memory_budget_mb,
        compressor=spec.compressor,
        device=hardware.device_type,
        gpu_name=hardware.device_name if hardware.device_type == "cuda" else None,
        gpu_total_memory_bytes=hardware.total_memory_bytes,
        gpu_compute_capability=hardware.compute_capability,
        peak_memory_bytes=generation.peak_memory_bytes,
        cache_bytes_total=cache_stats.get("bytes_total"),
        cache_bytes_on_device=cache_stats.get("bytes_on_device"),
        cache_bytes_offloaded=cache_stats.get("bytes_offloaded"),
        cache_compression_ratio=cache_stats.get("compression_ratio"),
        cache_final_tokens=cache_stats.get("total_tokens"),
        ttft_ms=generation.ttft_ms,
        tpot_ms=generation.tpot_ms,
        prefill_ms=generation.prefill_ms,
        decode_ms=generation.decode_ms,
        total_ms=generation.total_ms,
        latency_p50_ms=latencies.p50_ms,
        latency_p90_ms=latencies.p90_ms,
        tokens_per_second=generation.tokens_per_second,
        quality_metric=quality.metric if quality else None,
        quality_value=quality.value if quality else None,
        quality_reference=reference,
        seed=spec.seed,
        torch_version=hardware.torch_version,
        cuda_version=hardware.cuda_version,
        platform=hardware.platform,
        python_version=hardware.python_version,
        environment={
            "device_name": hardware.device_name,
            "device_total_memory_bytes": hardware.total_memory_bytes,
            "compute_capability": hardware.compute_capability,
            "torch_version": hardware.torch_version,
            "cuda_version": hardware.cuda_version,
            "quality_chunk_size": spec.quality_chunk_size,
            "quality_used_cache": quality.details.get("used_cache")
            if quality and quality.details
            else None,
            "quality_is_interpretable": quality.is_interpretable if quality else None,
            "quality_caveat": quality.caveat if quality else None,
        },
        # The HF backend mirrors K/V into the UniqKache cache in addition to the
        # model's own DynamicCache, so its memory figures overstate the cache's
        # footprint. The adapter's docstring promises records carry that caveat;
        # append it rather than overwrite any notes the spec author wrote.
        notes=(
            (spec.notes + " " if spec.notes else "")
            + "memory figures include the HF backend's K/V mirror; see "
            "uniqkache.models.hf_backend's docstring"
        )
        if built.is_hf_backend
        else spec.notes,
        status="research prototype" if spec.policy in {"adaptive"} else "experimental",
    )

    problems = validate_record(record)
    for problem in problems:
        _log.warning("record integrity: %s: %s", run_id, problem)

    return RunOutcome(record=record, generation=generation, quality=quality, problems=problems)


def _policy_config(cache: KVCache) -> dict[str, Any]:
    """Serialise the policy and cache configuration actually used."""
    config: dict[str, Any] = {"cache_config": cache.config.to_dict()}
    if cache.policy is not None:
        config["policy"] = cache.policy.state_dict()
    if cache.compressor is not None:
        config["compressor"] = cache.compressor.state_dict()
    return config


def run_config(
    config: ExperimentConfig,
    *,
    output_dir: str | Path,
    repo_path: str | Path | None = None,
    write: bool = True,
) -> list[RunOutcome]:
    """Run every spec in an experiment and write the results.

    A failure in one run does not abort the sweep: the failure is logged and
    re-raised only after all other runs have been attempted, so a single bad
    configuration does not discard the rest of an expensive sweep. The raised
    error lists which runs failed.
    """
    started = time.time()
    memo: dict[tuple[Any, ...], float] = {}
    outcomes: list[RunOutcome] = []
    failures: list[tuple[int, str, str]] = []

    for index, spec in enumerate(config.runs):
        _log.info("[%d/%d] %s", index + 1, len(config.runs), spec.model)
        try:
            outcomes.append(run_spec(spec, repo_path=repo_path, reference_memo=memo))
        except Exception as exc:
            _log.error("run %d failed: %s: %s", index, type(exc).__name__, exc)
            failures.append((index, spec.model, f"{type(exc).__name__}: {exc}"))

    if write and outcomes:
        paths = write_results(outcomes, output_dir, config.name)
        _log.info("wrote results to %s", ", ".join(str(p) for p in paths.values()))

    elapsed = time.time() - started
    _log.info(
        "experiment %r finished: %d/%d runs succeeded in %.1fs",
        config.name,
        len(outcomes),
        len(config.runs),
        elapsed,
    )

    if failures:
        detail = "; ".join(f"run[{i}] {model}: {err}" for i, model, err in failures)
        raise BackendError(
            f"{len(failures)} of {len(config.runs)} runs failed in experiment "
            f"{config.name!r}: {detail}"
        )

    return outcomes


def write_results(
    outcomes: list[RunOutcome],
    output_dir: str | Path,
    name: str,
) -> dict[str, Path]:
    """Write records as JSONL and CSV, plus the resolved config.

    Both formats are written because they serve different readers: JSONL keeps
    full nested fidelity for analysis, CSV is what a spreadsheet or a plot
    script consumes.
    """
    from uniqkache.metrics.report import records_to_csv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{name}-{stamp}"

    records = [outcome.record for outcome in outcomes]

    # Every artifact is written with "\n" endings explicitly, on every platform.
    # `.gitattributes` normalises these files to LF, so a Windows run would
    # otherwise produce a file that differs from the index in line endings and
    # make git warn on each add -- recurring warning noise on a data artifact,
    # which is the kind of thing that trains people to ignore warnings. The CSV
    # writer needs `lineterminator` because it emits "\r\n" by default.
    jsonl_path = output_dir / f"{base}.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), default=str) + "\n")

    csv_path = records_to_csv(records, output_dir / f"{base}.csv")

    config_path = output_dir / f"{base}.config.json"
    config_path.write_text(
        json.dumps(
            {
                "name": name,
                "runs": [record.to_dict() for record in records],
                "problems": {o.record.run_id: o.problems for o in outcomes if o.problems},
            },
            indent=2,
            default=str,
        )
        # json.dumps does not end with a newline, and a text file without one
        # shows up as "\ No newline at end of file" in every diff.
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return {"jsonl": jsonl_path, "csv": csv_path, "config": config_path}


__all__ = [
    "PRECISION_DTYPES",
    "RunOutcome",
    "build_model_for_spec",
    "resolve_dtype",
    "run_config",
    "run_spec",
    "write_results",
]
