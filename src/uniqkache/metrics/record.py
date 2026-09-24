"""The benchmark result record.

Every number this project reports is produced through this schema, so that a
result can be traced back to the exact conditions that produced it. The field
list is not decoration: a memory figure without a precision, a context length and
a git commit is not reproducible, and an irreproducible result is not evidence.

Two conventions
---------------
**Missing is not zero.** A metric that could not be measured is ``None`` and is
serialised as ``null``, never as ``0``. Peak GPU memory on a CPU-only run is
``None``, not ``0`` — reporting zero would look like a perfect result.

**Quality travels with performance.** :meth:`BenchmarkRecord.quality_claimed`
exists so that a record cannot be presented as a performance win without a
quality measurement alongside it. See
:func:`uniqkache.metrics.record.validate_record` for the enforcement.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch

from uniqkache.utils.device import describe_hardware
from uniqkache.utils.errors import ConfigError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)

SCHEMA_VERSION = "1.1.0"


@dataclass
class MetricAggregate:
    """Statistical summary of a metric across multiple repetitions."""

    mean: float
    std: float
    min: float
    max: float
    values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricAggregate:
        return cls(
            mean=float(data["mean"]),
            std=float(data["std"]),
            min=float(data["min"]),
            max=float(data["max"]),
            values=[float(v) for v in data.get("values", [])],
        )


def git_commit(repo_path: str | None = None) -> str | None:
    """Return the current commit hash, or ``None`` outside a git repository.

    Returns ``None`` rather than a placeholder such as ``"unknown"``: a record
    that claims a commit it cannot verify is worse than one that admits it has
    none.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.warning("could not read git commit: %s", exc)
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_is_dirty(repo_path: str | None = None) -> bool | None:
    """Whether the working tree has uncommitted changes, or ``None`` if unknown.

    A dirty tree means the recorded commit does not fully describe the code that
    ran, so this is surfaced rather than ignored.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        _log.warning("could not read git status: %s", exc)
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


@dataclass
class BenchmarkRecord:
    """One benchmark run, fully described.

    See the module docstring for the missing-vs-zero convention.
    """

    # -- identity ---------------------------------------------------------
    run_id: str
    schema_version: str = SCHEMA_VERSION
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    git_dirty: bool | None = None

    # -- model ------------------------------------------------------------
    model: str = ""
    model_revision: str | None = None
    model_num_parameters: int | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    weights_are_random: bool = False
    tokenizer: str | None = None
    # Whether the model was resolved strictly from the local cache, with no
    # network access. False means a download was permitted (and the record does
    # not say whether one actually happened — only that it was allowed).
    model_loaded_offline: bool | None = None

    # -- workload ---------------------------------------------------------
    task: str = "generation"
    dataset: str | None = None
    context_length: int | None = None
    generated_tokens: int | None = None
    batch_size: int = 1
    precision: str = "float32"

    # -- cache configuration ----------------------------------------------
    policy: str = "full_cache"
    policy_config: dict[str, Any] = field(default_factory=dict)
    capacity: int | list[int] | None = None
    capacity_schedule: str | None = None
    attention_sinks: int = 0
    compressor: str | None = None
    # The budget the user asked for, in MiB, when the capacity was derived from
    # bytes rather than stated in tokens. `capacity` alone cannot distinguish
    # "asked for 4096 tokens" from "asked for 512 MiB and got 4096 tokens".
    memory_budget_mb: float | None = None

    # -- hardware ---------------------------------------------------------
    device: str = "cpu"
    gpu_name: str | None = None
    gpu_total_memory_bytes: int | None = None
    gpu_compute_capability: str | None = None

    # -- memory -----------------------------------------------------------
    peak_memory_bytes: int | None = None
    cache_bytes_total: int | None = None
    cache_bytes_on_device: int | None = None
    cache_bytes_offloaded: int | None = None
    cache_compression_ratio: float | None = None
    cache_final_tokens: int | None = None
    tokens_per_layer: list[int] = field(default_factory=list)
    utilization_per_layer: list[float] | None = None
    # Extra K/V bytes held on top of the model-native cache (currently: the HF
    # adapter's mirror duplication). Subtract from `cache_bytes_total` for the
    # model-native footprint. None when there is no known holder overhead.
    mirror_overhead_bytes: int | None = None

    # -- latency ----------------------------------------------------------
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None
    total_ms: float | None = None
    latency_p50_ms: float | None = None
    latency_p90_ms: float | None = None
    tokens_per_second: float | None = None

    # -- quality ----------------------------------------------------------
    quality_metric: str | None = None
    quality_value: float | None = None
    quality_reference: float | None = None

    # -- multi-seed / repetitions -----------------------------------------
    repetitions: int = 1
    seeds: list[int] = field(default_factory=list)
    aggregates: dict[str, MetricAggregate] = field(default_factory=dict)
    repetition_records: list[dict[str, Any]] = field(default_factory=list)

    # -- environment ------------------------------------------------------
    seed: int = 0
    torch_version: str = ""
    cuda_version: str | None = None
    platform: str = ""
    python_version: str = ""
    environment: dict[str, Any] = field(default_factory=dict)

    # -- provenance -------------------------------------------------------
    notes: str = ""
    status: str = "experimental"

    # ------------------------------------------------------------------

    @property
    def quality_delta(self) -> float | None:
        """Quality change relative to the reference, or ``None``.

        For perplexity, lower is better, so the sign is inverted to keep
        "positive means better" consistent across metrics. A caller that forgets
        this would report a perplexity increase as an improvement.
        """
        if self.quality_value is None or self.quality_reference is None:
            return None
        if self.quality_metric in {"perplexity", "perplexity_delta"}:
            return self.quality_reference - self.quality_value
        return self.quality_value - self.quality_reference

    @property
    def quality_claimed(self) -> bool:
        """Whether this record makes a quality claim that needs support."""
        return self.quality_metric is not None and self.quality_value is not None

    def aggregate(self, metric: str) -> MetricAggregate | None:
        """Return the MetricAggregate for ``metric``, or None."""
        return self.aggregates.get(metric)

    def mean(self, metric: str) -> float | None:
        """Return the mean of ``metric``, or None."""
        if metric in self.aggregates:
            return self.aggregates[metric].mean
        val = getattr(self, metric, None)
        return float(val) if val is not None else None

    def std(self, metric: str) -> float | None:
        """Return the standard deviation of ``metric``, or None."""
        if metric in self.aggregates:
            return self.aggregates[metric].std
        return None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["quality_delta"] = self.quality_delta
        return data

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten nested dictionaries for CSV output.

        Nested keys are joined with ``.`` so no information is dropped in the
        CSV path. Dropping the policy config would make a CSV row less
        reproducible than the JSONL row it came from.
        """
        flat: dict[str, Any] = {}

        def _walk(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    _walk(f"{prefix}.{key}" if prefix else str(key), item)
            elif isinstance(value, list):
                flat[prefix] = "|".join(str(v) for v in value)
            else:
                flat[prefix] = value

        for key, value in self.to_dict().items():
            _walk(key, value)
        return flat

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkRecord:
        """Reconstruct a record, ignoring unknown keys with a warning.

        Unknown keys are reported rather than silently dropped, because a
        silently dropped field means a result that cannot be fully reproduced
        from its own record.
        """
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(data) - known - {"quality_delta"}
        if unknown:
            _log.warning(
                "ignoring unknown field(s) %s when loading a benchmark record; "
                "the record may have been produced by a different schema version",
                sorted(unknown),
            )
        kwargs = {k: v for k, v in data.items() if k in known}
        if "aggregates" in kwargs and isinstance(kwargs["aggregates"], dict):
            kwargs["aggregates"] = {
                k: MetricAggregate.from_dict(v) if isinstance(v, dict) else v
                for k, v in kwargs["aggregates"].items()
            }
        return cls(**kwargs)


def validate_record(record: BenchmarkRecord) -> list[str]:
    """Return a list of integrity problems with ``record``.

    This is the mechanical enforcement of the project's central reporting rule:
    a performance or memory result may not be presented without a quality
    measurement. It returns problems rather than raising so that a caller can
    decide whether to warn, annotate or reject — but the benchmark runner
    surfaces them, so a record cannot pass silently.
    """
    problems: list[str] = []

    if not record.model:
        problems.append("model is empty; the record does not say what was measured")
    if record.git_commit is None:
        problems.append("git_commit is missing; the result cannot be tied to a code version")
    if record.git_dirty:
        problems.append(
            "working tree was dirty; the recorded commit does not describe the code that ran"
        )

    # Performance reported without quality.
    reports_performance = any(
        v is not None for v in (record.ttft_ms, record.tpot_ms, record.tokens_per_second)
    )
    if reports_performance and not record.quality_claimed:
        problems.append(
            "performance metrics are present but no quality metric was measured. "
            "A memory or latency result without a quality measurement cannot show "
            "whether the gain came from sacrificing accuracy."
        )

    # Memory reported without saying which mechanism produced it.
    if (
        record.cache_bytes_total is not None
        and record.capacity is None
        and record.policy != "full_cache"
    ):
        problems.append(
            f"policy {record.policy!r} has no capacity recorded; a bounded policy "
            "without its budget is not reproducible"
        )

    if record.context_length is not None and record.context_length <= 0:
        problems.append(f"context_length must be positive, got {record.context_length}")

    # Throughput and TPOT are both derived from the decode step count, so a
    # record that reports them without saying how many tokens were decoded
    # cannot be checked: the same tok/s is a different claim at 4 tokens than at
    # 400.
    if record.tpot_ms is not None and not record.generated_tokens:
        problems.append(
            "tpot_ms is reported but generated_tokens is missing or zero; the "
            "decode step count is needed to interpret a per-token latency"
        )

    # A record that names a compressor must show that it ran. `1.0` means "no
    # saving", `None` means "not measured"; both mean the claimed mechanism is
    # invisible in the record, which is exactly the silent no-op that made this
    # check necessary (a `--compressor int8` run once wrote `ratio=1.0` while
    # claiming the compressor).
    if record.compressor is not None and (
        record.cache_compression_ratio is None or record.cache_compression_ratio <= 1.0
    ):
        problems.append(
            f"compressor {record.compressor!r} is recorded but cache_compression_ratio "
            f"is {record.cache_compression_ratio}; the record claims a compression that "
            "is not visible in its own numbers"
        )

    if record.weights_are_random and record.quality_claimed:
        problems.append(
            "quality was measured on a randomly-initialised model. That is a valid "
            "diagnostic of cache behaviour but must not be reported as model quality; "
            "see docs/research.md."
        )

    if record.repetitions < 1:
        problems.append(f"repetitions must be >= 1, got {record.repetitions}")
    if record.repetitions > 1:
        if record.seeds and len(record.seeds) != record.repetitions:
            problems.append(
                f"seeds count ({len(record.seeds)}) does not match repetitions ({record.repetitions})"
            )
        if not record.aggregates:
            problems.append(f"repetitions is {record.repetitions} but aggregates is missing")
        else:
            for metric_name, agg in record.aggregates.items():
                if agg.values and len(agg.values) != record.repetitions:
                    problems.append(
                        f"aggregate {metric_name!r} values count ({len(agg.values)}) does not match repetitions ({record.repetitions})"
                    )

    return problems


def environment_snapshot() -> dict[str, Any]:
    """Collect environment metadata for a record."""
    hardware = describe_hardware()
    return {
        "platform": hardware.platform,
        "python_version": hardware.python_version,
        "torch_version": hardware.torch_version,
        "cuda_version": hardware.cuda_version,
        "device_type": hardware.device_type,
        "device_name": hardware.device_name,
        "device_count": hardware.device_count,
        "device_total_memory_bytes": hardware.total_memory_bytes,
        "compute_capability": hardware.compute_capability,
        "processor": platform.processor(),
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }


def build_record(
    *,
    run_id: str,
    model: str,
    **kwargs: Any,
) -> BenchmarkRecord:
    """Build a record with the environment fields filled in automatically.

    Raises
    ------
    ConfigError
        If ``model`` or ``run_id`` is empty.
    """
    if not model:
        raise ConfigError("build_record requires a non-empty model identifier")
    if not run_id:
        raise ConfigError("build_record requires a non-empty run_id")

    env = environment_snapshot()
    defaults: dict[str, Any] = {
        "platform": env["platform"],
        "python_version": env["python_version"],
        "torch_version": env["torch_version"],
        "cuda_version": env["cuda_version"],
        "device": env["device_type"],
        "gpu_name": env["device_name"] if env["device_type"] == "cuda" else None,
        "gpu_total_memory_bytes": env["device_total_memory_bytes"],
        "gpu_compute_capability": env["compute_capability"],
        "environment": env,
    }
    # Caller-supplied values win over environment defaults.
    defaults.update(kwargs)
    return BenchmarkRecord(run_id=run_id, model=model, **defaults)


__all__ = [
    "SCHEMA_VERSION",
    "BenchmarkRecord",
    "MetricAggregate",
    "build_record",
    "environment_snapshot",
    "git_commit",
    "git_is_dirty",
    "validate_record",
]
