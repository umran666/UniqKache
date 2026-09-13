"""Experiment configuration.

Every experiment in this repository is described by a configuration file, and
every result is reproducible from that file. That requirement is why the config
is a typed dataclass with validation rather than a loose dictionary: a config
that silently omits a field produces a result that cannot be recreated.

Example
-------
.. code-block:: json

    {
      "name": "policy-sweep",
      "runs": [
        {"model": "synthetic:small", "policy": "full_cache", "context_length": 4096},
        {"model": "synthetic:small", "policy": "sliding_window",
         "context_length": 4096, "keep_ratio": 0.25, "attention_sinks": 4}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from uniqkache.policies import available_policies, resolve_name
from uniqkache.utils.errors import ConfigError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)


@dataclass
class RunSpec:
    """A single benchmark run.

    Parameters
    ----------
    model:
        Model identifier. ``synthetic:<preset>`` selects the built-in
        zero-download model; anything else is treated as a Hugging Face
        repository id.
    policy:
        Registered policy name. Aliases (``h2o``, ``streamingllm``) are
        accepted and resolved to canonical names, and the canonical name is what
        gets recorded.
    context_length:
        Prompt length in tokens.
    keep_ratio:
        Fraction of ``context_length`` to retain. ``1.0`` is a full cache,
        ``0.25`` keeps a quarter. Mutually exclusive with ``capacity``. This is
        the knob behind the 100/75/50/25/10 percent sweeps.
    capacity:
        Explicit per-layer token budget, overriding ``keep_ratio``.
    attention_sinks:
        Leading tokens protected from eviction.
    """

    model: str = "synthetic:tiny"
    policy: str = "full_cache"
    context_length: int = 1024
    batch_size: int = 1
    capacity: int | None = None
    keep_ratio: float | None = None
    attention_sinks: int = 0
    precision: str = "float32"
    device: str = "auto"
    seed: int = 0
    max_new_tokens: int = 8
    task: str = "generation"
    dataset: str | None = None
    measure_quality: bool = True
    quality_chunk_size: int = 1
    compressor: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.context_length < 2:
            raise ConfigError(f"context_length must be >= 2, got {self.context_length}")
        if self.capacity is not None and self.keep_ratio is not None:
            raise ConfigError(
                "pass either capacity or keep_ratio, not both: they both set the "
                "token budget and silently preferring one would make the config "
                "ambiguous"
            )
        if self.keep_ratio is not None and not 0.0 < self.keep_ratio <= 1.0:
            raise ConfigError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.attention_sinks < 0:
            raise ConfigError(f"attention_sinks must be >= 0, got {self.attention_sinks}")
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_new_tokens < 0:
            raise ConfigError(f"max_new_tokens must be >= 0, got {self.max_new_tokens}")
        if self.quality_chunk_size < 1:
            raise ConfigError(f"quality_chunk_size must be >= 1, got {self.quality_chunk_size}")

        # Resolve the policy name early so a typo fails before any model loads.
        try:
            self.policy = resolve_name(self.policy)
            if self.policy not in available_policies():
                raise ConfigError(
                    f"unknown policy {self.policy!r}. Available: {', '.join(available_policies())}"
                )
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(str(exc)) from exc

        if self.resolved_capacity is not None and self.attention_sinks > self.resolved_capacity:
            raise ConfigError(
                f"attention_sinks {self.attention_sinks} exceeds the resolved capacity "
                f"{self.resolved_capacity}; the protected tokens would leave no room "
                "for any evictable token"
            )
        if self.policy != "full_cache" and self.resolved_capacity is None:
            raise ConfigError(
                f"policy {self.policy!r} evicts, but no budget was given. Set "
                "keep_ratio or capacity, or use policy 'full_cache'."
            )

    @property
    def resolved_capacity(self) -> int | None:
        """The token budget this run will actually use."""
        if self.capacity is not None:
            return self.capacity
        if self.keep_ratio is not None:
            return max(1, int(self.context_length * self.keep_ratio))
        return None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["resolved_capacity"] = self.resolved_capacity
        return data

    def with_overrides(self, **kwargs: Any) -> RunSpec:
        """Return a copy with fields replaced, re-validating the result."""
        return replace(self, **kwargs)


@dataclass
class ExperimentConfig:
    """A named collection of runs."""

    name: str
    runs: list[RunSpec]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.runs:
            raise ConfigError(f"experiment {self.name!r} contains no runs")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Parse an experiment config, reporting unknown keys."""
        known = {"name", "runs", "description"}
        unknown = set(data) - known
        if unknown:
            raise ConfigError(
                f"unknown top-level key(s) {sorted(unknown)} in experiment config. "
                f"Expected any of: {sorted(known)}"
            )
        if "runs" not in data:
            raise ConfigError("experiment config must contain a 'runs' list")

        runs: list[RunSpec] = []
        spec_fields = set(RunSpec.__dataclass_fields__)  # type: ignore[attr-defined]
        # `resolved_capacity` is emitted by RunSpec.to_dict() as a convenience
        # for records but is derived, not an input, so it is accepted and
        # ignored on the way back in.
        derived = {"resolved_capacity"}
        for index, raw in enumerate(data["runs"]):
            if not isinstance(raw, dict):
                raise ConfigError(f"runs[{index}] must be an object, got {type(raw).__name__}")
            extra = set(raw) - spec_fields - derived
            if extra:
                raise ConfigError(
                    f"runs[{index}] has unknown key(s) {sorted(extra)}. "
                    f"Valid keys: {sorted(spec_fields)}"
                )
            runs.append(RunSpec(**{k: v for k, v in raw.items() if k in spec_fields}))

        return cls(
            name=data.get("name", "unnamed"),
            runs=runs,
            description=data.get("description", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "runs": [run.to_dict() for run in self.runs],
        }


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from JSON.

    Raises
    ------
    ConfigError
        If the file is missing, malformed, or fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be an object, got {type(data).__name__}")

    config = ExperimentConfig.from_dict(data)
    _log.info("loaded experiment %r with %d run(s) from %s", config.name, len(config.runs), path)
    return config


def percent_sweep(
    *,
    model: str,
    policy: str,
    context_length: int,
    ratios: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25, 0.1),
    attention_sinks: int = 4,
    **kwargs: Any,
) -> ExperimentConfig:
    """Build the standard retention-ratio sweep.

    Produces one run per ratio, which is the sweep the project's research plan
    calls for. ``attention_sinks`` is applied only to bounded runs: a full cache
    has nothing to protect, and setting sinks there would record a meaningless
    non-zero value.
    """
    runs: list[RunSpec] = []
    for ratio in ratios:
        if ratio >= 1.0:
            runs.append(
                RunSpec(
                    model=model,
                    policy="full_cache",
                    context_length=context_length,
                    attention_sinks=0,
                    **kwargs,
                )
            )
        else:
            runs.append(
                RunSpec(
                    model=model,
                    policy=policy,
                    context_length=context_length,
                    keep_ratio=ratio,
                    attention_sinks=attention_sinks,
                    **kwargs,
                )
            )
    return ExperimentConfig(
        name=f"{policy}-retention-sweep",
        description=(
            f"Retention ratio sweep for {policy} at context {context_length}. "
            "Ratio 1.0 is the full-cache reference."
        ),
        runs=runs,
    )


__all__ = [
    "ExperimentConfig",
    "RunSpec",
    "load_config",
    "percent_sweep",
]
