"""Types for the action-selection controller.

The controller's job is to choose, at each decision point, among six actions:

.. code-block:: text

    retain  evict  compress  offload  prefetch  recompute

Two things about this vocabulary matter more than the implementation:

**The actions are not interchangeable.** ``evict`` and ``recompute`` destroy or
discard information. ``compress`` and ``offload`` do not — they change how the
information is stored. ``retain`` does nothing. A controller that reports
"memory reduced" must say *which* of these happened, because only some of them
can degrade quality. :class:`Decision` therefore carries
``expected_quality_risk`` explicitly, and :func:`quality_risk_of` is the single
source of truth for it.

**The objective must be stated.** A controller cannot be evaluated without
knowing what it was asked to optimise. :class:`Constraints` makes the objective
and its budgets explicit, and :class:`Decision` records which constraint was
binding. "It used less memory" is not a result unless the objective said memory
was the thing to minimise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ActionKind(str, Enum):
    """The six actions the controller can choose."""

    RETAIN = "retain"
    EVICT = "evict"
    COMPRESS = "compress"
    OFFLOAD = "offload"
    PREFETCH = "prefetch"
    RECOMPUTE = "recompute"


class QualityRisk(str, Enum):
    """How much a decision can plausibly cost in model quality."""

    NONE = "none"
    LOW = "low"
    HIGH = "high"


class Objective(str, Enum):
    """Which formulation of the optimisation the controller is solving.

    The two formulations in the project's research question, made explicit:

    ``MAX_QUALITY_SUBJECT_TO``
        Maximise quality subject to ``memory <= memory_budget`` and
        ``latency <= latency_budget``. Budgets bind; quality is the objective.
    ``MIN_COST_SUBJECT_TO_QUALITY``
        Minimise memory and latency cost subject to
        ``quality >= quality_floor``. Quality binds; cost is the objective.

    These are genuinely different problems with different optimal policies. A
    result reported under one must not be compared against a baseline tuned for
    the other without saying so.
    """

    MAX_QUALITY_SUBJECT_TO = "max_quality_subject_to"
    MIN_COST_SUBJECT_TO_QUALITY = "min_cost_subject_to_quality"


_QUALITY_RISK: dict[ActionKind, QualityRisk] = {
    ActionKind.RETAIN: QualityRisk.NONE,
    ActionKind.PREFETCH: QualityRisk.NONE,
    ActionKind.OFFLOAD: QualityRisk.NONE,
    ActionKind.COMPRESS: QualityRisk.LOW,
    ActionKind.EVICT: QualityRisk.HIGH,
    ActionKind.RECOMPUTE: QualityRisk.NONE,
}


def quality_risk_of(kind: ActionKind) -> QualityRisk:
    """Return the quality risk associated with an action.

    This is the single source of truth for the mapping, so that no reporting
    path can disagree with another about whether an action is lossy.
    """
    return _QUALITY_RISK[kind]


@dataclass
class Constraints:
    """Explicit budgets and the objective they bound.

    All budgets are optional; ``None`` means "unconstrained", which is recorded
    as such rather than being silently treated as zero.

    Attributes
    ----------
    objective:
        Which formulation is being solved.
    memory_budget_bytes:
        Cap on device-resident cache bytes.
    latency_budget_ms:
        Cap on per-token decode latency.
    quality_floor:
        Minimum acceptable quality metric. Only meaningful under
        ``MIN_COST_SUBJECT_TO_QUALITY``.
    """

    objective: Objective = Objective.MAX_QUALITY_SUBJECT_TO
    memory_budget_bytes: int | None = None
    latency_budget_ms: float | None = None
    quality_floor: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["objective"] = self.objective.value
        return data


@dataclass
class ControllerState:
    """What the controller observes at a decision point.

    Every field is a measurement or a budget. Nothing here is inferred, and
    nothing is a prediction — the controller is given facts and must decide.

    Attributes
    ----------
    layer_idx:
        Layer about to be computed, i.e. the execution cursor.
    step:
        Decode step.
    num_tokens:
        Tokens currently cached in the cursor layer.
    capacity:
        Token budget for the cursor layer, if bounded.
    device_bytes, offloaded_bytes:
        Current cache footprint, reported separately so that offloading cannot
        be mistaken for a reduction in total memory.
    device_byte_budget:
        Cap on device-resident bytes for this decision, if any.
    latency_budget_ms:
        Per-token latency cap, if any. Measured, not modelled: when this is
        ``None`` the controller has no latency information and must not behave
        as though latency were free.
    recent_tpot_ms:
        Observed time-per-output-token, or ``None`` when unmeasured.
    quality_signal:
        Optional current quality estimate, used only under
        ``MIN_COST_SUBJECT_TO_QUALITY``.
    """

    layer_idx: int
    step: int
    num_tokens: int
    capacity: int | None
    device_bytes: int
    offloaded_bytes: int
    device_byte_budget: int | None = None
    latency_budget_ms: float | None = None
    recent_tpot_ms: float | None = None
    quality_signal: float | None = None
    num_layers: int = 1
    is_offloaded: bool = False

    @property
    def total_bytes(self) -> int:
        return self.device_bytes + self.offloaded_bytes

    @property
    def memory_pressure(self) -> float:
        """Device occupancy as a fraction of the device budget.

        Returns 0.0 when there is no budget — not 1.0, and not NaN. An
        unconstrained run is not under pressure, and pretending otherwise would
        make the controller evict for no reason.
        """
        if not self.device_byte_budget:
            return 0.0
        return min(1.0, self.device_bytes / self.device_byte_budget)

    @property
    def latency_pressure(self) -> float:
        """Observed latency as a fraction of the latency budget, or 0.0."""
        if not self.latency_budget_ms or self.recent_tpot_ms is None:
            return 0.0
        return min(1.0, self.recent_tpot_ms / self.latency_budget_ms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Decision:
    """A controller's chosen action, with the reasoning recorded.

    Attributes
    ----------
    kind:
        The action chosen.
    layer_idx:
        Target layer, or ``None`` for "all layers".
    reason:
        Human-readable justification. Required: a decision without a stated
        reason cannot be audited, and auditing is how a policy's behaviour gets
        attributed to a mechanism rather than guessed at.
    expected_quality_risk:
        Derived from ``kind`` via :func:`quality_risk_of`, never set by hand, so
        it cannot drift out of sync with the action.
    binding_constraint:
        Which budget forced this decision (``"memory"``, ``"latency"``,
        ``"quality"``), or ``"none"`` when nothing bound.
    alternatives:
        Other actions that were available and rejected, for the record.
    """

    kind: ActionKind
    layer_idx: int | None
    reason: str
    binding_constraint: str = "none"
    alternatives: list[str] = field(default_factory=list)

    @property
    def expected_quality_risk(self) -> QualityRisk:
        """Quality risk implied by the action. Derived, not stored."""
        return quality_risk_of(self.kind)

    @property
    def is_lossy(self) -> bool:
        """Whether this action can degrade model quality."""
        return self.expected_quality_risk is not QualityRisk.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "layer_idx": self.layer_idx,
            "reason": self.reason,
            "binding_constraint": self.binding_constraint,
            "alternatives": list(self.alternatives),
            "expected_quality_risk": self.expected_quality_risk.value,
            "is_lossy": self.is_lossy,
        }


__all__ = [
    "ActionKind",
    "Constraints",
    "ControllerState",
    "Decision",
    "Objective",
    "QualityRisk",
    "quality_risk_of",
]
