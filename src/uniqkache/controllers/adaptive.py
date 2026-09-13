"""Adaptive controller: chooses among the six cache actions.

STATUS: **research prototype — decision rules unvalidated.**

What this is
------------
A concrete implementation of the second half of the project's research question:
given explicit constraints, choose among ``retain``, ``evict``, ``compress``,
``offload``, ``prefetch`` and ``recompute``.

The rules below are a *hypothesis about which action is appropriate when*, not a
finding. They are written as an explicit, ordered chain so that:

* every decision carries a reason and a record of the alternatives rejected;
* a unit test can assert the rule that fired;
* an ablation can disable a rule and see what changes.

What this is **not**
--------------------
* It is not validated. Nothing here has been benchmarked against the fixed
  policies in this repository. Do not describe it as an improvement.
* The thresholds are priors, not fitted values.
* ``recompute`` is exposed as an action but no runtime in this milestone acts on
  it automatically; the caller must supply a recomputation callback or the
  controller raises rather than silently no-opping.

Decision chain
--------------
Evaluated in order; the first rule that applies wins.

1. **Needed-now prefetch** — the layer about to be computed is offloaded.
2. **Quality floor** — under ``MIN_COST_SUBJECT_TO_QUALITY`` with a quality
   signal below the floor, lossy actions are removed from consideration.
3. **Over memory budget** — pick the least lossy action that fits, preferring
   ``offload`` → ``compress`` → ``recompute`` → ``evict``, and dropping
   candidates that would breach the latency budget.
4. **Latency pressure only** — nothing to do about memory; retain.
5. **Default** — retain.

The ordering in rule 3 is the substantive claim: it encodes the belief that
*moving or re-encoding bytes is preferable to discarding them*. That belief is
what the ablation is meant to test, not to assume.
"""

from __future__ import annotations

from collections.abc import Callable

from uniqkache.cache.kv_cache import KVCache
from uniqkache.controllers.types import (
    ActionKind,
    Constraints,
    ControllerState,
    Decision,
    Objective,
)
from uniqkache.utils.errors import UniqKacheError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)

#: Actions ordered from least to most quality-destructive. The controller walks
#: this order when it needs to shed memory, which is the operational form of the
#: claim that representation changes beat information loss.
_LOSS_ORDER: tuple[ActionKind, ...] = (
    ActionKind.OFFLOAD,
    ActionKind.COMPRESS,
    ActionKind.RECOMPUTE,
    ActionKind.EVICT,
)

#: Above this latency pressure, an action that moves bytes is considered too
#: expensive to be worth its memory saving. A prior, not a measurement.
_LATENCY_PRESSURE_LIMIT = 0.9


class AdaptiveController:
    """Constraint-driven action selection over a :class:`KVCache`.

    Parameters
    ----------
    constraints:
        Explicit budgets and the objective they bound.
    allow_recompute:
        Whether ``recompute`` is an admissible action. Off by default, because
        no runtime in this milestone acts on it automatically.
    latency_pressure_limit:
        Fraction of the latency budget above which byte-moving actions are
        rejected as too expensive. Prior, not fitted.
    """

    name = "adaptive"

    def __init__(
        self,
        constraints: Constraints | None = None,
        *,
        allow_recompute: bool = False,
        latency_pressure_limit: float = _LATENCY_PRESSURE_LIMIT,
    ) -> None:
        if not 0.0 <= latency_pressure_limit <= 1.0:
            raise UniqKacheError(
                f"latency_pressure_limit must be in [0, 1], got {latency_pressure_limit}"
            )
        self.constraints = constraints or Constraints()
        self.allow_recompute = allow_recompute
        self.latency_pressure_limit = latency_pressure_limit
        self.trace: list[Decision] = []

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def decide(self, state: ControllerState) -> Decision:
        """Choose an action for the current state.

        Rules are evaluated in the order documented in the module docstring.
        The first rule that returns a decision wins, and its ``alternatives``
        field lists what else was available.
        """
        for rule in (
            self._rule_needed_now,
            self._rule_quality_floor,
            self._rule_over_memory_budget,
            self._rule_latency_pressure,
            self._rule_default,
        ):
            decision = rule(state)
            if decision is not None:
                self.trace.append(decision)
                return decision

        # Every rule above returns a decision (the last is unconditional), so
        # reaching here means the chain was edited incorrectly.
        raise UniqKacheError(
            "controller decision chain produced no decision; the final rule must be unconditional"
        )

    def _rule_needed_now(self, state: ControllerState) -> Decision | None:
        """Rule 1: the cursor layer is offloaded and is about to be used."""
        if not state.is_offloaded:
            return None
        return Decision(
            kind=ActionKind.PREFETCH,
            layer_idx=state.layer_idx,
            reason=(
                f"layer {state.layer_idx} is offloaded and is the execution cursor; "
                "bring it back before compute"
            ),
            binding_constraint="none",
            alternatives=["proceed without prefetch (would block on transfer)"],
        )

    def _rule_quality_floor(self, state: ControllerState) -> Decision | None:
        """Rule 2: quality is binding and already below the floor.

        Under this rule the controller refuses to take a lossy action. If it
        cannot shed memory losslessly it retains and reports infeasibility,
        rather than quietly trading quality away to satisfy a memory budget the
        objective never said was allowed to win.
        """
        if self.constraints.objective is not Objective.MIN_COST_SUBJECT_TO_QUALITY:
            return None
        floor = self.constraints.quality_floor
        if floor is None or state.quality_signal is None:
            return None
        if state.quality_signal >= floor:
            return None

        over_budget = (
            state.device_byte_budget is not None and state.device_bytes > state.device_byte_budget
        )
        if not over_budget:
            return Decision(
                kind=ActionKind.RETAIN,
                layer_idx=state.layer_idx,
                reason=(
                    f"quality {state.quality_signal:.4f} is below the floor {floor:.4f}; "
                    "retaining because no memory pressure requires action"
                ),
                binding_constraint="quality",
                alternatives=["evict", "compress", "offload"],
            )

        if state.latency_pressure < self.latency_pressure_limit:
            return Decision(
                kind=ActionKind.COMPRESS,
                layer_idx=None,
                reason=(
                    f"quality {state.quality_signal:.4f} < floor {floor:.4f} and device "
                    f"bytes {state.device_bytes} > budget {state.device_byte_budget}; "
                    "compressing preserves every cached token, unlike eviction"
                ),
                binding_constraint="quality",
                alternatives=["offload", "evict (rejected: lossy, quality is binding)"],
            )

        return Decision(
            kind=ActionKind.RETAIN,
            layer_idx=state.layer_idx,
            reason=(
                "quality below floor and latency pressure too high to compress or "
                "offload; retaining and reporting the constraint conflict instead of "
                "silently trading quality"
            ),
            binding_constraint="quality",
            alternatives=["evict (rejected: quality is binding)", "compress (rejected: latency)"],
        )

    def _rule_over_memory_budget(self, state: ControllerState) -> Decision | None:
        """Rule 3: device bytes exceed the device budget. Shed memory."""
        budget = state.device_byte_budget
        if budget is None or state.device_bytes <= budget:
            return None

        over_by = state.device_bytes - budget
        candidates = self._admissible_actions(state)
        rejected: list[str] = []

        for kind in candidates:
            if kind is ActionKind.EVICT:
                return Decision(
                    kind=kind,
                    layer_idx=state.layer_idx,
                    reason=(
                        f"device bytes {state.device_bytes} exceed budget {budget} by "
                        f"{over_by}; no lossless action available"
                        + (f" ({'; '.join(rejected)})" if rejected else "")
                        + ". Evicting, which can degrade quality — reported as lossy."
                    ),
                    binding_constraint="memory",
                    alternatives=rejected,
                )
            return Decision(
                kind=kind,
                layer_idx=None,
                reason=(
                    f"device bytes {state.device_bytes} exceed budget {budget} by "
                    f"{over_by}; choosing {kind.value} as the least quality-destructive "
                    "action that fits the latency budget"
                ),
                binding_constraint="memory",
                alternatives=rejected + [k.value for k in candidates if k is not kind],
            )

        return None  # unreachable: EVICT is always admissible

    def _admissible_actions(self, state: ControllerState) -> list[ActionKind]:
        """Least-lossy-first list of actions permitted in this state."""
        out: list[ActionKind] = []
        for kind in _LOSS_ORDER:
            if kind is ActionKind.RECOMPUTE and not self.allow_recompute:
                continue
            # Byte-moving and byte-rewriting actions cost time; under high
            # latency pressure only eviction (which is free) remains.
            if (
                kind in {ActionKind.OFFLOAD, ActionKind.COMPRESS, ActionKind.RECOMPUTE}
                and state.latency_pressure >= self.latency_pressure_limit
            ):
                continue
            out.append(kind)
        # EVICT is always last-resort admissible; the caller distinguishes the
        # case where it is the only option by inspecting the reason string.
        if ActionKind.EVICT not in out:
            out.append(ActionKind.EVICT)
        return out

    def _rule_latency_pressure(self, state: ControllerState) -> Decision | None:
        """Rule 4: latency budget is under pressure but memory is fine."""
        if state.latency_pressure < self.latency_pressure_limit:
            return None
        return Decision(
            kind=ActionKind.RETAIN,
            layer_idx=state.layer_idx,
            reason=(
                f"latency {state.recent_tpot_ms:.3f}ms is at "
                f"{state.latency_pressure:.0%} of budget {state.latency_budget_ms}ms, but "
                "device memory is within budget; retaining because no cache action "
                "reduces attention cost"
            ),
            binding_constraint="latency",
            alternatives=["compress", "offload (both would add transfer cost)"],
        )

    def _rule_default(self, state: ControllerState) -> Decision:
        """Rule 5: nothing is binding. Do nothing."""
        if state.device_byte_budget is None:
            detail = "no device byte budget is set"
        else:
            detail = f"device bytes {state.device_bytes} within budget {state.device_byte_budget}"
        return Decision(
            kind=ActionKind.RETAIN,
            layer_idx=state.layer_idx,
            reason=f"retaining: {detail}",
            binding_constraint="none",
            alternatives=[],
        )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(
        self,
        cache: KVCache,
        decision: Decision,
        *,
        recompute_fn: Callable[[int], int] | None = None,
    ) -> int:
        """Execute ``decision`` against ``cache``.

        Returns the number of units affected: tokens evicted, layers compressed
        or offloaded, layers prefetched, or tokens recomputed. Returns 0 for
        ``retain``.

        Raises
        ------
        UniqKacheError
            If the decision is ``recompute`` and no ``recompute_fn`` was given.
            Recomputing is the runtime's responsibility — re-running the model
            for a layer is not something the cache can do — so we refuse rather
            than report success for work that never happened.
        """
        kind = decision.kind
        layer = decision.layer_idx

        if kind is ActionKind.RETAIN:
            return 0
        if kind is ActionKind.EVICT:
            return cache.enforce_capacity(layer)
        if kind is ActionKind.COMPRESS:
            return cache.compress(layer)
        if kind is ActionKind.OFFLOAD:
            return cache.offload(layer, target="cpu")
        if kind is ActionKind.PREFETCH:
            return cache.prefetch(layer)
        if kind is ActionKind.RECOMPUTE:
            if recompute_fn is None:
                raise UniqKacheError(
                    "decision is RECOMPUTE but no recompute_fn was supplied. Recomputing "
                    "requires re-running the model, which the cache cannot do. Pass "
                    "recompute_fn=... or set allow_recompute=False."
                )
            if layer is None:
                raise UniqKacheError("RECOMPUTE requires a concrete layer index")
            return recompute_fn(layer)

        raise UniqKacheError(f"unhandled action kind {kind!r}")

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the decision trace."""
        self.trace.clear()

    def state_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "constraints": self.constraints.to_dict(),
            "allow_recompute": self.allow_recompute,
            "latency_pressure_limit": self.latency_pressure_limit,
            "validated": False,
        }

    def __repr__(self) -> str:
        return (
            f"AdaptiveController(objective={self.constraints.objective.value!r}, "
            f"memory_budget={self.constraints.memory_budget_bytes}, "
            f"latency_budget_ms={self.constraints.latency_budget_ms})"
        )


__all__ = ["AdaptiveController"]
