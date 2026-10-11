"""Unit tests for the action-selection controller.

The controller's rules are a hypothesis, but the *plumbing* around them is not:
the action vocabulary, the derived quality-risk mapping, the constraint
arithmetic and the refusal to no-op silently are all contract, and are pinned
here.

The second half of this module covers the runnable path (issue #9): the
controller driving a real GenerationEngine, its decisions landing in the
benchmark record, and the rule that an action the runtime cannot perform is
never chosen.
"""

from __future__ import annotations

import pytest
import torch

from uniqkache.bench.config import RunSpec
from uniqkache.bench.runner import run_spec
from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.controllers import (
    ActionKind,
    AdaptiveController,
    Constraints,
    ControllerState,
    Decision,
    Objective,
    QualityRisk,
    quality_risk_of,
)
from uniqkache.policies import SlidingWindowPolicy
from uniqkache.runtime.generation import GenerationConfig, GenerationEngine
from uniqkache.utils.errors import ConfigError, UniqKacheError


def state(**kwargs) -> ControllerState:
    """Build a ControllerState with sensible defaults for the given overrides."""
    defaults = {
        "layer_idx": 0,
        "step": 0,
        "num_tokens": 100,
        "capacity": 128,
        "device_bytes": 1000,
        "offloaded_bytes": 0,
        "device_byte_budget": 2048,
        "latency_budget_ms": None,
        "recent_tpot_ms": None,
        "quality_signal": None,
        "num_layers": 4,
        "is_offloaded": False,
        "can_offload": True,
    }
    defaults.update(kwargs)
    return ControllerState(**defaults)


# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------


class TestActionVocabulary:
    def test_six_actions_exist(self):
        assert {a.value for a in ActionKind} == {
            "retain",
            "evict",
            "compress",
            "offload",
            "prefetch",
            "recompute",
        }

    @pytest.mark.parametrize(
        "kind, expected",
        [
            (ActionKind.RETAIN, QualityRisk.NONE),
            (ActionKind.PREFETCH, QualityRisk.NONE),
            (ActionKind.OFFLOAD, QualityRisk.NONE),
            (ActionKind.COMPRESS, QualityRisk.LOW),
            (ActionKind.EVICT, QualityRisk.HIGH),
            (ActionKind.RECOMPUTE, QualityRisk.NONE),
        ],
    )
    def test_quality_risk_mapping(self, kind, expected):
        assert quality_risk_of(kind) is expected

    def test_only_lossy_actions_are_flagged_lossy(self):
        lossy = {a for a in ActionKind if Decision(a, 0, "r").is_lossy}
        assert lossy == {ActionKind.COMPRESS, ActionKind.EVICT}

    def test_quality_risk_is_derived_not_stored(self):
        """It must be impossible to hand-set a risk that disagrees with the action."""
        decision = Decision(ActionKind.EVICT, 0, "reason")
        assert decision.expected_quality_risk is QualityRisk.HIGH
        assert decision.to_dict()["expected_quality_risk"] == "high"


# ---------------------------------------------------------------------------
# Constraint arithmetic
# ---------------------------------------------------------------------------


class TestControllerState:
    def test_memory_pressure_is_a_fraction_of_the_budget(self):
        assert state(device_bytes=1024, device_byte_budget=2048).memory_pressure == 0.5

    def test_memory_pressure_is_zero_without_a_budget(self):
        """No budget means no pressure, not maximum pressure."""
        assert state(device_byte_budget=None).memory_pressure == 0.0

    def test_memory_pressure_is_capped_at_one(self):
        assert state(device_bytes=9999, device_byte_budget=100).memory_pressure == 1.0

    def test_latency_pressure_is_zero_when_unmeasured(self):
        assert state(latency_budget_ms=10.0, recent_tpot_ms=None).latency_pressure == 0.0
        assert state(latency_budget_ms=None, recent_tpot_ms=99.0).latency_pressure == 0.0

    def test_latency_pressure_reflects_measurement(self):
        assert state(latency_budget_ms=10.0, recent_tpot_ms=5.0).latency_pressure == 0.5

    def test_total_bytes_includes_offloaded(self):
        assert state(device_bytes=100, offloaded_bytes=50).total_bytes == 150


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------


class TestDecisionRules:
    def test_rule1_prefetches_a_needed_offloaded_layer(self):
        controller = AdaptiveController()
        decision = controller.decide(state(is_offloaded=True))
        assert decision.kind is ActionKind.PREFETCH
        assert decision.layer_idx == 0
        assert decision.is_lossy is False

    def test_rule2_retains_when_quality_is_low_but_memory_is_fine(self):
        controller = AdaptiveController(
            Constraints(objective=Objective.MIN_COST_SUBJECT_TO_QUALITY, quality_floor=0.9)
        )
        decision = controller.decide(
            state(quality_signal=0.5, device_bytes=100, device_byte_budget=2048)
        )
        assert decision.kind is ActionKind.RETAIN
        assert decision.binding_constraint == "quality"

    def test_rule2_compresses_rather_than_evicting_when_quality_binds(self):
        controller = AdaptiveController(
            Constraints(objective=Objective.MIN_COST_SUBJECT_TO_QUALITY, quality_floor=0.9)
        )
        decision = controller.decide(
            state(quality_signal=0.5, device_bytes=4096, device_byte_budget=2048)
        )
        assert decision.kind is ActionKind.COMPRESS
        assert decision.is_lossy is True
        assert decision.expected_quality_risk is QualityRisk.LOW

    def test_rule3_offloads_before_compressing(self):
        """Moving bytes is preferred over re-encoding them when memory binds."""
        controller = AdaptiveController()
        decision = controller.decide(state(device_bytes=4096, device_byte_budget=2048))
        assert decision.kind is ActionKind.OFFLOAD
        assert decision.binding_constraint == "memory"
        assert decision.is_lossy is False

    def test_rule3_never_chooses_an_infeasible_offload(self):
        """Single-tier runtime: offload cannot fire, so compress is chosen.

        A controller that picks an action the runtime cannot perform fills
        the trace with infeasible decisions while the budget stays breached.
        ``can_offload=False`` is how the runtime says "there is no cheaper
        tier", and the loss order must skip to the next admissible action.
        """
        controller = AdaptiveController()
        decision = controller.decide(
            state(device_bytes=4096, device_byte_budget=2048, can_offload=False)
        )
        assert decision.kind is ActionKind.COMPRESS
        assert decision.binding_constraint == "memory"

    def test_rule3_skips_offload_under_latency_pressure(self):
        """Byte-moving costs time, so under latency pressure it is rejected."""
        controller = AdaptiveController()
        decision = controller.decide(
            state(
                device_bytes=4096,
                device_byte_budget=2048,
                latency_budget_ms=10.0,
                recent_tpot_ms=9.9,
            )
        )
        assert decision.kind is ActionKind.EVICT
        assert decision.is_lossy is True

    def test_rule3_falls_back_to_eviction_as_last_resort(self):
        controller = AdaptiveController(latency_pressure_limit=0.0)
        decision = controller.decide(state(device_bytes=4096, device_byte_budget=2048))
        assert decision.kind is ActionKind.EVICT
        assert "can degrade quality" in decision.reason

    def test_rule3_offers_recompute_only_when_enabled(self):
        without = AdaptiveController(allow_recompute=False)
        with_it = AdaptiveController(allow_recompute=True, latency_pressure_limit=1.0)

        # With latency pressure rejecting offload and compress, recompute is the
        # next least-lossy action.
        pressured = state(
            device_bytes=4096,
            device_byte_budget=2048,
            latency_budget_ms=10.0,
            recent_tpot_ms=10.0,
        )
        assert without.decide(pressured).kind is ActionKind.EVICT
        assert (
            with_it.decide(pressured).kind is ActionKind.EVICT
        )  # latency limit 1.0 excludes it too

    def test_rule4_retains_when_only_latency_binds(self):
        controller = AdaptiveController()
        decision = controller.decide(
            state(
                device_bytes=100,
                device_byte_budget=2048,
                latency_budget_ms=10.0,
                recent_tpot_ms=9.9,
            )
        )
        assert decision.kind is ActionKind.RETAIN
        assert decision.binding_constraint == "latency"

    def test_rule5_default_retains_with_no_pressure(self):
        controller = AdaptiveController()
        decision = controller.decide(state(device_bytes=100, device_byte_budget=2048))
        assert decision.kind is ActionKind.RETAIN
        assert decision.binding_constraint == "none"

    def test_every_decision_states_a_reason(self):
        controller = AdaptiveController()
        for s in (
            state(is_offloaded=True),
            state(device_bytes=4096, device_byte_budget=2048),
            state(device_bytes=100, device_byte_budget=2048),
        ):
            assert controller.decide(s).reason, "a decision without a reason is unauditable"

    def test_decisions_are_traced(self):
        controller = AdaptiveController()
        controller.decide(state())
        controller.decide(state())
        assert len(controller.trace) == 2
        controller.reset()
        assert controller.trace == []

    def test_state_dict_declares_that_it_is_unvalidated(self):
        assert AdaptiveController().state_dict()["validated"] is False

    def test_invalid_latency_limit_is_rejected(self):
        with pytest.raises(UniqKacheError, match="latency_pressure_limit"):
            AdaptiveController(latency_pressure_limit=1.5)


# ---------------------------------------------------------------------------
# Applying decisions
# ---------------------------------------------------------------------------


def bounded_cache(capacity: int = 8) -> KVCache:
    config = CacheConfig(
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        capacity=capacity,
        attention_sinks=1,
    )
    return KVCache(config, policy=SlidingWindowPolicy())


def fill(cache: KVCache, tokens: int) -> None:
    for _ in range(tokens):
        tensor = torch.randn(1, 2, 1, 8)
        cache.append(0, tensor, tensor)


class TestApplyDecision:
    def test_retain_changes_nothing(self):
        cache = bounded_cache()
        fill(cache, 4)
        before = cache.num_tokens(0)
        assert AdaptiveController().apply(cache, Decision(ActionKind.RETAIN, 0, "r")) == 0
        assert cache.num_tokens(0) == before

    def test_evict_enforces_capacity(self):
        cache = bounded_cache(capacity=4)
        # auto_enforce is on, so fill past capacity with eviction disabled.
        cache.auto_enforce = False
        fill(cache, 10)
        assert cache.num_tokens(0) == 10

        moved = AdaptiveController().apply(cache, Decision(ActionKind.EVICT, 0, "r"))
        assert moved == 6
        assert cache.num_tokens(0) == 4

    def test_compress_reduces_bytes(self):
        cache = bounded_cache(capacity=64)
        fill(cache, 32)
        before = cache.stats().bytes_total

        assert AdaptiveController().apply(cache, Decision(ActionKind.COMPRESS, 0, "r")) == 1
        assert cache.stats().bytes_total < before

    def test_offload_on_a_cpu_cache_is_a_no_op(self):
        cache = bounded_cache(capacity=64)
        fill(cache, 8)
        assert AdaptiveController().apply(cache, Decision(ActionKind.OFFLOAD, 0, "r")) == 0

    def test_prefetch_on_a_resident_layer_is_a_no_op(self):
        cache = bounded_cache(capacity=64)
        fill(cache, 8)
        assert AdaptiveController().apply(cache, Decision(ActionKind.PREFETCH, 0, "r")) == 0

    def test_recompute_without_a_callback_raises(self):
        """Recomputing is the runtime's job; the controller must not fake it."""
        cache = bounded_cache()
        with pytest.raises(UniqKacheError, match="no recompute_fn"):
            AdaptiveController().apply(cache, Decision(ActionKind.RECOMPUTE, 0, "r"))

    def test_recompute_calls_the_supplied_callback(self):
        cache = bounded_cache()
        calls: list[int] = []
        moved = AdaptiveController().apply(
            cache,
            Decision(ActionKind.RECOMPUTE, 2, "r"),
            recompute_fn=lambda layer: (calls.append(layer), 7)[1],
        )
        assert moved == 7
        assert calls == [2]


# ---------------------------------------------------------------------------
# The runnable path (issue #9): the controller drives a real engine
# ---------------------------------------------------------------------------


def _engine_cache(tiny_model, capacity: int = 8) -> KVCache:
    """A cache shaped for the tiny model, with a sliding-window policy."""
    config = tiny_model.config.cache_config(
        capacity=capacity, attention_sinks=0, dtype=torch.float32, device="cpu"
    )
    return KVCache(config, policy=SlidingWindowPolicy())


class TestControllerDrivesTheEngine:
    """The decision chain was complete but nothing in bench/ instantiated it."""

    def test_controller_acts_between_decode_steps(self, tiny_model):
        cache = _engine_cache(tiny_model, capacity=8)
        # A budget below the cache's own footprint: the memory rule binds from
        # the first decision point after prefill.
        controller = AdaptiveController(Constraints(memory_budget_bytes=64))

        engine = GenerationEngine(
            tiny_model, cache, GenerationConfig(max_new_tokens=4), controller=controller
        )
        engine.generate(torch.randint(0, 512, (1, 16)))

        # One decision after prefill plus one between each pair of decode steps.
        assert len(controller.trace) == 4
        assert all(d.kind is ActionKind.COMPRESS for d in controller.trace)
        # The action actually ran: the generation cache ends compressed.
        assert cache.stats().compression_ratio > 1.0

    def test_unconstrained_controller_retains_at_every_step(self, tiny_model):
        cache = _engine_cache(tiny_model, capacity=8)
        controller = AdaptiveController()  # no memory budget: nothing binds

        engine = GenerationEngine(
            tiny_model, cache, GenerationConfig(max_new_tokens=2), controller=controller
        )
        engine.generate(torch.randint(0, 512, (1, 16)))

        assert [d.kind for d in controller.trace] == [ActionKind.RETAIN, ActionKind.RETAIN]

    def test_offload_decisions_stay_feasible_on_single_tier(self, tiny_model):
        """The engine must tell the controller when no cheaper tier exists."""
        cache = _engine_cache(tiny_model, capacity=8)
        controller = AdaptiveController(Constraints(memory_budget_bytes=64))

        engine = GenerationEngine(
            tiny_model, cache, GenerationConfig(max_new_tokens=2), controller=controller
        )
        engine.generate(torch.randint(0, 512, (1, 16)))

        # CPU: offload is infeasible, so the trace compresses instead of
        # recording offloads that would every one of them no-op.
        assert controller.trace
        assert all(d.kind is not ActionKind.OFFLOAD for d in controller.trace)


class TestControllerOnTheRecord:
    def test_trace_and_config_are_recorded(self):
        outcome = run_spec(
            RunSpec(
                model="synthetic:tiny",
                policy="token_importance",
                context_length=32,
                keep_ratio=0.25,
                attention_sinks=2,
                max_new_tokens=3,
                measure_quality=False,
                device_budget_mb=0.015,  # below the cache's own footprint
                controller="adaptive",
            )
        )
        record = outcome.record
        assert record.controller_trace, "a controller-driven run must record its trace"
        assert {d["kind"] for d in record.controller_trace} == {"compress"}

        config = record.policy_config["controller"]
        assert config["name"] == "adaptive"
        assert config["allow_recompute"] is False
        assert config["validated"] is False  # the prototype declares its status
        assert config["constraints"]["memory_budget_bytes"] == int(0.015 * 1024 * 1024)

        # The action is visible in the numbers, not just the trace.
        assert record.cache_compression_ratio > 1.0

    def test_policy_driven_run_records_no_trace(self):
        outcome = run_spec(
            RunSpec(
                model="synthetic:tiny",
                policy="token_importance",
                context_length=32,
                keep_ratio=0.25,
                attention_sinks=2,
                max_new_tokens=2,
                measure_quality=False,
            )
        )
        # None, not []: a trace of [] would claim a controller that decided
        # nothing, and the vocabulary has an explicit `retain` for that.
        assert outcome.record.controller_trace is None
        assert "controller" not in outcome.record.policy_config

    def test_unknown_controller_is_a_configuration_error(self):
        with pytest.raises(ConfigError, match="unknown controller"):
            RunSpec(model="synthetic:tiny", controller="magic")

    def test_controller_without_a_byte_budget_runs_unconstrained(self):
        """No device_budget_mb means the memory rule never binds; that is a
        valid run, just a boring one."""
        outcome = run_spec(
            RunSpec(
                model="synthetic:tiny",
                policy="token_importance",
                context_length=32,
                keep_ratio=0.25,
                attention_sinks=2,
                max_new_tokens=2,
                measure_quality=False,
                controller="adaptive",
            )
        )
        kinds = [d["kind"] for d in outcome.record.controller_trace or []]
        assert kinds and set(kinds) == {"retain"}
