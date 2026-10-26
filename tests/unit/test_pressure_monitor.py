"""Unit tests for the memory-pressure monitor (issue #33).

The monitor is the missing feedback loop: today the token budget is fixed at
config time and ``PolicyState.memory_pressure`` is derived from that static
budget, so nothing reacts to *actual* device memory approaching a ceiling.
These tests pin the mapping (observed bytes -> token capacity), the two
stability rules (shrink-only, minimum interval), and the record-level
consequence: the runtime budget that was actually realised.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from uniqkache.bench.config import RunSpec
from uniqkache.bench.runner import run_spec
from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.policies import SlidingWindowPolicy
from uniqkache.runtime.generation import GenerationConfig, GenerationEngine
from uniqkache.runtime.pressure import MemoryPressureMonitor
from uniqkache.utils.errors import ConfigError

# synthetic:tiny's shape: 4 layers x 2 kv heads x 32 head_dim x 2 (K and V) x 4 B.
BYTES_PER_TOKEN = 2048
BUDGET = 10 * BYTES_PER_TOKEN  # a ceiling that fits exactly 10 tokens


def _config(capacity: int | None = 16) -> CacheConfig:
    return CacheConfig(
        num_layers=4,
        num_kv_heads=2,
        head_dim=32,
        dtype=torch.float32,
        device="cpu",
        capacity=capacity,
    )


def _monitor(**kwargs) -> MemoryPressureMonitor:
    return MemoryPressureMonitor(_config(), BUDGET, **kwargs)


# ---------------------------------------------------------------------------
# The byte -> token mapping
# ---------------------------------------------------------------------------


class TestMapping:
    def test_over_budget_observation_tightens_to_the_budget(self):
        monitor = _monitor()
        new_capacity = monitor.observe(
            step=0, cache_bytes=32 * BYTES_PER_TOKEN, current_capacity=32
        )
        # Aimed at the low watermark, not the ceiling: the tightened budget
        # must leave the next observation quiet, not back at pressure.
        assert new_capacity == 7  # floor(10 tokens * 0.75)
        assert monitor.events[-1].old_capacity == 32
        assert monitor.events[-1].new_capacity == 7

    def test_under_budget_observation_does_nothing(self):
        monitor = _monitor()
        assert monitor.observe(step=0, cache_bytes=5 * BYTES_PER_TOKEN, current_capacity=32) is None
        assert monitor.events == []

    def test_non_cache_bytes_are_reserved_before_the_cache(self):
        """A whole-device ceiling must pay for weights and activations first."""
        monitor = _monitor()
        # 6 tokens' worth of non-cache allocation on top of 32 tokens of cache.
        new_capacity = monitor.observe(
            step=0,
            cache_bytes=32 * BYTES_PER_TOKEN,
            current_capacity=32,
            total_device_bytes=38 * BYTES_PER_TOKEN,
        )
        assert new_capacity == 1  # floor(7 - 6) tokens' worth

    def test_an_unbounded_cache_can_be_given_a_budget(self):
        """The "would OOM at full cache" case: no budget yet, pressure binding."""
        monitor = _monitor()
        new_capacity = monitor.observe(
            step=0, cache_bytes=32 * BYTES_PER_TOKEN, current_capacity=None
        )
        assert new_capacity == 7
        assert monitor.events[-1].old_capacity is None

    def test_already_tight_enough_is_not_re_tightened(self):
        """A cache already within the budget maps to itself: no event, no churn."""
        monitor = _monitor()
        assert monitor.observe(step=0, cache_bytes=7 * BYTES_PER_TOKEN, current_capacity=7) is None
        assert monitor.events == []

    def test_pressure_is_reported_against_the_budget(self):
        monitor = _monitor()
        monitor.observe(step=0, cache_bytes=20 * BYTES_PER_TOKEN, current_capacity=20)
        assert monitor.events[-1].pressure == pytest.approx(2.0)

    def test_exhausted_budget_floors_at_one_token(self):
        """Issue #114: non-cache bytes eating the whole low-watermark share
        must floor the derived capacity at 1, never 0 — the cache rejects
        capacity 0 and the run would crash mid-generation."""
        monitor = _monitor()
        # 9 tokens of non-cache allocation against a 7-token low-watermark
        # target: tokens_for_bytes would derive 0.
        new_capacity = monitor.observe(
            step=0,
            cache_bytes=0,
            current_capacity=64,
            total_device_bytes=9 * BYTES_PER_TOKEN,
        )
        assert new_capacity == 1
        assert monitor.events[-1].new_capacity == 1


# ---------------------------------------------------------------------------
# Stability rules
# ---------------------------------------------------------------------------


class TestStability:
    def test_minimum_interval_holds_back_retightening(self):
        monitor = _monitor(min_interval=8)
        assert monitor.observe(step=0, cache_bytes=32 * BYTES_PER_TOKEN, current_capacity=32) == 7
        # Still over budget, but too soon: no second adaptation.
        assert (
            monitor.observe(step=4, cache_bytes=32 * BYTES_PER_TOKEN, current_capacity=10) is None
        )
        assert len(monitor.events) == 1

    def test_repeated_observations_never_grow_the_budget(self):
        """Shrink-only is what makes the loop stable; growth would oscillate."""
        monitor = _monitor(min_interval=1)
        capacities = [32]
        for step in range(6):
            new = monitor.observe(
                step=step, cache_bytes=40 * BYTES_PER_TOKEN, current_capacity=capacities[-1]
            )
            if new is not None:
                assert new < capacities[-1]
                capacities.append(new)
        assert len(capacities) >= 2
        assert capacities == sorted(capacities, reverse=True)

    def test_low_pressure_after_tightening_is_stable(self):
        """The tightened budget must land below the watermark, not on its edge."""
        monitor = _monitor(high_watermark=0.9, low_watermark=0.75)
        monitor.observe(step=0, cache_bytes=BUDGET, current_capacity=BUDGET // BYTES_PER_TOKEN)
        new_capacity = monitor.events[-1].new_capacity
        assert new_capacity * BYTES_PER_TOKEN / BUDGET < 0.9

    def test_target_capacity_follows_the_low_watermark(self):
        monitor = _monitor(low_watermark=0.5)
        assert monitor.target_capacity == 5


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_positive_budget_is_rejected(self):
        with pytest.raises(ConfigError, match="device_byte_budget must be > 0"):
            MemoryPressureMonitor(_config(), 0)

    def test_watermark_ordering_is_enforced(self):
        with pytest.raises(ConfigError, match="watermarks must satisfy"):
            _monitor(high_watermark=0.5, low_watermark=0.9)

    def test_min_interval_must_be_positive(self):
        with pytest.raises(ConfigError, match="min_interval must be >= 1"):
            _monitor(min_interval=0)

    def test_per_layer_capacity_list_is_refused(self):
        config = CacheConfig(
            num_layers=2,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
            capacity=[4, 8],
        )
        with pytest.raises(ConfigError, match="per-layer capacity list"):
            MemoryPressureMonitor(config, BUDGET)


# ---------------------------------------------------------------------------
# Driving a real cache and a real run
# ---------------------------------------------------------------------------


class TestMonitorDrivesACache:
    def _cache(self, tiny_model, capacity: int) -> KVCache:
        config = tiny_model.config.cache_config(
            capacity=capacity, attention_sinks=0, dtype=torch.float32, device="cpu"
        )
        return KVCache(config, policy=SlidingWindowPolicy())

    def test_engine_tightens_between_steps_and_stays_in_budget(self, tiny_model):
        cache = self._cache(tiny_model, capacity=64)
        budget = 5 * BYTES_PER_TOKEN  # the ceiling fits 5 tokens
        monitor = MemoryPressureMonitor(cache.config, budget)

        engine = GenerationEngine(
            tiny_model,
            cache,
            GenerationConfig(max_new_tokens=3),
            pressure_monitor=monitor,
        )
        engine.generate(torch.randint(0, 512, (1, 16)))

        assert monitor.events, "pressure crossed the watermark but nothing was tightened"
        # Aimed at the low watermark: floor(5 * 0.75) = 3 tokens.
        assert monitor.events[-1].new_capacity == 3
        final_capacity = cache.config.capacity_for_layer(0)
        assert final_capacity == 3
        assert cache.store.bytes_on_device() <= budget
        # Every layer was brought within the tightened budget.
        assert all(t <= final_capacity for t in cache.store.tokens_per_layer())

    def test_monitor_without_a_policy_cannot_enforce_and_says_so(self, tiny_model):
        """A bounded cache needs a policy to decide what survives."""
        config = tiny_model.config.cache_config(
            capacity=None, attention_sinks=0, dtype=torch.float32, device="cpu"
        )
        cache = KVCache(config, policy=None)
        monitor = MemoryPressureMonitor(cache.config, 5 * BYTES_PER_TOKEN)

        assert monitor.update(cache, step=0) is None
        assert cache.config.capacity is None

    def test_per_layer_budget_installed_later_is_skipped(self, tiny_model):
        cache = self._cache(tiny_model, capacity=64)
        monitor = MemoryPressureMonitor(cache.config, 5 * BYTES_PER_TOKEN)
        cache.set_capacity([4, 8, 4, 8])
        assert monitor.update(cache, step=0) is None
        assert monitor.events == []

    def test_state_dict_reports_the_adaptations(self):
        monitor = _monitor()
        monitor.observe(step=3, cache_bytes=32 * BYTES_PER_TOKEN, current_capacity=32)
        state = monitor.state_dict()
        assert state["device_byte_budget"] == BUDGET
        assert state["adaptations"][0]["step"] == 3

    def test_exhausted_budget_applies_the_one_token_floor_without_crashing(self, tiny_model):
        """Issue #114 end-to-end: update() must apply capacity 1, not raise
        CacheConfigError from set_capacity(0)."""
        cache = self._cache(tiny_model, capacity=64)
        budget = 4 * BYTES_PER_TOKEN
        monitor = MemoryPressureMonitor(cache.config, budget)

        # Observed total allocation leaves nothing for the cache: the
        # low-watermark target is 3 tokens, non-cache share is 4.
        new_capacity = monitor.update(
            cache, step=0, total_device_bytes=4 * BYTES_PER_TOKEN
        )
        assert new_capacity == 1
        assert cache.config.capacity_for_layer(0) == 1


class TestAdaptiveEvictionOnTheRecord:
    """The record must carry the realised budget, not just the initial value."""

    SPEC: ClassVar[dict] = {
        "model": "synthetic:tiny",
        "policy": "sliding_window",
        "context_length": 64,
        "keep_ratio": 0.25,
        "attention_sinks": 2,
        "max_new_tokens": 4,
        "measure_quality": False,
        "device_budget_mb": 0.015,  # ~7 tokens' worth; the cache holds 16
        "seed": 0,
    }

    def test_adaptive_run_records_the_tightened_budget(self):
        outcome = run_spec(RunSpec(**self.SPEC, adaptive_eviction=True))
        record = outcome.record

        assert record.adaptation_events, "an adaptive run that tightened must show it"
        event = record.adaptation_events[0]
        assert event["old_capacity"] == 16
        # The ceiling fits 7 tokens; the tightened budget aims at 75% of it.
        assert event["new_capacity"] == 5
        assert event["pressure"] > 1.0
        # The realised budget is the tightened one, and it was enforced.
        assert record.capacity == event["new_capacity"]
        assert record.cache_bytes_total <= int(0.015 * 1024 * 1024)

    def test_static_run_records_no_adaptation(self):
        outcome = run_spec(RunSpec(**self.SPEC))
        assert outcome.record.adaptation_events == []
        assert outcome.record.capacity == 16

    def test_adaptive_eviction_requires_a_device_budget(self):
        with pytest.raises(ConfigError, match="adaptive_eviction needs a device byte budget"):
            RunSpec(
                model="synthetic:tiny",
                policy="sliding_window",
                context_length=64,
                keep_ratio=0.25,
                max_new_tokens=2,
                adaptive_eviction=True,
            )
