"""Unit tests for memory accounting, latency measurement and quality evaluation."""

from __future__ import annotations

import math
import time

import pytest
import torch

from tests.conftest import HEAD_DIM, NUM_KV_HEADS, NUM_LAYERS
from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.types import CacheConfig
from uniqkache.metrics.latency import LatencyStats, Timer, percentile, throughput
from uniqkache.metrics.memory import snapshot, theoretical_kv_bytes
from uniqkache.metrics.quality import (
    exact_match,
    needle_retrieval,
    perplexity,
    random_token_ids,
)
from uniqkache.models.synthetic import build_model
from uniqkache.policies import FullCachePolicy, SlidingWindowPolicy
from uniqkache.utils.errors import BackendError

# ---------------------------------------------------------------------------
# Memory accounting
# ---------------------------------------------------------------------------


class TestCacheMemoryAccounting:
    def test_bytes_match_the_arithmetic(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(10), kv_factory(10))
        stats = full_cache.stats()

        # 2 (k and v) * 2 kv heads * 10 tokens * 8 head_dim * 4 bytes
        expected = 2 * NUM_KV_HEADS * 10 * HEAD_DIM * 4
        assert stats.bytes_on_device == expected
        assert stats.bytes_total == expected

    def test_unbounded_cache_reports_no_utilization(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(4), kv_factory(4))
        assert full_cache.stats().utilization is None

    def test_utilization_is_fraction_of_capacity(self, kv_factory):
        config = CacheConfig(
            num_layers=1,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            capacity=8,
            attention_sinks=0,
        )
        cache = KVCache(config, policy=SlidingWindowPolicy(), auto_enforce=False)
        cache.append(0, kv_factory(6), kv_factory(6))
        assert cache.stats().utilization == pytest.approx(6 / 8)

    def test_device_and_offloaded_are_reported_separately(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(4), kv_factory(4))
        stats = full_cache.stats()
        assert stats.bytes_offloaded == 0
        assert stats.bytes_on_device == stats.bytes_total

    def test_compression_ratio_is_one_when_uncompressed(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(16), kv_factory(16))
        assert full_cache.stats().compression_ratio == pytest.approx(1.0)

    def test_compression_ratio_exceeds_one_when_compressed(self, full_cache, kv_factory):
        full_cache.append(0, kv_factory(64), kv_factory(64))
        full_cache.compress(0)
        assert full_cache.stats().compression_ratio > 1.0

    def test_theoretical_bytes_matches_a_real_cache(self, full_cache, kv_factory):
        # Populate every layer, so the cache total covers all of them.
        for layer in range(NUM_LAYERS):
            full_cache.append(layer, kv_factory(12), kv_factory(12))
        predicted = theoretical_kv_bytes(full_cache.config, 12)
        assert full_cache.stats().bytes_on_device == predicted
        assert full_cache.stats().total_tokens == 12 * NUM_LAYERS

    def test_snapshot_reports_none_for_unmeasurable_gpu_fields(self, full_cache, kv_factory):
        """A CPU run must report None, not 0, for GPU memory."""
        full_cache.append(0, kv_factory(4), kv_factory(4))
        snap = snapshot(full_cache, device="cpu")
        assert snap.device_peak_bytes is None
        assert snap.device_current_bytes is None

    def test_counters_track_activity(self, kv_factory):
        config = CacheConfig(
            num_layers=1,
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.float32,
            capacity=4,
            attention_sinks=1,
        )
        cache = KVCache(config, policy=SlidingWindowPolicy())
        for _ in range(10):
            cache.append(0, kv_factory(1), kv_factory(1))
            cache.advance()

        stats = cache.stats()
        assert stats.evictions > 0
        assert stats.total_tokens == 4
        assert stats.max_tokens_in_layer == 4

    def test_stats_are_json_serialisable(self, full_cache, kv_factory):
        import json

        full_cache.append(0, kv_factory(4), kv_factory(4))
        payload = json.dumps(full_cache.stats().to_dict())
        assert "bytes_total" in payload


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


class TestLatency:
    def test_percentile_of_empty_is_none(self):
        assert percentile([], 0.5) is None

    def test_percentile_of_single_sample(self):
        assert percentile([4.0], 0.9) == 4.0

    def test_percentile_interpolates(self):
        assert percentile([0.0, 10.0], 0.5) == pytest.approx(5.0)

    def test_percentile_rejects_bad_q(self):
        with pytest.raises(ValueError, match="q must be in"):
            percentile([1.0], 1.5)

    def test_stats_of_empty_samples_are_all_none(self):
        stats = LatencyStats.from_samples([])
        assert stats.count == 0
        assert stats.mean_ms is None
        assert stats.p50_ms is None
        assert stats.summary() == "latency: no samples"

    def test_stats_summarise_samples(self):
        stats = LatencyStats.from_samples([1.0, 2.0, 3.0, 4.0])
        assert stats.count == 4
        assert stats.mean_ms == pytest.approx(2.5)
        assert stats.min_ms == 1.0
        assert stats.max_ms == 4.0
        assert stats.p50_ms == pytest.approx(2.5)

    def test_raw_samples_are_dropped_by_default(self):
        assert LatencyStats.from_samples([1.0, 2.0]).samples == []
        assert LatencyStats.from_samples([1.0, 2.0], keep_samples=True).samples == [1.0, 2.0]

    def test_throughput_of_degenerate_input_is_none(self):
        assert throughput(0, 100.0) is None
        assert throughput(10, 0.0) is None
        assert throughput(10, 1000.0) == pytest.approx(10.0)

    def test_timer_elapsed_is_none_until_used(self):
        # A timer that has not run must not report a duration; ``0.0`` would be
        # indistinguishable from an instantaneous measurement.
        assert Timer("cpu").elapsed_ms is None

    @staticmethod
    def _burn(seconds: float) -> None:
        """Busy-wait until ``seconds`` of wall clock have actually passed.

        Deliberately not ``time.sleep``. ``sleep`` is a *minimum* hint that the
        OS may cut short -- on Windows the ~15.6 ms timer granularity makes an
        early return routine, and the previous version of this test was measured
        failing at 8.39 ms for a requested 10 ms.

        The deadline is taken *inside* the caller's timing block on purpose. If
        it were computed before entering the block, the ``synchronize`` in
        ``Timer.__enter__`` would be charged against the wait and the body would
        consume slightly less than requested -- which is how the first rewrite
        of this test still managed to fail, at 9.91 ms.
        """
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            pass

    def test_timer_measures_elapsed_time(self):
        # Spin for double the asserted duration, so ordinary scheduling jitter
        # cannot push the measurement below the bound.
        with Timer("cpu") as timer:
            self._burn(0.02)
        assert timer.elapsed_ms is not None
        assert timer.elapsed_ms >= 10.0

    def test_timer_elapsed_grows_with_the_body(self):
        # The stronger property: the timer tracks the work done, not a constant.
        def measure(seconds: float) -> float:
            with Timer("cpu") as timer:
                self._burn(seconds)
            assert timer.elapsed_ms is not None
            return timer.elapsed_ms

        short = measure(0.005)
        long = measure(0.050)
        assert long > short


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


class TestQualityPrimitives:
    def test_random_token_ids_are_deterministic(self):
        a = random_token_ids(100, 20, seed=1)
        b = random_token_ids(100, 20, seed=1)
        c = random_token_ids(100, 20, seed=2)
        assert torch.equal(a, b)
        assert not torch.equal(a, c)

    def test_random_token_ids_respect_vocab_bounds(self):
        tokens = random_token_ids(7, 100, seed=0)
        assert tokens.min() >= 0 and tokens.max() < 7

    def test_random_token_ids_reject_bad_arguments(self):
        with pytest.raises(BackendError, match="vocab_size"):
            random_token_ids(0, 10)
        with pytest.raises(BackendError, match="length"):
            random_token_ids(10, 0)

    def test_exact_match_identical(self):
        a = torch.tensor([[1, 2, 3]])
        assert exact_match(a, a) == 1.0

    def test_exact_match_partial(self):
        a = torch.tensor([[1, 2, 3, 4]])
        b = torch.tensor([[1, 9, 3, 9]])
        assert exact_match(a, b) == pytest.approx(0.5)

    def test_exact_match_rejects_shape_mismatch(self):
        with pytest.raises(BackendError, match="identical shapes"):
            exact_match(torch.tensor([[1, 2]]), torch.tensor([[1, 2, 3]]))


class TestPerplexity:
    def test_perplexity_is_finite_and_near_uniform_for_an_untrained_model(self, model_config):
        """The diagnostic must not be saturated at the full-cache reference."""
        model = build_model(config=model_config, seed=0, device="cpu", dtype=torch.float32)
        tokens = random_token_ids(model_config.vocab_size, 64, seed=0)

        result = perplexity(model, tokens)
        assert result.value is not None
        assert math.isfinite(result.value), "perplexity saturated; the diagnostic is useless"
        # An untrained model should sit near uniform prediction: ln(vocab).
        assert result.value == pytest.approx(model_config.vocab_size, rel=0.5)

    def test_perplexity_flags_random_weights_as_not_interpretable(self, model_config):
        model = build_model(config=model_config, seed=0, device="cpu", dtype=torch.float32)
        result = perplexity(model, random_token_ids(model_config.vocab_size, 32, seed=0))
        assert result.is_interpretable is False
        assert "NOT valid as model quality" in result.caveat

    def test_cached_and_uncached_perplexity_agree_for_a_full_cache(self, model_config):
        """A full cache must reproduce the no-cache result: it discards nothing."""
        model = build_model(config=model_config, seed=0, device="cpu", dtype=torch.float32)
        tokens = random_token_ids(model_config.vocab_size, 48, seed=1)

        uncached = perplexity(model, tokens)
        cache = KVCache(model_config.cache_config(dtype=torch.float32), policy=FullCachePolicy())
        cached = perplexity(model, tokens, cache=cache, chunk_size=1)

        assert cached.value == pytest.approx(uncached.value, rel=1e-3)

    def test_eviction_measurably_changes_perplexity(self, model_config):
        """The quality metric must be sensitive to what the cache retains.

        Note carefully what is asserted: that perplexity *changes*, not that it
        gets worse. On a randomly-initialised model the direction is not
        guaranteed, because there is no learned long-range dependency for
        eviction to destroy — the attention pattern is arbitrary to begin with.
        Degradation *is* observed at aggressive retention on larger synthetic
        configurations (see ``docs/research.md``), but it is not a reliable
        property of this diagnostic.

        What would invalidate the whole pipeline is perplexity being *invariant*
        to cache contents: that would mean either eviction is not happening or
        the metric is not reading the cache. That is what this test rules out.
        """
        model = build_model(config=model_config, seed=0, device="cpu", dtype=torch.float32)
        tokens = random_token_ids(model_config.vocab_size, 128, seed=2)

        reference = perplexity(model, tokens)

        config = model_config.cache_config(capacity=4, attention_sinks=1, dtype=torch.float32)
        evicting = KVCache(config, policy=SlidingWindowPolicy())
        altered = perplexity(model, tokens, cache=evicting, chunk_size=1)

        assert altered.value is not None and reference.value is not None
        relative_change = abs(altered.value - reference.value) / reference.value
        assert relative_change > 1e-3, (
            f"perplexity was insensitive to retaining 4 tokens instead of all "
            f"{tokens.shape[1]} ({altered.value:.4f} vs {reference.value:.4f}); either "
            "eviction is not happening or the quality metric is not reading the cache"
        )

    def test_too_short_a_sequence_is_reported_not_scored(self, model_config):
        model = build_model(config=model_config, seed=0, device="cpu", dtype=torch.float32)
        with pytest.raises(BackendError, match="at least 2 tokens"):
            perplexity(model, torch.tensor([[1]]))

    def test_invalid_chunk_size_is_rejected(self, model_config):
        model = build_model(config=model_config, seed=0, device="cpu", dtype=torch.float32)
        with pytest.raises(BackendError, match="chunk_size"):
            perplexity(model, torch.tensor([[1, 2, 3]]), chunk_size=0)


class TestNeedleRetrieval:
    def test_needle_retrieval_returns_a_bounded_score(self, model_config):
        model = build_model(config=model_config, seed=0, device="cpu", dtype=torch.float32)
        cache = KVCache(model_config.cache_config(dtype=torch.float32), policy=FullCachePolicy())
        result = needle_retrieval(
            model,
            haystack_length=64,
            needle=torch.tensor([[7, 8, 9]]),
            vocab_size=model_config.vocab_size,
            cache=cache,
        )
        assert result.value is not None
        assert 0.0 <= result.value <= 1.0

    def test_needle_retrieval_rejects_a_too_short_haystack(self, model_config):
        model = build_model(config=model_config, seed=0, device="cpu", dtype=torch.float32)
        with pytest.raises(BackendError, match="too short"):
            needle_retrieval(
                model, haystack_length=2, needle=torch.tensor([[1, 2, 3]]), vocab_size=16
            )

    def test_needle_retrieval_rejects_bad_depth(self, model_config):
        model = build_model(config=model_config, seed=0, device="cpu", dtype=torch.float32)
        with pytest.raises(BackendError, match="depth"):
            needle_retrieval(
                model, haystack_length=32, needle=torch.tensor([[1]]), vocab_size=16, depth=2.0
            )

    def test_multi_token_needle_without_a_cache_is_rejected(self, model_config):
        model = build_model(config=model_config, seed=0, device="cpu", dtype=torch.float32)
        with pytest.raises(BackendError, match="supply a cache"):
            needle_retrieval(
                model, haystack_length=32, needle=torch.tensor([[1, 2]]), vocab_size=16
            )
