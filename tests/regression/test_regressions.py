"""Regression tests.

Every test in this file pins a bug that was actually found and fixed while
building UniqKache, or a measurement error that would have produced a wrong
conclusion. They exist because these particular failures were *silent* or
*plausible-looking*: each one would have been reported as a result rather than
noticed as a defect.

The project's rule is that a discovered bug gets a regression test in the same
change that fixes it. See CONTRIBUTING.md.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tests.conftest import HEAD_DIM, NUM_KV_HEADS
from uniqkache.bench.config import RunSpec
from uniqkache.bench.runner import _warmup
from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.store import _gather_quantized
from uniqkache.cache.types import CacheConfig
from uniqkache.compression.quantize import Int8KVCompressor, quantize
from uniqkache.metrics.quality import perplexity, random_token_ids
from uniqkache.metrics.report import load_records, records_to_markdown
from uniqkache.models.synthetic import build_model, get_preset
from uniqkache.policies import FullCachePolicy, SlidingWindowPolicy

# ---------------------------------------------------------------------------
# Bug 1 — quantisation granularity: the reduction axis was confused with the
# granularity it produces, so evicting a compressed layer indexed past the end
# of a size-1 dimension and raised IndexError.
# ---------------------------------------------------------------------------


class TestQuantisationAxisRegression:
    """Pins: per-token reduces over head_dim; per-channel reduces over sequence.

    The original code documented the axes backwards *and* gathered the wrong
    affine parameter when evicting a quantised layer. The symptom was an
    ``IndexError: index 1 is out of bounds for dimension 0 with size 1`` — which
    at least failed loudly. Had the shapes happened to line up, the failure mode
    would instead have been silently corrupted dequantisation of every surviving
    token.
    """

    def test_reduction_axis_determines_the_scale_shape(self):
        x = torch.randn(1, NUM_KV_HEADS, 32, HEAD_DIM)
        # Reduce over the sequence -> one scale per channel -> sequence extent 1.
        per_channel = quantize(x, axis=2)
        assert per_channel.scale.shape[2] == 1
        assert per_channel.scale.shape[3] == HEAD_DIM

        # Reduce over head_dim -> one scale per token -> sequence extent T.
        per_token = quantize(x, axis=3)
        assert per_token.scale.shape[2] == 32
        assert per_token.scale.shape[3] == 1

    @pytest.mark.parametrize("axis", [2, 3])
    def test_gathering_a_quantised_tensor_preserves_dequantisation(self, axis: int):
        """The exact operation that raised: eviction on a compressed layer."""
        x = torch.randn(1, NUM_KV_HEADS, 32, HEAD_DIM)
        q = quantize(x, axis=axis)
        indices = torch.tensor([0, 7, 19, 31])

        gathered = _gather_quantized(q, indices).dequantize()
        expected = q.dequantize()[:, :, indices, :]
        assert torch.allclose(gathered, expected, atol=1e-4)

    def test_evicting_a_compressed_layer_does_not_raise(self):
        config = CacheConfig(
            num_layers=1,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            capacity=64,
            attention_sinks=1,
        )
        cache = KVCache(config, policy=SlidingWindowPolicy(), auto_enforce=False)
        keys = torch.randn(1, NUM_KV_HEADS, 32, HEAD_DIM)
        cache.append(0, keys, keys)
        cache.compress(0)

        dropped = cache.evict(0, keep=torch.arange(8))
        assert dropped == 24
        assert cache.store.layer(0).is_compressed, "compression must survive eviction"
        restored, _ = cache.get(0)
        assert restored.shape[2] == 8

    def test_default_compressor_uses_kivi_granularity(self):
        """Keys per-channel, values per-token — the KIVI recommendation."""
        compressor = Int8KVCompressor()
        assert compressor.key_axis == 2
        assert compressor.value_axis == 3


# ---------------------------------------------------------------------------
# Bug 1b — asymmetric int8 quantisation silently degraded when a slice's values
# were all one sign.
# ---------------------------------------------------------------------------


class TestAsymmetricQuantisationRegression:
    """Pins: the observed range must include zero.

    The affine map is ``q = x/scale + zp`` with ``zp = qmin - xmin/scale``. When
    every value in a slice is positive, ``xmin > 0`` pushes ``zp`` below
    ``qmin``; storing it as an int8 clamps it, while the quantised data was
    computed with the unclamped value. The two then disagree and dequantisation
    is wrong.

    Measured before the fix: max error 0.115 for asymmetric versus 0.016 for
    symmetric — asymmetric was several times *worse* than the scheme it exists
    to improve on, while looking perfectly healthy.
    """

    @pytest.mark.parametrize("axis", [2, 3])
    def test_asymmetric_is_no_worse_than_symmetric(self, axis: int):
        x = torch.randn(1, NUM_KV_HEADS, 32, HEAD_DIM)
        asym_error = (quantize(x, axis=axis, symmetric=False).dequantize() - x).abs().max().item()
        sym_error = (quantize(x, axis=axis, symmetric=True).dequantize() - x).abs().max().item()
        assert asym_error < 2.0 * sym_error, (
            f"asymmetric quantisation (error {asym_error:.5f}) is much worse than "
            f"symmetric ({sym_error:.5f}) on axis {axis}; the zero-point is probably "
            "not representable"
        )

    def test_all_positive_slice_round_trips_accurately(self):
        """The exact pathological case: every value positive, so xmin > 0."""
        x = torch.rand(1, NUM_KV_HEADS, 16, HEAD_DIM) + 0.5  # strictly positive
        q = quantize(x, axis=3, symmetric=False)
        error = (q.dequantize() - x).abs().max().item()
        # Range is ~1.0 over 255 levels, so the error must be ~1/255.
        assert error < 0.02, f"positive-only slice quantised poorly: {error}"

    def test_zero_point_stays_representable(self):
        x = torch.rand(1, NUM_KV_HEADS, 16, HEAD_DIM) + 0.5
        q = quantize(x, axis=3, symmetric=False)
        assert q.zero_point.min().item() >= -128
        assert q.zero_point.max().item() <= 127

    def test_symmetric_mode_has_a_zero_zero_point(self):
        q = quantize(torch.randn(1, NUM_KV_HEADS, 8, HEAD_DIM), axis=3, symmetric=True)
        assert torch.all(q.zero_point == 0)


# ---------------------------------------------------------------------------
# Bug 2 — the synthetic model's perplexity saturated at infinity, making the
# quality diagnostic useless.
# ---------------------------------------------------------------------------


class TestPerplexitySaturationRegression:
    """Pins: untrained models must not saturate the quality diagnostic.

    PyTorch's default ``nn.Embedding`` initialisation is ``N(0, 1)``. With a
    small hidden size that produced logits of magnitude ~137 and a cross-entropy
    around 118, so perplexity was ``inf`` for the *full-cache reference*.

    That is fatal for this project: the whole purpose of the synthetic model is
    that perplexity must *respond* when a policy discards information, and it
    cannot respond from infinity. Scaling initialisation to ``std=0.02``, as
    GPT-2 and Llama do, restores a usable dynamic range.
    """

    def test_untrained_logits_are_not_enormous(self):
        model = build_model(config=get_preset("tiny"), seed=0, device="cpu", dtype=torch.float32)
        tokens = random_token_ids(model.config.vocab_size, 32, seed=0)
        with torch.no_grad():
            logits, _ = model(tokens)
        # Sanity bound: default N(0,1) embedding init pushed this past 100.
        assert logits.abs().max().item() < 20.0, (
            "logits are large enough that perplexity may saturate; check weight "
            "initialisation in SyntheticCausalLM._init_weights"
        )

    def test_full_cache_perplexity_is_finite_and_near_uniform(self):
        config = get_preset("tiny")
        model = build_model(config=config, seed=0, device="cpu", dtype=torch.float32)
        result = perplexity(model, random_token_ids(config.vocab_size, 64, seed=0))

        assert result.value is not None
        assert math.isfinite(result.value)
        # An untrained model should be near uniform: perplexity ~= vocab_size.
        assert 0.25 * config.vocab_size < result.value < 4.0 * config.vocab_size

    def test_the_diagnostic_has_room_to_move(self):
        """Eviction must measurably change perplexity, in some direction.

        Direction is deliberately not asserted. On a randomly-initialised model
        there is no learned long-range dependency for eviction to destroy, so
        aggressive eviction does not reliably make perplexity worse — an earlier
        version of this test asserted that it did and was wrong. The property
        that must hold is *sensitivity*: an invariant perplexity would mean the
        metric is not reading the cache at all.
        """
        config = get_preset("tiny")
        model = build_model(config=config, seed=0, device="cpu", dtype=torch.float32)
        tokens = random_token_ids(config.vocab_size, 96, seed=3)

        reference = perplexity(model, tokens)
        bounded_config = config.cache_config(capacity=4, attention_sinks=1, dtype=torch.float32)
        altered = perplexity(
            model, tokens, cache=KVCache(bounded_config, policy=SlidingWindowPolicy()), chunk_size=1
        )

        assert reference.value is not None and altered.value is not None
        relative_change = abs(altered.value - reference.value) / reference.value
        assert relative_change > 1e-3, (
            "perplexity is invariant to cache contents; the quality diagnostic "
            "cannot detect information loss"
        )


# ---------------------------------------------------------------------------
# Bug 3 — warmup did not exercise the eviction path, so the first bounded run
# absorbed a one-time cost and looked anomalous.
# ---------------------------------------------------------------------------


class TestWarmupCoversEvictionRegression:
    """Pins: warmup must exercise every code path the measurement will take.

    With a warmup that only ran the plain forward pass, the first *bounded* run
    paid the first-call cost of the eviction path (policy scoring, a descending
    argsort, a gather) inside its timed prefill. It measured ~292ms TTFT against
    ~40ms for its neighbours — a 7x artefact that would have been read as a
    property of the policy rather than of the harness.
    """

    def test_warmup_on_a_bounded_spec_actually_evicts(self, monkeypatch):
        calls: list[int] = []
        original = KVCache.enforce_capacity

        def counting_enforce(self, layer_idx=None):
            calls.append(1)
            return original(self, layer_idx)

        monkeypatch.setattr(KVCache, "enforce_capacity", counting_enforce)

        spec = RunSpec(
            model="synthetic:tiny",
            policy="sliding_window",
            context_length=64,
            keep_ratio=0.5,
            attention_sinks=2,
            measure_quality=False,
        )
        from uniqkache.bench.runner import _build_synthetic, resolve_dtype

        built = _build_synthetic(spec, resolve_dtype("float32"), "cpu")
        _warmup(spec, built, dtype=torch.float32, device="cpu")

        assert calls, (
            "warmup did not exercise the eviction path; the first bounded run would "
            "absorb its one-time cost inside the timed region"
        )

    def test_warmup_on_an_unbounded_spec_needs_no_eviction(self):
        spec = RunSpec(
            model="synthetic:tiny", policy="full_cache", context_length=64, measure_quality=False
        )
        from uniqkache.bench.runner import _build_synthetic, resolve_dtype

        built = _build_synthetic(spec, resolve_dtype("float32"), "cpu")
        # Must simply run cleanly: an unbounded cache never evicts.
        _warmup(spec, built, dtype=torch.float32, device="cpu")


# ---------------------------------------------------------------------------
# Bug 4 — `--sweep` validated a single budget-less spec before expanding it,
# so a valid sweep was rejected as a configuration error.
# ---------------------------------------------------------------------------


class TestSweepExpansionRegression:
    """Pins: a sweep must not be validated as a single run first.

    Each sweep run has a *different* budget, so validating one budget-less spec
    up front rejected a perfectly valid sweep with "policy evicts, but no budget
    was given".
    """

    def test_sweep_runs_each_carry_a_budget(self):
        from uniqkache.bench.config import percent_sweep

        config = percent_sweep(model="synthetic:tiny", policy="sliding_window", context_length=1024)
        for run in config.runs:
            if run.policy != "full_cache":
                assert run.resolved_capacity is not None

    def test_budgetless_evicting_spec_is_still_rejected_on_its_own(self):
        """The validation that caused the false rejection is still correct alone."""
        from uniqkache.utils.errors import ConfigError

        with pytest.raises(ConfigError, match="no budget was given"):
            RunSpec(policy="sliding_window", context_length=1024)


# ---------------------------------------------------------------------------
# Bug 5 — a policy alias was accepted by the registry but not re-exported,
# so the CLI failed at import time with a confusing ImportError.
# ---------------------------------------------------------------------------


class TestPublicApiRegression:
    """Pins: anything the CLI imports must be exported from its package."""

    def test_policy_helpers_are_importable_from_the_package(self):
        from uniqkache.policies import (  # noqa: F401
            available_aliases,
            available_policies,
            resolve_name,
        )

        assert resolve_name("h2o") == "attention_based"

    def test_metrics_report_cli_is_importable(self):
        from uniqkache.metrics.report import main  # noqa: F401

    def test_bench_cli_is_importable(self):
        from uniqkache.bench.cli import main  # noqa: F401


# ---------------------------------------------------------------------------
# Bug 6 — the results reporter globbed `*.json`, which matches the
# `<name>-<timestamp>.config.json` files the runner writes beside every result.
# Those are experiment configs, not records: they carry `name`, `runs` and
# `problems` and no `run_id`, so parsing one raised
# `TypeError: missing 1 required positional argument: 'run_id'`.
#
# The symptom was that `make results-table` failed on precisely the directory
# it is designed to be pointed at -- one the runner had populated itself.
# ---------------------------------------------------------------------------


class TestResultsReporterRegression:
    """Pins: a runner-populated results directory must be summarisable."""

    def _write_runner_output(self, directory: Path) -> None:
        """Write a results directory shaped exactly as the runner writes it."""
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": "full_cache-synthetic:tiny-128-deadbeef",
            "schema_version": "1.0.0",
            "model": "synthetic:tiny",
            "policy": "full_cache",
            "context_length": 128,
            "generated_tokens": 4,
            "ttft_ms": 10.0,
            "tpot_ms": 1.0,
            "tokens_per_second": 100.0,
            "quality_metric": "perplexity",
            "quality_value": 500.0,
        }
        (directory / "run-20260101-000000.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
        (directory / "run-20260101-000000.config.json").write_text(
            json.dumps(
                {
                    "name": "single-run",
                    "runs": [{"model": "synthetic:tiny", "policy": "full_cache"}],
                    "problems": {"full_cache-synthetic:tiny-128-deadbeef": ["a warning"]},
                }
            ),
            encoding="utf-8",
        )

    def test_a_runner_populated_directory_loads(self, tmp_path: Path):
        self._write_runner_output(tmp_path)
        records = load_records(tmp_path)
        assert len(records) == 1
        assert records[0].policy == "full_cache"

    def test_the_config_file_is_never_parsed_as_a_record(self, tmp_path: Path):
        self._write_runner_output(tmp_path)
        # The exact failure mode: `name`/`runs`/`problems` are not record fields,
        # and `run_id` is absent. Loading must not reach that code path at all.
        records = load_records(tmp_path)
        assert all(record.run_id for record in records)

    def test_it_renders_a_markdown_table_with_a_quality_column(self, tmp_path: Path):
        self._write_runner_output(tmp_path)
        rendered = records_to_markdown(load_records(tmp_path))
        assert "full_cache" in rendered
        # Quality is a mandatory column; a table without it would let a
        # memory/latency trade-off read as a win.
        assert "quality" in rendered.lower()


# ---------------------------------------------------------------------------
# Invariants that must never regress, regardless of which bug exposed them
# ---------------------------------------------------------------------------


class TestStandingInvariants:
    def test_a_full_cache_discards_nothing(self):
        """The baseline's defining property: occupancy equals the sequence length."""
        config = CacheConfig(
            num_layers=2, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, dtype=torch.float32
        )
        cache = KVCache(config, policy=FullCachePolicy())
        for _ in range(20):
            tensor = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
            for layer in range(2):
                cache.append(layer, tensor, tensor)
            cache.advance()
        assert cache.num_tokens(0) == 20
        assert cache.stats().evictions == 0

    def test_offloading_never_changes_total_bytes(self):
        """Offload relocates; it does not free. A CPU cache cannot offload."""
        config = CacheConfig(
            num_layers=1, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, dtype=torch.float32
        )
        cache = KVCache(config, policy=FullCachePolicy())
        tensor = torch.randn(1, NUM_KV_HEADS, 8, HEAD_DIM)
        cache.append(0, tensor, tensor)
        before = cache.stats()
        cache.offload(0, target="cpu")
        after = cache.stats()
        assert after.bytes_total == before.bytes_total

    def test_eviction_never_increases_occupancy(self):
        config = CacheConfig(
            num_layers=1,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            capacity=16,
            attention_sinks=1,
        )
        cache = KVCache(config, policy=SlidingWindowPolicy())
        previous = 0
        for _ in range(40):
            tensor = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
            cache.append(0, tensor, tensor)
            cache.advance()
            current = cache.num_tokens(0)
            assert current >= previous or current == config.capacity
            assert current <= config.capacity
            previous = current
