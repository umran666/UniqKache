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

from tests.conftest import HEAD_DIM, NUM_KV_HEADS, good_record
from uniqkache.bench.config import RunSpec
from uniqkache.bench.runner import RunOutcome, _warmup, write_results
from uniqkache.cache.kv_cache import KVCache
from uniqkache.cache.store import _gather_quantized
from uniqkache.cache.types import CacheConfig
from uniqkache.compression.quantize import Int8KVCompressor, quantize
from uniqkache.metrics.quality import perplexity, random_token_ids
from uniqkache.metrics.record import validate_record
from uniqkache.metrics.report import load_records, records_to_markdown
from uniqkache.models.synthetic import build_cache_for_model, build_model, get_preset
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
# Bug 7 — the quality pass discarded the attention weights, so `cum_attention`
# stayed all-zero for the whole evaluation. Every attention-based policy then
# scored every token equally and silently fell back to a deterministic tie
# order, meaning the reported quality number described a policy that was never
# actually run.
#
# The symptom was subtle rather than loud: four different policies
# (sliding_window, lru, token_importance, adaptive) produced *bit-identical*
# perplexity at an equal budget, which is not what four different retention
# rules should do.
# ---------------------------------------------------------------------------


class TestQualityPassRecordsAttentionRegression:
    """Pins: a policy that uses attention must be fed attention when scored."""

    def _cache(self, policy_name: str, capacity: int = 8):
        model = build_model(preset="tiny", device="cpu")
        cache = build_cache_for_model(
            model,
            capacity=capacity,
            attention_sinks=1,
            policy_name=policy_name,
            device="cpu",
        )
        return model, cache

    def test_attention_accumulates_during_a_cached_perplexity_pass(self):
        model, cache = self._cache("attention_based")
        tokens = random_token_ids(model.config.vocab_size, 48, seed=0)

        perplexity(model, tokens, cache=cache, chunk_size=1)

        totals = [
            float(cache.store.layer(i).metadata.cum_attention.abs().sum())
            for i in range(model.config.num_layers)
        ]
        assert any(total > 0 for total in totals), (
            "no attention was recorded during the quality pass; an attention-based "
            "policy would have scored every token equally"
        )

    def test_a_non_attention_policy_does_not_pay_for_attention(self):
        """The guard must not over-correct and materialise attention needlessly.

        Materialising attention costs memory proportional to the context length,
        so requesting it for a policy that does not read it would distort that
        policy's own memory measurement.
        """
        model, cache = self._cache("sliding_window")
        assert cache.policy is not None and not cache.policy.uses_attention

        tokens = random_token_ids(model.config.vocab_size, 48, seed=0)
        perplexity(model, tokens, cache=cache, chunk_size=1)

        totals = [
            float(cache.store.layer(i).metadata.cum_attention.abs().sum())
            for i in range(model.config.num_layers)
        ]
        assert all(total == 0 for total in totals)

    def test_the_quality_diagnostic_distinguishes_retention_rules(self):
        """The point of the fix: two different rules must not score identically.

        Before it, an attention-based policy and a recency-based policy could
        return the same perplexity at the same budget, because neither was
        reading attention.
        """
        values = {}
        for policy_name in ("attention_based", "sliding_window"):
            model, cache = self._cache(policy_name)
            tokens = random_token_ids(model.config.vocab_size, 48, seed=0)
            values[policy_name] = perplexity(model, tokens, cache=cache, chunk_size=1).value

        assert values["attention_based"] is not None
        assert values["sliding_window"] is not None
        assert values["attention_based"] != values["sliding_window"]


# ---------------------------------------------------------------------------
# Bug 8 — the runner wrote its artifacts with the platform's line endings, so
# on Windows every generated file disagreed with the LF normalisation in
# `.gitattributes` and git warned on each add. Harmless in itself, but a warning
# that fires on every result commit is one people learn to skip, and the next
# warning to appear alongside it would be skipped too.
# ---------------------------------------------------------------------------


class TestResultArtifactLineEndingsRegression:
    """Pins: results are written with LF endings on every platform."""

    def test_write_results_emits_lf_only(self, tmp_path: Path):
        outcome = RunOutcome(record=good_record(), generation=None, quality=None, problems=[])
        paths = write_results([outcome], tmp_path, "line-endings")

        assert set(paths) == {"jsonl", "csv", "config"}
        for label, path in paths.items():
            # Read as bytes: text mode would translate the endings away and
            # hide exactly the bug this test exists for.
            raw = path.read_bytes()
            assert b"\r\n" not in raw, f"{label} was written with CRLF endings"
            assert raw.endswith(b"\n"), f"{label} does not end with a newline"

    def test_the_written_jsonl_round_trips(self, tmp_path: Path):
        outcome = RunOutcome(record=good_record(), generation=None, quality=None, problems=[])
        paths = write_results([outcome], tmp_path, "round-trip")
        assert load_records(paths["jsonl"])[0].run_id == "r1"

    def test_the_written_config_is_not_mistaken_for_a_record(self, tmp_path: Path):
        outcome = RunOutcome(record=good_record(), generation=None, quality=None, problems=[])
        write_results([outcome], tmp_path, "config-vs-record")
        # Bug 6's failure mode, checked against real runner output rather than a
        # hand-built fixture: the config file sits beside the record and must be
        # skipped by the loader.
        assert len(load_records(tmp_path)) == 1


# ---------------------------------------------------------------------------
# Bug 9 — `KVCache.evict()` counted an eviction when nothing was dropped.
#
# The increment was unconditional, whereas `enforce_capacity` increments only
# when something was evicted. A no-op evict on an empty cache, or one retaining
# every token, bumped `stats().evictions` — so a result could quote a counter
# claiming an eviction that removed zero tokens.
# ---------------------------------------------------------------------------


class TestEvictAccountingRegression:
    """Pins: an eviction counter must not claim a no-op dropped tokens."""

    def test_noop_evict_on_an_empty_cache_reports_zero_evictions(self):
        config = CacheConfig(
            num_layers=1, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, dtype=torch.float32
        )
        cache = KVCache(config, policy=FullCachePolicy())

        cache.evict(0, indices=torch.tensor([0]))
        # Before the fix this reported 1: an eviction that dropped nothing.
        assert cache.stats().evictions == 0
        assert cache.state_dict()["evictions"] == 0

    def test_the_two_eviction_paths_agree_on_a_noop(self):
        config = CacheConfig(
            num_layers=1,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            capacity=8,
            attention_sinks=1,
        )
        cache = KVCache(config, policy=SlidingWindowPolicy())
        for _ in range(8):
            tensor = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM)
            cache.append(0, tensor, tensor)

        # evict(keep=all) keeps every token; enforce_capacity on an at-budget
        # cache drops nothing. Both must report zero evictions and agree.
        assert cache.evict(0, keep=torch.arange(cache.num_tokens(0))) == 0
        assert cache.evict(0) == 0
        assert cache.stats().evictions == 0


# ---------------------------------------------------------------------------
# Bug 10 — the HF model revision was present in the record schema but could not
# be specified in a run and was therefore never forwarded to the loader.
# ---------------------------------------------------------------------------


class TestHFRevisionPropagationRegression:
    """Pins: a RunSpec revision reaches the HF loader without downloading."""

    def test_run_spec_revision_reaches_hf_loader(self, monkeypatch):
        from typing import ClassVar

        from uniqkache.models import hf_backend

        spec = RunSpec(model="org/model", model_revision="abc123", policy="full_cache")
        captured = {}

        class StubBackend:
            identifier = spec.model
            revision = spec.model_revision
            tokenizer = None
            weights_are_random = False
            vocab_size = 32
            num_parameters = 7
            config: ClassVar[dict[str, object]] = {"stub": True}

        def stub_from_pretrained(model_id, **kwargs):
            captured["model_id"] = model_id
            captured.update(kwargs)
            return StubBackend()

        monkeypatch.setattr(hf_backend.HFBackend, "from_pretrained", stub_from_pretrained)

        built = hf_backend.build_hf_model(spec, dtype=torch.float32, device="cpu")

        assert captured == {
            "model_id": "org/model",
            "dtype": torch.float32,
            "device": "cpu",
            "revision": "abc123",
            "local_files_only": False,
        }
        assert built.revision == "abc123"

    def test_revision_round_trips_through_experiment_config(self):
        from uniqkache.bench.config import ExperimentConfig

        config = ExperimentConfig.from_dict(
            {
                "name": "pinned",
                "runs": [
                    {
                        "model": "org/model",
                        "model_revision": "v1.2.3",
                        "policy": "full_cache",
                    }
                ],
            }
        )

        restored = ExperimentConfig.from_dict(config.to_dict())
        assert restored.runs[0].model_revision == "v1.2.3"

    def test_model_revision_populated_in_record_for_hf_runs(self, monkeypatch):
        from typing import ClassVar

        from uniqkache.metrics.record import build_record
        from uniqkache.models import hf_backend

        spec = RunSpec(
            model="org/model",
            model_revision="commit-sha-456",
            policy="full_cache",
        )

        class StubBackend:
            identifier = spec.model
            revision = spec.model_revision
            tokenizer = None
            weights_are_random = False
            vocab_size = 32
            num_parameters = 7
            config: ClassVar[dict[str, object]] = {"stub": True}

        monkeypatch.setattr(
            hf_backend.HFBackend,
            "from_pretrained",
            lambda model_id, **kwargs: StubBackend(),
        )

        built = hf_backend.build_hf_model(spec, dtype=torch.float32, device="cpu")
        record = build_record(
            run_id="test-run",
            model=built.identifier,
            model_revision=built.revision,
        )
        assert record.model == "org/model"
        assert record.model_revision == "commit-sha-456"


# ---------------------------------------------------------------------------
# Bug 13 — Benchmark runs reached the network unconditionally and did not record it.
# An --offline flag must be threaded from RunSpec to from_pretrained (local_files_only),
# and the record must state whether the model was resolved offline.
# ---------------------------------------------------------------------------


class TestOfflineNetworkHonestyRegression:
    """Pins: --offline threads to local_files_only, and record states offline resolution."""

    def test_run_spec_offline_reaches_hf_loader(self, monkeypatch):
        from typing import ClassVar

        from uniqkache.models import hf_backend

        spec = RunSpec(model="org/model", offline=True, policy="full_cache")
        captured = {}

        class StubBackend:
            identifier = spec.model
            revision = spec.model_revision
            tokenizer = None
            weights_are_random = False
            vocab_size = 32
            num_parameters = 7
            config: ClassVar[dict[str, object]] = {"stub": True}

        def stub_from_pretrained(model_id, **kwargs):
            captured["model_id"] = model_id
            captured.update(kwargs)
            return StubBackend()

        monkeypatch.setattr(hf_backend.HFBackend, "from_pretrained", stub_from_pretrained)

        built = hf_backend.build_hf_model(spec, dtype=torch.float32, device="cpu")

        assert captured == {
            "model_id": "org/model",
            "dtype": torch.float32,
            "device": "cpu",
            "revision": None,
            "local_files_only": True,
        }
        assert built.is_hf_backend is True

    def test_offline_round_trips_through_experiment_config(self):
        from uniqkache.bench.config import ExperimentConfig

        config = ExperimentConfig.from_dict(
            {
                "name": "offline-exp",
                "runs": [
                    {
                        "model": "org/model",
                        "offline": True,
                        "policy": "full_cache",
                    }
                ],
            }
        )

        restored = ExperimentConfig.from_dict(config.to_dict())
        assert restored.runs[0].offline is True

    def test_model_loaded_offline_recorded_for_hf_and_synthetic(self, monkeypatch):
        from typing import ClassVar

        from uniqkache.bench.runner import run_spec
        from uniqkache.models import hf_backend

        spec_hf_offline = RunSpec(
            model="org/model",
            offline=True,
            policy="full_cache",
            context_length=8,
            max_new_tokens=2,
            measure_quality=False,
        )

        class StubBackend:
            identifier = "org/model"
            revision = None
            tokenizer = None
            weights_are_random = False
            vocab_size = 32
            num_parameters = 7
            config: ClassVar[dict[str, object]] = {"stub": True}
            mirrored_bytes = 0

            def forward(self, input_ids, cache=None, start_pos=0, **kwargs):
                return torch.zeros(1, input_ids.shape[1], 32), None

            def cache_config(
                self,
                capacity=None,
                attention_sinks=0,
                dtype=None,
                device=None,
                batch_size=1,
            ):
                return CacheConfig(
                    num_layers=1, num_kv_heads=1, head_dim=8, dtype=torch.float32, capacity=capacity
                )

        monkeypatch.setattr(
            hf_backend.HFBackend,
            "from_pretrained",
            lambda model_id, **kwargs: StubBackend(),
        )

        outcome_hf = run_spec(spec_hf_offline)
        assert outcome_hf.record.model_loaded_offline is True

        spec_hf_online = spec_hf_offline.with_overrides(offline=False)
        outcome_online = run_spec(spec_hf_online)
        assert outcome_online.record.model_loaded_offline is False

        spec_synthetic = RunSpec(
            model="synthetic:tiny",
            context_length=8,
            max_new_tokens=2,
            measure_quality=False,
            policy="full_cache",
        )
        outcome_synthetic = run_spec(spec_synthetic)
        assert outcome_synthetic.record.model_loaded_offline is None


# ---------------------------------------------------------------------------
# Bug N — `--compressor int8` was a silent no-op: the runner set the compressor
# on the cache but never called compress(), so the record claimed a mechanism
# that never ran, with cache_compression_ratio stuck at 1.0 and zero integrity
# problems. This is exactly the "record claims a mechanism that never ran"
# failure mode the project's reporting rules forbid.
# ---------------------------------------------------------------------------


class TestCompressorIsNotASilentNoOp:
    """Pins: a `--compressor int8` run must be visible in its own record."""

    def _run(self, tmp_path, compressor: str | None):
        from uniqkache.bench.cli import main

        args = [
            "--model",
            "synthetic:tiny",
            "--context-length",
            "64",
            "--max-new-tokens",
            "2",
            "--no-quality",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
        if compressor is not None:
            args += ["--compressor", compressor]
        assert main(args) == 0
        (record,) = load_records(tmp_path)
        return record

    def test_the_compressed_run_differs_from_the_plain_run(self, tmp_path):
        plain = self._run(tmp_path / "plain", None)
        compressed = self._run(tmp_path / "compressed", "int8")

        # Before the fix both records were byte-identical in these fields, and
        # the compressed one claimed compressor="int8" with ratio 1.0.
        assert compressed.compressor == "int8"
        assert compressed.cache_compression_ratio > 1.0
        assert compressed.cache_bytes_total < plain.cache_bytes_total
        assert compressed.cache_final_tokens == plain.cache_final_tokens

    def test_validate_record_catches_the_original_failure_mode(self):
        """The exact record the bug produced must be flagged, forever."""
        problems = validate_record(good_record(compressor="int8", cache_compression_ratio=1.0))
        assert any("compression that is not visible" in p for p in problems)


# ---------------------------------------------------------------------------
# Bug 71 — Declared transformers>=4.40 support conflicted with DynamicCache.layers API
# ---------------------------------------------------------------------------


class TestTransformers440DynamicCacheCompatibilityRegression:
    """Pins: HFBackend._mirror_into_cache supports DynamicCache without .layers.

    In transformers 4.40 - 4.48, DynamicCache stored KV tensors in key_cache
    and value_cache lists without a .layers attribute. Accessing
    self._hf_cache.layers unconditionally caused an AttributeError when
    mirroring into UniqKache cache.
    """

    def test_transformers_440_dynamic_cache_mirrors_successfully(self):
        from uniqkache.models.hf_backend import HFBackend

        class _StubConfig:
            model_type = "stub"
            num_hidden_layers = 1
            num_attention_heads = 4
            num_key_value_heads = 2
            hidden_size = 32
            vocab_size = 32

        # Mimics Transformers 4.40 DynamicCache (key_cache and value_cache, no layers)
        class _DynamicCache440:
            def __init__(self, keys, values):
                self.key_cache = [keys]
                self.value_cache = [values]

            def __getitem__(self, idx):
                return self.key_cache[idx], self.value_cache[idx]

            def __len__(self):
                return len(self.key_cache)

        k = torch.randn(1, 2, 3, 8)
        v = torch.randn(1, 2, 3, 8)

        class _StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _StubConfig()
                self.weight = torch.nn.Parameter(torch.zeros(1, 1))

            def forward(self, input_ids, **kwargs):
                return type(
                    "ModelOutput",
                    (),
                    {
                        "logits": torch.zeros(1, input_ids.shape[1], 32),
                        "past_key_values": _DynamicCache440(k, v),
                    },
                )()

        backend = HFBackend(_StubModel(), identifier="stub-llama", weights_are_random=True)
        # Pre-seed the internal cache slot so forward() does not try to build a
        # real transformers.DynamicCache; the stub ignores past_key_values and
        # returns its own cache structure, which is what this test exercises.
        # CI runs without the hf extra, so _require_transformers must not be hit.
        backend._hf_cache = None
        cache = KVCache(backend.cache_config(capacity=None), policy=None)
        input_ids = torch.tensor([[1, 2, 3]])

        # Should not raise AttributeError: 'DynamicCache440' object has no attribute 'layers'
        logits, _ = backend.forward(input_ids, cache=cache, start_pos=0)
        assert logits.shape == (1, 3, 32)
        assert cache.num_tokens(0) == 3
        per_token = 2 * 8 * 4  # 2 heads * 8 head_dim * 4 bytes/float32 = 64
        assert (
            backend.mirrored_bytes == 2 * 3 * per_token
        )  # k + v = 2 * (3 tokens * 64 bytes) = 384

    def test_legacy_tuple_past_key_values_mirrors_successfully(self):
        from uniqkache.models.hf_backend import HFBackend

        class _StubConfig:
            model_type = "stub"
            num_hidden_layers = 1
            num_attention_heads = 4
            num_key_value_heads = 2
            hidden_size = 32
            vocab_size = 32

        k = torch.randn(1, 2, 3, 8)
        v = torch.randn(1, 2, 3, 8)

        class _StubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _StubConfig()
                self.weight = torch.nn.Parameter(torch.zeros(1, 1))

            def forward(self, input_ids, **kwargs):
                return type(
                    "ModelOutput",
                    (),
                    {
                        "logits": torch.zeros(1, input_ids.shape[1], 32),
                        "past_key_values": ((k, v),),
                    },
                )()

        backend = HFBackend(_StubModel(), identifier="stub-llama", weights_are_random=True)
        # See the sibling 4.40 test: pre-seed so forward() never calls
        # _require_transformers() (CI has no hf extra).
        backend._hf_cache = None
        cache = KVCache(backend.cache_config(capacity=None), policy=None)
        input_ids = torch.tensor([[1, 2, 3]])

        logits, _ = backend.forward(input_ids, cache=cache, start_pos=0)
        assert logits.shape == (1, 3, 32)
        assert cache.num_tokens(0) == 3
        assert backend.mirrored_bytes > 0


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
