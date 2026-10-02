"""Benchmark evidence must be valid, reproducible, and metric-specific."""

import json
from dataclasses import replace

import pytest
import torch

from tests.conftest import good_record
from uniqkache.bench import runner
from uniqkache.bench.config import RunSpec, load_config
from uniqkache.metrics.quality import QualityResult
from uniqkache.metrics.record import validate_record
from uniqkache.metrics.report import load_records, summarize
from uniqkache.utils.errors import ConfigError


@pytest.mark.parametrize("capacity", [0, -1])
def test_invalid_capacity_fails_before_model_build(capacity):
    with pytest.raises(ConfigError, match="capacity must be >= 1"):
        RunSpec(policy="sliding_window", capacity=capacity)


@pytest.mark.parametrize(
    "budget", [{"capacity": 4}, {"keep_ratio": 1.0}, {"memory_budget_mb": 1.0}]
)
def test_full_cache_cannot_be_a_bounded_reference(budget):
    with pytest.raises(ConfigError, match="unbounded reference"):
        RunSpec(policy="full_cache", **budget)


@pytest.mark.parametrize(
    "name,value",
    [
        ("ttft_ms", float("nan")),
        ("tpot_ms", float("inf")),
        ("quality_value", float("nan")),
        ("cache_bytes_total", -1),
        ("tokens_per_second", -1.0),
        ("quality_value", -1.0),
    ],
)
def test_invalid_measurements_are_integrity_failures(name, value):
    problems = validate_record(good_record(**{name: value}))
    assert any(name in problem for problem in problems)


def test_retrieval_quality_is_a_probability():
    assert any(
        "[0, 1]" in p
        for p in validate_record(
            good_record(quality_metric="needle_retrieval", quality_value=2.0, quality_reference=0.0)
        )
    )


def test_reference_memo_separates_config_revision_and_device(monkeypatch):
    spec = RunSpec(policy="sliding_window", capacity=4, context_length=8)
    built = runner._build_synthetic(spec, torch.float32, "cpu")
    memo = {}
    calls = []
    monkeypatch.setattr(runner, "_make_cache", lambda *args, **kwargs: None)

    def quality(*args, **kwargs):
        calls.append(kwargs.get("device"))
        return QualityResult(
            metric="perplexity", value=float(len(calls)), num_tokens=7, is_interpretable=True
        )

    monkeypatch.setattr(runner, "perplexity", quality)
    prompt = torch.ones(1, 8, dtype=torch.long)

    def reference(spec=spec, model=built, device="cpu"):
        return runner._quality_reference(
            spec, model, prompt, dtype=torch.float32, device=device, memo=memo
        )

    assert reference() == reference() == 1.0
    assert reference(model=replace(built, config={**built.config, "hidden_size": 64})) == 2.0
    assert reference(spec=spec.with_overrides(model_revision="other")) == 3.0
    assert reference(device="cuda:0") == 4.0


def test_quality_summary_keeps_incompatible_metrics_separate():
    records = [
        good_record(quality_value=400),
        good_record(quality_value=600),
        good_record(quality_metric="needle_retrieval", quality_value=0.5),
    ]
    summary = summarize(records)["policies"]["full_cache"]
    assert summary["mean_quality"] is None
    assert summary["quality_by_metric"] == {
        "perplexity": {"runs": 2, "mean": 500.0},
        "needle_retrieval": {"runs": 1, "mean": 0.5},
    }


def test_saved_needle_configuration_replays_and_retains_details(tmp_path):
    spec = RunSpec(
        context_length=8,
        quality_metric="needle_retrieval",
        needle_length=3,
        needle_depth=0.25,
        max_new_tokens=2,
        device="cpu",
    )
    outcome = runner.run_spec(spec)
    paths = runner.write_results([outcome], tmp_path, "needle")
    config = load_config(paths["config"])
    assert config.runs[0].to_dict() == spec.to_dict()
    replay = runner.run_spec(config.runs[0])
    assert replay.record.quality_value == outcome.record.quality_value
    restored = load_records(tmp_path)
    assert len(restored) == 1
    assert restored[0].quality_details == outcome.quality.details
    assert restored[0].quality_details["depth"] == 0.25
    assert "quality_details.depth" in paths["csv"].read_text()


def test_failed_numerical_quality_is_flagged_and_saved_as_null(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner,
        "perplexity",
        lambda *args, **kwargs: QualityResult(
            metric="perplexity", value=float("nan"), num_tokens=7, is_interpretable=False
        ),
    )
    outcome = runner.run_spec(RunSpec(context_length=8, max_new_tokens=2, device="cpu"))
    assert any("quality_value must be finite" in p for p in outcome.problems)
    paths = runner.write_results([outcome], tmp_path, "failed")

    def reject_constant(value):
        raise AssertionError(f"nonstandard JSON number: {value}")

    payload = json.loads(paths["jsonl"].read_text(), parse_constant=reject_constant)
    assert payload["quality_value"] is None
    diagnostics = json.loads(paths["diagnostics"].read_text())
    assert diagnostics["problems"][outcome.record.run_id]


def test_missing_source_spec_is_not_fabricated(tmp_path):
    with pytest.raises(ConfigError, match="source RunSpec"):
        runner.write_results(
            [runner.RunOutcome(good_record(), None, None, [])], tmp_path, "missing"
        )
    assert not list(tmp_path.glob("*.*"))
