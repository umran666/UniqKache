"""Issue #61: one-row inputs with a batch-two cache crashed the output projection."""

from __future__ import annotations

import math

import pytest

from uniqkache.bench.cli import main
from uniqkache.bench.config import RunSpec
from uniqkache.bench.runner import run_spec
from uniqkache.metrics.report import load_records


@pytest.mark.regression
@pytest.mark.parametrize("metric", ["perplexity", "needle_retrieval"])
@pytest.mark.parametrize("policy", ["full_cache", "sliding_window"])
def test_batch_two_benchmark_runs_warmup_generation_and_quality(monkeypatch, metric, policy):
    from uniqkache.models.synthetic import SyntheticCausalLM

    batches = []
    original_forward = SyntheticCausalLM.forward

    def checked_forward(self, input_ids, cache=None, **kwargs):
        batches.append(input_ids.shape[0])
        assert input_ids.shape[0] == 2
        if cache is not None:
            assert cache.config.batch_size == 2
        return original_forward(self, input_ids, cache=cache, **kwargs)

    monkeypatch.setattr(SyntheticCausalLM, "forward", checked_forward)
    outcome = run_spec(
        RunSpec(
            context_length=8,
            batch_size=2,
            max_new_tokens=2,
            device="cpu",
            quality_metric=metric,
            needle_length=2,
            policy=policy,
            capacity=4 if policy == "sliding_window" else None,
        )
    )
    assert batches and set(batches) == {2}
    assert outcome.record.batch_size == 2
    assert outcome.generation.generated_ids.shape == (2, 2)
    assert math.isfinite(outcome.record.quality_value)
    assert math.isfinite(outcome.record.quality_reference)
    if metric == "perplexity":
        assert outcome.quality.num_tokens == 2 * (8 - 1)
    else:
        assert outcome.quality.num_tokens == 2 * 8
        assert outcome.quality.details["batch_size"] == 2
        assert len(outcome.quality.details["expected"]) == 2 * 2


@pytest.mark.regression
def test_batch_size_cli_writes_a_successful_batch_two_record(tmp_path):
    exit_code = main(
        [
            "--model",
            "synthetic:tiny",
            "--context-length",
            "8",
            "--batch-size",
            "2",
            "--max-new-tokens",
            "2",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
    )
    assert exit_code == 0
    records = load_records(tmp_path)
    assert len(records) == 1
    assert records[0].batch_size == 2
    assert math.isfinite(records[0].quality_value)


@pytest.mark.regression
def test_reference_memo_keeps_batch_workloads_separate():
    base = RunSpec(
        context_length=8, max_new_tokens=2, policy="sliding_window", capacity=16, device="cpu"
    )
    memo = {}
    first = run_spec(base, reference_memo=memo)
    second = run_spec(base.with_overrides(batch_size=2), reference_memo=memo)
    standalone = run_spec(base.with_overrides(batch_size=2))
    assert second.record.quality_reference == standalone.record.quality_reference
    assert len(memo) == 2
    assert first.record.batch_size == 1
