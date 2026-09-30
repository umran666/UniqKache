"""Issue #80: each synthetic repetition must reproduce its standalone seeded model."""

from __future__ import annotations

import pytest
import torch

from uniqkache.bench import runner
from uniqkache.bench.config import RunSpec


@pytest.mark.regression
@pytest.mark.parametrize("model", ["synthetic:tiny", "tiny"])
@pytest.mark.parametrize("policy", ["full_cache", "sliding_window"])
def test_synthetic_repetitions_match_standalone_weights_tokens_and_quality(
    monkeypatch, model, policy
):
    spec = RunSpec(
        model=model,
        policy=policy,
        capacity=16 if policy == "sliding_window" else None,
        context_length=8,
        max_new_tokens=3,
        repetitions=3,
        seed=10,
        device="cpu",
    )
    snapshots = []
    original_run = runner._run_single_spec

    def capture_run(rep_spec, built, **kwargs):
        weights = torch.cat([p.detach().flatten().cpu() for p in built.model.parameters()])
        outcome = original_run(rep_spec, built, **kwargs)
        snapshots.append((rep_spec.seed, weights, outcome.generation.generated_ids.clone()))
        return outcome

    monkeypatch.setattr(runner, "_run_single_spec", capture_run)
    aggregate = runner.run_spec(spec).record
    repetitions = list(snapshots)
    assert [row[0] for row in repetitions] == aggregate.seeds == [10, 11, 12]

    # Previously seeds 11 and 12 reused seed 10's weights and could not be reproduced.
    for (seed, weights, tokens), child in zip(repetitions, aggregate.repetition_records):
        standalone = runner.run_spec(spec.with_overrides(seed=seed, repetitions=1))
        _, standalone_weights, standalone_tokens = snapshots[-1]
        torch.testing.assert_close(weights, standalone_weights, rtol=0, atol=0)
        torch.testing.assert_close(tokens, standalone_tokens, rtol=0, atol=0)
        assert child["quality_value"] == standalone.record.quality_value
        assert child["quality_reference"] == standalone.record.quality_reference

    assert not torch.equal(repetitions[0][1], repetitions[1][1])
    assert not torch.equal(repetitions[1][1], repetitions[2][1])
