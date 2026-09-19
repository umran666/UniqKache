"""Regression test: acceptance sweep comparing uniform vs attention-proportional allocation at equal total budget."""

from uniqkache.bench.config import RunSpec
from uniqkache.bench.runner import run_spec


def test_layerwise_budget_acceptance_sweep():
    # Model: synthetic:tiny has 4 layers.
    # Total context: 64 tokens.
    # Scalar capacity = 16 per layer (total budget across 4 layers = 64 tokens).
    # Sinks = 4 per layer.

    # 1. Baseline: Scalar capacity (capacity_schedule=None)
    spec_scalar = RunSpec(
        model="synthetic:tiny",
        policy="sliding_window",
        context_length=64,
        capacity=16,
        attention_sinks=4,
        max_new_tokens=4,
        measure_quality=False,
    )
    outcome_scalar = run_spec(spec_scalar)
    rec_scalar = outcome_scalar.record

    assert rec_scalar.capacity == 16
    assert rec_scalar.capacity_schedule is None
    assert len(rec_scalar.tokens_per_layer) == 4
    assert rec_scalar.tokens_per_layer == [16, 16, 16, 16]
    assert rec_scalar.utilization_per_layer == [1.0, 1.0, 1.0, 1.0]

    # 2. Uniform capacity schedule
    spec_uniform = RunSpec(
        model="synthetic:tiny",
        policy="sliding_window",
        context_length=64,
        capacity=16,
        attention_sinks=4,
        capacity_schedule="uniform",
        max_new_tokens=4,
        measure_quality=False,
    )
    outcome_uniform = run_spec(spec_uniform)
    rec_uniform = outcome_uniform.record

    assert rec_uniform.capacity_schedule == "uniform"
    assert isinstance(rec_uniform.capacity, list)
    assert len(rec_uniform.capacity) == 4
    assert sum(rec_uniform.capacity) == 64
    assert rec_uniform.capacity == [16, 16, 16, 16]
    assert rec_uniform.tokens_per_layer == [16, 16, 16, 16]
    assert rec_uniform.utilization_per_layer == [1.0, 1.0, 1.0, 1.0]

    # 3. Attention-proportional capacity schedule
    spec_attn = RunSpec(
        model="synthetic:tiny",
        policy="sliding_window",
        context_length=64,
        capacity=16,
        attention_sinks=4,
        capacity_schedule="attention_proportional",
        max_new_tokens=4,
        measure_quality=False,
    )
    outcome_attn = run_spec(spec_attn)
    rec_attn = outcome_attn.record

    assert rec_attn.capacity_schedule == "attention_proportional"
    assert isinstance(rec_attn.capacity, list)
    assert len(rec_attn.capacity) == 4
    # Total budget must be strictly preserved: 16 * 4 = 64
    assert sum(rec_attn.capacity) == 64
    # Invariant: each layer must receive at least attention_sinks
    assert all(c >= 4 for c in rec_attn.capacity)
    # Occupancy per layer recorded
    assert len(rec_attn.tokens_per_layer) == 4
    assert all(t <= c for t, c in zip(rec_attn.tokens_per_layer, rec_attn.capacity))
    # Per-layer utilization recorded
    assert rec_attn.utilization_per_layer is not None
    assert len(rec_attn.utilization_per_layer) == 4
