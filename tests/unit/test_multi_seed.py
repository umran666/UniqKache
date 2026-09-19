"""Unit tests for multi-seed runs and aggregated statistics in BenchmarkRecord."""

from __future__ import annotations

import json

import pytest

from tests.conftest import good_record
from uniqkache.bench.cli import _spec_from_args, build_parser
from uniqkache.bench.config import RunSpec, percent_sweep
from uniqkache.metrics.record import BenchmarkRecord, MetricAggregate, validate_record
from uniqkache.metrics.report import records_to_markdown
from uniqkache.utils.errors import ConfigError


class TestMetricAggregate:
    def test_metric_aggregate_math(self):
        agg = MetricAggregate(
            mean=20.0,
            std=10.0,
            min=10.0,
            max=30.0,
            values=[10.0, 20.0, 30.0],
        )
        assert agg.mean == 20.0
        assert agg.std == 10.0
        assert agg.min == 10.0
        assert agg.max == 30.0
        assert agg.values == [10.0, 20.0, 30.0]

    def test_single_value_std_is_zero(self):
        agg = MetricAggregate(
            mean=15.0,
            std=0.0,
            min=15.0,
            max=15.0,
            values=[15.0],
        )
        assert agg.std == 0.0

    def test_metric_aggregate_serialization(self):
        agg = MetricAggregate(
            mean=12.5,
            std=2.5,
            min=10.0,
            max=15.0,
            values=[10.0, 15.0],
        )
        d = agg.to_dict()
        assert d == {
            "mean": 12.5,
            "std": 2.5,
            "min": 10.0,
            "max": 15.0,
            "values": [10.0, 15.0],
        }
        restored = MetricAggregate.from_dict(d)
        assert restored == agg


class TestBenchmarkRecordMultiSeed:
    def test_record_with_aggregates_and_helpers(self):
        agg = MetricAggregate(
            mean=10.0,
            std=1.0,
            min=9.0,
            max=11.0,
            values=[9.0, 10.0, 11.0],
        )
        record = good_record(
            repetitions=3,
            seeds=[42, 43, 44],
            aggregates={"ttft_ms": agg},
            repetition_records=[{"seed": 42}, {"seed": 43}, {"seed": 44}],
        )
        assert record.repetitions == 3
        assert record.seeds == [42, 43, 44]
        assert record.aggregate("ttft_ms") == agg
        assert record.mean("ttft_ms") == 10.0
        assert record.std("ttft_ms") == 1.0

        assert record.aggregate("unknown") is None
        assert record.mean("unknown") is None
        assert record.std("unknown") is None

    def test_backward_compatibility_v1_0_0(self):
        # A dictionary representing a schema 1.0.0 record (lacks repetitions, seeds, aggregates)
        v1_data = {
            "run_id": "legacy-run",
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
        record = BenchmarkRecord.from_dict(v1_data)
        assert record.schema_version == "1.0.0"
        assert record.repetitions == 1
        assert record.seeds == []
        assert record.aggregates == {}
        assert record.repetition_records == []
        assert record.aggregate("ttft_ms") is None

    def test_serialization_round_trip(self):
        agg = MetricAggregate(
            mean=25.0,
            std=2.0,
            min=23.0,
            max=27.0,
            values=[23.0, 25.0, 27.0],
        )
        record = good_record(
            schema_version="1.1.0",
            repetitions=3,
            seeds=[1, 2, 3],
            aggregates={"tpot_ms": agg},
            repetition_records=[{"run_id": "r1"}, {"run_id": "r2"}, {"run_id": "r3"}],
        )
        payload = record.to_dict()
        assert payload["schema_version"] == "1.1.0"
        assert payload["repetitions"] == 3
        assert payload["seeds"] == [1, 2, 3]
        assert "tpot_ms" in payload["aggregates"]

        # Ensure JSON-serializable
        json_str = json.dumps(payload)
        restored = BenchmarkRecord.from_dict(json.loads(json_str))
        assert restored.repetitions == 3
        assert restored.seeds == [1, 2, 3]
        assert isinstance(restored.aggregates["tpot_ms"], MetricAggregate)
        assert restored.aggregates["tpot_ms"] == agg
        assert len(restored.repetition_records) == 3

    def test_validate_record_multi_seed(self):
        agg = MetricAggregate(
            mean=10.0,
            std=1.0,
            min=9.0,
            max=11.0,
            values=[9.0, 10.0, 11.0],
        )
        valid = good_record(
            repetitions=3,
            seeds=[10, 11, 12],
            aggregates={"ttft_ms": agg},
        )
        assert validate_record(valid) == []

        # repetitions < 1
        invalid_rep = good_record(repetitions=0)
        problems = validate_record(invalid_rep)
        assert any("repetitions must be >= 1" in p for p in problems)

        # seed count mismatch
        mismatched_seeds = good_record(
            repetitions=3,
            seeds=[10, 11],
            aggregates={"ttft_ms": agg},
        )
        problems = validate_record(mismatched_seeds)
        assert any("seeds count (2) does not match repetitions (3)" in p for p in problems)

        # missing aggregates when repetitions > 1
        missing_agg = good_record(
            repetitions=3,
            seeds=[10, 11, 12],
            aggregates={},
        )
        problems = validate_record(missing_agg)
        assert any("repetitions is 3 but aggregates is missing" in p for p in problems)

        # values length mismatch in aggregate
        bad_agg = MetricAggregate(
            mean=10.0,
            std=1.0,
            min=9.0,
            max=11.0,
            values=[9.0, 10.0],  # only 2 values for 3 repetitions
        )
        mismatched_values = good_record(
            repetitions=3,
            seeds=[10, 11, 12],
            aggregates={"ttft_ms": bad_agg},
        )
        problems = validate_record(mismatched_values)
        assert any(
            "aggregate 'ttft_ms' values count (2) does not match repetitions (3)" in p
            for p in problems
        )


class TestRunSpecRepetitions:
    def test_valid_repetitions(self):
        spec = RunSpec(policy="full_cache", context_length=128, repetitions=5)
        assert spec.repetitions == 5
        assert spec.to_dict()["repetitions"] == 5

    def test_invalid_repetitions(self):
        with pytest.raises(ConfigError, match="repetitions must be >= 1"):
            RunSpec(policy="full_cache", context_length=128, repetitions=0)

        with pytest.raises(ConfigError, match="repetitions must be >= 1"):
            RunSpec(policy="full_cache", context_length=128, repetitions=-2)


class TestCliRepetitions:
    def test_cli_repetitions_argument(self):
        parser = build_parser()
        args = parser.parse_args(["--repetitions", "3"])
        assert args.repetitions == 3

        spec = _spec_from_args(args)
        assert spec.repetitions == 3

    def test_percent_sweep_propagates_repetitions(self):
        config = percent_sweep(
            model="synthetic:tiny",
            policy="sliding_window",
            context_length=128,
            repetitions=4,
        )
        assert len(config.runs) > 0
        for run in config.runs:
            assert run.repetitions == 4


class TestReportingMultiSeed:
    def test_records_to_markdown_single_seed(self):
        record = good_record()
        rendered = records_to_markdown([record])
        assert "TTFT (ms)" in rendered
        assert "+/- std" not in rendered

    def test_records_to_markdown_multi_seed(self):
        agg_ttft = MetricAggregate(
            mean=12.34, std=0.56, min=11.5, max=13.0, values=[11.5, 12.5, 13.0]
        )
        agg_tpot = MetricAggregate(
            mean=1.23, std=0.04, min=1.19, max=1.27, values=[1.19, 1.23, 1.27]
        )
        agg_tps = MetricAggregate(
            mean=810.0, std=15.0, min=795.0, max=825.0, values=[795.0, 810.0, 825.0]
        )
        agg_qual = MetricAggregate(
            mean=500.0, std=2.5, min=497.0, max=502.0, values=[497.0, 501.0, 502.0]
        )

        record = good_record(
            repetitions=3,
            seeds=[1, 2, 3],
            ttft_ms=12.34,
            tpot_ms=1.23,
            tokens_per_second=810.0,
            quality_value=500.0,
            aggregates={
                "ttft_ms": agg_ttft,
                "tpot_ms": agg_tpot,
                "tokens_per_second": agg_tps,
                "quality_value": agg_qual,
            },
        )
        rendered = records_to_markdown([record])
        assert "TTFT (ms) +/- std" in rendered
        assert "TPOT (ms) +/- std" in rendered
        assert "tok/s +/- std" in rendered
        assert "quality +/- std" in rendered
        assert "12.34 +/- 0.56" in rendered
        assert "1.230 +/- 0.040" in rendered
        assert "810.0 +/- 15.0" in rendered
        assert "500.0000 +/- 2.5000" in rendered

    def test_records_to_markdown_mixed(self):
        rec_single = good_record(run_id="single", ttft_ms=10.0)
        agg_ttft = MetricAggregate(
            mean=12.34, std=0.56, min=11.5, max=13.0, values=[11.5, 12.5, 13.0]
        )
        rec_multi = good_record(
            run_id="multi",
            repetitions=3,
            ttft_ms=12.34,
            aggregates={"ttft_ms": agg_ttft},
        )
        rendered = records_to_markdown([rec_single, rec_multi])
        # Mixed: table should include +/- std headers because at least one record has aggregates
        assert "TTFT (ms) +/- std" in rendered
        # Single record formats as scalar
        assert "10.00" in rendered
        # Multi record formats as mean +/- std
        assert "12.34 +/- 0.56" in rendered
