"""Regression tests for multi-seed runs and aggregated statistics.

Pins:
1. A multi-seed run (repetitions >= 2) computes mean, sample std, min, max,
   and per-repetition values for all measured metrics.
2. Identical seeds reproduce identical aggregates (deterministic reproducibility).
3. A 3-seed sweep produces a report table with std columns and mean +/- std cells.
4. Backward compatibility: schema 1.0.0 records coexist cleanly with schema 1.1.0 records.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from uniqkache.bench.cli import main
from uniqkache.bench.config import RunSpec
from uniqkache.bench.runner import run_spec
from uniqkache.metrics.report import load_records, records_to_markdown


class TestMultiSeedRegression:
    def test_three_seed_run_produces_aggregates_and_std_report(self, tmp_path: Path):
        """Acceptance test 1: A 3-seed run produces a report table with std columns."""
        args = [
            "--model",
            "synthetic:tiny",
            "--context-length",
            "64",
            "--max-new-tokens",
            "2",
            "--repetitions",
            "3",
            "--seed",
            "100",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
        exit_code = main(args)
        assert exit_code == 0

        records = load_records(tmp_path)
        assert len(records) == 1
        record = records[0]

        assert record.repetitions == 3
        assert record.seeds == [100, 101, 102]
        assert len(record.repetition_records) == 3
        assert "ttft_ms" in record.aggregates
        assert "tpot_ms" in record.aggregates
        assert "quality_value" in record.aggregates

        ttft_agg = record.aggregates["ttft_ms"]
        assert len(ttft_agg.values) == 3
        assert ttft_agg.min <= ttft_agg.mean <= ttft_agg.max
        assert ttft_agg.std >= 0.0

        rendered = records_to_markdown([record])
        assert "TTFT (ms) +/- std" in rendered
        assert "TPOT (ms) +/- std" in rendered
        assert "quality +/- std" in rendered
        assert "+/-" in rendered

    def test_identical_seeds_reproduce_identical_aggregates(self, monkeypatch):
        """Acceptance test 2: Identical seeds reproduce identical aggregates."""
        spec1 = RunSpec(
            model="synthetic:tiny",
            policy="full_cache",
            context_length=64,
            max_new_tokens=4,
            seed=42,
            repetitions=3,
            measure_quality=True,
        )
        spec2 = RunSpec(
            model="synthetic:tiny",
            policy="full_cache",
            context_length=64,
            max_new_tokens=4,
            seed=42,
            repetitions=3,
            measure_quality=True,
        )

        clock = 0.0

        def fake_perf_counter():
            nonlocal clock
            clock += 0.01
            return clock

        monkeypatch.setattr("time.perf_counter", fake_perf_counter)

        clock = 0.0
        outcome1 = run_spec(spec1)
        clock = 0.0
        outcome2 = run_spec(spec2)

        rec1 = outcome1.record
        rec2 = outcome2.record

        assert rec1.repetitions == rec2.repetitions == 3
        assert rec1.seeds == rec2.seeds == [42, 43, 44]
        assert set(rec1.aggregates.keys()) == set(rec2.aggregates.keys())

        # For identical seeds and deterministic clock, all aggregates match
        for metric, agg1 in rec1.aggregates.items():
            agg2 = rec2.aggregates[metric]
            assert len(agg1.values) == len(agg2.values) == 3
            assert agg1.mean == pytest.approx(agg2.mean)
            assert agg1.std == pytest.approx(agg2.std)
            assert agg1.min == pytest.approx(agg2.min)
            assert agg1.max == pytest.approx(agg2.max)
            assert agg1.values == pytest.approx(agg2.values)

    def test_schema_1_0_0_and_1_1_0_coexistence(self, tmp_path: Path):
        """Legacy schema 1.0.0 records coexist with 1.1.0 records without crashing reporting."""
        v1_record = {
            "run_id": "v1-legacy",
            "schema_version": "1.0.0",
            "model": "synthetic:tiny",
            "policy": "full_cache",
            "context_length": 64,
            "generated_tokens": 2,
            "ttft_ms": 15.0,
            "tpot_ms": 1.5,
            "tokens_per_second": 66.7,
            "quality_metric": "perplexity",
            "quality_value": 450.0,
        }
        (tmp_path / "v1.jsonl").write_text(json.dumps(v1_record) + "\n", encoding="utf-8")

        # Run a 1.1.0 multi-seed run
        args = [
            "--model",
            "synthetic:tiny",
            "--context-length",
            "64",
            "--max-new-tokens",
            "2",
            "--repetitions",
            "2",
            "--seed",
            "200",
            "--output-dir",
            str(tmp_path),
            "--quiet",
        ]
        assert main(args) == 0

        records = load_records(tmp_path)
        assert len(records) == 2
        rendered = records_to_markdown(records)
        assert "TTFT (ms) +/- std" in rendered
        assert "15.00" in rendered  # Legacy scalar row
        assert "+/-" in rendered  # Multi-seed row
