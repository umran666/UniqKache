"""Unit tests for verify_results.py script.

Ensures that:
1. Committed experiment results are parsed and grouped by config name.
2. Only QUALITY_FIELDS are checked; timing fields (ttft_ms, tpot_ms, etc.) are excluded per F10.
3. Any drift in quality fields or token counts is caught and reported (F14).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.conftest import good_record
from uniqkache.bench.runner import RunOutcome

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "experiments" / "scripts" / "verify_results.py"
_spec = importlib.util.spec_from_file_location("verify_results", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
verify_results = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_results)


class TestRecordsByConfig:
    def test_loads_and_groups_records_by_config_name(self, tmp_path: Path):
        # Create a config.json with a specific name
        config_path = tmp_path / "run1.config.json"
        config_path.write_text(json.dumps({"name": "my-config-name"}), encoding="utf-8")

        # Create corresponding jsonl
        record = good_record(
            model="synthetic:tiny",
            policy="sliding_window",
            context_length=128,
            quality_value=502.059692,
            quality_reference=502.059692,
            capacity=64,
            cache_final_tokens=64,
            ttft_ms=45.6,
            tpot_ms=12.3,
            peak_memory_bytes=1048576,
        )
        jsonl_path = tmp_path / "run1.jsonl"
        jsonl_path.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")

        records_by_config = verify_results._records_by_config(tmp_path)
        assert "my-config-name" in records_by_config

        key = "synthetic:tiny|sliding_window|128"
        data = records_by_config["my-config-name"][key]

        # Quality fields must be present
        for field in verify_results.QUALITY_FIELDS:
            assert field in data

        # Timing and memory fields must NOT be in the compared data
        for timing_field in ("ttft_ms", "tpot_ms", "peak_memory_bytes", "tokens_per_second"):
            assert timing_field not in data

    def test_fallback_to_stem_when_config_json_missing(self, tmp_path: Path):
        record = good_record(model="synthetic:tiny", policy="full_cache", context_length=128)
        jsonl_path = tmp_path / "sweep-test.jsonl"
        jsonl_path.write_text(json.dumps(record.to_dict()) + "\n", encoding="utf-8")

        records_by_config = verify_results._records_by_config(tmp_path)
        assert "sweep-test" in records_by_config
        key = "synthetic:tiny|full_cache|128"
        assert key in records_by_config["sweep-test"]

    def test_skips_empty_and_malformed_lines(self, tmp_path: Path):
        record = good_record(model="synthetic:tiny", policy="full_cache", context_length=128)
        jsonl_path = tmp_path / "empty-and-bad.jsonl"
        jsonl_path.write_text(
            "\n   \n" + json.dumps(record.to_dict()) + "\n{not valid json}\n",
            encoding="utf-8",
        )

        records_by_config = verify_results._records_by_config(tmp_path)
        assert "empty-and-bad" in records_by_config
        key = "synthetic:tiny|full_cache|128"
        assert key in records_by_config["empty-and-bad"]


class TestVerifyResultsComparison:
    def test_empty_results_dir_returns_1(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        with patch.object(sys, "argv", ["verify_results.py", "--results-dir", str(tmp_path)]):
            rc = verify_results.main()
            assert rc == 1
            captured = capsys.readouterr()
            assert "no committed records found" in captured.out

    def test_timing_differences_are_ignored(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Timing and memory differences between runs do NOT cause failure (F10)."""
        baseline_record = good_record(
            model="synthetic:tiny",
            policy="sliding_window",
            context_length=128,
            quality_value=502.059692,
            quality_reference=502.059692,
            capacity=256,
            cache_final_tokens=132,
            ttft_ms=50.0,
            tpot_ms=10.0,
            tokens_per_second=100.0,
            peak_memory_bytes=100000,
        )
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(json.dumps(baseline_record.to_dict()) + "\n", encoding="utf-8")
        config_json = tmp_path / "test.config.json"
        config_json.write_text(json.dumps({"name": "test-config"}), encoding="utf-8")

        # Mock run outcome with identical quality but wildly different timing & memory
        fresh_record = good_record(
            model="synthetic:tiny",
            policy="sliding_window",
            context_length=128,
            quality_value=502.059692,
            quality_reference=502.059692,
            capacity=256,
            cache_final_tokens=132,
            ttft_ms=150.0,
            tpot_ms=30.0,
            tokens_per_second=33.3,
            peak_memory_bytes=999999,
        )

        mock_config = MagicMock()
        mock_config.name = "test-config"

        monkeypatch.setattr(verify_results, "CONFIGS", ("test-config.json",))
        monkeypatch.setattr(
            verify_results,
            "load_config",
            lambda p: mock_config,
        )
        monkeypatch.setattr(
            verify_results,
            "run_config",
            lambda cfg, output_dir, write: [
                RunOutcome(record=fresh_record, generation=None, quality=None, problems=[])
            ],
        )
        monkeypatch.setattr(Path, "exists", lambda self: True)

        with patch.object(sys, "argv", ["verify_results.py", "--results-dir", str(tmp_path)]):
            rc = verify_results.main()
            assert rc == 0

    def test_quality_value_drift_is_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        """Quality drift causes failure with mismatch details (F14)."""
        baseline_record = good_record(
            model="synthetic:tiny",
            policy="sliding_window",
            context_length=128,
            quality_value=502.059692,
            capacity=256,
            cache_final_tokens=132,
        )
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(json.dumps(baseline_record.to_dict()) + "\n", encoding="utf-8")
        config_json = tmp_path / "test.config.json"
        config_json.write_text(json.dumps({"name": "test-config"}), encoding="utf-8")

        fresh_record = good_record(
            model="synthetic:tiny",
            policy="sliding_window",
            context_length=128,
            quality_value=503.111111,  # Drifted!
            capacity=256,
            cache_final_tokens=132,
        )

        mock_config = MagicMock()
        mock_config.name = "test-config"

        monkeypatch.setattr(verify_results, "CONFIGS", ("test-config.json",))
        monkeypatch.setattr(verify_results, "load_config", lambda p: mock_config)
        monkeypatch.setattr(
            verify_results,
            "run_config",
            lambda cfg, output_dir, write: [
                RunOutcome(record=fresh_record, generation=None, quality=None, problems=[])
            ],
        )
        monkeypatch.setattr(Path, "exists", lambda self: True)

        with patch.object(sys, "argv", ["verify_results.py", "--results-dir", str(tmp_path)]):
            rc = verify_results.main()
            assert rc == 1
            captured = capsys.readouterr()
            assert "mismatch(es); timing fields are excluded by design (F10)" in captured.out
            assert "quality_value is 503.111111, committed value is 502.059692" in captured.out

    def test_token_count_drift_is_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        """Final token count drift (e.g. from missing --max-new-tokens) is caught."""
        baseline_record = good_record(
            model="synthetic:tiny",
            policy="full_cache",
            context_length=192,
            cache_final_tokens=193,
        )
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(json.dumps(baseline_record.to_dict()) + "\n", encoding="utf-8")
        config_json = tmp_path / "test.config.json"
        config_json.write_text(json.dumps({"name": "test-config"}), encoding="utf-8")

        fresh_record = good_record(
            model="synthetic:tiny",
            policy="full_cache",
            context_length=192,
            cache_final_tokens=199,  # Drifted token count!
        )

        mock_config = MagicMock()
        mock_config.name = "test-config"

        monkeypatch.setattr(verify_results, "CONFIGS", ("test-config.json",))
        monkeypatch.setattr(verify_results, "load_config", lambda p: mock_config)
        monkeypatch.setattr(
            verify_results,
            "run_config",
            lambda cfg, output_dir, write: [
                RunOutcome(record=fresh_record, generation=None, quality=None, problems=[])
            ],
        )
        monkeypatch.setattr(Path, "exists", lambda self: True)

        with patch.object(sys, "argv", ["verify_results.py", "--results-dir", str(tmp_path)]):
            rc = verify_results.main()
            assert rc == 1
            captured = capsys.readouterr()
            assert "cache_final_tokens is 199, committed value is 193" in captured.out

    def test_missing_committed_baseline_is_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ):
        baseline_record = good_record(
            model="synthetic:tiny",
            policy="full_cache",
            context_length=128,
        )
        jsonl_path = tmp_path / "test.jsonl"
        jsonl_path.write_text(json.dumps(baseline_record.to_dict()) + "\n", encoding="utf-8")
        config_json = tmp_path / "test.config.json"
        config_json.write_text(json.dumps({"name": "other-config"}), encoding="utf-8")

        mock_config = MagicMock()
        mock_config.name = "expected-config"

        monkeypatch.setattr(verify_results, "CONFIGS", ("test-config.json",))
        monkeypatch.setattr(verify_results, "load_config", lambda p: mock_config)
        monkeypatch.setattr(Path, "exists", lambda self: True)

        with patch.object(sys, "argv", ["verify_results.py", "--results-dir", str(tmp_path)]):
            rc = verify_results.main()
            assert rc == 1
            captured = capsys.readouterr()
            assert "no committed baseline for 'expected-config'" in captured.out
