"""End-to-end tests for the benchmark CLI's documented flag surface.

Why these exist
---------------
Every test here calls :func:`uniqkache.bench.cli.main` with an argument vector,
exactly as a user would. Before this module existed, no test invoked the CLI at
all: 285 tests passed while ``--compressor``, ``--no-quality``, ``--sweep``,
``--markdown`` and ``--print-records`` had zero coverage, which is how a
``--compressor int8`` run could write a record claiming a mechanism that never
executed (see the bug pinned in ``tests/regression/test_regressions.py``).

Everything runs on the CPU-only ``synthetic:tiny`` model with a short context,
no GPU and no downloads, so the tests live in ``tests/unit`` and are covered by
``make test-fast``.
"""

from __future__ import annotations

import json

import pytest

from uniqkache.bench.cli import main
from uniqkache.metrics.report import load_records

# Small but comfortably above the largest swept capacity, so every sweep row
# actually exercises its budget rather than degenerating into a full cache.
CTX = 64
NEW_TOKENS = 2


def _run_cli(tmp_path, *args: str) -> int:
    return main(["--output-dir", str(tmp_path), "--quiet", *args])


def _load_single_record(tmp_path):
    records = load_records(tmp_path)
    assert len(records) == 1
    return records[0]


class TestCompressorFlag:
    """`--compressor int8` must produce a record that shows the mechanism ran."""

    def test_compressor_reduces_the_recorded_cache_bytes(self, tmp_path):
        plain_dir = tmp_path / "plain"
        compressed_dir = tmp_path / "compressed"

        base = [
            "--model",
            "synthetic:tiny",
            "--context-length",
            str(CTX),
            "--max-new-tokens",
            str(NEW_TOKENS),
            "--no-quality",
        ]
        assert _run_cli(plain_dir, *base) == 0
        assert _run_cli(compressed_dir, *base, "--compressor", "int8") == 0

        plain = _load_single_record(plain_dir)
        compressed = _load_single_record(compressed_dir)

        assert plain.cache_compression_ratio == pytest.approx(1.0)
        assert compressed.compressor == "int8"
        assert compressed.cache_compression_ratio > 1.0
        assert compressed.cache_bytes_total < plain.cache_bytes_total
        # Compression changes representation, not occupancy.
        assert compressed.cache_final_tokens == plain.cache_final_tokens

    def test_unknown_compressor_is_a_configuration_error(self, tmp_path):
        # argparse rejects values outside choices=[...] with exit code 2.
        with pytest.raises(SystemExit) as excinfo:
            _run_cli(tmp_path, "--compressor", "lz4")
        assert excinfo.value.code == 2


class TestNoQualityFlag:
    def test_no_quality_produces_a_record_that_validate_record_flags(self, tmp_path):
        assert (
            _run_cli(
                tmp_path,
                "--model",
                "synthetic:tiny",
                "--context-length",
                str(CTX),
                "--max-new-tokens",
                str(NEW_TOKENS),
                "--no-quality",
            )
            == 0
        )
        record = _load_single_record(tmp_path)
        assert record.quality_metric is None
        assert record.quality_value is None


class TestSweepFlag:
    def test_sweep_expands_to_the_documented_ratios(self, tmp_path):
        assert (
            _run_cli(
                tmp_path,
                "--model",
                "synthetic:tiny",
                "--policy",
                "sliding_window",
                "--context-length",
                str(CTX),
                "--max-new-tokens",
                str(NEW_TOKENS),
                "--sweep",
                "--no-quality",
            )
            == 0
        )
        records = load_records(tmp_path)
        assert len(records) == 5, "the documented sweep is 100/75/50/25/10 %"

        expected_capacities = [int(CTX * r) for r in (0.75, 0.5, 0.25, 0.1)]
        bounded = sorted((r.capacity for r in records if r.capacity is not None), reverse=True)
        assert bounded == expected_capacities

        references = [r for r in records if r.capacity is None]
        assert len(references) == 1
        assert references[0].policy == "full_cache"
        for record in records:
            if record.capacity is not None:
                assert record.policy == "sliding_window"
                assert record.attention_sinks == 4

    def test_sweep_rejects_an_explicit_budget(self, tmp_path):
        for flag, value in (("--capacity", "32"), ("--keep-ratio", "0.5")):
            with pytest.raises(SystemExit) as excinfo:
                _run_cli(tmp_path, "--sweep", flag, value)
            assert excinfo.value.code == 2

    def test_sweep_and_config_are_mutually_exclusive(self, tmp_path):
        config_path = tmp_path / "exp.json"
        config_path.write_text(
            json.dumps({"name": "x", "runs": [{"model": "synthetic:tiny", "context_length": 8}]}),
            encoding="utf-8",
        )
        with pytest.raises(SystemExit) as excinfo:
            _run_cli(tmp_path, "--sweep", "--config", str(config_path))
        assert excinfo.value.code == 2


class TestOutputFlags:
    def test_print_records_and_markdown_emit_to_stdout(self, tmp_path, capsys):
        assert (
            _run_cli(
                tmp_path,
                "--model",
                "synthetic:tiny",
                "--context-length",
                str(CTX),
                "--max-new-tokens",
                str(NEW_TOKENS),
                "--no-quality",
                "--print-records",
                "--markdown",
            )
            == 0
        )
        out = capsys.readouterr().out
        assert '"run_id"' in out  # --print-records emits JSON
        assert "Results: single-run" in out  # --markdown emits the table

    def test_integrity_warnings_are_surfaced_on_stderr(self, tmp_path, capsys):
        # --no-quality must surface the "performance without quality" warning.
        assert (
            _run_cli(
                tmp_path,
                "--model",
                "synthetic:tiny",
                "--context-length",
                str(CTX),
                "--max-new-tokens",
                str(NEW_TOKENS),
                "--no-quality",
            )
            == 0
        )
        err = capsys.readouterr().err
        assert "integrity warnings" in err
        assert "no quality metric was measured" in err

    def test_list_policies_exits_ok_without_running(self, tmp_path, capsys):
        assert main(["--list-policies"]) == 0
        out = capsys.readouterr().out
        assert "full_cache" in out
        assert "sliding_window" in out
