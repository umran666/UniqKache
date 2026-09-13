"""Unit tests for experiment configuration and benchmark records.

These are the parts that make a result *reproducible* and *honest*. A config that
silently drops a field, or a record that cannot support its own claims, is a
correctness bug in a research project just as much as a broken cache is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.conftest import good_record
from uniqkache.bench.config import (
    ExperimentConfig,
    RunSpec,
    load_config,
    percent_sweep,
)
from uniqkache.metrics.record import BenchmarkRecord, validate_record
from uniqkache.metrics.report import (
    load_records,
    records_to_csv,
    records_to_markdown,
    summarize,
)
from uniqkache.utils.errors import ConfigError

# ---------------------------------------------------------------------------
# RunSpec validation
# ---------------------------------------------------------------------------


class TestRunSpec:
    def test_keep_ratio_resolves_to_a_capacity(self):
        spec = RunSpec(policy="sliding_window", context_length=1024, keep_ratio=0.25)
        assert spec.resolved_capacity == 256

    def test_explicit_capacity_wins_over_nothing(self):
        spec = RunSpec(policy="sliding_window", context_length=1024, capacity=99)
        assert spec.resolved_capacity == 99

    def test_aliases_are_canonicalised(self):
        assert RunSpec(policy="h2o", keep_ratio=0.5).policy == "attention_based"
        assert RunSpec(policy="streamingllm", keep_ratio=0.5).policy == "sliding_window"

    def test_evicting_policy_without_a_budget_is_rejected(self):
        """A bounded policy with no budget would silently never evict."""
        with pytest.raises(ConfigError, match="no budget was given"):
            RunSpec(policy="sliding_window", context_length=1024)

    def test_capacity_and_keep_ratio_together_are_rejected(self):
        with pytest.raises(ConfigError, match="not both"):
            RunSpec(context_length=1024, capacity=10, keep_ratio=0.5)

    def test_full_cache_without_a_budget_is_allowed(self):
        assert RunSpec(policy="full_cache", context_length=1024).resolved_capacity is None

    def test_unknown_policy_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown policy"):
            RunSpec(policy="nope", context_length=1024, keep_ratio=0.5)

    def test_sinks_exceeding_capacity_are_rejected(self):
        with pytest.raises(ConfigError, match="exceeds the resolved capacity"):
            RunSpec(policy="sliding_window", context_length=100, keep_ratio=0.1, attention_sinks=50)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"context_length": 1}, "context_length"),
            ({"keep_ratio": 0.0}, "keep_ratio"),
            ({"keep_ratio": 1.5}, "keep_ratio"),
            ({"attention_sinks": -1}, "attention_sinks"),
            ({"batch_size": 0}, "batch_size"),
            ({"max_new_tokens": -1}, "max_new_tokens"),
        ],
    )
    def test_invalid_fields_are_rejected(self, kwargs, match):
        base = {"context_length": 128, "policy": "full_cache"}
        base.update(kwargs)
        with pytest.raises(ConfigError, match=match):
            RunSpec(**base)


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_percent_sweep_produces_one_run_per_ratio(self):
        config = percent_sweep(model="synthetic:tiny", policy="sliding_window", context_length=1024)
        assert len(config.runs) == 5
        capacities = [run.resolved_capacity for run in config.runs]
        assert capacities[0] is None, "the 100% run must be an unbounded full cache"
        assert all(c is not None for c in capacities[1:])

    def test_sweep_uses_full_cache_at_full_retention(self):
        config = percent_sweep(model="synthetic:tiny", policy="sliding_window", context_length=512)
        assert config.runs[0].policy == "full_cache"

    def test_empty_experiment_is_rejected(self):
        with pytest.raises(ConfigError, match="contains no runs"):
            ExperimentConfig(name="empty", runs=[])

    def test_round_trip_through_dict(self):
        config = percent_sweep(model="synthetic:tiny", policy="sliding_window", context_length=256)
        restored = ExperimentConfig.from_dict(config.to_dict())
        assert len(restored.runs) == len(config.runs)
        assert restored.name == config.name

    def test_unknown_top_level_key_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown top-level key"):
            ExperimentConfig.from_dict({"name": "x", "runs": [{}], "bogus": 1})

    def test_unknown_run_key_is_rejected(self):
        with pytest.raises(ConfigError, match="unknown key"):
            ExperimentConfig.from_dict(
                {"name": "x", "runs": [{"context_length": 64, "policy": "full_cache", "typo": 1}]}
            )

    def test_missing_runs_is_rejected(self):
        with pytest.raises(ConfigError, match="must contain a 'runs' list"):
            ExperimentConfig.from_dict({"name": "x"})

    def test_load_config_from_disk(self, tmp_path: Path):
        path = tmp_path / "exp.json"
        path.write_text(
            json.dumps(
                {
                    "name": "on-disk",
                    "runs": [
                        {"model": "synthetic:tiny", "policy": "full_cache", "context_length": 64}
                    ],
                }
            ),
            encoding="utf-8",
        )
        config = load_config(path)
        assert config.name == "on-disk"
        assert len(config.runs) == 1

    def test_load_missing_file_is_reported(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "absent.json")

    def test_load_malformed_json_is_reported(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid JSON"):
            load_config(path)


# ---------------------------------------------------------------------------
# Record integrity
# ---------------------------------------------------------------------------


class TestRecordValidation:
    def test_a_complete_record_has_no_problems(self):
        assert validate_record(good_record()) == []

    def test_missing_git_commit_is_flagged(self):
        problems = validate_record(good_record(git_commit=None))
        assert any("git_commit is missing" in p for p in problems)

    def test_dirty_working_tree_is_flagged(self):
        problems = validate_record(good_record(git_dirty=True))
        assert any("dirty" in p for p in problems)

    def test_performance_without_quality_is_flagged(self):
        """The central reporting rule of the project, enforced mechanically."""
        problems = validate_record(good_record(quality_metric=None, quality_value=None))
        assert any("no quality metric was measured" in p for p in problems)

    def test_quality_on_random_weights_is_flagged(self):
        problems = validate_record(good_record(weights_are_random=True))
        assert any("randomly-initialised" in p for p in problems)

    def test_tpot_without_generated_tokens_is_flagged(self):
        """A per-token latency is meaningless without the step count it averages."""
        problems = validate_record(good_record(generated_tokens=None))
        assert any("generated_tokens is missing" in p for p in problems)

    def test_no_decode_means_no_tpot_and_no_problem(self):
        """Zero generated tokens is consistent as long as TPOT is absent too."""
        problems = validate_record(good_record(tpot_ms=None, generated_tokens=0))
        assert not any("generated_tokens" in p for p in problems)

    def test_bounded_policy_without_capacity_is_flagged(self):
        problems = validate_record(
            good_record(
                policy="sliding_window",
                capacity=None,
                cache_bytes_total=1024,
                quality_value=1.0,
            )
        )
        assert any("no capacity recorded" in p for p in problems)

    def test_empty_model_is_flagged(self):
        assert any("model is empty" in p for p in validate_record(good_record(model="")))

    def test_non_positive_context_length_is_flagged(self):
        assert any("context_length" in p for p in validate_record(good_record(context_length=0)))


class TestRecordShape:
    def test_quality_delta_inverts_perplexity_sign(self):
        """Lower perplexity is better, so a drop must read as a positive delta."""
        record = good_record(
            quality_metric="perplexity", quality_value=400.0, quality_reference=500.0
        )
        assert record.quality_delta == pytest.approx(100.0)

    def test_quality_delta_is_none_without_a_reference(self):
        assert good_record(quality_reference=None).quality_delta is None

    def test_flat_dict_flattens_without_dropping_keys(self):
        record = good_record(model_config={"a": 1, "b": {"c": 2}})
        flat = record.to_flat_dict()
        assert flat["model_config.a"] == 1
        assert flat["model_config.b.c"] == 2
        assert "run_id" in flat

    def test_flat_dict_keeps_nested_environment_fields(self):
        """Flattening must not drop fields; a dropped field is a lost reproduction path."""
        flat = good_record(
            environment={"torch_version": "2.6.0", "device_name": "cpu"}
        ).to_flat_dict()
        assert flat["environment.torch_version"] == "2.6.0"
        assert flat["environment.device_name"] == "cpu"

    def test_from_dict_ignores_unknown_keys(self):
        payload = good_record().to_dict()
        payload["from_the_future"] = 1
        restored = BenchmarkRecord.from_dict(payload)
        assert restored.run_id == "r1"

    def test_record_is_json_serialisable(self):
        payload = json.dumps(good_record().to_dict(), default=str)
        assert "run_id" in payload


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


class TestReporting:
    def test_markdown_always_includes_a_quality_column(self):
        """A table without quality is the artefact this project exists to prevent."""
        rendered = records_to_markdown([good_record()])
        assert "quality" in rendered
        assert "perplexity" in rendered

    def test_markdown_marks_random_weight_quality(self):
        rendered = records_to_markdown([good_record(weights_are_random=True)])
        assert "random weights" in rendered

    def test_markdown_lists_integrity_warnings(self):
        rendered = records_to_markdown([good_record(git_commit=None)])
        assert "Integrity warnings" in rendered

    def test_markdown_renders_missing_values_as_na_not_zero(self):
        record = good_record(ttft_ms=None, tpot_ms=None, tokens_per_second=None)
        rendered = records_to_markdown([record])
        assert "n/a" in rendered

    def test_markdown_of_no_records(self):
        assert "No records" in records_to_markdown([])

    def test_csv_round_trips_nested_fields(self, tmp_path: Path):
        path = records_to_csv([good_record()], tmp_path / "out.csv")
        text = path.read_text(encoding="utf-8")
        assert "model_config" in text or "run_id" in text
        assert "synthetic:tiny" in text

    def test_csv_uses_lf_line_endings(self, tmp_path: Path):
        # `.gitattributes` normalises result files to LF. The csv module emits
        # "\r\n" by default, so without an explicit terminator a CSV written on
        # Windows differs from the index and git warns on every add. Read as
        # bytes: text mode would translate the endings away and hide the bug.
        path = records_to_csv([good_record()], tmp_path / "out.csv")
        raw = path.read_bytes()
        assert b"\r\n" not in raw
        assert raw.endswith(b"\n")

    def test_summarize_groups_by_policy(self):
        summary = summarize([good_record(), good_record(policy="lru", capacity=8)])
        assert summary["total_runs"] == 2
        assert set(summary["policies"]) == {"full_cache", "lru"}

    def test_load_records_from_jsonl(self, tmp_path: Path):
        path = tmp_path / "r.jsonl"
        path.write_text(
            json.dumps(good_record().to_dict()) + "\n" + json.dumps(good_record().to_dict()) + "\n",
            encoding="utf-8",
        )
        assert len(load_records(path)) == 2

    def test_load_records_reports_a_bad_line(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text("{oops}\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="invalid JSON"):
            load_records(path)

    def test_load_records_from_a_directory(self, tmp_path: Path):
        (tmp_path / "a.jsonl").write_text(
            json.dumps(good_record().to_dict()) + "\n", encoding="utf-8"
        )
        (tmp_path / "b.jsonl").write_text(
            json.dumps(good_record().to_dict()) + "\n", encoding="utf-8"
        )
        assert len(load_records(tmp_path)) == 2

    def test_load_records_from_an_empty_directory_is_reported(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="no result files"):
            load_records(tmp_path)

    def test_load_records_from_a_missing_path_is_reported(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="does not exist"):
            load_records(tmp_path / "nope")

    def test_load_records_skips_experiment_config_files(self, tmp_path: Path):
        """Regression: the runner writes `.config.json` beside every result.

        Those files carry `name`, `runs` and `problems` and no `run_id`, so
        parsing one as a record raised `TypeError: missing 'run_id'`. The
        command was therefore broken on exactly the directory it is pointed at
        -- one the runner had populated itself.
        """
        (tmp_path / "run-20260101-000000.jsonl").write_text(
            json.dumps(good_record().to_dict()) + "\n", encoding="utf-8"
        )
        (tmp_path / "run-20260101-000000.config.json").write_text(
            json.dumps(
                {
                    "name": "some-experiment",
                    "runs": [{"model": "synthetic:tiny", "policy": "full_cache"}],
                    "problems": {"r1": ["something"]},
                }
            ),
            encoding="utf-8",
        )
        records = load_records(tmp_path)
        assert len(records) == 1
        assert records[0].run_id == "r1"

    def test_load_records_does_not_descend_into_subdirectories(self, tmp_path: Path):
        """Superseded results must not be folded into a current summary.

        Withdrawn results live in `experiments/results/superseded/`. Including
        them would misreport the project's current state, so loading stays
        non-recursive and they are read only by pointing --input at them.
        """
        (tmp_path / "current.jsonl").write_text(
            json.dumps(good_record().to_dict()) + "\n", encoding="utf-8"
        )
        superseded = tmp_path / "superseded"
        superseded.mkdir()
        (superseded / "old.jsonl").write_text(
            json.dumps(good_record(run_id="withdrawn").to_dict()) + "\n", encoding="utf-8"
        )
        records = load_records(tmp_path)
        assert len(records) == 1
        assert records[0].run_id == "r1"
