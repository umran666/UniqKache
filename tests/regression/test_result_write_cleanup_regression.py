"""PR #75: failed exclusive writes must remove their partial result artifacts."""

from __future__ import annotations

import csv

import pytest

from uniqkache.bench import runner
from uniqkache.bench.config import RunSpec
from uniqkache.metrics.record import BenchmarkRecord


@pytest.mark.regression
@pytest.mark.parametrize("stage", ["csv", "config", "diagnostics"])
def test_failed_write_removes_partial_artifacts_and_preserves_previous_results(
    tmp_path, monkeypatch, stage
):
    outcome = runner.RunOutcome(BenchmarkRecord(run_id="original"), None, None, [], spec=RunSpec())
    previous = runner.write_results([outcome], tmp_path, "experiment")
    contents = {path: path.read_bytes() for path in previous.values()}

    if stage == "csv":

        def fail_rows(self, rows):
            raise OSError("simulated CSV write failure after header")

        monkeypatch.setattr(csv.DictWriter, "writerows", fail_rows)
    else:
        original_dumps = runner.json.dumps

        def fail_config(value, **kwargs):
            if kwargs.get("indent") == 2 and (
                (stage == "config" and "runs" in value)
                or (stage == "diagnostics" and "problems" in value)
            ):
                raise OSError("simulated config serialization failure after file creation")
            return original_dumps(value, **kwargs)

        monkeypatch.setattr(runner.json, "dumps", fail_config)

    with pytest.raises(OSError, match="simulated"):
        runner.write_results([outcome], tmp_path, "experiment")

    # Previously ownership was recorded after writing, leaving a partial CSV or config.
    assert set(tmp_path.iterdir()) == set(previous.values())
    assert {path: path.read_bytes() for path in previous.values()} == contents
