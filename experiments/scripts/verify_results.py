"""Verify committed result artifacts against a fresh run of the same configs.

Why this exists (F14)
---------------------
A documented command can *succeed* while producing a different number. Re-running
the committed experiment configs and diffing the quality fields against the
committed artifacts is the only check that catches that.

**Quality fields must match exactly. Timing fields are excluded, deliberately:**
F10 established that run-to-run variance at this scale exceeds the between-policy
differences (one fixed budget measured 88.8 ms in one pass and 58.5 ms in
another), so a strict diff on latency would fail spuriously and train people to
ignore the target. The exclusion below is that decision, made explicit.

Usage
-----
.. code-block:: bash

    make verify-results
    # or directly:
    python experiments/scripts/verify_results.py --results-dir experiments/results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uniqkache.bench.config import load_config
from uniqkache.bench.runner import run_config
from uniqkache.metrics.record import BenchmarkRecord

# Fields compared strictly. Everything else (ttft_ms, tpot_ms, tokens_per_second,
# peak_memory_bytes, latency percentiles) is excluded as measurement noise per F10.
QUALITY_FIELDS = (
    "quality_metric",
    "quality_value",
    "quality_reference",
    "policy",
    "capacity",
    "context_length",
    "cache_final_tokens",
)

CONFIGS = ("correctness.json", "needle_comparison.json")


def _records_by_config(results_dir: Path) -> dict[str, dict[str, dict]]:
    """Committed records grouped by the config they were produced from, then by identity.

    Each config keeps its own baselines so a later commit adding a new config
    (say needle_comparison.json) does not overwrite the correctness baseline under
    the same (model, policy, context) key.
    """
    out: dict[str, dict[str, dict]] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        base = path.stem
        source_config_name: str | None = None
        config_path = results_dir / f"{base}.config.json"
        if config_path.exists():
            try:
                source_config_name = json.loads(config_path.read_text(encoding="utf-8")).get("name")
            except Exception:
                pass
        source_config_name = source_config_name or base
        records: dict[str, dict] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = BenchmarkRecord.from_dict(json.loads(line))
            except Exception:
                continue
            key = f"{record.model}|{record.policy}|{record.context_length}"
            records[key] = {field: getattr(record, field) for field in QUALITY_FIELDS}
        out[source_config_name] = records
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument(
        "--output-dir", default=None, help="Where fresh runs write; a temp dir when omitted"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    committed_by_config = _records_by_config(results_dir)
    if not committed_by_config:
        print(f"no committed records found under {results_dir}")
        return 1

    import tempfile

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = args.output_dir or tmp
        for config_name in CONFIGS:
            config_path = Path("experiments/configs") / config_name
            if not config_path.exists():
                failures.append(f"missing config: {config_path}")
                continue
            print(f"--- re-running {config_name} ---")
            config = load_config(config_path)
            outcomes = run_config(
                config,
                output_dir=Path(output_dir) / config_name,
                write=True,
            )
            committed = committed_by_config.get(config_name)
            if committed is None:
                failures.append(
                    f"no committed baseline for {config_name}; cannot verify"
                )
                continue
            for outcome in outcomes:
                fresh = outcome.record
                key = f"{fresh.model}|{fresh.policy}|{fresh.context_length}"
                baseline = committed.get(key)
                if baseline is None:
                    failures.append(
                        f"{config_name}: no committed record for {key} to compare against"
                    )
                    continue
                for field in QUALITY_FIELDS:
                    got = getattr(fresh, field)
                    want = baseline[field]
                    if field in {"quality_value", "quality_reference"}:
                        # Exact match is the requirement (F14); a tolerance would
                        # hide a real change behind "close enough".
                        ok = (
                            got is not None and want is not None and float(got) == float(want)
                        ) or (got is None and want is None)
                    else:
                        ok = got == want
                    if not ok:
                        failures.append(
                            f"{config_name} {key}: {field} is {got!r}, committed value is {want!r}"
                        )

    if failures:
        print(f"\n{len(failures)} mismatch(es); timing fields are excluded by design (F10):")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(
        "\nverify-results: all committed quality fields reproduced exactly "
        "(timing fields excluded per F10)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
