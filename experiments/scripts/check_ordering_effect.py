"""Check whether sweep timings track the retention budget or the run position.

Why this exists
---------------
A retention sweep changes two things at once: the token budget, and the position
of the run within the sweep. On hardware whose clocks ramp up under sustained
load, later runs in a sweep are faster for reasons that have nothing to do with
the budget. A monotonic TTFT curve across a sweep then *looks* like a finding
("less retention is faster") when it is an artefact of the measurement order.

The original UniqKache retention sweep showed exactly that shape:
43.2 / 37.3 / 36.6 / 36.0 / 33.9 ms as retention fell. This script decides
whether that was the budget or the position.

Method
------
Run the same sweep twice, once in forward order (100, 75, 50, 25, 10 %) and once
in reverse (10, 25, 50, 75, 100 %), and compare.

* If TTFT tracks the **budget**, each budget gets a similar TTFT in both orders,
  and the two curves mirror each other.
* If TTFT tracks the **position**, the forward and reverse curves have the same
  shape — a decrease from the first run to the last — and the budget ordering
  reverses with it.

Usage
-----
.. code-block:: bash

    python experiments/scripts/check_ordering_effect.py
    python experiments/scripts/check_ordering_effect.py --context-length 512 --device cuda

Run it on your own hardware before quoting a latency sweep from it. The answer
can differ by device: a datacentre GPU with fixed clocks may show no effect.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running straight from a checkout without an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from uniqkache.bench.config import percent_sweep  # noqa: E402
from uniqkache.bench.runner import run_config  # noqa: E402

FORWARD = (1.0, 0.75, 0.5, 0.25, 0.1)
REVERSE = (0.1, 0.25, 0.5, 0.75, 1.0)


def _run(label: str, ratios: tuple[float, ...], args: argparse.Namespace) -> list[tuple]:
    config = percent_sweep(
        model=args.model,
        policy=args.policy,
        context_length=args.context_length,
        ratios=ratios,
        device=args.device,
        measure_quality=False,  # quality is irrelevant to a timing-order question
        max_new_tokens=args.max_new_tokens,
    )
    outcomes = run_config(config, output_dir=args.output_dir, write=False)

    rows: list[tuple] = []
    print(f"--- {label} ---")
    for position, outcome in enumerate(outcomes, start=1):
        record = outcome.record
        if record.ttft_ms is None:
            print(f"  position {position}: no timing available")
            continue
        rows.append((position, record.capacity, record.ttft_ms, record.tpot_ms))
        print(
            f"  position {position}: policy={record.policy:<15s} "
            f"budget={str(record.capacity):>4s} "
            f"TTFT={record.ttft_ms:7.2f}ms TPOT={record.tpot_ms:6.2f}ms"
        )
    return rows


def _describe(label: str, rows: list[tuple]) -> None:
    if len(rows) < 2:
        return
    first, last = rows[0][2], rows[-1][2]
    direction = "decreasing" if last < first else "increasing"
    print(f"  -> {label}: TTFT is {direction} from the first run to the last ({first:.1f} -> {last:.1f} ms)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="synthetic:tiny")
    parser.add_argument("--policy", default="sliding_window")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="experiments/results")
    args = parser.parse_args()

    forward = _run("forward order (100 -> 10%)", FORWARD, args)
    print()
    reverse = _run("reverse order (10 -> 100%)", REVERSE, args)

    print()
    _describe("forward", forward)
    _describe("reverse", reverse)
    print()

    if len(forward) >= 2 and len(reverse) >= 2:
        forward_decreasing = forward[-1][2] < forward[0][2]
        reverse_decreasing = reverse[-1][2] < reverse[0][2]
        if forward_decreasing and reverse_decreasing:
            print(
                "VERDICT: both orders decrease with run position, so the trend follows the\n"
                "         MEASUREMENT ORDER, not the retention budget. Do not report this\n"
                "         sweep's TTFT curve as an effect of the budget. See docs/research.md, F10."
            )
        else:
            print(
                "VERDICT: the two orders do not share a shape, so run position does not\n"
                "         explain the trend. The budget may be the cause -- but with one\n"
                "         sample per point that is still not established. Repeat before claiming."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
