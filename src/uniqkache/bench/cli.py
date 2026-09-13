"""Command-line interface for the UniqKache benchmark.

Documented invocations
----------------------
.. code-block:: bash

    # Full-cache reference at 32k context
    python -m uniqkache.bench \\
        --model synthetic:small \\
        --context-length 32768 \\
        --policy full_cache \\
        --batch-size 1

    # The same workload under an evicting policy
    python -m uniqkache.bench \\
        --model synthetic:small \\
        --context-length 32768 \\
        --policy sliding_window \\
        --keep-ratio 0.25 \\
        --batch-size 1

    # A whole experiment from a config file
    python -m uniqkache.bench --config experiments/configs/sweep_policies.json

Results are written as JSONL and CSV under ``--output-dir``. Every record
carries the metadata needed to reproduce it; see
:class:`~uniqkache.metrics.record.BenchmarkRecord`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uniqkache.bench.config import ExperimentConfig, RunSpec, load_config, percent_sweep
from uniqkache.bench.runner import run_config
from uniqkache.metrics.report import records_to_markdown
from uniqkache.policies import available_aliases, available_policies
from uniqkache.utils.errors import UniqKacheError
from uniqkache.utils.logging import get_logger

_log = get_logger(__name__)

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_RUN_FAILED = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m uniqkache.bench",
        description=(
            "Run reproducible KV-cache benchmarks. Every result is written with the "
            "model, revision, precision, hardware, policy configuration, seed and git "
            "commit that produced it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Policies: "
            + ", ".join(available_policies())
            + "\nAliases:  "
            + ", ".join(f"{a}->{t}" for a, t in sorted(available_aliases().items()))
            + "\nModels:   synthetic:tiny | synthetic:small | synthetic:medium | <hf-repo-id>"
        ),
    )

    parser.add_argument(
        "--model",
        default="synthetic:tiny",
        help="Model identifier. 'synthetic:<preset>' uses the built-in zero-download "
        "model; anything else is treated as a Hugging Face repo id. Default: %(default)s",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=1024,
        help="Prompt length in tokens. Default: %(default)s",
    )
    parser.add_argument(
        "--policy",
        default="full_cache",
        help="Cache policy. Default: %(default)s",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size. Default: %(default)s",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=None,
        help="Explicit per-layer token budget. Mutually exclusive with --keep-ratio.",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=None,
        help="Fraction of the context to retain, in (0, 1]. 1.0 is a full cache. "
        "Mutually exclusive with --capacity.",
    )
    parser.add_argument(
        "--attention-sinks",
        type=int,
        default=4,
        help="Leading tokens protected from eviction. Default: %(default)s",
    )
    parser.add_argument(
        "--precision",
        default="float32",
        help="Model/storage precision: float32, float16 or bfloat16. Default: %(default)s",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu' or 'cuda'. Default: %(default)s",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Default: %(default)s")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="Tokens to generate during the decode phase. Default: %(default)s",
    )
    parser.add_argument("--task", default="generation", help="Task label for the record")
    parser.add_argument("--dataset", default=None, help="Dataset identifier for the record")
    parser.add_argument(
        "--compressor",
        default=None,
        choices=["int8"],
        help="Compress the cache before generation. Off by default.",
    )

    quality = parser.add_argument_group("quality measurement")
    quality.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip quality measurement. NOT recommended: a record without quality "
        "cannot show whether a memory saving cost accuracy, and will be flagged.",
    )
    quality.add_argument(
        "--quality-chunk-size",
        type=int,
        default=1,
        help="Tokens per forward pass during quality evaluation. 1 reproduces true "
        "streaming decode. Default: %(default)s",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir",
        default="experiments/results",
        help="Where to write results. Default: %(default)s",
    )
    output.add_argument(
        "--config",
        default=None,
        help="Run an experiment config file instead of a single ad-hoc run.",
    )
    output.add_argument(
        "--sweep",
        action="store_true",
        help="Expand the single run into the standard retention-ratio sweep (100/75/50/25/10%%).",
    )
    output.add_argument(
        "--print-records",
        action="store_true",
        help="Print records as JSON to stdout in addition to writing files.",
    )
    output.add_argument(
        "--markdown",
        action="store_true",
        help="Print a Markdown summary table to stdout.",
    )
    output.add_argument("--repo-path", default=None, help="Repository root for git metadata")
    output.add_argument("--quiet", action="store_true", help="Reduce log output")

    parser.add_argument(
        "--list-policies",
        action="store_true",
        help="List registered policies and exit.",
    )
    return parser


def _spec_from_args(args: argparse.Namespace) -> RunSpec:
    """Build a RunSpec from CLI arguments."""
    return RunSpec(
        model=args.model,
        policy=args.policy,
        context_length=args.context_length,
        batch_size=args.batch_size,
        capacity=args.capacity,
        keep_ratio=args.keep_ratio,
        attention_sinks=args.attention_sinks,
        precision=args.precision,
        device=args.device,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        task=args.task,
        dataset=args.dataset,
        measure_quality=not args.no_quality,
        quality_chunk_size=args.quality_chunk_size,
        compressor=args.compressor,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.quiet:
        import logging

        logging.getLogger("uniqkache").setLevel(logging.WARNING)

    if args.list_policies:
        print("Registered policies:")
        for name in available_policies():
            print(f"  {name}")
        aliases = available_aliases()
        if aliases:
            print("\nAliases:")
            for alias, target in sorted(aliases.items()):
                print(f"  {alias} -> {target}")
        return EXIT_OK

    # ---- assemble the experiment ----------------------------------------
    try:
        if args.config:
            if args.sweep:
                parser.error("--config and --sweep are mutually exclusive")
            config = load_config(args.config)
        elif args.sweep:
            # Built directly from arguments rather than by expanding a single
            # RunSpec: a sweep's whole point is that each run has a different
            # budget, so validating one budget-less spec first would reject a
            # perfectly valid sweep.
            if args.capacity is not None or args.keep_ratio is not None:
                parser.error(
                    "--sweep derives its own budget per ratio; do not pass "
                    "--capacity or --keep-ratio alongside it"
                )
            config = percent_sweep(
                model=args.model,
                policy=args.policy,
                context_length=args.context_length,
                attention_sinks=args.attention_sinks,
                batch_size=args.batch_size,
                precision=args.precision,
                device=args.device,
                seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                task=args.task,
                dataset=args.dataset,
                measure_quality=not args.no_quality,
                quality_chunk_size=args.quality_chunk_size,
            )
        else:
            config = ExperimentConfig(name="single-run", runs=[_spec_from_args(args)])
    except UniqKacheError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    # ---- run -------------------------------------------------------------
    try:
        outcomes = run_config(
            config, output_dir=args.output_dir, repo_path=args.repo_path, write=True
        )
    except UniqKacheError as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return EXIT_RUN_FAILED

    records = [outcome.record for outcome in outcomes]

    if args.print_records:
        for record in records:
            print(json.dumps(record.to_dict(), indent=2, default=str))

    if args.markdown:
        print(records_to_markdown(records, title=f"Results: {config.name}"))

    # Surface integrity problems on the console, not just in the log. A run
    # whose record cannot support its own claims should be visible immediately.
    flagged = [(o.record.run_id, o.problems) for o in outcomes if o.problems]
    if flagged:
        print(
            f"\n{len(flagged)} run(s) produced records with integrity warnings:",
            file=sys.stderr,
        )
        for run_id, problems in flagged:
            for problem in problems:
                print(f"  {run_id}: {problem}", file=sys.stderr)

    print(f"wrote {len(records)} record(s) to {Path(args.output_dir)}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
