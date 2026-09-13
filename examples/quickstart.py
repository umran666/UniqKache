"""Quickstart: run a cached generation and read the cache statistics.

Run it:

.. code-block:: bash

    python examples/quickstart.py

    # force CPU, or a different retention budget
    python examples/quickstart.py --device cpu --keep-ratio 0.25

This example deliberately shows the two things that matter for reading any UniqKache
result:

1. ``stats()`` reports memory, and it reports ``bytes_on_device`` and
   ``bytes_offloaded`` separately, because only one of them frees anything.
2. The model's weights are random, so the generated tokens are meaningless. The
   ``weights_are_random`` flag is checked and printed, because a result from this
   model is a diagnostic of cache behaviour and never a language-modelling result.
"""

from __future__ import annotations

import argparse

import torch

from uniqkache.models import build_cache_for_model, build_model
from uniqkache.runtime import GenerationConfig, GenerationEngine
from uniqkache.utils.device import describe_hardware, resolve_device
from uniqkache.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="tiny", help="synthetic preset: tiny, small, medium")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu' or 'cuda'")
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--policy",
        default="full_cache",
        help="full_cache, sliding_window, lru, attention_based, token_importance, adaptive",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=None,
        help="Fraction of the context to retain. Required for an evicting policy.",
    )
    parser.add_argument("--attention-sinks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    hardware = describe_hardware()
    print(f"device: {hardware.device_type} ({hardware.device_name or 'cpu'})")
    print(f"torch:  {hardware.torch_version}  cuda: {hardware.cuda_version}")

    # ---- model ---------------------------------------------------------
    model = build_model(preset=args.preset, seed=args.seed, device=device)
    print(
        f"model:  synthetic:{args.preset}  "
        f"params={sum(p.numel() for p in model.parameters()):,}  "
        f"random_weights={model.weights_are_random}"
    )
    if model.weights_are_random:
        print(
            "        NOTE: weights are random. Generated text is meaningless. This run is a\n"
            "        diagnostic of cache behaviour, not a language-modelling result."
        )

    # ---- cache ---------------------------------------------------------
    capacity = None
    if args.keep_ratio is not None:
        capacity = max(1, int(args.context_length * args.keep_ratio))
    if capacity is None and args.policy != "full_cache":
        parser.error(f"policy {args.policy!r} evicts, so --keep-ratio is required")

    cache = build_cache_for_model(
        model,
        capacity=capacity,
        attention_sinks=args.attention_sinks if capacity is not None else 0,
        policy_name=args.policy,
        device=device,
    )
    print(
        f"cache:  policy={args.policy} capacity={capacity} "
        f"sinks={cache.config.attention_sinks} "
        f"bytes/token={cache.config.bytes_per_token()}"
    )

    # ---- warmup --------------------------------------------------------
    # Both the plain forward path and the eviction path must be compiled before
    # anything is timed, or the first timed run absorbs one-time kernel
    # compilation cost and reports a fake TTFT. Two separate measurement bugs
    # came from getting this wrong; see docs/research.md, F1 and F2.
    prompt = torch.randint(0, model.config.vocab_size, (1, args.context_length), device=device)
    _warmup(model, args, device)

    # ---- generate ------------------------------------------------------
    engine = GenerationEngine(model, cache, GenerationConfig(max_new_tokens=args.max_new_tokens))
    result = engine.generate(prompt)

    print()
    print(result.summary())

    # ---- cache statistics ----------------------------------------------
    stats = cache.stats()
    print()
    print(stats.summary())
    print()
    print(
        f"  occupancy        : {stats.total_tokens} tokens across {len(cache.store.layers)} layers"
    )
    print(f"  bytes on device  : {stats.bytes_on_device:,}")
    print(f"  bytes offloaded  : {stats.bytes_offloaded:,}   <- frees nothing; costs bandwidth")
    print(f"  compression ratio: {stats.compression_ratio:.3f}   <- representation, not occupancy")
    print(f"  evictions        : {stats.evictions}")
    # `total_tokens` is summed across layers, so reconcile it against the
    # per-layer cost rather than passing it to bytes_for_tokens(), which expects
    # a per-layer count and would over-count by num_layers.
    theoretical = cache.config.bytes_per_token_per_layer() * stats.total_tokens
    print(f"  theoretical bytes: {theoretical:,}")

    if result.peak_memory_bytes is None:
        print("  peak memory      : None (not measurable on this device -- not 0)")
    else:
        print(f"  peak memory      : {result.peak_memory_bytes / 1024**2:.1f} MiB")

    return 0


def _warmup(model, args, device) -> None:
    """Run both code paths once so no timed run pays compilation cost."""
    from uniqkache.models import build_cache_for_model as _build

    short = torch.randint(0, model.config.vocab_size, (1, 8), device=device)

    # Phase 1: the plain decode path.
    plain = _build(model, capacity=None, policy_name="full_cache", device=device)
    GenerationEngine(model, plain, GenerationConfig(max_new_tokens=2)).generate(short)

    # Phase 2: the eviction path, forced with a tiny capacity.
    if args.policy != "full_cache":
        tiny_capacity = max(2, min(args.attention_sinks + 1, 4))
        evicting = _build(
            model,
            capacity=tiny_capacity,
            attention_sinks=0,
            policy_name=args.policy,
            device=device,
        )
        GenerationEngine(model, evicting, GenerationConfig(max_new_tokens=2)).generate(short)


if __name__ == "__main__":
    raise SystemExit(main())
