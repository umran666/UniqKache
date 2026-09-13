# Examples

Runnable demonstrations of the public API. Each one is self-contained, needs no download, and
runs on CPU or CUDA.

```bash
python examples/quickstart.py
python examples/quickstart.py --device cuda --context-length 128 --policy sliding_window --keep-ratio 0.25
python examples/custom_policy.py
```

| Example | Shows |
| --- | --- |
| `quickstart.py` | Build a model and a cache, run a cached generation, and read the cache statistics. Demonstrates why `bytes_on_device` and `bytes_offloaded` are separate, and includes the warmup that timing requires. |
| `custom_policy.py` | Write and register a new retention policy, then run it through the benchmark harness. The starting point for a research contribution. |

## A note on the numbers these print

The synthetic models have **randomly initialised weights**, so the generated tokens are
meaningless. These examples are for learning the interface and for checking that cache
accounting is right, not for producing results.

In particular:

- **Memory numbers are meaningful.** The accounting is arithmetic and exact to the byte, and
  each example prints the measured footprint next to the theoretical one so you can check it.
- **Quality numbers from these models are a diagnostic**, not a language-modelling result.
- **Timing numbers from these examples are indicative only.** They use a minimal warmup and a
  single run. Latency at this model scale is not currently measurable above run-to-run noise —
  see `docs/research.md`, F10. For a measurement, use the benchmark harness:

```bash
python -m uniqkache.bench --model synthetic:tiny --context-length 192 --policy full_cache
```

Full methodology: [`docs/benchmarks.md`](../docs/benchmarks.md).
