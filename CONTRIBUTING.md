# Contributing to UniqKache

This is a research repository. The bar for a code contribution is *correctness and
reproducibility*; the bar for a research contribution is *evidence*. Both are enforced in
review, and both are described below.

The single rule everything else follows from:

> **A claim must be earned by an experiment, not asserted by a description.**

If you are unsure whether something counts as a finding, it does not yet. Write it down as a
hypothesis in [docs/research.md](docs/research.md) and design the experiment.

---

## The contribution pipeline

```
Issue → Hypothesis → Implementation → Baseline → Experiment → Benchmark → PR → Review → Merge → Research result
```

| Step | What it means here | Where it lives |
| --- | --- | --- |
| **Issue** | State the problem or the question before writing code. Use a template. | GitHub issue |
| **Hypothesis** | A falsifiable prediction, including **what would prove it wrong**. | Issue body / `docs/research.md` |
| **Implementation** | The smallest change that tests the hypothesis. No speculative generality. | Branch |
| **Baseline** | The method you are compared against, running in the same harness. | `baselines/` |
| **Experiment** | A declared configuration: model, context, budget, seed, precision. | `experiments/configs/*.json` |
| **Benchmark** | The run that produces records, including a quality measurement. | `experiments/results/` |
| **PR** | Diff + the numbers it produced + the limitations it did not remove. | Pull request |
| **Review** | A reviewer checks the claim against the evidence. | Review |
| **Merge** | Squash-merged into `main` once tests and benchmarks pass. | `main` |
| **Research result** | The finding, written up **including negative results**. | `docs/research.md` |

Two things this pipeline forbids:

- **Reporting a win without a baseline.** "Our policy uses less memory" is not a result until
  it is measured against `full_cache` at the same context length, precision and seed.
- **Deleting a failure.** If an experiment did not work, it goes in the
  [Failed Experiments](docs/research.md#failed-experiments) section. A documented failure is
  a contribution; a deleted one is a defect.

---

## Development setup

```bash
git clone https://github.com/umran666/UniqKache.git
cd UniqKache
python -m pip install -e ".[dev]"
pre-commit install          # optional but recommended
```

Verify:

```bash
pytest -q                                     # full suite, ~8s, no GPU needed
ruff check src tests benchmarks
python -m uniqkache.bench --list-policies
```

The test suite is deliberately runnable **without a GPU and without any download**. If your
change breaks that, it will be caught in review.

### Multiple interpreters

`ModuleNotFoundError: No module named 'torch'` almost always means `python` on your `PATH` is
not the interpreter you installed into. Confirm which one you are using:

```bash
python -c "import sys, torch; print(sys.executable, torch.__version__)"
```

Then invoke that interpreter explicitly (on Windows, typically the full path to
`Python310\python.exe`).

---

## Coding standards

- **Line length** 100, enforced by `ruff`. Format with `ruff format`; do not hand-align.
- **Lint rules** are configured in `pyproject.toml` (`E, F, I, UP, B, SIM, C4, RET, PIE, RUF`).
  `make lint` must pass with zero findings.
- **Python 3.10+**. Use `from __future__ import annotations` and `X | None`, not `Optional[X]`.
- **Types on public functions.** Dataclasses for anything crossing a module boundary.
- **Docstrings explain *why*, not *what*.** The code says what it does. A docstring earns its
  place by recording a decision, a constraint, or a failure mode that is not visible in the
  code.
- **No silent failures.** `except Exception: pass` is banned. Log it and carry on, or let it
  propagate. A swallowed error is a bug wearing a disguise.
- **No fabricated success.** Never hardcode `success: True`, a plausible-looking metric, or a
  hardware constant you did not measure. If a value is unknown, it is `None`.
- **Missing is not zero.** An unmeasured metric is `None`/`null`. Reporting `0` for peak
  memory on CPU reads as a perfect result.
- **No marketing language.** Not "revolutionary", "state-of-the-art", "breakthrough",
  "best-in-class", "novel" — unless a citation or a benchmark in this repository supports it,
  and even then, state the measurement rather than the adjective.

---

## Branch naming

| Prefix | Use |
| --- | --- |
| `main` | Released, green. Protected; merges only via PR. |
| `develop` | Integration branch for work in progress. |
| `feature/*` | A new capability. `feature/paged-store` |
| `research/*` | An experiment that may not work out. `research/adaptive-pressure-weighting` |
| `benchmark/*` | Benchmark harness or measurement work. `benchmark/latency-percentiles` |
| `bugfix/*` | A defect fix. `bugfix/quant-axis-gather` |
| `docs/*` | Documentation only. `docs/architecture-diagram` |

Branch from `main` unless the work depends on an unmerged `develop` change.

---

## Commit conventions

[Conventional Commits](https://www.conventionalcommits.org/). Types in use:
`feat`, `fix`, `test`, `docs`, `bench`, `refactor`, `chore`, `perf`.

```
feat: add sliding window policy
feat: add KV cache abstraction
feat: add full cache baseline
feat: add benchmark runner
fix: correct int8 quantisation axis for keys
test: add cache correctness tests
docs: add architecture documentation
bench: add retention sweep at 32k context
```

Rules:

- One logical change per commit. If the subject line needs "and", it is two commits.
- **No giant initial commit.** History is reviewable because it was built incrementally.
- **Commit the results that back a documented claim**, together with the config that produced
  them — that is what `experiments/results/` is for, and the file names are timestamped so a
  re-run never overwrites an earlier one. See
  [experiments/results/README.md](experiments/results/README.md) for the convention.
- **Do not commit exploratory runs.** Ad-hoc output goes to an ignored path
  (`experiments/results/runs/`). A result file that nothing cites is clutter.
- **If a committed result stops being current, move it to `experiments/results/superseded/`
  and say why** rather than deleting it. Withdrawn results are the evidence for a failed
  experiment, and `docs/research.md` cites them by name.
- Never commit a `.env`, a token, or a machine-specific path.

---

## Testing requirements

**No pull request may change core cache behaviour without tests.** That includes any change to
`cache/`, `policies/`, `compression/`, `offload/`, `prefetch/` or `controllers/`.

The suite has three layers:

| Layer | Location | Purpose |
| --- | --- | --- |
| Unit | `tests/unit/` | One behaviour each. Fast. No GPU, no downloads. |
| Integration | `tests/integration/` | End-to-end inference. The load-bearing one is `test_incremental_matches_single_shot.py`: incremental decode must equal a single-shot forward pass. |
| Regression | `tests/regression/` | Pins a specific bug that was fixed. Never delete one. |

Requirements for a PR that touches runtime behaviour:

1. `pytest -q` passes in full.
2. A new unit test covers the behaviour you added.
3. If you fixed a bug, a regression test in `tests/regression/` names the bug and the
   symptom. Add a comment saying what the wrong behaviour looked like.
4. If your change alters a *result*, say so explicitly in the PR description. A change that
   moves a number is a research event, not a refactor.

Mark tests that need a GPU with `@pytest.mark.gpu` and slow ones with `@pytest.mark.slow`, so
the default run stays fast and hardware-free.

### Coverage floor

CI runs a coverage gate (`--cov-fail-under=70`) over `tests/unit`, `tests/regression` and
`tests/integration`. The floor is set to the measured coverage when the gate was introduced, so
it can only move in one direction deliberately:

- **Raising** the floor is always welcome; do it in the PR that adds the tests.
- **Lowering** the floor requires a sentence in the PR description saying why the uncovered code
  is acceptable. Merging a change that silently lets coverage regress is how a test suite stops
  describing the code it claims to cover.

Local check: `python -m pytest tests/unit tests/regression tests/integration -q --cov=uniqkache
--cov-report=term-missing`.

---

## Benchmark requirements

Every benchmark result in a PR must be produced by a **declared configuration**, not an ad-hoc
command whose arguments are lost.

```bash
# a single declared run
python -m uniqkache.bench --config experiments/configs/sweep_policies.json

# ad hoc is fine for exploration; it is not fine as evidence
python -m uniqkache.bench --model synthetic:tiny --context-length 192 --policy full_cache
```

A result is only acceptable as evidence if:

- **A quality metric is present.** `validate_record` will flag a record with latency or memory
  and no quality, and the runner prints that warning. Do not paste a flagged record as a win.
- **A baseline is present at the same context length, precision and seed.**
- **The budget is recorded.** A bounded policy without its capacity is not reproducible.
- **The git commit is recorded and the tree was clean.** Commit before you benchmark, or state
  the dirty flag.
- **The hardware is named.** Never compare numbers across machines as if they were comparable;
  this repository has not validated any cross-device claim.
- **Retention sweeps use 100 / 75 / 50 / 25 / 10 %** unless you state why a different set is
  needed. `--sweep` produces exactly that.

Report the raw table, not a summary of it. If a row is inconvenient, that is the row the
reviewer most wants to see.

### What counts as a win

Only one of these, and only when the others are stated alongside:

| Form | Required evidence |
| --- | --- |
| Same quality, less memory | quality within noise, `cache_bytes_total` lower, budget recorded |
| Same quality, lower latency | quality within noise, TTFT/TPOT lower, same precision and device |
| Same memory, better quality | both budgets equal, quality metric higher, baseline present |
| A frontier movement | both axes measured; described as a trade-off, **not** as an improvement |

A memory reduction that costs quality is **a trade-off**, and must be described as one.

---

## Research contribution guidelines

For anything that proposes a *method* rather than a fix, use the
[research proposal template](.github/ISSUE_TEMPLATE/research_proposal.yml). It requires:
Problem, Hypothesis, Related work, Proposed method, Experiment design, Expected outcome and
**Failure criteria**. The last one is mandatory — a hypothesis you cannot falsify is not a
hypothesis.

Then:

1. **Search for prior work first.** If the method exists, reproduce it and cite it. A
   reproduction is a legitimate contribution; presenting one as new is not.
2. **Classify your contribution honestly** as one of: (a) a reproduced technique, (b) an
   engineering improvement, (c) a genuinely new idea, (d) a validated finding. Most
   contributions are (a) or (b), and that is fine.
3. **Do not claim novelty before comparing against existing methods.** State the comparison
   you have not yet run, rather than implying it.
4. **Write the failure criteria before the result.** If you would not know a negative result
   when you saw it, the experiment is not designed.
5. **Record negative results** in [docs/research.md](docs/research.md#failed-experiments).
6. **Cite the original paper** for any reproduced method, in `baselines/README.md` and in the
   policy's docstring.

Reproductions must be *faithful*. If you simplify a published method, say so in the docstring
and in `baselines/README.md`, because a simplified reproduction that underperforms may be
underperforming because of the simplification.

---

## Pull request checklist

Copy this into your PR description:

```markdown
## What this changes
<one paragraph>

## Pipeline stage
- [ ] Issue: #
- [ ] Hypothesis (falsifiable, with failure criteria):
- [ ] Baseline compared against:
- [ ] Experiment config:
- [ ] Benchmark result (quality + performance):

## Evidence
| Policy | Budget | Cache bytes | TTFT ms | TPOT ms | Quality |
| --- | --- | --- | --- | --- | --- |

## Tests
- [ ] `pytest -q` passes
- [ ] New/updated tests: <names>
- [ ] Regression test added for a bug fix: <name or n/a>

## Honesty checks
- [ ] A quality metric is present for every performance claim
- [ ] `validate_record` warnings are quoted, not hidden
- [ ] Limitations introduced or not removed are stated below
- [ ] No marketing adjectives

## Limitations
<what this does NOT show>

## Files changed
<list>
```

A reviewer will check the claim against the evidence, and will ask for the baseline or the
quality measurement if it is missing. That is not a rejection of the work — it is the review
doing its job.

---

## After merge

Add the outcome to [docs/research.md](docs/research.md): under Results if it worked, under
Failed Experiments if it did not, and under Open Questions if it raised a new one. If it
changes what the project claims, update the README's status table in the same PR.

## License

By contributing you agree that your contributions are licensed under Apache-2.0.
