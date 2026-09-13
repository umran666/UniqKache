<!--
  Pull request. Fill in every section. A PR that reports a result without a baseline or a
  quality measurement will be asked for them before review — that is the process working,
  not a rejection of the work.
-->

## What this changes

<!-- One paragraph. What is different after this PR, and why. -->

## Pipeline stage

- **Issue:** #
- **Hypothesis** (falsifiable, with failure criteria):
- **Baseline compared against:**
- **Experiment config:** `experiments/configs/...`
- **Benchmark result** (quality + performance, or "n/a — no result claimed"):

## Evidence

<!--
  Paste the raw table. Include every row, including the inconvenient ones.
  If you are not claiming a result, write "no result claimed" and delete the table.
-->

| Policy | Budget | Cache bytes | TTFT ms | TPOT ms | Quality |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

**Integrity warnings** (from `validate_record`, quoted not hidden):

```
```

## Tests

- [ ] `pytest -q` passes
- [ ] `ruff check src tests benchmarks` passes
- [ ] New or updated tests: <!-- names -->
- [ ] Regression test added for a bug fix: <!-- name, or "n/a" -->

## Honesty checks

- [ ] A quality metric is present for every performance or memory claim
- [ ] Any `validate_record` warnings are quoted above rather than omitted
- [ ] Performance claims name the baseline, the budget, the precision and the device
- [ ] No marketing adjectives ("revolutionary", "state-of-the-art", "breakthrough", "novel")
      unless a citation or a benchmark in this repository supports it
- [ ] A negative or null result, if any, is recorded rather than dropped

## Limitations

<!--
  What this does NOT show. What was true before that is still true. What a reader might
  wrongly conclude from the numbers above. Every result has a limitation; a result without
  one has not been examined closely enough.
-->

## Files changed

<!--
  Group by module. Note any change that alters a *result* — that is a research event, not a
  refactor, and must be called out explicitly.
-->
