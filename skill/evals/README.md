# Evals

Two tiers, because *did it run* and *is it right* are different questions and only one
of them can be answered by a string match.

| Tier | Where | Asks | Cost |
|---|---|---|---|
| **1 · Invariants** | `tests/test_invariants.py` | Are the numbers self-consistent? | Free, runs in CI |
| **2 · Judgement** | `evals.json` | Does the assistant reason well about them? | Model-graded, run occasionally |

## Why the split exists

The previous version of `evals.json` asserted three kinds of thing: a file exists, a
substring appears in the reply, and the HTML is over 500KB.

Every corruption found in August 2026 — five currencies summed as one, 9.7% of demand
silently undated, 63% of purchase quantity discarded by a lead-time trim, a planning
master joining zero rows — produced a complete run, a written report, and the string
`PURCHASE-REQUEST` in the output. **The suite would have gone green on all four.**

That is not thin coverage. It is the wrong category of test: the entire risk in this
domain is a run that completes perfectly and is wrong, so an assertion that only proves
the run completed proves nothing about the risk.

It had also rotted in ways nothing surfaced. All three evals asserted on
`inventory_report_*.html`, which the pipeline stopped producing; one asserted that file
exceeded 500KB, which stopped being true when charts became inline SVG rather than
embedded PNGs; one asserted a hardcoded part number and a hardcoded quantity from a
single run; and every fixture path pointed at un-anonymised customer files under one
user's `~/Downloads`, so the suite could not run on another machine or be shared at all.

## Tier 1 — invariants

Properties that must hold of *any* run, checked on data built to violate them. They are
pytest, so they run with everything else and cost nothing.

```bash
python -m pytest tests/test_invariants.py -q
```

Currently covering: money conservation through conversion and the exclusion of unrated
currency from every total; quantity conservation through date bucketing and the
lead-time trim; a join that matches nothing being reported rather than producing silent
NaNs; recommendation coherence (no buy without a requirement, no push-out of a shortage,
no policy stock on an order-on-demand item); and the report agreeing with itself.

Each of the four August corruptions has one. They were verified by mutation — the guard
was removed, the test was confirmed to fail, and the guard restored — because a green
test that would not have caught the bug is worse than no test, since it also supplies
confidence.

**When adding one, mutate first.** Break the code the test is meant to protect, watch it
go red, then fix it. A property test written against already-correct code often asserts
something that cannot fail.

## Tier 2 — judgement

`evals.json`. Assertions of type `rubric` are the point; the mechanical ones only
establish that a run happened at all.

A rubric carries a `criterion` that must be judged true of the reply, and `fails_if` —
behaviours that make it false *even when the criterion looks satisfied*. That second
field is where most of the value is: "reports the total in one currency" is satisfiable
by an assistant that also quietly drops the lines it could not convert, and only the
`fails_if` catches it.

The scenarios deliberately include cases with no clean answer — a capital target that
may be unreachable under its own service constraint, two signals that contradict each
other, a board-pack framing that invites suppressing a caveat. What is being graded is
whether the trade-off is *surfaced*, not whether a particular number is produced.

There is no runner in this repo yet; the assertion types are documented in
`evals.json` under `_assertion_types` so one can be written against a stable contract.

## Fixtures

```bash
python skill/evals/fixtures.py <target_dir>
```

Generates four variants from `sample_data/` — which is synthetic (`SKU-002`,
`CUST-018`) — so the suite travels with the repo:

| Variant | Defect |
|---|---|
| `clean` | none, the baseline |
| `multi_currency` | USD/GBP/INR, all rated. The raw sum is ~27× the true total |
| `unrated_currency` | adds JPY, which `fx_rates.json` does not carry |
| `mangled_dates` | ship date half text, half spreadsheet-swapped ISO |

Fixtures are generated, never committed. **Real extracts must not become fixtures** —
they are not anonymised, and a 5MB file "reduced to a few hundred rows" still names real
customers. Reproduce the *shape* of a defect, as `fixtures.py` and
`tests/test_mixed_formats.py` both do, never the data that exhibited it.
