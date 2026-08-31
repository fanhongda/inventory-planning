"""
The checks each gate runs.

One function per checkpoint, each taking what that point of the run actually holds and
returning a `GateReport`. They are separate from `gates.py` on purpose: that module is
the vocabulary — severity, finding, threshold, override — and this one is the
judgement, which is the part that changes as more ways of being wrong turn up.

Every check here earned its place by having produced a wrong report first. None of
them is a general-purpose data-quality rule; a run does not stop because a column is
untidy, it stops because a specific figure downstream would be a fabrication.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from .dimensions import normalise_frame
from .gates import BLOCK, SEVERE, WARN, Finding, GateReport, GateThresholds

_MIN_KEY_DISTINCT = 10


# ── Stage: intake ────────────────────────────────────────────────────────────


def gate_intake(intake_result, plan, thresholds: GateThresholds) -> GateReport:
    """
    Everything that can be judged from the documents alone, before any arithmetic.

    This is the most valuable checkpoint by a wide margin: every silent failure the
    pipeline has produced was visible here and invisible in the report it went on to
    write.
    """
    report = GateReport(stage="intake")
    documents = getattr(intake_result, "documents", {}) or {}

    report.findings.extend(_check_sku_agreement(documents, thresholds))
    report.findings.extend(_check_semantic_failures(documents, thresholds))
    report.findings.extend(_check_dimension_collisions(documents))
    # Order matters: where the family column is absent altogether, the per-SKU count
    # is the same gap said twice at a finer grain, and the severe finding is the one
    # that should be read. A reader who sees both reads neither.
    absent = _check_product_dimension(documents, plan)
    report.findings.extend(absent)
    if not absent:
        report.findings.extend(_check_product_family_coverage(documents, thresholds))
    return report


def _check_product_dimension(documents: Dict[str, Any], plan) -> List[Finding]:
    """
    No product family column anywhere — the whole grouping axis is missing.

    Distinct from `product_family_missing`, which counts individual unclassified items.
    This is the case where no master carries the column at all, so every family in the
    output is `_infer_family`'s reading of the part number and there is no subset of
    rows that is trustworthy. `product_family_source == "master"` selects nothing.

    Severe rather than blocking. Replenishment does not use the family: safety stock,
    reorder points and order quantities come out identical either way, and withholding
    a plan a buyer can act on because the rollup above it is unreliable helps nobody.
    What is owed is that nobody reads the rollup as if it meant something.
    """
    masters = [dt for dt in ("item_master", "planning_master") if dt in documents]
    if not masters:
        return []
    if any("product_family" in documents[dt].frame.columns
           and documents[dt].frame["product_family"].notna().any() for dt in masters):
        return []

    names = ", ".join(f"{dt} ({documents[dt].source_name})" for dt in masters)
    return [Finding(
        stage="intake", check="no_product_dimension", severity=SEVERE,
        what=(f"No product family anywhere in the master data ({names}). Nothing "
              f"mapped to `product_family`, so every family in this run is guessed "
              f"from the part number."),
        why=("The guess is only as good as the numbering scheme, and where the scheme "
             "does not encode the family — most SAP material numbers do not — the "
             "groups are arbitrary. Unlike a partial gap there is no trustworthy "
             "subset: `product_family_source` reads `inferred` on every row."),
        impacts=[
            "Revenue and forecast value by product line: groups are the numbering "
            "scheme's, not the business's — do not present these",
            "Days-on-hand by product line: same denominator problem, unusable",
            "S&OP worksheet: cannot be split for review by the team owning each line",
            "Family-scoped rules in planning_parameters.md match the guess, so a rule "
            "written for one line may be applying to another — check `rules_applied`",
            "Not affected: safety stock, reorder points, purchase quantities and every "
            "per-SKU recommendation. Those never use the family.",
        ],
        fix=("Map the product family in the master. The column is usually there under "
             "a name no alias matched — run `python -m inventory_planning.explain "
             "<master file>` to see the unmatched headers. Where a master carries a "
             "two-level hierarchy, the finer level is the family and the coarser one "
             "is `business_unit`."),
        evidence={"masters": masters},
    )]


def _check_sku_agreement(documents: Dict[str, Any],
                         thresholds: GateThresholds) -> List[Finding]:
    """
    A document whose item numbers appear in no other document.

    Every join downstream is on `sku`. A key that agrees with nothing does not raise:
    the merges return empty, safety stock falls to zero, annual value is zero so every
    item lands in one ABC class, and the report is full of confident zeroes. The intake
    already computes this and prints a warning; a warning printed above sixty lines of
    output is not a control.
    """
    floor = thresholds["sku_agreement_floor"]
    keyed = {dt: doc for dt, doc in documents.items()
             if "sku" in getattr(doc, "frame", pd.DataFrame()).columns and len(doc.frame)}
    if len(keyed) < 2:
        return []

    skus = {dt: set(doc.frame["sku"].dropna().astype(str)) for dt, doc in keyed.items()}
    out: List[Finding] = []
    for doc_type, own in skus.items():
        if len(own) < _MIN_KEY_DISTINCT:
            continue
        others = set().union(*(s for dt, s in skus.items() if dt != doc_type))
        if not others:
            continue
        share = len(own & others) / len(own)
        if share >= floor:
            continue
        doc = keyed[doc_type]
        mapped = getattr(getattr(doc.route, "adapter", None), "column_map", {}).get("sku", "?")
        # The suggestion is the useful half. That `sku` matches nothing is a puzzle;
        # that `Alternate material` would have matched 82% is an answer.
        better = [(len(values & others) / max(len(values), 1), name)
                  for name, values in getattr(doc, "key_candidates", {}).items()
                  if len(values) >= _MIN_KEY_DISTINCT]
        best_share, best_col = max(better, default=(0.0, None))
        fix = (f"Map `sku` to {best_col!r} in an adapter for this export — it would "
               f"match {best_share:.0%}."
               if best_col and best_share > max(share + 0.2, floor)
               else "Check which column the rest of the business uses as the item "
                    "number, and map `sku` to it in an adapter for this export.")
        out.append(Finding(
            stage="intake", check="sku_agreement", severity=BLOCK,
            what=(f"{doc_type} ({doc.source_name}) keys on {mapped!r}, and only "
                  f"{share:.0%} of its {len(own):,} item numbers appear in any other "
                  f"document."),
            why=("Every join in the pipeline is on `sku`. At this rate the merges "
                 "return nothing, and nothing reads as zero rather than as missing — "
                 "a full report of confident zeroes with no error in it."),
            fix=fix,
            evidence={"doc_type": doc_type, "source": doc.source_name,
                      "mapped_from": mapped, "agreement": round(share, 4),
                      "distinct_skus": len(own)},
        ))
    return out


def _check_semantic_failures(documents: Dict[str, Any],
                             thresholds: GateThresholds) -> List[Finding]:
    """
    An assertion failing on nearly every row is the wrong column, not dirty data.

    The ceiling is what keeps this usable. 8% of lead times negative is a finding the
    report carries and the run survives; 100% negative means the two dates are the
    wrong way round or one of them is not a date. The first is data, the second is
    mapping, and only the second is worth stopping for.
    """
    ceiling = thresholds["semantic_failure_ceiling"]
    out: List[Finding] = []
    for doc_type, doc in sorted(documents.items()):
        report = getattr(doc, "test_report", None)
        if report is None:
            continue
        for result in report.results:
            if result.passed or result.name.startswith("required_field"):
                continue
            if getattr(result, "severity", "error") != "error":
                continue
            rate = float(getattr(result, "failure_rate", 0.0) or 0.0)
            if rate < ceiling:
                continue
            out.append(Finding(
                stage="intake", check="semantic_failure", severity=BLOCK,
                what=(f"{doc_type} ({doc.source_name}): {result.name} fails on "
                      f"{rate:.0%} of rows ({result.failed_rows:,}/{result.total_rows:,})."),
                why=("At this rate it is not dirt in the data but the wrong column, or "
                     "two columns the wrong way round. Every figure derived from it "
                     "carries the same error, uniformly and invisibly."),
                fix=(f"Look at the source columns behind {result.name} — run "
                     f"`python -m inventory_planning.explain <file>` — and correct the "
                     f"mapping in an adapter."),
                evidence={"doc_type": doc_type, "source": doc.source_name,
                          "test": result.name, "failure_rate": round(rate, 4)},
            ))
    return out


def _check_dimension_collisions(documents: Dict[str, Any]) -> List[Finding]:
    """
    One business unit spelled two ways splits every rollup that groups by it.

    A warning rather than a block: the fold is applied, so the numbers that follow are
    right. What the reader has to see is that it happened, because `Water` and `Water
    Segment` becoming one row is a judgement about their business and not about their
    data.
    """
    out: List[Finding] = []
    for doc_type, doc in sorted(documents.items()):
        frame = getattr(doc, "frame", None)
        if frame is None or not len(frame):
            continue
        _, collisions = normalise_frame(frame, ("business_unit", "product_family",
                                                "country", "region"))
        for collision in collisions:
            out.append(Finding(
                stage="intake", check="dimension_spelling", severity=WARN,
                what=(f"{doc_type} ({doc.source_name}) spells one "
                      f"{collision.column} more than one way — {collision.describe()}"),
                why=("Grouped on the raw value these are separate rows: the rollup "
                     "splits, each half looks smaller than the business is, and a "
                     "segmented forecast is fitted to half the history twice."),
                fix=(f"They have been folded onto {collision.kept!r} for this run. Fix "
                     f"the spelling at source so the next export does not need it."),
                evidence={"doc_type": doc_type, "column": collision.column,
                          "kept": collision.kept, "merged": collision.merged},
            ))
    return out


def _check_product_family_coverage(documents: Dict[str, Any],
                                   thresholds: GateThresholds) -> List[Finding]:
    """
    Count the SKUs that sell and have no family, and let the run continue.

    This was a BLOCK and is deliberately not one any more. The argument for stopping
    was that a partly-filled family column is invisible — `_infer_family` supplies the
    rest from the part-number prefix and nothing marks the boundary. The argument
    against stopping is stronger: an item selling without a master row is *normal* on a
    live catalogue, it is the ordinary state of anything new or anything transferred in,
    and a planner who cannot get a report until master data is perfect gets no reports.

    What the invisibility argument actually demanded was not a stop but a *count*, and
    an output that can be reconciled against it. `assemble` now tags every family with
    `product_family_source`, so an inferred one is inferred on the face of the report
    and the number here is the number a reader can find in the sheet.

    Split into two findings because the two have different owners. A SKU with no master
    row at all is a master-data gap — somebody has to create the item. A SKU with a row
    and a blank family is a classification gap — somebody has to decide what it is.
    """
    floor = thresholds["product_family_coverage_floor"]
    demand_skus: set = set()
    for doc_type in ("sales_history", "demand_timeseries"):
        doc = documents.get(doc_type)
        if doc is not None and "sku" in doc.frame.columns:
            demand_skus |= set(doc.frame["sku"].dropna().astype(str))
    if not demand_skus:
        return []

    mastered: set = set()
    covered: set = set()
    sources: List[str] = []
    for doc_type in ("item_master", "planning_master"):
        doc = documents.get(doc_type)
        if doc is None or "sku" not in doc.frame.columns:
            continue
        frame = doc.frame
        mastered |= set(frame["sku"].dropna().astype(str))
        if "product_family" in frame.columns:
            covered |= set(frame.loc[frame["product_family"].notna(), "sku"]
                           .dropna().astype(str))
        sources.append(f"{doc_type} ({doc.source_name})")
    if not sources:
        return []

    no_master = sorted(demand_skus - mastered)
    no_family = sorted((demand_skus & mastered) - covered)
    share = len(demand_skus & covered) / len(demand_skus)

    out: List[Finding] = []
    if no_master:
        out.append(Finding(
            stage="intake", check="sku_missing_from_master", severity=WARN,
            what=(f"{len(no_master):,} of the {len(demand_skus):,} SKUs with demand "
                  f"have no row in {' or '.join(sources)}."),
            why=("They are planned on config defaults for MOQ and order multiple, and "
                 "their product family is guessed from the part number. Everything "
                 "about them is a fallback, and the report says so per row rather than "
                 "leaving it to be assumed."),
            fix=("Create the items in the master, or confirm they are out of scope. "
                 f"First few: {', '.join(no_master[:5])}"),
            evidence={"count": len(no_master), "demand_skus": len(demand_skus),
                      "examples": no_master[:20]},
        ))
    if no_family:
        out.append(Finding(
            stage="intake", check="product_family_missing", severity=WARN,
            what=(f"{len(no_family):,} SKUs with demand are in the master but carry no "
                  f"product family."),
            why=("Their family is filled from the part-number prefix, so a rollup by "
                 "product line mixes master data with a guess. `product_family_source` "
                 "on every row says which, and these are the rows reading `inferred`."),
            fix=("Classify them in the master. Until then, filter the rollup on "
                 f"`product_family_source == \"master\"` to see only what is known. "
                 f"First few: {', '.join(no_family[:5])}"),
            evidence={"count": len(no_family), "examples": no_family[:20]},
        ))
    if out and share < floor:
        # The headline, once, so the coverage number appears whichever of the two
        # gaps dominates.
        out[0] = Finding(
            stage=out[0].stage, check=out[0].check, severity=WARN,
            what=(f"Product family coverage is {share:.0%} of the "
                  f"{len(demand_skus):,} SKUs with demand. " + out[0].what),
            why=out[0].why, fix=out[0].fix,
            evidence={**out[0].evidence, "coverage": round(share, 4)},
        )
    return out


# ── Stage: demand ────────────────────────────────────────────────────────────


def gate_demand(time_series: pd.DataFrame, as_of, stale_days: Optional[float],
                thresholds: GateThresholds) -> GateReport:
    """What the demand history can and cannot support before anything is fitted to it."""
    report = GateReport(stage="demand")
    if time_series is None or not len(time_series.columns):
        report.findings.append(Finding(
            stage="demand", check="demand_series", severity=BLOCK,
            what="The demand history produced no series at all.",
            why=("There is nothing to forecast, so every downstream number would come "
                 "from a default rather than from this business."),
            fix=("Check that the sales history's date and quantity columns mapped — "
                 "`python -m inventory_planning.explain <file>`."),
        ))
        return report

    active = (time_series.fillna(0) > 0).sum()
    months = float(active.median())
    if months < thresholds["median_history_months"]:
        report.findings.append(Finding(
            stage="demand", check="history_depth", severity=WARN,
            what=(f"The median SKU has demand in only {months:.0f} of "
                  f"{len(time_series)} periods."),
            why=("Below roughly twelve observations nothing can be held out, so the "
                 "model competition does not run and every SKU falls back to the "
                 "pattern route. `selected_by` on the forecast says which."),
            fix="Extend the history window if a longer extract is available.",
            evidence={"median_active_periods": months, "periods": len(time_series)},
        ))

    if stale_days is not None and stale_days > thresholds["extract_staleness_days"]:
        report.findings.append(Finding(
            stage="demand", check="extract_staleness", severity=WARN,
            what=f"The newest date in the data is {stale_days:.0f} days old ({as_of}).",
            why=("The run anchors to that date rather than to today, so the plan is "
                 "made as of then. Open orders placed since are not in it, and the "
                 "recommendations are that old."),
            fix="Pull a current extract before acting on the purchase recommendations.",
            evidence={"as_of": str(as_of), "stale_days": stale_days},
        ))
    return report


# ── Stage: forecast ──────────────────────────────────────────────────────────


def gate_forecast(forecast_detail: pd.DataFrame, time_series: pd.DataFrame,
                  thresholds: GateThresholds) -> GateReport:
    """
    Whether the forecast covers the business it is supposed to be planning.

    A forecast is not produced for every SKU — too short a series, no demand at all —
    and that is correct. What is not correct is the recommender then reading a missing
    forecast as a forecast of nothing: the SKU is planned against zero demand, holds no
    safety stock, and is never bought until it stocks out.
    """
    report = GateReport(stage="forecast")
    if time_series is None or not len(time_series.columns):
        return report

    with_demand = {str(c) for c in time_series.columns
                   if float(time_series[c].fillna(0).sum()) > 0}
    if not with_demand:
        return report

    forecast_skus = (set(forecast_detail["sku"].astype(str))
                     if forecast_detail is not None and len(forecast_detail) else set())
    share = len(with_demand & forecast_skus) / len(with_demand)
    if share >= thresholds["forecast_coverage_floor"]:
        return report

    missing = sorted(with_demand - forecast_skus)
    report.findings.append(Finding(
        stage="forecast", check="forecast_coverage", severity=BLOCK,
        what=(f"{share:.0%} of the {len(with_demand):,} SKUs that sold got a forecast "
              f"— {len(missing):,} did not."),
        why=("A SKU with no forecast is planned against zero demand: no safety stock, "
             "no reorder point, and no purchase recommendation until it stocks out. "
             "Nothing in the report distinguishes it from an item that genuinely has "
             "no demand."),
        fix=("Check the demand series for these SKUs — usually a history too short to "
             "model, sometimes a date column that parsed for some rows and not others. "
             f"First few: {', '.join(missing[:5])}"),
        evidence={"coverage": round(share, 4), "with_demand": len(with_demand),
                  "uncovered": len(missing), "examples": missing[:20]},
    ))
    return report


# ── Stage: plan ──────────────────────────────────────────────────────────────


def gate_plan(forecast_summary: pd.DataFrame, inventory: pd.DataFrame,
              thresholds: GateThresholds, attributes: pd.DataFrame = None,
              classified: pd.DataFrame = None) -> GateReport:
    """
    Whether the items the business *intends to stock* joined to a stock position.

    The first version of this check counted every forecast SKU and blocked below 80%,
    and it was wrong about the business rather than about the arithmetic. **An
    inventory extract is not expected to contain every item that sold.** A
    make-to-order item is bought against a customer order and never held; a
    non-stocking item is one the policy has decided not to carry. Both sell, neither
    has a position, and an extract that omits them is correct. On a real run 45% of
    forecast SKUs had no inventory row and the gate stopped a report that was fine.

    Worse than being wrong, it was wrong in the way that gets a gate removed: the fix
    reached for was `allow_degraded`, which switches off *every* gate at every stage.
    A check that fires on the ordinary state of the data costs more than it protects.

    So the denominator is now the items that should have a position — the ERP's `MTS`
    where it says, the pipeline's stocking class otherwise — and the make-to-order and
    non-stocking items are counted separately and reported as context rather than as a
    fault.

    It no longer blocks either. A missing position is visible in the output: the SKU
    comes out at zero on hand and lands in SHORTAGE-RISK. Blocking is reserved for
    damage a reader cannot see, and this is not that. The genuine catastrophe — an item
    key that agrees with nothing — is caught at intake by `sku_agreement`, which
    compares the keys directly instead of inferring a join failure from a coverage
    number that has an innocent explanation.
    """
    report = GateReport(stage="plan")
    if (forecast_summary is None or not len(forecast_summary)
            or inventory is None or not len(inventory)):
        return report

    forecast_skus = set(forecast_summary["sku"].astype(str))
    stocked = set(inventory["sku"].dropna().astype(str))
    expected, held_by_design, basis = _expected_to_hold(
        forecast_skus, attributes, classified)

    missing_by_design = sorted((forecast_skus - stocked) & held_by_design)
    if not expected:
        return report

    covered = expected & stocked
    share = len(covered) / len(expected)
    if share >= thresholds["position_coverage_floor"]:
        return report

    missing = sorted(expected - stocked)
    context = ""
    if missing_by_design:
        context = (f" A further {len(missing_by_design):,} SKUs have no position and "
                   f"are not expected to — they are make-to-order or non-stocking, and "
                   f"are excluded from the figure above.")

    report.findings.append(Finding(
        stage="plan", check="position_coverage", severity=SEVERE,
        what=(f"{share:.0%} of the {len(expected):,} SKUs that should be stocked have "
              f"a row in the inventory snapshot — {len(missing):,} do not.{context}"),
        why=("For an item the policy says to hold, an absent row is read as a position "
             "of zero rather than as unknown. That is right when the shelf really is "
             "empty and wrong when the extract simply did not cover the item, and "
             f"nothing in the output separates the two. Stocking intent read from "
             f"{basis}."),
        impacts=[
            "Purchase recommendations for the affected SKUs: each reads as a shortage "
            "and may ask you to buy stock the warehouse already holds",
            "Inventory projection and days of cover: computed from zero on hand",
            "Excess and value-at-risk totals: understated, since these SKUs carry no "
            "value into them",
            "Not affected: every SKU that did join, and the demand forecast for all of "
            "them — the forecast never reads the stock position.",
        ],
        fix=("Check the inventory extract covers the same plants and item range as the "
             "sales history. Where it is a genuine zero-stock position that the export "
             "omits, the report is right and this is noise — say so and move on. "
             f"First few: {', '.join(missing[:5])}"),
        evidence={"coverage": round(share, 4), "expected_to_hold": len(expected),
                  "unpositioned": len(missing), "basis": basis,
                  "missing_by_design": len(missing_by_design),
                  "examples": missing[:20]},
    ))
    return report


def _expected_to_hold(forecast_skus: set, attributes, classified) -> tuple:
    """
    Split the forecast SKUs into those that should carry stock and those that should not.

    Authority order matters. `stocking_policy` is the ERP's — a decision a person made
    about whether to hold the item — and outranks `stocking_class`, which is the
    pipeline's own reading of the demand. Where neither is available every SKU is
    treated as expected to hold, and the finding says so: an unqualified count is worth
    less, but silently narrowing the check to nothing would be worse.
    """
    frames = [f for f in (attributes, classified)
              if f is not None and len(f) and "sku" in f.columns]

    policy = {}
    for frame in frames:
        if "stocking_policy" in frame.columns:
            part = frame[["sku", "stocking_policy"]].dropna()
            policy.update({str(k): str(v).strip().upper()
                           for k, v in zip(part["sku"], part["stocking_policy"])})
    if policy:
        held = {s for s in forecast_skus if policy.get(s) == "MTO"}
        # Returned even when nothing is left. A business whose every item is bought to
        # order is a real one, and there the right answer is that no SKU was expected
        # to carry stock — falling through to the inferred class would put them all
        # back and report a missing position for each.
        return forecast_skus - held, held, "the ERP's stocking policy (MTS/MTO)"

    klass = {}
    for frame in frames:
        if "stocking_class" in frame.columns:
            part = frame[["sku", "stocking_class"]].dropna()
            klass.update({str(k): str(v).strip().lower()
                          for k, v in zip(part["sku"], part["stocking_class"])})
    if klass:
        held = {s for s in forecast_skus
                if klass.get(s, "").startswith("non-stocking")}
        return forecast_skus - held, held, "the pipeline's stocking class"

    return set(forecast_skus), set(), "nothing — no stocking policy or class was available"
