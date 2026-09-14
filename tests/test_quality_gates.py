"""
The gates stop a run that would produce a complete, plausible, wrong report.

Each test here corresponds to a failure that reached a planner's desk as a finished
document: a join that matched nothing and reported zeroes, a master half-filled with
product families and half-filled with a guess, a business unit spelled two ways that
split every rollup it appeared in.

The other half of the specification matters as much: a gate that stops a run on
ordinary dirt is a gate that gets deleted. `test_dirt_does_not_stop_a_run` is the
guard on that, and it is why `semantic_failure_ceiling` is set where it is.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.orchestrator import InventoryPlanner
from inventory_planning.quality import (BLOCK, SEVERE, WARN, DataQualityError,
                                        Finding, GateReport, GateThresholds, assess)
from inventory_planning.quality import checks as quality_checks
from inventory_planning.quality.dimensions import canonical, normalise

SAMPLE = Path(__file__).parents[1] / "sample_data"
CONFIG = Path(__file__).parents[1] / "config"


# ── Dimension folding ────────────────────────────────────────────────────────


class TestOneThingSpelledSeveralWays:

    def test_case_and_space_are_not_meaning(self):
        assert canonical("INDIA") == canonical("India") == canonical("  india ")

    def test_a_trailing_noise_word_is_not_meaning(self):
        """One business unit, spelled with and without a `Segment` suffix."""
        assert canonical("Hydraulics") == canonical("Hydraulics Segment")
        assert canonical("Fittings") == canonical("Fittings Segment")

    def test_a_noise_word_alone_is_still_a_value(self):
        """Stripping the suffix off `Segment` would leave nothing to group by."""
        assert canonical("Segment") == "segment"

    def test_the_common_spelling_survives(self):
        """
        Not the first seen and not the alphabetically smallest — either would put the
        odd spelling on every report.
        """
        s = pd.Series(["INDIA"] + ["India"] * 20)
        assert set(normalise(s, "country").values) == {"India"}

    def test_a_tie_keeps_the_shorter_spelling(self):
        s = pd.Series(["Hydraulics Segment", "Hydraulics"])
        assert set(normalise(s, "bu").values) == {"Hydraulics"}

    def test_every_merge_is_reported(self):
        """The fold is a judgement about the business — it may not happen silently."""
        s = pd.Series(["Hydraulics"] * 5 + ["Hydraulics Segment"] * 2 + ["Sensors"])
        result = normalise(s, "business_unit")
        assert len(result.collisions) == 1
        collision = result.collisions[0]
        assert collision.kept == "Hydraulics"
        assert collision.merged == ["Hydraulics Segment"]

    def test_distinct_values_are_left_alone(self):
        s = pd.Series(["Hydraulics", "Sensors", "Pumps"])
        assert normalise(s, "bu").collisions == []


# ── Thresholds ───────────────────────────────────────────────────────────────


class TestThresholds:

    def test_the_shipped_config_parses(self):
        assert GateThresholds.load(CONFIG)["sku_agreement_floor"] > 0

    def test_an_unknown_threshold_is_refused(self, tmp_path):
        """
        A typo that silently introduces a threshold nothing reads is the same class of
        bug as everything else here: it looks configured and it is not.
        """
        (tmp_path / "quality_gates.json").write_text(
            json.dumps({"sku_agreemnt_floor": 0.5}), encoding="utf-8")
        with pytest.raises(ValueError, match="unknown threshold"):
            GateThresholds.load(tmp_path)

    def test_documentation_keys_are_not_thresholds(self, tmp_path):
        (tmp_path / "quality_gates.json").write_text(
            json.dumps({"_sku_agreement_floor": "prose", "sku_agreement_floor": 0.4}),
            encoding="utf-8")
        assert GateThresholds.load(tmp_path)["sku_agreement_floor"] == 0.4


# ── Stage checks, in isolation ───────────────────────────────────────────────


class _Doc:
    """The parts of a LoadedDocument the checks read."""

    def __init__(self, frame, source_name="f.xlsx", results=(), key_candidates=None):
        self.frame = frame
        self.source_name = source_name
        self.test_report = type("R", (), {"results": list(results)})()
        self.key_candidates = key_candidates or {}
        self.route = type("Route", (), {"adapter": type("A", (), {"column_map": {"sku": "Item"}})()})()


def _result(documents):
    return type("IntakeResult", (), {"documents": documents})()


class TestTheJoinThatMatchesNothing:

    def _documents(self, overlap: int):
        sales = [f"P-{i:03d}" for i in range(100)]
        stock = [f"P-{i:03d}" for i in range(overlap)] + \
                [f"X-{i:03d}" for i in range(100 - overlap)]
        return {
            "sales_history": _Doc(pd.DataFrame({"sku": sales})),
            "inventory": _Doc(pd.DataFrame({"sku": stock}), "stock.xlsx"),
        }

    def test_a_key_agreeing_with_nothing_blocks(self):
        report = quality_checks.gate_intake(
            _result(self._documents(overlap=3)), None, GateThresholds.load(None))
        blocking = [f for f in report.blocking if f.check == "sku_agreement"]
        assert blocking, "a 3% join must not be allowed to produce a report"
        assert not report.passed

    def test_a_healthy_join_passes(self):
        report = quality_checks.gate_intake(
            _result(self._documents(overlap=95)), None, GateThresholds.load(None))
        assert not [f for f in report.findings if f.check == "sku_agreement"]

    def test_the_finding_says_what_to_do(self):
        """A finding that names a symptom and no action is not worth stopping for."""
        docs = self._documents(overlap=3)
        # The column that *would* have joined: it holds the numbers the stock report
        # actually uses. That is the half of the finding a planner can act on.
        docs["sales_history"].key_candidates = {
            "Alternate material": {f"P-{i:03d}" for i in range(3)}
                                  | {f"X-{i:03d}" for i in range(97)}}
        report = quality_checks.gate_intake(_result(docs), None, GateThresholds.load(None))
        finding = next(f for f in report.blocking if f.check == "sku_agreement")
        assert "Alternate material" in finding.fix
        assert finding.why and finding.what


class TestAWiderGrainIsNotABrokenJoin:
    """
    The one document whose item list is *meant* to be bigger than everyone else's.

    A whole-warehouse stock snapshot holds every material in the DC, most of which
    have no demand, no PO and no SO. Measured outward — the share of its own items
    appearing elsewhere — it looks exactly like a stock file numbered in a different
    system, and the check condemned it on that alone: 9,132 materials against 1,869
    with demand read as 13% agreement and stopped the run. The planner's answer was
    `allow_degraded=True`, which then waved through every other blocking finding for
    the rest of the run.

    A genuinely broken key fails in *both* directions. So which direction is low is
    the whole distinction, and the fix is in the check rather than in a waiver — a
    waiver on `sku_agreement`/`inventory` would have silenced the broken case too.
    """

    def _documents(self, snapshot_size: int, covered: int, demand: int = 1869):
        sold = [f"{1000000 + i}" for i in range(demand)]
        stock = ([f"{1000000 + i}" for i in range(covered)]
                 + [f"{9000000 + i}" for i in range(snapshot_size - covered)])
        return {
            "sales_history": _Doc(pd.DataFrame({"sku": sold}), "otd.xlsx"),
            "inventory": _Doc(pd.DataFrame({"sku": stock}), "stock.xlsx"),
        }

    def test_a_superset_warns_rather_than_blocks(self):
        report = quality_checks.gate_intake(
            _result(self._documents(snapshot_size=9132, covered=1187)),
            None, GateThresholds.load(None))
        finding = next(f for f in report.findings if f.check == "sku_agreement")
        assert finding.severity == WARN
        assert finding not in report.blocking
        assert report.passed, "a wider grain must not stop the run"

    def test_it_reports_both_directions(self):
        report = quality_checks.gate_intake(
            _result(self._documents(snapshot_size=9132, covered=1187)),
            None, GateThresholds.load(None))
        finding = next(f for f in report.findings if f.check == "sku_agreement")
        assert finding.evidence["superset"] is True
        assert finding.evidence["agreement"] < 0.2      # outward, low by construction
        assert finding.evidence["coverage"] > 0.5       # inward, which is what matters
        assert "of the item numbers the rest of the run uses" in finding.what

    def test_a_snapshot_covering_almost_nothing_still_blocks(self):
        """Wide and empty is the broken case, not the superset one."""
        report = quality_checks.gate_intake(
            _result(self._documents(snapshot_size=9132, covered=100)),
            None, GateThresholds.load(None))
        finding = next(f for f in report.blocking if f.check == "sku_agreement")
        assert finding.severity == BLOCK
        assert not report.passed

    def test_two_numbering_systems_of_the_same_size_still_block(self):
        report = quality_checks.gate_intake(
            _result(self._documents(snapshot_size=1869, covered=40)),
            None, GateThresholds.load(None))
        assert [f for f in report.blocking if f.check == "sku_agreement"]


class TestDirtDoesNotStopARun:

    def _po(self, failure_rate: float):
        result = type("T", (), {
            "passed": False, "name": "lead_time_days >= 0", "severity": "error",
            "failure_rate": failure_rate, "failed_rows": int(failure_rate * 1000),
            "total_rows": 1000})()
        return {"po_history": _Doc(pd.DataFrame({"sku": [f"P-{i}" for i in range(50)]}),
                                   results=[result])}

    def test_a_few_bad_rows_are_reported_and_survived(self):
        """
        8% of lead times negative is a data-quality finding the report is built to
        carry. Halting on it would make the pipeline unusable on every real export.
        """
        report = quality_checks.gate_intake(
            _result(self._po(0.08)), None, GateThresholds.load(None))
        assert report.passed

    def test_a_column_that_is_wrong_everywhere_blocks(self):
        """100% failure is not dirt — it is the wrong column, or two columns swapped."""
        report = quality_checks.gate_intake(
            _result(self._po(1.0)), None, GateThresholds.load(None))
        assert [f for f in report.blocking if f.check == "semantic_failure"]


class TestPartialFamilyCoverage:

    def _documents(self, covered: int):
        skus = [f"P-{i:03d}" for i in range(100)]
        master = pd.DataFrame({
            "sku": skus,
            "product_family": ["valve"] * covered + [None] * (100 - covered),
        })
        return {
            "sales_history": _Doc(pd.DataFrame({"sku": skus})),
            "item_master": _Doc(master, "master.xlsx"),
        }

    def test_a_half_filled_family_column_is_counted_not_blocked(self):
        """
        Deliberately not a stop. An item selling without a classified master row is the
        ordinary state of anything new or transferred in, and a planner who cannot get
        a report until master data is perfect gets no reports. What is owed is a count
        the reader can reconcile against the sheet.
        """
        report = quality_checks.gate_intake(
            _result(self._documents(covered=50)), None, GateThresholds.load(None))
        assert report.passed, "a classification gap must not stop the run"
        finding = next(f for f in report.warnings
                       if f.check == "product_family_missing")
        assert "50" in finding.what
        assert "product_family_source" in finding.why

    def test_a_sku_with_no_master_row_is_a_separate_finding(self):
        """
        Different gap, different owner: one needs an item created, the other needs an
        item classified.
        """
        skus = [f"P-{i:03d}" for i in range(100)]
        docs = {
            "sales_history": _Doc(pd.DataFrame({"sku": skus})),
            "item_master": _Doc(pd.DataFrame({
                "sku": skus[:60], "product_family": ["valve"] * 60}), "master.xlsx"),
        }
        report = quality_checks.gate_intake(_result(docs), None,
                                            GateThresholds.load(None))
        assert report.passed
        finding = next(f for f in report.warnings
                       if f.check == "sku_missing_from_master")
        assert "40" in finding.what
        assert "P-060" in finding.fix

    def test_full_coverage_passes(self):
        report = quality_checks.gate_intake(
            _result(self._documents(covered=100)), None, GateThresholds.load(None))
        assert report.passed


class TestForecastAndPositionCoverage:

    def test_skus_that_sold_but_got_no_forecast_block(self):
        ts = pd.DataFrame({f"P-{i}": [10.0] * 12 for i in range(10)})
        detail = pd.DataFrame({"sku": ["P-0", "P-1"], "forecast_qty": [1.0, 1.0]})
        report = quality_checks.gate_forecast(detail, ts, GateThresholds.load(None))
        finding = next(f for f in report.blocking if f.check == "forecast_coverage")
        assert "planned against zero demand" in finding.why

    def test_a_sku_with_no_demand_is_not_expected_to_have_one(self):
        ts = pd.DataFrame({f"P-{i}": [0.0] * 12 for i in range(8)}
                          | {f"Q-{i}": [10.0] * 12 for i in range(2)})
        detail = pd.DataFrame({"sku": ["Q-0", "Q-1"], "forecast_qty": [1.0, 1.0]})
        assert quality_checks.gate_forecast(detail, ts, GateThresholds.load(None)).passed

    def _summary(self, n=10):
        return pd.DataFrame({"sku": [f"P-{i}" for i in range(n)]})

    def test_stocked_items_absent_from_the_stock_report_are_severe(self):
        """
        An item the policy says to hold, with no row, is read as a position of zero —
        right when the shelf is empty, wrong when the extract did not cover it, and the
        output does not separate the two.
        """
        classified = pd.DataFrame({"sku": [f"P-{i}" for i in range(10)],
                                   "stocking_class": ["stocking-high"] * 10})
        report = quality_checks.gate_plan(
            self._summary(), pd.DataFrame({"sku": ["P-0", "P-1"]}),
            GateThresholds.load(None), classified=classified)
        finding = next(f for f in report.severe if f.check == "position_coverage")
        assert "20%" in finding.what

    def test_it_does_not_block(self):
        """
        It did, and it was the wrong call. A missing position is visible in the output —
        the SKU comes out at zero and lands in SHORTAGE-RISK — and blocking is for
        damage a reader cannot see. Blocking here also got the whole mechanism switched
        off with `allow_degraded`, which disables every gate at every stage.
        """
        classified = pd.DataFrame({"sku": [f"P-{i}" for i in range(10)],
                                   "stocking_class": ["stocking-high"] * 10})
        report = quality_checks.gate_plan(
            self._summary(), pd.DataFrame({"sku": ["P-0"]}),
            GateThresholds.load(None), classified=classified)
        assert report.passed, "a visible gap must not stop the run"

    def test_non_stocking_items_are_not_expected_to_have_a_position(self):
        """
        The case that made this fire on healthy data. A non-stocking item sells and is
        never held; an extract that omits it is correct, not incomplete.
        """
        classified = pd.DataFrame({
            "sku": [f"P-{i}" for i in range(10)],
            "stocking_class": ["stocking-high"] * 2 + ["non-stocking"] * 8,
        })
        report = quality_checks.gate_plan(
            self._summary(), pd.DataFrame({"sku": ["P-0", "P-1"]}),
            GateThresholds.load(None), classified=classified)
        assert report.findings == [], "only the two stocked SKUs count, and both joined"

    def test_the_erp_policy_outranks_the_inferred_class(self):
        """
        `stocking_policy` is a decision someone made; `stocking_class` is the pipeline's
        reading of the demand. Where both speak, the decision wins.
        """
        attributes = pd.DataFrame({
            "sku": [f"P-{i}" for i in range(10)],
            "stocking_policy": ["MTO"] * 8 + ["MTS"] * 2,
            # The pipeline disagrees and would call all ten stocked.
            "stocking_class": ["stocking-high"] * 10,
        })
        report = quality_checks.gate_plan(
            self._summary(), pd.DataFrame({"sku": ["P-8", "P-9"]}),
            GateThresholds.load(None), attributes=attributes)
        assert report.findings == []

    def test_the_excluded_items_are_still_counted_in_the_finding(self):
        """Excluded from the ratio, not from the reader's view."""
        classified = pd.DataFrame({
            "sku": [f"P-{i}" for i in range(10)],
            "stocking_class": ["stocking-high"] * 5 + ["non-stocking"] * 5,
        })
        report = quality_checks.gate_plan(
            self._summary(), pd.DataFrame({"sku": ["P-0"]}),
            GateThresholds.load(None), classified=classified)
        finding = next(f for f in report.severe if f.check == "position_coverage")
        assert finding.evidence["missing_by_design"] == 5
        assert "not expected to" in finding.what

    def test_with_no_policy_at_all_every_sku_counts_and_it_says_so(self):
        report = quality_checks.gate_plan(
            self._summary(), pd.DataFrame({"sku": ["P-0"]}), GateThresholds.load(None))
        finding = next(f for f in report.severe if f.check == "position_coverage")
        assert "no stocking policy or class was available" in finding.why


# ── End to end ───────────────────────────────────────────────────────────────


class TestTheGatesInARun:

    def test_the_sample_data_passes_every_gate(self):
        planner = InventoryPlanner(output_dir=Path("/tmp") / "gate_sample",
                                   interactive=False)
        results = planner.run_planning(**planner.load_all(sorted(SAMPLE.glob("*.csv"))))
        assert [g.stage for g in planner.gate_reports] == [
            "intake", "demand", "forecast", "plan"]
        assert all(g.passed for g in planner.gate_reports)
        assert results["gate_reports"] is planner._gate_reports

    def test_a_broken_join_stops_the_run_before_any_report(self, tmp_path):
        rng = np.random.default_rng(3)
        n = 400
        pd.DataFrame({
            "Part Number": [f"P-{i % 40:03d}" for i in range(n)],
            "Billto Customer Name": "ACME",
            "Invdate Date": pd.to_datetime("2024-01-01") + pd.to_timedelta(
                rng.integers(0, 540, n), "D"),
            "Shipped Quantity": rng.integers(1, 40, n),
            "Sales Revenue (USD)": rng.integers(10, 900, n),
        }).to_excel(tmp_path / "sales.xlsx", index=False)
        # A stock report for a different item range entirely — the mapping succeeded
        # and the two documents describe different businesses.
        pd.DataFrame({
            "Material": [f"Z-{i:03d}" for i in range(40)],
            "Closing Stock": rng.integers(0, 100, 40),
        }).to_excel(tmp_path / "inventory.xlsx", index=False)
        pd.DataFrame({
            "Material": [f"P-{i:03d}" for i in range(40)],
            "Product Group": "valve",
            "Min Order Qty": 25,
        }).to_excel(tmp_path / "item master.xlsx", index=False)

        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False)
        with pytest.raises(DataQualityError) as exc:
            planner.load_all(sorted(tmp_path.glob("*.xlsx")))
        message = str(exc.value)
        assert "sku_agreement" in message
        assert "confident zeroes" in message, "the message has to say why it matters"
        assert "allow_degraded" in message, "and how to proceed if the gate is wrong"

    def test_an_override_runs_and_is_recorded(self, tmp_path):
        """
        A threshold is a judgement and judgements are sometimes wrong on data nobody
        anticipated. Overriding is allowed; overriding silently is not.
        """
        rng = np.random.default_rng(3)
        n = 400
        pd.DataFrame({
            "Part Number": [f"P-{i % 40:03d}" for i in range(n)],
            "Billto Customer Name": "ACME",
            "Invdate Date": pd.to_datetime("2024-01-01") + pd.to_timedelta(
                rng.integers(0, 540, n), "D"),
            "Shipped Quantity": rng.integers(1, 40, n),
            "Sales Revenue (USD)": rng.integers(10, 900, n),
        }).to_excel(tmp_path / "sales.xlsx", index=False)
        pd.DataFrame({
            "Material": [f"Z-{i:03d}" for i in range(40)],
            "Closing Stock": rng.integers(0, 100, 40),
        }).to_excel(tmp_path / "inventory.xlsx", index=False)
        pd.DataFrame({
            "Material": [f"P-{i:03d}" for i in range(40)],
            "Product Group": "valve",
            "Min Order Qty": 25,
        }).to_excel(tmp_path / "item master.xlsx", index=False)

        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False,
                                   allow_degraded=True)
        planner.load_all(sorted(tmp_path.glob("*.xlsx")))
        intake = next(g for g in planner.gate_reports if g.stage == "intake")
        assert not intake.passed
        assert intake.overridden
        assert "OVERRIDDEN" in intake.summary()


# ── The whole grouping axis missing ──────────────────────────────────────────


class TestNoProductFamilyAtAll:
    """
    Severe, not blocking. Replenishment never uses the family — safety stock, reorder
    points and order quantities come out identical with it and without — so a plan a
    buyer can act on is not withheld because the rollup above it would be unreliable.
    What is owed is that nobody reads the rollup as if it meant something.
    """

    def _documents(self, family=None):
        skus = [f"P-{i:03d}" for i in range(60)]
        master = pd.DataFrame({"sku": skus, "min_order_qty": 25})
        if family is not None:
            master["product_family"] = family
        return {
            "sales_history": _Doc(pd.DataFrame({"sku": skus})),
            "item_master": _Doc(master, "master.xlsx"),
        }

    def test_the_run_is_not_stopped(self):
        report = quality_checks.gate_intake(
            _result(self._documents()), None, GateThresholds.load(None))
        assert report.passed

    def test_it_is_severe_not_a_passing_mention(self):
        report = quality_checks.gate_intake(
            _result(self._documents()), None, GateThresholds.load(None))
        finding = next(f for f in report.severe if f.check == "no_product_dimension")
        assert finding.severity == SEVERE

    def test_it_names_the_outputs_it_reaches(self):
        """A severity with no impact list is an adjective, not a warning."""
        report = quality_checks.gate_intake(
            _result(self._documents()), None, GateThresholds.load(None))
        finding = next(f for f in report.severe if f.check == "no_product_dimension")
        joined = " ".join(finding.impacts).lower()
        assert "by product line" in joined
        assert "days-on-hand" in joined
        assert "s&op worksheet" in joined

    def test_it_says_what_is_still_trustworthy(self):
        """
        The half a planner acts on. A warning that does not bound itself gets read as
        "the whole report is suspect", and the purchase recommendations — which do not
        use the family at all — are exactly as good as they were.
        """
        report = quality_checks.gate_intake(
            _result(self._documents()), None, GateThresholds.load(None))
        finding = next(f for f in report.severe if f.check == "no_product_dimension")
        not_affected = next(i for i in finding.impacts if i.startswith("Not affected"))
        assert "safety stock" in not_affected
        assert "purchase quantities" in not_affected

    def test_the_per_sku_count_is_not_also_reported(self):
        """One gap said twice at two grains is a reader who reads neither."""
        report = quality_checks.gate_intake(
            _result(self._documents()), None, GateThresholds.load(None))
        assert not [f for f in report.findings if f.check == "product_family_missing"]

    def test_a_populated_family_raises_nothing(self):
        report = quality_checks.gate_intake(
            _result(self._documents(family="valve")), None, GateThresholds.load(None))
        assert not [f for f in report.findings if f.check == "no_product_dimension"]


# ── The run health document ──────────────────────────────────────────────────


def _finding(severity, check="c", impacts=()):
    return Finding(stage="intake", check=check, severity=severity, what="what",
                   why="why", fix="fix", impacts=list(impacts))


class TestRunHealth:

    def test_a_clean_run_says_so_plainly(self):
        health = assess("r1", [GateReport(stage="intake")])
        assert health.clean
        assert "no input was missing" in health.markdown()

    def test_severe_findings_lead_the_document(self):
        gate = GateReport(stage="intake", findings=[
            _finding(WARN, "spelling"), _finding(SEVERE, "no_family", ["a table"])])
        text = assess("r1", [gate]).markdown()
        assert text.index("Do not rely on these") < text.index("Noted")

    def test_a_blocking_finding_appears_only_when_it_was_overridden(self):
        """
        An unoverridden block never produced an output for this document to describe,
        so listing it would be describing a report that does not exist.
        """
        gate = GateReport(stage="intake", findings=[_finding(BLOCK, "join")])
        assert assess("r1", [gate], allow_degraded=False).critical == []
        forced = assess("r1", [gate], allow_degraded=True)
        assert [f.check for f in forced.critical] == ["join"]
        assert "OVERRIDDEN" in forced.markdown()

    def test_impacts_are_collected_across_findings(self):
        gate = GateReport(stage="intake", findings=[
            _finding(SEVERE, "a", ["one", "two"]), _finding(SEVERE, "b", ["two", "three"])])
        assert assess("r1", [gate]).affected_outputs == ["one", "two", "three"]

    def test_missing_inputs_are_reported_beside_the_findings(self):
        """
        A different kind of statement — nothing is wrong, something is absent — but a
        reader needs both for the same purpose, and two places neither of which is
        complete is how one gets missed.
        """
        plan = type("P", (), {"degradations": ["Safety stock will be understated"]})()
        text = assess("r1", [GateReport(stage="intake")], plan=plan).markdown()
        assert "could not measure" in text
        assert "Safety stock will be understated" in text


class TestHealthReachesTheOutputs:

    def test_the_run_writes_a_health_document(self, tmp_path):
        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False)
        results = planner.run_planning(**planner.load_all(sorted(SAMPLE.glob("*.csv"))))
        health = results["run_health"]
        assert (tmp_path / "out" / f"run_health_{planner.run.run_id}.md").exists()
        assert (tmp_path / "out" / f"quality_gates_{planner.run.run_id}.json").exists()
        assert health.to_dict()["run_id"] == planner.run.run_id
