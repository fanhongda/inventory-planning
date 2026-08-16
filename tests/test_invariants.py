"""
Properties that must hold on any run, checked on data built to break them.

The existing skill evals assert that a file was written, that a string appears in the
output, and that the HTML is over 500KB. Every one of the four corruptions found in
August 2026 — five currencies summed as one, 9.7% of demand silently undated, 63% of
purchase quantity discarded by a lead-time trim, a master joining zero rows — produced a
complete run, a written report, and the string `PURCHASE-REQUEST` in the output. Those
evals would have gone green on all four.

That is not thin coverage, it is the wrong category of test. The whole risk in this
domain is a run that completes perfectly and is wrong, so the assertions have to be
about the numbers rather than about the plumbing.

Each test below is a property, not an example: it states something that must be true of
*any* run, and the fixture is constructed to violate it if the guard is removed. The
four named above each have one here, so a regression reintroduces a failing test rather
than a plausible total.

These are deliberately end-to-end. Unit tests for the same guards live in
`test_currency.py`, `test_mixed_formats.py` and `test_cadence.py`; what this file adds
is that the property survives the whole pipeline, which is where the originals were lost.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.fx import FxTable, convert_money
from inventory_planning.ingest_bridge import IngestBridge
from inventory_planning.orchestrator import InventoryPlanner
from inventory_planning.policy.cadence import CadenceAnalyzer

SAMPLE = Path(__file__).parents[1] / "sample_data"


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    """One complete pipeline pass over the synthetic sample data."""
    out = tmp_path_factory.mktemp("invariants")
    planner = InventoryPlanner(output_dir=out, interactive=False)
    inputs = planner.load_all(sorted(SAMPLE.glob("*.csv")))
    results = planner.run_planning(**inputs)
    policy = planner.run_policy_analysis(
        results, inventory_df=inputs["inventory_df"], open_po_df=inputs["open_po_df"]
    )
    return {"inputs": inputs, "results": results, "policy": policy}


@pytest.fixture(scope="module")
def recs(full_run):
    return full_run["results"]["recommendations"]


# ── Money ────────────────────────────────────────────────────────────────────


class TestMoneyIsConserved:
    """
    Currency conversion has one arithmetic property and one honesty property, and the
    honesty one is what actually failed: the raw total was a valid number, its sum was
    a valid number, and the ₹/$ distinction lived in a column that had been dropped.
    """

    @staticmethod
    def _rates(tmp_path):
        import json

        cfg = tmp_path / "config"
        cfg.mkdir(exist_ok=True)
        (cfg / "fx_rates.json").write_text(json.dumps({
            "reporting_currency": "USD",
            "rates": {"USD": 1.0, "GBP": 1.25, "INR": 0.012},
        }), encoding="utf-8")
        return FxTable.load(cfg)

    @staticmethod
    def _lines(currencies):
        return pd.DataFrame({
            "sku": [f"S{i}" for i in range(len(currencies))],
            "po_date": pd.to_datetime(["2025-06-01"] * len(currencies)),
            "currency": currencies,
            "po_amount": [1000.0] * len(currencies),
        })

    def test_the_converted_total_equals_the_sum_of_line_conversions(self, tmp_path):
        table = self._rates(tmp_path)
        df = self._lines(["USD", "GBP", "INR", "GBP"])
        out, _ = convert_money(df, "po_history", table)
        expected = 1000 * (1.0 + 1.25 + 0.012 + 1.25)
        assert out["po_amount"].sum() == pytest.approx(expected)

    def test_no_unrated_line_contributes_to_a_total(self, tmp_path):
        """
        The failure exactly: a missing rate becoming 1.0, so ₹126,550 reports as
        $126,550 and outranks every genuine dollar line in the ABC ranking.
        """
        table = self._rates(tmp_path)
        out, report = convert_money(self._lines(["USD", "JPY"]), "po_history", table)
        assert out["po_amount"].sum() == pytest.approx(1000.0), "only the USD line counts"
        assert report.rows_unrated == 1

    def test_an_unrated_currency_is_never_silently_absent(self, tmp_path):
        """Excluding the money is only half the guard — the exclusion must be named,
        or the total is quietly understated instead of quietly overstated."""
        table = self._rates(tmp_path)
        _, report = convert_money(self._lines(["USD", "JPY", "JPY"]), "po_history", table)
        assert report.has_gap
        assert "JPY" in report.unrated
        assert any("JPY" in line for line in report.summary())

    def test_a_multi_currency_extract_never_reports_the_raw_sum(self, tmp_path):
        """A mixed-scale total is a valid number and a meaningless one."""
        table = self._rates(tmp_path)
        df = self._lines(["USD", "INR", "INR", "INR"])
        out, report = convert_money(df, "po_history", table)
        raw = report.value_before["po_amount"]
        assert out["po_amount"].sum() < raw
        assert out["po_amount"].sum() == pytest.approx(1000 + 3 * 12.0)


# ── Quantity ─────────────────────────────────────────────────────────────────


class TestQuantityIsConserved:
    """
    Every quantity that enters must either be counted or be reported as uncounted.
    Both August losses were of the second kind: rows dropped for a reason that was
    real, in a place where the loss had no voice.
    """

    def test_demand_is_either_placed_in_a_month_or_reported_as_undated(self):
        """The 9.7% loss: rows with no usable date fell out of one side of the
        subtraction while the other side kept everything, so every SKU read as
        over-bought by whatever went missing."""
        months = pd.period_range(end=pd.Period("2026-07", freq="M"), periods=12, freq="M")
        sales = pd.DataFrame({
            "sku": ["A"] * 12,
            "demand_date": [m.to_timestamp() + pd.Timedelta(days=5) for m in months],
            "qty": [100.0] * 12,
        })
        sales.loc[sales.index[:4], "demand_date"] = pd.NaT      # 400 units undated
        po = pd.DataFrame({
            "sku": ["A"] * 12, "po_number": [f"P{i}" for i in range(12)],
            "po_date": [m.to_timestamp() + pd.Timedelta(days=3) for m in months],
            "po_qty": [100.0] * 12,
        })

        result = CadenceAnalyzer().analyze(
            po_history=po, sales_history=sales, as_of=date(2026, 7, 31)
        )
        counted = float(result.frame.iloc[0]["demand_qty"])
        assert counted == pytest.approx(800.0), "only the dated demand can be placed"
        # The 400 that could not be placed has to be audible somewhere.
        assert any("cannot be placed in a month" in note for note in result.notes)
        assert any("40%" in note or "33%" in note for note in result.notes)

    def test_an_unusable_lead_time_does_not_remove_the_order(self, tmp_path):
        """
        The 63% loss. An implausible receipt date makes the *lead time* unusable and
        implicates nothing else on the line — the order was still raised, for that
        quantity, on that date.
        """
        n = 40
        pd.DataFrame({
            "Material": [f"{600000 + i % 5}" for i in range(n)],
            "Vendor": ["V1"] * n,
            "PO Number": [f"45{i:06d}" for i in range(n)],
            "PO Date": pd.date_range("2026-01-01", periods=n, freq="D"),
            # Half the lines receive before they were ordered — a broken receipt date.
            "GR Date": [
                (pd.Timestamp("2026-01-01") + pd.Timedelta(days=i))
                - pd.Timedelta(days=5 if i % 2 else -20)
                for i in range(n)
            ],
            "PO quantity": [100.0] * n,
            "Net Price": [10.0] * n,
        }).to_excel(tmp_path / "po_history.xlsx", index=False)

        df = IngestBridge(verbose=False).load([tmp_path / "po_history.xlsx"])["po_history_df"]
        assert df["po_qty"].sum() == pytest.approx(n * 100.0), "every order survives"
        # ...while the unusable lead times are gone from the lead-time statistic.
        assert df["lead_time_days"].isna().sum() == n // 2


# ── Joins ────────────────────────────────────────────────────────────────────


class TestAJoinThatMatchesNothingIsReported:
    """
    A merge that finds no rows is not an error. The costs, the lead times and the
    policy simply never arrive, and every figure derived from them reads as absent
    rather than as wrong — which is why it ran for months.
    """

    def test_a_master_keyed_on_the_wrong_column_is_named(self, tmp_path):
        from inventory_planning.ingest.intake import Intake

        skus = [f"700{i:04d}" for i in range(60)]
        rows = [{"Part Number": s, "Billto Customer Name": "ACME",
                 "Invdate Date": pd.Timestamp("2025-01-10") + pd.DateOffset(months=k),
                 "Shipped Quantity": 10}
                for s in skus for k in range(14)]
        pd.DataFrame(rows).to_excel(tmp_path / "sales.xlsx", index=False)
        pd.DataFrame({
            "Material": [f"LOCAL-{i:04d}" for i in range(60)],
            "Alternate material ": skus,
            "SS": range(10, 70),
            "Std cost": np.linspace(5, 90, 60).round(2),
        }).to_excel(tmp_path / "master.xlsx", index=False)

        notes = " ".join(Intake(verbose=False)
                         .load_files(sorted(tmp_path.glob("*.xlsx"))).notes)
        assert "keys on something the other documents do not use" in notes
        assert "Alternate material" in notes, "name the column that would have joined"


# ── Recommendations ──────────────────────────────────────────────────────────


class TestRecommendationsAreCoherent:
    """
    Whether the advice is *good* needs a planner. Whether it contradicts itself does
    not, and a recommendation that contradicts its own inputs is wrong regardless of
    anyone's judgement.
    """

    def test_nothing_is_ordered_in_negative_quantity(self, recs):
        assert (recs["suggested_po_qty"] >= 0).all()
        assert (recs["net_requirement"] >= 0).all()

    def test_a_purchase_request_has_a_requirement_behind_it(self, recs):
        buys = recs[recs["recommended_action"] == "PURCHASE-REQUEST"]
        assert (buys["net_requirement"] > 0).all()
        assert (buys["suggested_po_qty"] > 0).all(), "a buy signal with no quantity is not a buy"

    def test_only_a_buy_action_carries_a_quantity(self, recs):
        """A push-out or a hold with an order quantity attached is a contradiction."""
        with_qty = recs[recs["suggested_po_qty"] > 0]
        assert set(with_qty["recommended_action"]) <= {"PURCHASE-REQUEST", "ORDER-FOR-BACKLOG"}

    def test_stock_is_never_pushed_out_of_a_shortage(self, recs):
        """Delaying inbound supply on an item that is about to run out is the single
        most expensive thing this report could get wrong."""
        pushed = recs[recs["recommended_action"] == "PUSH-OUT-OPEN-PO"]
        assert not (pushed["inventory_status"] == "SHORTAGE-RISK").any()

    def test_hold_excess_agrees_with_the_projection(self, recs):
        held = recs[recs["recommended_action"] == "HOLD-EXCESS"]
        assert (held["inventory_status"] == "EXCESS").all()

    def test_an_order_on_demand_item_gets_no_policy_stock_buy(self, recs):
        """Non-stocking means no cycle or safety stock to plan, so the only buy that
        can be justified is the realizable order book."""
        non_stocking = recs[recs["stocking_class"] == "non-stocking"]
        assert set(non_stocking["recommended_action"]) <= {"ORDER-FOR-BACKLOG", "NO-ACTION"}
        assert (non_stocking["safety_stock"].fillna(0) == 0).all()

    def test_every_sku_gets_exactly_one_action(self, recs):
        assert recs["recommended_action"].notna().all()
        assert not recs["sku"].duplicated().any()


class TestTheReportAgreesWithItself:
    """
    The report states the same quantity in several places. Two of them disagreeing is
    not a rounding question — it means two code paths computed it differently, and the
    reader has no way to know which one to believe.
    """

    def test_should_be_is_the_sum_of_its_parts(self, full_run):
        frame = full_run["policy"]["should_be"].frame
        parts = frame[["cycle_value", "safety_value", "pipeline_value"]].sum().sum()
        assert full_run["policy"]["should_be"].total_should_be_value == pytest.approx(parts, rel=1e-6)

    def test_the_kpi_tile_and_the_section_report_one_number(self, full_run, tmp_path):
        """The headline and the detail are rendered by different methods over the same
        object; a divergence there is invisible until someone adds the two up."""
        from inventory_planning.reporting.kpi_report import KPIReport, _money

        should_be = full_run["policy"]["should_be"]
        html = KPIReport().render(should_be=should_be, output_path=tmp_path / "r.html")
        total = _money(should_be.total_should_be_value)
        assert html.count(total) >= 2, f"{total} should appear as both tile and section"
