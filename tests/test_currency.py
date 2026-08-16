"""
Money columns arriving in five currencies and being added together.

Found by running the real PL30 extract. The open-PO export carries a `Currency` column
— AUD, EUR, GBP, INR, USD — and a `PO Total Value` in whichever of those the supplier
invoices in. Nothing read that column, so committed spend summed to 97,174,359 and was
printed with a dollar sign. Restated, it is 11,153,060. The purchase history is worse:
7,312,164,360 raw against 402,281,808 USD, because 7.0 billion of INR was being added
to 240 million of USD.

None of it raised anything. It could not — every value was a valid number, the sum was
a valid number, and the ₹/$ distinction lived in a column that had been dropped at
ingest. What the total loses is the ranking: an INR line outranks every dollar line on
value alone, so ABC classification, excess value, value at risk and the whole
efficient frontier get sorted by the accident of which currency a supplier bills in.

The tests below hold three properties:

  the conversion happens at all, on the canonical field names
  an unrated currency blanks its money rather than defaulting to 1.0
  a rate the export itself booked outranks the configured table
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.fx import FxTable, convert_money


@pytest.fixture
def config_dir(tmp_path):
    d = tmp_path / "config"
    d.mkdir()
    (d / "fx_rates.json").write_text(json.dumps({
        "reporting_currency": "USD",
        "rates": {
            "USD": 1.0,
            "GBP": 1.25,
            "INR": [
                {"effective_from": "2024-01-01", "rate": 0.012},
                {"effective_from": "2025-01-01", "rate": 0.011},
            ],
        },
    }), encoding="utf-8")
    return d


@pytest.fixture
def table(config_dir):
    return FxTable.load(config_dir)


def _po_history(rows):
    return pd.DataFrame(rows)


class TestTheRateTable:

    def test_reporting_currency_converts_to_itself(self, table):
        assert table.rate_for("USD") == 1.0

    def test_a_flat_rate_applies_at_any_date(self, table):
        assert table.rate_for("GBP", "2019-06-01") == 1.25
        assert table.rate_for("GBP", "2026-06-01") == 1.25

    def test_a_dated_rate_follows_the_date(self, table):
        assert table.rate_for("INR", "2024-06-01") == 0.012
        assert table.rate_for("INR", "2025-06-01") == 0.011

    def test_a_row_older_than_every_quote_takes_the_earliest(self, table):
        """An approximate rate on a 2018 line beats blanking eight years of history."""
        assert table.rate_for("INR", "2018-03-01") == 0.012

    def test_an_undated_row_takes_the_newest_quote(self, table):
        assert table.rate_for("INR", None) == 0.011

    def test_an_unlisted_currency_has_no_rate(self, table):
        assert table.rate_for("JPY") is None

    def test_a_missing_rate_file_is_not_an_error(self, tmp_path):
        """It yields an empty table, so the gap gets reported rather than assumed away."""
        empty = FxTable.load(tmp_path / "nowhere")
        assert empty.reporting_currency == "USD"
        assert empty.rate_for("GBP") is None
        assert empty.rate_for("USD") == 1.0


class TestConversion:

    def test_money_is_restated_and_the_rate_is_recorded(self, table):
        df = _po_history([
            {"sku": "A", "po_date": "2025-03-01", "currency": "GBP",
             "po_amount": 1000.0, "unit_cost": 100.0},
            {"sku": "B", "po_date": "2025-03-01", "currency": "USD",
             "po_amount": 1000.0, "unit_cost": 100.0},
        ])
        out, report = convert_money(df, "po_history", table)

        assert out.loc[0, "po_amount"] == pytest.approx(1250.0)
        assert out.loc[0, "unit_cost"] == pytest.approx(125.0)
        assert out.loc[1, "po_amount"] == pytest.approx(1000.0)
        # The rate travels with the row, so any figure can be traced back or undone.
        assert out.loc[0, "fx_rate_applied"] == pytest.approx(1.25)
        assert out.loc[0, "currency_original"] == "GBP"
        assert report.rows_converted == 1
        assert report.rows_domestic == 1

    def test_the_rate_follows_the_line_date_not_the_run_date(self, table):
        """Purchase history spans years; one rate would read a currency move as a trend."""
        df = _po_history([
            {"sku": "A", "po_date": "2024-06-01", "currency": "INR", "po_amount": 100000.0},
            {"sku": "A", "po_date": "2025-06-01", "currency": "INR", "po_amount": 100000.0},
        ])
        out, _ = convert_money(df, "po_history", table)
        assert out.loc[0, "po_amount"] == pytest.approx(1200.0)
        assert out.loc[1, "po_amount"] == pytest.approx(1100.0)

    def test_an_unrated_currency_is_blanked_not_defaulted(self, table):
        """
        The whole failure this guards: a missing rate silently becoming 1.0, so
        ₹126,550 reports as $126,550 and outranks every genuine dollar line.
        """
        df = _po_history([
            {"sku": "A", "po_date": "2025-03-01", "currency": "JPY", "po_amount": 500000.0},
            {"sku": "B", "po_date": "2025-03-01", "currency": "USD", "po_amount": 100.0},
        ])
        out, report = convert_money(df, "po_history", table)

        assert pd.isna(out.loc[0, "po_amount"])
        assert out.loc[1, "po_amount"] == pytest.approx(100.0)
        assert report.rows_unrated == 1
        assert report.unrated == {"JPY": 1}
        assert report.has_gap

    def test_blanking_is_nan_not_zero(self, table):
        """Zero claims the line was free; NaN says its value is not known."""
        df = _po_history([
            {"sku": "A", "po_date": "2025-03-01", "currency": "JPY", "po_amount": 500000.0},
        ])
        out, _ = convert_money(df, "po_history", table)
        assert out["po_amount"].sum(skipna=True) == 0.0    # nothing counted
        assert out["po_amount"].isna().all()               # and it is visibly absent

    def test_the_exports_own_rate_outranks_the_table(self, table):
        """It is what the transaction settled at; the table is a planning average."""
        df = _po_history([
            {"sku": "A", "po_date": "2025-03-01", "currency": "GBP",
             "po_amount": 1000.0, "fx_rate": 1.31},
        ])
        out, report = convert_money(df, "po_history", table)
        assert out.loc[0, "po_amount"] == pytest.approx(1310.0)
        assert report.rows_from_row_rate == 1

    def test_a_booked_rate_of_one_on_a_foreign_line_is_rejected(self, table):
        """
        The real backlog extract carries `Exchange Rate` = 1 on all 3,169 INR lines.
        Taken at face value it left ₹1.19bn in rupees while the log called them
        restated — the failure the whole module exists to prevent, reintroduced by the
        rule that trusts the source.
        """
        df = _po_history([
            {"sku": "A", "po_date": "2025-03-01", "currency": "INR",
             "po_amount": 100000.0, "fx_rate": 1.0},
        ])
        out, report = convert_money(df, "po_history", table)
        assert out.loc[0, "po_amount"] == pytest.approx(1100.0)   # the table's rate
        assert report.rows_rate_rejected == 1
        assert report.rows_from_row_rate == 0

    def test_an_inverted_booked_rate_is_rejected(self, table):
        """94.7 rupees per dollar where the conversion needs 0.01056 the other way."""
        df = _po_history([
            {"sku": "A", "po_date": "2025-03-01", "currency": "INR",
             "po_amount": 100000.0, "fx_rate": 90.9},
        ])
        out, report = convert_money(df, "po_history", table)
        assert out.loc[0, "po_amount"] == pytest.approx(1100.0)
        assert report.rows_rate_rejected == 1

    def test_a_booked_rate_near_the_table_is_still_preferred(self, table):
        """The guard rejects nonsense, not ordinary daily variation."""
        df = _po_history([
            {"sku": "A", "po_date": "2025-03-01", "currency": "INR",
             "po_amount": 100000.0, "fx_rate": 0.0114},
        ])
        out, report = convert_money(df, "po_history", table)
        assert out.loc[0, "po_amount"] == pytest.approx(1140.0)
        assert report.rows_from_row_rate == 1
        assert report.rows_rate_rejected == 0

    def test_a_booked_rate_rescues_an_unrated_currency(self, table):
        df = _po_history([
            {"sku": "A", "po_date": "2025-03-01", "currency": "JPY",
             "po_amount": 500000.0, "fx_rate": 0.0067},
        ])
        out, report = convert_money(df, "po_history", table)
        assert out.loc[0, "po_amount"] == pytest.approx(3350.0)
        assert report.rows_unrated == 0

    def test_a_document_with_no_currency_column_is_assumed_domestic(self, table):
        """The ordinary single-entity export, recorded as an assumption not a fact."""
        df = _po_history([{"sku": "A", "po_date": "2025-03-01", "po_amount": 1000.0}])
        out, report = convert_money(df, "po_history", table)
        assert out.loc[0, "po_amount"] == pytest.approx(1000.0)
        assert report.assumed_reporting_currency
        assert not report.has_gap

    def test_a_document_with_no_money_columns_is_left_alone(self, table):
        df = pd.DataFrame({"sku": ["A"], "qty": [5]})
        out, report = convert_money(df, "demand_timeseries", table)
        assert report.skipped
        assert "fx_rate_applied" not in out.columns

    def test_case_and_whitespace_in_the_code_do_not_break_the_lookup(self, table):
        df = _po_history([
            {"sku": "A", "po_date": "2025-03-01", "currency": " gbp ", "po_amount": 1000.0},
        ])
        out, _ = convert_money(df, "po_history", table)
        assert out.loc[0, "po_amount"] == pytest.approx(1250.0)


class TestThroughTheBridge:
    """The conversion has to happen on the canonical frames, not just in isolation."""

    def test_a_multi_currency_po_history_totals_in_one_currency(self, tmp_path, config_dir):
        from inventory_planning.ingest_bridge import IngestBridge

        (config_dir / "node_config.json").write_text(
            json.dumps({"location_id": "DC-01", "currency": "USD"}), encoding="utf-8"
        )
        rng = np.random.default_rng(7)
        n = 120
        pd.DataFrame({
            "Material": [f"{100000 + i % 20}" for i in range(n)],
            "Vendor": ["V1"] * n,
            "PO Number": [f"45{i:06d}" for i in range(n)],
            "PO Date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "GR Date": pd.date_range("2025-02-10", periods=n, freq="D"),
            "PO quantity": rng.integers(10, 100, n),
            "Net Price": 100.0,
            "PO Total Value": 1000.0,
            # Half the lines in a currency worth ~1/100th of the reporting one. Summed
            # raw, they dominate; converted, they are a rounding error.
            "Currency": ["INR" if i % 2 else "USD" for i in range(n)],
        }).to_excel(tmp_path / "po_history.xlsx", index=False)

        out = IngestBridge(config_dir=config_dir, verbose=False).load(
            [tmp_path / "po_history.xlsx"]
        )
        df = out["po_history_df"]
        assert df is not None and len(df)

        inr = df[df["currency_original"] == "INR"]
        usd = df[df["currency_original"] == "USD"]
        assert len(inr) and len(usd)
        # 1000 INR at 0.011 is 11, not 1000. Before the fix these were equal.
        assert inr["po_amount"].mean() == pytest.approx(11.0)
        assert usd["po_amount"].mean() == pytest.approx(1000.0)

        report = out["_fx"].reports["po_history"]
        assert report.is_multi_currency
        assert report.value_after["po_amount"] < report.value_before["po_amount"]
