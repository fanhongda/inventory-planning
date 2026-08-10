"""
A required field that never arrived must stop the run, not crash it later.

From a production run. The sales history routed correctly and mapped seven of its
columns, but its date column — `Invdate Date` — matched no alias, so `demand_date`
was absent. The contract test said exactly that:

    ✗ [structural] required_field:demand_date (19,598/19,598 rows)
      field absent after mapping and derivation

Nothing consulted it. The run proceeded and died in `SalesHistoryReader.to_time_series`
with `KeyError: 'demand_date'` — a traceback that names a column and never mentions
which of six files it came from.

The distinction that matters is between a *semantic* failure and a *structural* one. A
negative lead time on 8% of rows is a data-quality finding the report is built to
carry, and halting on it would make the pipeline unusable on real exports. A missing
required field is different in kind: nothing downstream guards against it, so the run
does not degrade — it raises.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.intake import Intake
from inventory_planning.orchestrator import InventoryPlanner

CONFIG_DIR = Path(__file__).parents[1] / "config"


def _sales(tmp_path, date_header: str, name: str = "sales.xlsx", n: int = 200):
    rng = np.random.default_rng(4)
    pd.DataFrame({
        "Part Number": [f"P-{i % 50:03d}" for i in range(n)],
        "Billto Customer Name": "ACME",
        "Order Number": [f"SO{i:05d}" for i in range(n)],
        date_header: pd.to_datetime("2025-01-01") + pd.to_timedelta(range(n), "D"),
        "Shipped Quantity": rng.integers(1, 40, n),
        "Sales Revenue (USD)": rng.integers(10, 900, n),
    }).to_excel(tmp_path / name, index=False)


def _inventory(tmp_path, name: str = "inv.xlsx"):
    rng = np.random.default_rng(5)
    pd.DataFrame({
        "Material": [f"P-{i % 50:03d}" for i in range(50)],
        "Closing Stock": rng.integers(0, 100, 50),
    }).to_excel(tmp_path / name, index=False)


class TestTheRunStopsWithAUsefulMessage:

    def test_a_missing_required_field_raises_before_planning(self, tmp_path):
        _sales(tmp_path, "Weird Timestamp Col")
        _inventory(tmp_path)
        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False)

        with pytest.raises(ValueError) as exc:
            planner.load_all(sorted(tmp_path.glob("*.xlsx")))
        message = str(exc.value)
        assert "demand_date" in message, "the field has to be named"
        assert "sales.xlsx" in message, "so does the file it was missing from"
        assert "explain" in message, "and what to do next"

    def test_it_is_not_a_keyerror_from_deep_in_a_reader(self, tmp_path):
        """The symptom being fixed: KeyError('demand_date') out of to_time_series."""
        _sales(tmp_path, "Weird Timestamp Col")
        _inventory(tmp_path)
        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False)

        with pytest.raises(ValueError):
            planner.load_all(sorted(tmp_path.glob("*.xlsx")))

    def test_a_good_file_is_not_blocked(self, tmp_path):
        _sales(tmp_path, "Invdate Date")
        _inventory(tmp_path)
        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False)
        inputs = planner.load_all(sorted(tmp_path.glob("*.xlsx")))
        assert inputs["sales_df"] is not None

    def test_a_semantic_failure_does_not_stop_the_run(self, tmp_path):
        """
        Negative quantities are returns. They fail an assertion, are reported, and the
        run continues — halting on them would make the pipeline useless on real data.
        """
        rng = np.random.default_rng(7)
        n = 200
        pd.DataFrame({
            "Part Number": [f"P-{i % 50:03d}" for i in range(n)],
            "Billto Customer Name": "ACME",
            "Order Number": [f"SO{i:05d}" for i in range(n)],
            "Invdate Date": pd.to_datetime("2025-01-01") + pd.to_timedelta(range(n), "D"),
            "Shipped Quantity": rng.integers(-5, 40, n),
            "Sales Revenue (USD)": rng.integers(10, 900, n),
        }).to_excel(tmp_path / "sales.xlsx", index=False)
        _inventory(tmp_path)

        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False)
        inputs = planner.load_all(sorted(tmp_path.glob("*.xlsx")))
        assert len(inputs["sales_df"]) > 0


class TestUnusableReporting:

    def test_unusable_names_the_document_and_the_field(self, tmp_path):
        _sales(tmp_path, "Weird Timestamp Col")
        result = Intake(verbose=False).load_files([tmp_path / "sales.xlsx"])
        unusable = result.unusable()
        assert len(unusable) == 1
        doc, fields = unusable[0]
        assert doc.doc_type == "sales_history"
        assert fields == ["demand_date"]

    def test_a_sound_document_reports_nothing(self, tmp_path):
        _sales(tmp_path, "Invdate Date")
        result = Intake(verbose=False).load_files([tmp_path / "sales.xlsx"])
        assert result.unusable() == []


class TestSapDateHeaders:
    """The specific headers from the reported export."""

    @pytest.mark.parametrize("header", [
        "Invdate Date", "Invoice Date", "Billing Date", "Posting Date", "Demand Date",
    ])
    def test_a_date_header_reaches_demand_date(self, tmp_path, header):
        _sales(tmp_path, header)
        result = Intake(verbose=False).load_files([tmp_path / "sales.xlsx"])
        frame = result.frame("sales_history")
        assert frame is not None and "demand_date" in frame.columns
        assert frame["demand_date"].notna().any()

    def test_orddate_reaches_order_date(self, tmp_path):
        rng = np.random.default_rng(8)
        n = 100
        base = pd.to_datetime("2025-01-01") + pd.to_timedelta(range(n), "D")
        pd.DataFrame({
            "Part Number": [f"P-{i % 20:03d}" for i in range(n)],
            "Billto Customer Name": "ACME",
            "Invdate Date": base,
            "Orddate Date": base - pd.Timedelta(days=120),
            "Shipped Quantity": rng.integers(1, 40, n),
        }).to_excel(tmp_path / "sales.xlsx", index=False)

        frame = Intake(verbose=False).load_files([tmp_path / "sales.xlsx"]).frame("sales_history")
        assert "order_date" in frame.columns
        assert frame["order_date"].notna().any()
