"""
Everything joins on `sku`, so a key that agrees with nothing fails in total silence.

Found by running the pipeline on a real six-file SAP extract. It completed, reported
`OK`, and produced a should-be inventory of $0 across 1,144 SKUs. Three separate
causes, none of which raised anything:

  a purchase history keyed on `000000000005432106` against a sales history keyed on
  `6013908` — MATNR padded in one report and not the other

  a backlog whose `sku` came from `Item` (SAP's line number: 10, 20, 30) rather than
  `Material`, because `open_so` had no field for a line number and `item` outranked
  `material` in the alias list

  a planner master keyed on a local code in `Material` while carrying the ERP number
  in `Alternate material`

The last is a per-file quirk and belongs in an adapter. What does not belong anywhere
except here is the silence: the merges return empty, safety stock falls to zero,
annual value is zero so every item lands in one ABC class, and the report is full of
confident zeroes.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.intake import Intake
from inventory_planning.ingest.profiler import Profiler
from inventory_planning.ingest.registry import AdapterRegistry


@pytest.fixture(scope="module")
def registry():
    return AdapterRegistry()


class TestItemIsNotTheSku:
    """SAP calls the line number `Item`. It is not the material."""

    @staticmethod
    def _backlog(n: int = 300):
        rng = np.random.default_rng(12)
        return pd.DataFrame({
            "Sales Docu No": [f"030{i // 8:06d}" for i in range(n)],
            "Item": [str(10 * (i % 12 + 1)) for i in range(n)],       # 10, 20, 30 …
            "Material": [f"100000{i % 60:03d}" for i in range(n)],
            "Sold To party": "ACME",
            "Req Del Date": pd.Timestamp("2026-10-30"),
            "Order Qty": rng.integers(1, 50, n),
            "Open Qty": rng.integers(1, 50, n),
        }).astype(str)

    def test_sku_takes_material(self, registry):
        profile = Profiler().profile(self._backlog(), source_name="backlog.xlsx")
        mapping = registry._assign_columns(profile, registry.contracts.get("open_so"))
        assert mapping.get("sku") == "Material"

    def test_the_line_number_has_somewhere_to_go(self, registry):
        """Without a home of its own, `Item` was claimed by sku."""
        profile = Profiler().profile(self._backlog(), source_name="backlog.xlsx")
        mapping = registry._assign_columns(profile, registry.contracts.get("open_so"))
        assert mapping.get("so_line_number") == "Item"

    def test_a_file_with_only_item_still_maps_it(self, registry):
        """Demoting the alias must not break a source where `Item` is the material."""
        frame = pd.DataFrame({
            "Item": [f"M-{i:04d}" for i in range(60)],
            "Customer": "ACME",
            "Req Del Date": pd.Timestamp("2026-10-30"),
            "Open Qty": range(1, 61),
        }).astype(str)
        profile = Profiler().profile(frame, source_name="b.xlsx")
        mapping = registry._assign_columns(profile, registry.contracts.get("open_so"))
        assert mapping.get("sku") == "Item"


class TestIsoDatesSurviveDayfirst:
    """
    `dayfirst=True` on an ISO column destroys every date whose day exceeds 12.

    pandas reads `2020-10-25` as year-day-month, month 25 is invalid, and the value
    becomes NaT. A purchase history with no null dates at all arrived reporting 59.9%
    null — almost exactly the 19/31 of days above the 12th — and 86% of its rows were
    then dropped as "implausible lead time".
    """

    @staticmethod
    def _po_history(n: int = 400):
        rng = np.random.default_rng(13)
        # Every day of the month, so the >12 failure is guaranteed to show.
        order = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(i)) for i in range(n)]
        return pd.DataFrame({
            "Material": [f"54{i % 40:05d}" for i in range(n)],
            "Vendor Name": "ACME",
            "PO Number": [f"45{i:06d}" for i in range(n)],
            "PO Date": [d.strftime("%Y-%m-%d %H:%M:%S") for d in order],
            "PO quantity": rng.integers(1, 100, n),
            "Open Quantity": 0,
            "GR Date": [(d + pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
                        for d in order],
        }).astype(str)

    def test_no_date_is_lost(self, tmp_path):
        self._po_history().to_excel(tmp_path / "ph.xlsx", index=False)
        frame = Intake(verbose=False).load_files([tmp_path / "ph.xlsx"]).frame("po_history")
        assert frame["po_date"].isna().sum() == 0, "ISO dates parsed with dayfirst"
        assert frame["receive_date"].isna().sum() == 0

    def test_the_lead_time_is_the_real_one(self, tmp_path):
        self._po_history().to_excel(tmp_path / "ph.xlsx", index=False)
        frame = Intake(verbose=False).load_files([tmp_path / "ph.xlsx"]).frame("po_history")
        lead = pd.to_numeric(frame["lead_time_days"], errors="coerce").dropna()
        assert (lead == 30).all(), f"expected 30 days, got {sorted(set(lead))[:5]}"

    def test_a_genuine_dayfirst_source_still_parses(self, tmp_path):
        """Demoting dayfirst for ISO must not break `25/12/2024`."""
        n = 60
        order = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(n)]
        pd.DataFrame({
            "Material": [f"54{i % 20:05d}" for i in range(n)],
            "Vendor Name": "ACME",
            "PO Number": [f"45{i:06d}" for i in range(n)],
            "PO Date": [d.strftime("%d/%m/%Y") for d in order],
            "PO quantity": range(1, n + 1),
            "Open Quantity": 0,
            "GR Date": [(d + pd.Timedelta(days=20)).strftime("%d/%m/%Y") for d in order],
        }).astype(str).to_excel(tmp_path / "ph.xlsx", index=False)

        frame = Intake(verbose=False).load_files([tmp_path / "ph.xlsx"]).frame("po_history")
        assert frame["po_date"].isna().sum() == 0
        lead = pd.to_numeric(frame["lead_time_days"], errors="coerce").dropna()
        assert (lead == 20).all()


class TestSkuAgreementIsReported:
    """
    A key that matches nothing must say so, and name the column that would.

    Knowing that `sku` joins to nothing is a puzzle. Knowing that `Alternate material`
    would have joined 99% is an answer.
    """

    @staticmethod
    def _pair(tmp_path):
        skus = [f"600{i:04d}" for i in range(60)]
        rows = []
        for s in skus:
            for k in range(14):
                rows.append({"Part Number": s, "Billto Customer Name": "ACME",
                             "Invdate Date": pd.Timestamp("2025-01-10") + pd.DateOffset(months=k),
                             "Shipped Quantity": 10})
        pd.DataFrame(rows).to_excel(tmp_path / "sales.xlsx", index=False)

        # A master keyed on a local code, with the real number in a second column.
        pd.DataFrame({
            "Material": [f"LOCAL-{i:04d}" for i in range(60)],
            "Alternate material ": skus,
            "SS": range(10, 70),
            "Std cost": np.linspace(5, 90, 60).round(2),
        }).to_excel(tmp_path / "master.xlsx", index=False)

    def test_the_mismatch_is_reported(self, tmp_path):
        self._pair(tmp_path)
        notes = " ".join(Intake(verbose=False)
                         .load_files(sorted(tmp_path.glob("*.xlsx"))).notes)
        assert "keys on something the other documents do not use" in notes

    def test_the_column_that_would_join_is_named(self, tmp_path):
        self._pair(tmp_path)
        notes = " ".join(Intake(verbose=False)
                         .load_files(sorted(tmp_path.glob("*.xlsx"))).notes)
        assert "Alternate material" in notes

    def test_agreeing_documents_are_silent(self, tmp_path):
        """The warning must not become wallpaper."""
        skus = [f"600{i:04d}" for i in range(60)]
        rows = []
        for s in skus:
            for k in range(14):
                rows.append({"Part Number": s, "Billto Customer Name": "ACME",
                             "Invdate Date": pd.Timestamp("2025-01-10") + pd.DateOffset(months=k),
                             "Shipped Quantity": 10})
        pd.DataFrame(rows).to_excel(tmp_path / "sales.xlsx", index=False)
        pd.DataFrame({"Material": skus, "Closing Stock": range(60)}).to_excel(
            tmp_path / "inv.xlsx", index=False)

        notes = " ".join(Intake(verbose=False)
                         .load_files(sorted(tmp_path.glob("*.xlsx"))).notes)
        assert "keys on something" not in notes


class TestOneMasterAdapterServesEveryRegion:
    """
    The FSP planning master is one export template shared across regions, so one adapter
    serves all of them. The ERP number sits in `Alternate material` for the IN30 export
    and in `Material` for other regions; the adapter coalesces the two.

    The regression this guards: the earlier IN30-only adapter hard-mapped `sku` to
    `Alternate material`, and its fingerprint matched every region sharing this template.
    On a region whose `Alternate material` is blank it re-keyed every row to a null SKU,
    and the rollup collapsed the whole master to one row — silently deleting the
    planner-baseline comparison.
    """

    ERP = [f"6{i:06d}" for i in range(60)]   # "6000000", "6000001", …

    @classmethod
    def _master(cls, *, erp_in_alternate: bool) -> pd.DataFrame:
        n = len(cls.ERP)
        # Numeric ID cells read back as "6000000.0" under dtype=str — the trailing ".0"
        # the material_number normalisation on the derived key has to strip.
        erp = pd.Series([float(x) for x in cls.ERP])
        local = pd.Series([f"LOCAL-{i:04d}" for i in range(n)])
        blank = pd.Series([np.nan] * n)
        material, alternate = (local, erp) if erp_in_alternate else (erp, blank)
        return pd.DataFrame({
            "Material": material,
            "Alternate material": alternate,
            "Material description": [f"ITEM {i}" for i in range(n)],
            "Stock Cat": "MTS",
            "Material Classification": "Runners",
            "vendor name": "ACME",
            "Std cost": np.linspace(5, 90, n).round(2),
            "SS": range(10, 10 + n),
        })

    def _frame(self, tmp_path, *, erp_in_alternate: bool):
        self._master(erp_in_alternate=erp_in_alternate).to_excel(
            tmp_path / "master.xlsx", index=False)
        return (Intake(verbose=False)
                .load_files([tmp_path / "master.xlsx"]).frame("planning_master"))

    def test_in30_layout_keys_on_alternate_material(self, tmp_path):
        frame = self._frame(tmp_path, erp_in_alternate=True)
        assert len(frame) == len(self.ERP)
        assert set(frame["sku"]) == set(self.ERP)

    def test_other_region_keys_on_material(self, tmp_path):
        """An empty `Alternate material` must fall through to `Material`, not collapse."""
        frame = self._frame(tmp_path, erp_in_alternate=False)
        assert len(frame) == len(self.ERP)
        assert set(frame["sku"]) == set(self.ERP)

    def test_the_derived_key_carries_the_planner_parameter(self, tmp_path):
        """The whole point of the master: its safety stock arrives, keyed correctly."""
        frame = self._frame(tmp_path, erp_in_alternate=False).set_index("sku")
        assert pd.to_numeric(frame["planner_safety_stock"]).notna().all()
        assert int(pd.to_numeric(frame.loc["6000000", "planner_safety_stock"])) == 10

    def test_a_region_without_the_alternate_column_keys_on_material(self, tmp_path):
        """
        The IN30 export drops `Alternate material` entirely (its secondary code is a
        different column, `Material II`). The coalesce input is absent, not just empty;
        the derivation must treat it as null and fall through to `Material`, not skip and
        leave the master keyless.
        """
        n = len(self.ERP)
        pd.DataFrame({
            "Material": pd.Series([float(x) for x in self.ERP]),
            "Material II": [f"ULF{i:05d}" for i in range(n)],
            "Material description": [f"ITEM {i}" for i in range(n)],
            "Stock Cat": "MTO",
            "Material Classification": "strangers",
            "vendor name": "ACME",
            "Std cost": np.linspace(5, 90, n).round(2),
            "SS": range(10, 10 + n),
        }).to_excel(tmp_path / "master.xlsx", index=False)
        frame = (Intake(verbose=False)
                 .load_files([tmp_path / "master.xlsx"]).frame("planning_master"))
        assert len(frame) == n
        assert set(frame["sku"]) == set(self.ERP)

