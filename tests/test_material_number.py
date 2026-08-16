"""
SAP pads material numbers; whether an export shows the padding is a property of the
report, not of the material.

From a production run where every derived number came out empty:

    Warning: 1144 SKUs have no supplier LT data — safety stock set to 0
    ⚠ 1144 SKUs have no unit cost — counted in units only
    ⚠ 1144 SKUs have no lead time
    SHOULD-BE $0   ACTUAL $0   baseline should-be: $0

327 lead-time records had been measured and not one matched. The purchase history
gave `000000000005432106`, sales gave `6013908`, inventory gave `408052` — MATNR
padded to 18 characters in one report and not in the others, so every join between
them matched nothing.

Nothing errors when that happens. The joins simply return empty, safety stock falls
to zero, the ABC split collapses to one class because annual value is zero
everywhere, and the report is full of confident zeroes. This is the failure mode the
whole contract layer exists to prevent, and it needed a normalization rather than an
alias.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.adapter import _normalize_material
from inventory_planning.ingest.contract import default_registry
from inventory_planning.ingest.intake import Intake
from inventory_planning.orchestrator import InventoryPlanner


class TestNormalization:

    @pytest.mark.parametrize("raw,expected", [
        ("000000000005432106", "5432106"),   # the padded form SAP stores
        ("5432106", "5432106"),              # already unpadded — unchanged
        ("  0000417975  ", "417975"),        # padding plus stray whitespace
        ("0012", "12"),
        ("000", "0"),                        # not the empty string
    ])
    def test_a_padded_number_loses_its_zeros(self, raw, expected):
        assert _normalize_material(pd.Series([raw])).iloc[0] == expected

    @pytest.mark.parametrize("raw", [
        "4190-6002",            # a leading zero inside a hyphenated code is meaningful
        "ULFMCV224151",
        "SERVICE_BSINES_SUP",
        "0ABC123",              # letters present, so not a padded number
    ])
    def test_a_non_numeric_code_is_untouched(self, raw):
        assert _normalize_material(pd.Series([raw])).iloc[0] == raw.strip().upper()

    @pytest.mark.parametrize("raw,expected", [
        ("6048713.0", "6048713"),            # a numeric cell read back as a float
        ("6048713.00", "6048713"),
        ("000006048713.0", "6048713"),       # padded *and* float-ified
        ("  6048713.0  ", "6048713"),
    ])
    def test_a_float_suffix_is_trimmed(self, raw, expected):
        """
        The same silent-empty-join as the padding, arriving by a different route: a
        spreadsheet storing an identifier column as numeric renders every value as
        `6048713.0`, which matches nothing. Found on the IN30 master, whose
        `Alternate material` key joined 0 SKUs until this was trimmed.
        """
        assert _normalize_material(pd.Series([raw])).iloc[0] == expected

    @pytest.mark.parametrize("raw", [
        "6048713.5",            # a real decimal, not a float-ified integer
        "4190.6002",            # a dotted code — the part after the point is not zeros
        "1.0.0",                # a version-like code
        "ABC.0",                # not all digits before the point
    ])
    def test_a_meaningful_dot_is_kept(self, raw):
        assert _normalize_material(pd.Series([raw])).iloc[0] == raw.strip().upper()

    def test_case_and_whitespace_still_normalize(self):
        assert _normalize_material(pd.Series([" abc-1 "])).iloc[0] == "ABC-1"

    def test_nulls_survive(self):
        out = _normalize_material(pd.Series(["001", None, np.nan]))
        assert out.iloc[0] == "1"
        assert out.isna().sum() == 2

    def test_every_contract_normalizes_its_sku(self):
        """One contract left on plain upper_strip reintroduces the whole failure."""
        offenders = [
            dt for dt, c in default_registry().all().items()
            if c.fields.get("sku") and c.fields["sku"].normalize != "material_number"
        ]
        assert not offenders, f"sku not normalized as a material number in: {offenders}"


class TestThePaddedAndUnpaddedJoin:

    @staticmethod
    def _files(tmp_path, n_skus: int = 30):
        rng = np.random.default_rng(41)
        mats = [str(5400000 + i) for i in range(n_skus)]

        rows = []
        for m in mats:
            # Calendar months, not 30-day steps: the latter drifts, leaving whole
            # months empty, and the classifier then reads every SKU as non-stocking —
            # which zeroes safety stock for a reason that has nothing to do with the
            # join under test.
            for k in range(18):
                rows.append({
                    "Part Number": m,
                    "Billto Customer Name": "ACME",
                    "Invdate Date": pd.Timestamp("2025-01-10") + pd.DateOffset(months=k),
                    "Shipped Quantity": int(rng.integers(5, 40)),
                    "Sales Revenue (USD)": int(rng.integers(50, 900)),
                })
        pd.DataFrame(rows).to_excel(tmp_path / "sales.xlsx", index=False)

        n = 240
        po = pd.Timestamp("2024-06-01") + pd.to_timedelta(rng.integers(0, 400, n), "D")
        pd.DataFrame({
            # Padded, exactly as the reported purchase history exports it.
            # Every SKU appears, so any unmatched one is a join failure and not a
            # sampling artefact.
            "Material": [mats[i % len(mats)].zfill(18) for i in range(n)],
            "Vendor Name": "ACME SUPPLY",
            "PO Number": [f"45{i:06d}" for i in range(n)],
            "PO Date": po,
            "PO quantity": rng.integers(10, 200, n),
            "Open Quantity": 0,
            "Net Price": np.round(rng.random(n) * 90, 2),
            "GR Date": po + pd.Timedelta(days=35),
        }).to_excel(tmp_path / "purchase history.xlsx", index=False)

        pd.DataFrame({
            "Material": mats,
            "Closing Stock": rng.integers(20, 400, len(mats)),
            "Std cost": np.round(rng.random(len(mats)) * 100 + 5, 2),
        }).to_excel(tmp_path / "inventory.xlsx", index=False)
        return mats

    def test_the_two_forms_land_on_one_sku(self, tmp_path):
        mats = self._files(tmp_path)
        result = Intake(verbose=False).load_files(sorted(tmp_path.glob("*.xlsx")))

        history = set(result.frame("po_history")["sku"])
        sales = set(result.frame("sales_history")["sku"])
        assert history & sales, "padded and unpadded forms still do not meet"
        assert history <= set(mats), f"unexpected sku forms: {sorted(history - set(mats))[:3]}"

    def test_lead_time_is_measured_for_the_skus_that_have_demand(self, tmp_path):
        self._files(tmp_path)
        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False)
        results = planner.run_planning(**planner.load_all(sorted(tmp_path.glob("*.xlsx"))))

        supplier_lt = results["supplier_lt"]
        demand_skus = set(results["classified_demand"]["sku"])
        matched = demand_skus & set(supplier_lt["sku"])
        assert len(matched) == len(demand_skus), (
            f"only {len(matched)}/{len(demand_skus)} demand SKUs found a lead time"
        )

    def test_safety_stock_is_not_zero_across_the_board(self, tmp_path):
        """The visible symptom: every SKU at zero because nothing joined."""
        self._files(tmp_path)
        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False)
        results = planner.run_planning(**planner.load_all(sorted(tmp_path.glob("*.xlsx"))))
        assert (results["safety_stock"]["safety_stock"] > 0).any()

    def test_unit_cost_reaches_the_attribute_frame(self, tmp_path):
        """Without cost, every value total is $0 and ABC collapses to one class."""
        from inventory_planning.policy.assemble import build_sku_attributes

        self._files(tmp_path)
        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False)
        inputs = planner.load_all(sorted(tmp_path.glob("*.xlsx")))
        results = planner.run_planning(**inputs)

        attributes, _ = build_sku_attributes(
            classified_demand=results["classified_demand"],
            supplier_lt=results["supplier_lt"],
            inventory=inputs["inventory_df"],
            forecast_summary=results.get("forecast_summary"),
        )
        assert attributes["unit_cost"].notna().any(), "no SKU picked up a cost"
        assert (attributes["annual_value"] > 0).any(), "annual value zero everywhere"
