"""
An exact header match must beat a fuzzy one.

Found in a live SAP purchase history, where `po_date` was assigned to `PO del date`
— the *delivery* date — while the actual `PO Date` column was claimed by nothing.
Lead time is `receive_date - po_date`, so every measured lead time came out negative
and the document failed its own contract test.

The cause was a one-character asymmetry between two lookup tables built from the same
alias list:

    alias_rank = {normalize_header(a): i for i, a in enumerate(aliases)}   # last wins
    token_rank.setdefault(frozenset(...), i)                              # first wins

`aliases` is `[canonical_name] + declared_aliases`, and `po_date` normalizes to
`po date` — the same string as its own first declared alias. The comprehension
therefore recorded `po date` at rank 1, so the exact header `PO Date` lost the +0.30
canonical-name bonus. `PO del date`, which matches the same alias only as a token
subset, was scored through `token_rank`, kept rank 0, and won the bonus:

    PO Date        exact match,  rank 1, no bonus            -> 1.280
    PO del date    subset match, rank 0, bonus, -0.20 penalty -> 1.400

75 fields across the seven contracts had the same hole, because declaring the
canonical spelling among the aliases is the normal thing to do.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.contract import default_registry
from inventory_planning.ingest.profiler import Profiler, normalize_header
from inventory_planning.ingest.registry import AdapterRegistry


@pytest.fixture(scope="module")
def registry():
    return AdapterRegistry()


def _assign(registry, frame, doc_type):
    profile = Profiler().profile(frame, source_name="x.xlsx")
    return registry._assign_columns(profile, registry.contracts.get(doc_type))


class TestExactMatchWins:

    def test_po_date_takes_the_order_date_not_the_delivery_date(self, registry):
        """The reported failure, reduced to its two competing columns."""
        frame = pd.DataFrame({
            "Material": [f"M-{i:03d}" for i in range(40)],
            "Vendor Name": ["ACME"] * 40,
            "PO Date": pd.to_datetime("2025-01-01") + pd.to_timedelta(range(40), "D"),
            "PO del date": pd.to_datetime("2025-03-01") + pd.to_timedelta(range(40), "D"),
            "PO quantity": range(1, 41),
            "GR Date": pd.to_datetime("2025-02-15") + pd.to_timedelta(range(40), "D"),
        }).astype(str)

        mapping = _assign(registry, frame, "po_history")
        assert mapping["po_date"] == "PO Date"
        assert mapping["receive_date"] == "GR Date"

    def test_the_resulting_lead_time_is_positive(self, registry):
        """
        The consequence, not just the mapping. With the delivery date standing in for
        the order date the lead time went negative on every row, and the contract's
        own `lead_time_days >= 0` assertion failed.
        """
        from inventory_planning.ingest.intake import Intake

        n = 40
        order = pd.to_datetime("2025-01-01") + pd.to_timedelta(range(n), "D")
        frame = pd.DataFrame({
            "Material": [f"M-{i:03d}" for i in range(n)],
            "Vendor Name": ["ACME"] * n,
            "PO Number": [f"45{i:06d}" for i in range(n)],
            "PO Date": order,
            "PO del date": order + pd.Timedelta(days=45),
            "PO quantity": range(1, n + 1),
            "Open Quantity": [0] * n,
            "GR Date": order + pd.Timedelta(days=30),
        })

        doc = Intake(verbose=False).load_frame(frame.astype(str), source_name="ph.xlsx",
                                               doc_type_hint="po_history")
        lead = pd.to_numeric(doc.frame["lead_time_days"], errors="coerce").dropna()
        assert len(lead), "no lead time was derived at all"
        assert (lead == 30).all(), f"expected 30 days, got {sorted(set(lead))[:5]}"


class TestRankTablesAgree:

    def test_the_canonical_name_keeps_rank_zero(self, registry):
        """
        Both tables must resolve the canonical spelling to rank 0. They are built from
        one list and consulted for the same purpose; disagreeing makes an exact match
        score below a fuzzy one.
        """
        offenders = []
        for doc_type, contract in registry.contracts.all().items():
            for name, spec in contract.fields.items():
                aliases = [name] + spec.aliases
                canonical = normalize_header(name)

                alias_rank = {}
                for i, alias in enumerate(aliases):
                    alias_rank.setdefault(normalize_header(alias), i)
                token_rank = {}
                for i, alias in enumerate(aliases):
                    key = frozenset(normalize_header(alias).split())
                    if key:
                        token_rank.setdefault(key, i)

                by_name = alias_rank.get(canonical)
                by_token = token_rank.get(frozenset(canonical.split()))
                if by_name != 0 or by_token != 0:
                    offenders.append(f"{doc_type}.{name} name={by_name} token={by_token}")
        assert not offenders, "canonical spelling not at rank 0:\n  " + "\n  ".join(offenders)

    @pytest.mark.parametrize("doc_type,field,header", [
        ("po_history", "po_date", "PO Date"),
        ("po_history", "po_qty", "PO quantity"),
        ("open_po", "open_qty", "Open Qty"),
        ("inventory", "qty_on_hand", "Qty On Hand"),
        ("open_so", "open_qty", "Open Qty"),
    ])
    def test_a_header_spelled_exactly_like_the_field_is_taken(
        self, registry, doc_type, field, header
    ):
        """
        A decoy sharing the field's tokens plus one more must not displace the exact
        spelling — that is the shape of the original bug.
        """
        frame = pd.DataFrame({
            "Material": [f"M-{i:03d}" for i in range(30)],
            header: list(range(1, 31)),
            f"Old {header}": list(range(101, 131)),
        }).astype(str)
        assert _assign(registry, frame, doc_type).get(field) == header


class TestRunTogetherWordsAreStillWords:
    """
    `OrderDate` matched none of an alias's three rules — not exact, not the same token
    set, not a superset of one — because the boundary between its two words is carried
    in the capitalisation and normalisation threw that away before anything looked. It
    went unmapped, and so did every `ShipDate`, `MaterialNo` and `PONumber` in every
    export that writes headers that way.

    The alternative was to list the run-together spelling of each alias by hand, which
    is the enumeration treadmill the contracts exist to avoid.
    """

    def test_a_camel_case_header_maps(self, registry):
        frame = pd.DataFrame({
            "Material": ["100000797", "100000798"],
            "Still to be delivered (qty)": [10, 20],
            "OrderDate": ["2026-06-16", "2026-06-17"],
        })
        assert _assign(registry, frame, "open_po").get("order_date") == "OrderDate"

    @pytest.mark.parametrize("header,expected", [
        ("OrderDate", "order date"),
        ("ShipDate", "ship date"),
        ("MaterialNo", "material no"),
        ("PONumber", "po number"),
        ("GRDate", "gr date"),
    ])
    def test_the_boundary_is_found(self, header, expected):
        assert normalize_header(header) == expected

    @pytest.mark.parametrize("header", ["MATNR", "ETA", "SKU", "EBELN", "WERKS"])
    def test_an_acronym_is_left_whole(self, header):
        """
        The rule that protects them is the second one: a split happens between an
        acronym and a following word, never inside the acronym. `MATNR` has no
        lower-to-upper transition and comes through untouched.
        """
        assert normalize_header(header) == header.lower()

    def test_separators_still_work_as_before(self):
        for header, expected in (("qty_on_hand", "qty on hand"),
                                 ("QUANTITY_ORDERED", "quantity ordered"),
                                 ("Sold-to Region", "sold to region"),
                                 ("Vendor/supplying plant", "vendor supplying plant")):
            assert normalize_header(header) == expected


class TestSiblingContractsAgreeOnTheSameParty:
    """
    `vendor supplying plant` was declared on `po_history` and not on `open_po`. Somebody
    met the header on a purchase history, added the alias to the contract in front of
    them, and the sibling describing the same real-world party was never touched — so
    one export's supplier mapped and the next one's did not.
    """

    def test_the_supplier_maps_on_an_open_po(self, registry):
        frame = pd.DataFrame({
            "Material": ["100000797", "100000798"],
            "Still to be delivered (qty)": [10, 20],
            "Vendor/supplying plant": ["8012822 ACME", "8012822 ACME"],
        })
        assert _assign(registry, frame, "open_po").get("supplier") == "Vendor/supplying plant"

    def test_a_single_word_alias_is_not_what_rescues_it(self, registry):
        """
        `vendor` alone cannot claim `Vendor/supplying plant`: subset matching ignores
        one-token aliases on purpose, because `date` inside `delivery date` is not
        evidence. The multi-word alias is doing the work, which is why it has to be
        declared on both contracts.
        """
        contract = registry.contracts.get("open_po")
        aliases = contract.fields["supplier"].aliases
        assert "vendor supplying plant" in aliases


class TestTheStockExportsOwnVocabulary:
    """
    MB52 names the two halves of a stock line `Unrestricted` and `Value Unrestricted`.

    The quantity side was in the aliases and the value side was not, so a stock export
    mapped its quantity and dropped its money — and `unit_cost` is derived from
    `inventory_value / qty_on_hand`, so losing one column left every SKU unpriced
    rather than leaving a visible blank. The value totals then cover a different
    population from the quantity totals, with nothing marking the boundary.
    """

    @pytest.mark.parametrize("header", [
        "Value Unrestricted", "Val. Unrestricted", "Unrestricted Value",
        "Value_Unrestricted", "Value Unrestricted Use", "Unrestricted Stock Value",
    ])
    def test_the_value_column_maps_however_it_is_spelled(self, registry, header):
        frame = pd.DataFrame({
            "Material": [f"{5164745 + i}" for i in range(12)],
            "Plant": ["5051"] * 12,
            "Unrestricted": [10] * 12,
            header: [1000.0] * 12,
        })
        assigned = _assign(registry, frame, "inventory")
        assert assigned.get("inventory_value") == header

    def test_the_quantity_side_is_not_stolen_by_the_value_side(self, registry):
        """Both halves carry the same word; each still has to land on its own field."""
        frame = pd.DataFrame({
            "Material": [f"{5164745 + i}" for i in range(12)],
            "Plant": ["5051"] * 12,
            "Unrestricted": [10] * 12,
            "Value Unrestricted": [1000.0] * 12,
        })
        assigned = _assign(registry, frame, "inventory")
        assert assigned.get("qty_on_hand") == "Unrestricted"
        assert assigned.get("inventory_value") == "Value Unrestricted"

    def test_the_unit_cost_that_falls_out_of_it(self, registry):
        """The real line: 6,187 on hand against 1,812,172 of value."""
        from inventory_planning.ingest.intake import Intake

        frame = pd.DataFrame({
            "Material": ["5164745"] + [f"{9000000 + i}" for i in range(11)],
            "Plant": ["5051"] * 12,
            "Unrestricted": [6187] + [10] * 11,
            "Value Unrestricted": [1812172.0] + [1000.0] * 11,
        })
        doc = Intake(verbose=False).load_frame(
            frame, source_name="stock.xlsx", doc_type_hint="inventory")
        row = doc.frame[doc.frame["sku"].astype(str) == "5164745"].iloc[0]
        assert row["unit_cost"] == pytest.approx(292.90, abs=0.01)
