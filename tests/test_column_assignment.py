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
