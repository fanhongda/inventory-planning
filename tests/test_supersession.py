"""
Tests for material renumbering — two item numbers made into one material.

The feature exists because the failure is silent. 7100014 renumbered to 7100015 gives
the old number stock against a history that stopped and the new number a history too
short to plan on, and every downstream figure is confidently wrong with no error
anywhere. So what has to hold is mostly about *what does not happen*:

  merge        every document is rewritten, and the quantities land on the successor
  everywhere   no document is left on the old number — a partial rewrite is worse
               than none, because position and demand end up under different keys
  grain        rows that become duplicates are recombined, not left to double-count
  parameters   a master row is a description, not an event: two lead times do not add
  ratio        a pack-size change scales the quantities and leaves the money alone
  refusal      a split or a loop is dropped and reported, never resolved by choosing
  phase        coexisting pairs are read and left entirely alone
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.contract import default_registry
from inventory_planning.ingest.intake import Intake
from inventory_planning.ingest.supersede import SupersessionMap

OLD = "7100014"
NEW = "7100015"

CONTRACTS = default_registry()


def build_map(rows, item_master=None):
    """A SupersessionMap from the canonical frame a substitution file becomes."""
    df = pd.DataFrame(rows) if rows else None
    return SupersessionMap.from_frames(substitution_df=df, item_master_df=item_master)


# ── Resolution ───────────────────────────────────────────────────────────────


def test_pair_resolves_to_the_successor():
    smap = build_map([{"old_sku": OLD, "new_sku": NEW}])
    assert len(smap) == 1
    pair = smap.pairs[0]
    assert (pair.old_sku, pair.new_sku, pair.ratio) == (OLD, NEW, 1.0)
    assert not smap.report.problems


def test_a_chain_resolves_to_the_number_that_is_alive_now():
    """A→B→C. A document written before either change still says A."""
    smap = build_map([
        {"old_sku": "A", "new_sku": "B"},
        {"old_sku": "B", "new_sku": "C"},
    ])
    terminal = {p.old_sku: p.new_sku for p in smap.pairs}
    assert terminal == {"A": "C", "B": "C"}
    assert next(p for p in smap.pairs if p.old_sku == "A").declared_as == "B"


def test_ratios_compound_along_a_chain():
    smap = build_map([
        {"old_sku": "A", "new_sku": "B", "ratio": 2},
        {"old_sku": "B", "new_sku": "C", "ratio": 5},
    ])
    assert next(p for p in smap.pairs if p.old_sku == "A").ratio == 10


def test_many_to_one_is_ordinary():
    """Two obsolete numbers consolidated onto one current part is not a conflict."""
    smap = build_map([
        {"old_sku": "A", "new_sku": "C"},
        {"old_sku": "B", "new_sku": "C"},
    ])
    assert len(smap) == 2
    assert not smap.report.problems


def test_one_to_many_is_refused_not_resolved():
    """
    A number with two successors is a split. Which of them a given transaction
    belongs to is a question about the transaction, so there is nothing to choose.
    """
    smap = build_map([
        {"old_sku": "A", "new_sku": "B"},
        {"old_sku": "A", "new_sku": "C"},
    ])
    assert len(smap) == 0
    assert any("split" in p for p in smap.report.problems)


def test_a_loop_is_dropped_with_the_rest_of_the_map_intact():
    smap = build_map([
        {"old_sku": "A", "new_sku": "B"},
        {"old_sku": "B", "new_sku": "A"},
        {"old_sku": "X", "new_sku": "Y"},
    ])
    assert {p.old_sku for p in smap.pairs} == {"X"}
    assert any("loops back" in p for p in smap.report.problems)


def test_self_mapping_is_a_maintenance_error():
    smap = build_map([{"old_sku": OLD, "new_sku": OLD}])
    assert len(smap) == 0
    assert smap.report.problems


def test_phase_pairs_are_read_and_never_merged():
    smap = build_map([
        {"old_sku": OLD, "new_sku": NEW, "relation": "phase"},
        {"old_sku": "A", "new_sku": "B", "relation": "supersede"},
    ])
    assert {p.old_sku for p in smap.pairs} == {"A"}
    assert smap.report.phase_pairs == 1


def test_an_absent_relation_column_means_supersede():
    """The value domain's default only reaches values that are present."""
    smap = build_map([{"old_sku": OLD, "new_sku": NEW}])
    assert len(smap) == 1


def test_an_item_master_column_proposes_and_never_merges():
    """
    The master was handed over for lead times and MOQs. A follow-up-material column on
    it is maintained years before anyone plans on it, and nobody chose it for this — so
    it cannot restructure every document in the run on its own.
    """
    master = pd.DataFrame({
        "sku": [OLD, "OTHER"],
        "successor_sku": [NEW, None],
        "discontinued_date": ["2026-03-01", None],
    })
    smap = SupersessionMap.from_frames(item_master_df=master)

    assert len(smap) == 0
    assert [(p.old_sku, p.new_sku) for p in smap.report.proposals] == [(OLD, NEW)]

    inv = pd.DataFrame({"sku": [OLD, NEW], "location_id": ["DC-01"] * 2,
                        "qty_on_hand": [150.0, 400.0]})
    out, records = smap.apply(inv, CONTRACTS.get("inventory"))
    assert set(out["sku"]) == {OLD, NEW}
    assert not records


def test_a_proposal_comes_back_as_a_list_that_can_be_handed_in():
    master = pd.DataFrame({
        "sku": [OLD], "successor_sku": [NEW], "discontinued_date": ["2026-03-01"],
    })
    block = SupersessionMap.from_frames(item_master_df=master).report.proposal_block()
    assert block.splitlines()[0].startswith("old_sku,new_sku,relation")
    assert f"{OLD},{NEW},supersede,2026-03-01," in block

    # And handing it back is what makes the merge happen.
    handed_back = pd.read_csv(pd.io.common.StringIO(block), dtype=str)
    smap = SupersessionMap.from_frames(substitution_df=handed_back)
    assert [(p.old_sku, p.new_sku) for p in smap.pairs] == [(OLD, NEW)]


def test_a_declared_pair_is_not_also_proposed():
    master = pd.DataFrame({"sku": [OLD], "successor_sku": [NEW]})
    smap = build_map([{"old_sku": OLD, "new_sku": NEW}], item_master=master)
    assert len(smap) == 1
    assert not smap.report.proposals
    assert not smap.report.problems


def test_the_declared_pair_wins_over_the_master_and_the_gap_is_named():
    """The act outranks the attribute — but a stale master is worth saying out loud."""
    master = pd.DataFrame({"sku": [OLD], "successor_sku": ["9999999"]})
    smap = build_map([{"old_sku": OLD, "new_sku": NEW}], item_master=master)
    assert [(p.old_sku, p.new_sku) for p in smap.pairs] == [(OLD, NEW)]
    assert any("9999999" in c for c in smap.report.challenges)


def test_a_map_can_be_withheld_wholesale():
    smap = SupersessionMap.from_frames(
        substitution_df=pd.DataFrame([{"old_sku": OLD, "new_sku": NEW}]),
        withheld="routed on a thin margin",
    )
    assert len(smap) == 0
    # Still reported — withholding is not hiding.
    assert [(p.old_sku, p.new_sku) for p in smap.report.pairs] == [(OLD, NEW)]
    assert "thin margin" in smap.report.summary()
    assert "NOTHING WAS MERGED" in smap.report.summary()


def test_an_assumed_relation_is_declared_as_assumed():
    assert build_map([{"old_sku": OLD, "new_sku": NEW}]).report.relation_assumed
    assert not build_map(
        [{"old_sku": OLD, "new_sku": NEW, "relation": "supersede"}]
    ).report.relation_assumed


# ── Application ──────────────────────────────────────────────────────────────


def test_quantities_land_on_the_successor_and_the_rows_combine():
    smap = build_map([{"old_sku": OLD, "new_sku": NEW}])
    inv = pd.DataFrame({
        "sku": [OLD, NEW, "UNRELATED"],
        "location_id": ["DC-01"] * 3,
        "qty_on_hand": [150.0, 400.0, 10.0],
    })
    out, records = smap.apply(inv, CONTRACTS.get("inventory"))

    assert OLD not in set(out["sku"])
    assert float(out.loc[out["sku"] == NEW, "qty_on_hand"].iloc[0]) == 550.0
    assert float(out.loc[out["sku"] == "UNRELATED", "qty_on_hand"].iloc[0]) == 10.0
    assert len(out) == 2
    assert records[0].rows == 1 and records[0].quantity == 150.0


def test_a_document_with_no_successor_row_just_gets_renamed():
    smap = build_map([{"old_sku": OLD, "new_sku": NEW}])
    inv = pd.DataFrame({
        "sku": [OLD], "location_id": ["DC-01"], "qty_on_hand": [150.0],
    })
    out, _ = smap.apply(inv, CONTRACTS.get("inventory"))
    assert list(out["sku"]) == [NEW]
    assert float(out["qty_on_hand"].iloc[0]) == 150.0


def test_transaction_lines_are_not_collapsed_by_the_merge():
    """
    Two receipts on different POs are two events, not duplicates. The natural key
    carries the PO, so the rename cannot make them collide.
    """
    smap = build_map([{"old_sku": OLD, "new_sku": NEW}])
    hist = pd.DataFrame({
        "po_number": ["P1", "P2"],
        "sku": [OLD, NEW],
        "receive_date": pd.to_datetime(["2026-01-05", "2026-02-05"]),
        "receive_qty": [100.0, 200.0],
        "lead_time_days": [90.0, 92.0],
    })
    out, _ = smap.apply(hist, CONTRACTS.get("po_history"))
    assert len(out) == 2
    assert set(out["sku"]) == {NEW}
    assert sorted(out["lead_time_days"]) == [90.0, 92.0]


def test_a_master_row_is_a_description_so_lead_times_do_not_add():
    """Two 90-day lead times are not a 180-day lead time."""
    smap = build_map([{"old_sku": OLD, "new_sku": NEW}])
    master = pd.DataFrame({
        "sku": [OLD, NEW],
        "lead_time_days": [90.0, 100.0],
        "min_order_qty": [50.0, 60.0],
        "unit_cost": [12.0, 13.0],
    })
    out, _ = smap.apply(master, CONTRACTS.get("item_master"))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["sku"] == NEW
    # The successor describes the successor.
    assert row["lead_time_days"] == 100.0
    assert row["min_order_qty"] == 60.0
    assert row["unit_cost"] == 13.0


def test_the_predecessor_fills_what_the_successor_leaves_blank():
    """A master row raised for a new number is routinely thinner than the old one."""
    smap = build_map([{"old_sku": OLD, "new_sku": NEW}])
    master = pd.DataFrame({
        "sku": [OLD, NEW],
        "lead_time_days": [90.0, None],
        "supplier": ["ACME", None],
        "min_order_qty": [50.0, 60.0],
    })
    out, _ = smap.apply(master, CONTRACTS.get("item_master"))
    row = out.iloc[0]
    assert row["lead_time_days"] == 90.0
    assert row["supplier"] == "ACME"
    assert row["min_order_qty"] == 60.0


def test_unit_cost_is_weighted_by_quantity_not_summed():
    """
    A per-unit price is one number the combined quantity was bought at. Summing it
    invents money; a flat average ignores that one row may be a hundred times the other.
    """
    smap = build_map([{"old_sku": OLD, "new_sku": NEW}])
    inv = pd.DataFrame({
        "sku": [OLD, NEW],
        "location_id": ["DC-01", "DC-01"],
        "qty_on_hand": [100.0, 300.0],
        "unit_cost": [10.0, 20.0],
    })
    out, _ = smap.apply(inv, CONTRACTS.get("inventory"))
    assert float(out["qty_on_hand"].iloc[0]) == 400.0
    assert float(out["unit_cost"].iloc[0]) == pytest.approx(17.5)


def test_two_empty_rows_leave_the_successors_price_standing():
    """
    An item in a stock report with nothing on the shelf is ordinary. There is no
    weighted price to compute from two zeroes, and the division must not blow up.
    """
    smap = build_map([{"old_sku": OLD, "new_sku": NEW}])
    inv = pd.DataFrame({
        "sku": [OLD, NEW],
        "location_id": ["DC-01", "DC-01"],
        "qty_on_hand": [0.0, 0.0],
        "unit_cost": [10.0, 20.0],
    })
    out, _ = smap.apply(inv, CONTRACTS.get("inventory"))
    assert len(out) == 1
    assert float(out["qty_on_hand"].iloc[0]) == 0.0
    assert float(out["unit_cost"].iloc[0]) == 20.0


def test_a_pack_size_change_scales_quantities_and_leaves_the_money_alone():
    """One old piece is ten new ones: the units multiply, the total value does not move."""
    smap = build_map([{"old_sku": OLD, "new_sku": NEW, "ratio": 10}])
    inv = pd.DataFrame({
        "sku": [OLD], "location_id": ["DC-01"],
        "qty_on_hand": [100.0], "unit_cost": [50.0],
    })
    out, _ = smap.apply(inv, CONTRACTS.get("inventory"))
    assert float(out["qty_on_hand"].iloc[0]) == 1000.0
    assert float(out["unit_cost"].iloc[0]) == 5.0
    value = out["qty_on_hand"].iloc[0] * out["unit_cost"].iloc[0]
    assert float(value) == 5000.0


def test_a_ratio_does_not_touch_rows_that_were_not_renamed():
    smap = build_map([{"old_sku": OLD, "new_sku": NEW, "ratio": 10}])
    inv = pd.DataFrame({
        "sku": [OLD, "UNRELATED"], "location_id": ["DC-01", "DC-01"],
        "qty_on_hand": [100.0, 7.0],
    })
    out, _ = smap.apply(inv, CONTRACTS.get("inventory"))
    assert float(out.loc[out["sku"] == "UNRELATED", "qty_on_hand"].iloc[0]) == 7.0


def test_a_declaration_the_transactions_contradict_is_challenged():
    """
    Not an inference — the premise of a claim somebody already made. An old number
    still shipping after its own effective date has not been superseded.
    """
    smap = build_map([
        {"old_sku": OLD, "new_sku": NEW, "effective_date": "2026-03-01"},
    ])
    sales = pd.DataFrame({
        "sku": [OLD, OLD],
        "ship_date": pd.to_datetime(["2026-01-10", "2026-06-10"]),
        "ship_qty": [5.0, 5.0],
    })
    challenges = smap.challenge({"sales_history": sales})
    assert len(challenges) == 1
    assert OLD in challenges[0] and "2026-03-01" in challenges[0]


def test_one_challenge_per_pair_not_one_per_document():
    """The same live material appears in every extract; that is one problem, not five."""
    smap = build_map([
        {"old_sku": OLD, "new_sku": NEW, "effective_date": "2026-03-01"},
    ])
    late = pd.to_datetime(["2026-06-10"])
    frames = {
        "sales_history": pd.DataFrame({"sku": [OLD], "ship_date": late, "ship_qty": [5.0]}),
        "po_history": pd.DataFrame({"sku": [OLD], "receive_date": late, "receive_qty": [5.0]}),
        "open_so": pd.DataFrame({"sku": [OLD], "order_date": late, "open_qty": [5.0]}),
    }
    challenges = smap.challenge(frames)
    assert len(challenges) == 1
    for doc_type in frames:
        assert doc_type in challenges[0]


def test_no_challenge_when_the_old_number_stopped_on_time():
    smap = build_map([
        {"old_sku": OLD, "new_sku": NEW, "effective_date": "2026-03-01"},
    ])
    sales = pd.DataFrame({
        "sku": [OLD], "ship_date": pd.to_datetime(["2026-01-10"]), "ship_qty": [5.0],
    })
    assert smap.challenge({"sales_history": sales}) == []


# ── End to end through intake ────────────────────────────────────────────────


@pytest.fixture
def exports(tmp_path):
    """A renumbering split across four documents, as it actually arrives."""
    pd.DataFrame({
        "Old Material": [OLD],
        "New Material": [NEW],
        "Effective Date": ["2026-03-01"],
        "Reason": ["supplier consolidated the two casting variants"],
    }).to_csv(tmp_path / "substitutions.csv", index=False)

    pd.DataFrame({
        "Material": [OLD, NEW, "OTHER-1"],
        "Plant": ["DC-01"] * 3,
        "Unrestricted Stock": [150, 400, 90],
        "Moving Average Price": [10.0, 10.0, 4.0],
    }).to_csv(tmp_path / "stock.csv", index=False)

    months = pd.date_range("2024-09-01", periods=18, freq="MS")
    rows = []
    for i, month in enumerate(months):
        # The old number sells for a year and stops; the new one takes over.
        if i < 12:
            rows.append({"Sales Order": f"S{i}", "Material": OLD,
                         "Ship Date": month.strftime("%Y-%m-%d"),
                         "Quantity": 100 + i, "Customer": "C1"})
        else:
            rows.append({"Sales Order": f"S{i}", "Material": NEW,
                         "Ship Date": month.strftime("%Y-%m-%d"),
                         "Quantity": 100 + i, "Customer": "C1"})
        rows.append({"Sales Order": f"T{i}", "Material": "OTHER-1",
                     "Ship Date": month.strftime("%Y-%m-%d"),
                     "Quantity": 20, "Customer": "C2"})
    pd.DataFrame(rows).to_csv(tmp_path / "sales.csv", index=False)

    pd.DataFrame({
        "PO Number": ["P1", "P2", "P3"],
        "Material": [OLD, NEW, "OTHER-1"],
        "Order Date": ["2025-01-05", "2026-04-05", "2025-06-05"],
        "Delivery Date": ["2025-04-05", "2026-07-05", "2025-09-05"],
        # P1 is the realistic case: supply raised against the old number before the
        # switch, still to arrive. It is inbound for the successor.
        "Order Quantity": [500, 600, 100],
        "Open Quantity": [500, 600, 100],
    }).to_csv(tmp_path / "open_po.csv", index=False)

    return tmp_path


def test_intake_merges_every_document_at_once(exports):
    result = Intake(verbose=False).load_files(sorted(exports.glob("*.csv")))

    assert "substitution" in result.documents
    applied = result.supersessions.applied
    assert [(p.old_sku, p.new_sku) for p in applied] == [(OLD, NEW)]

    # The point of the feature: no document is left holding the old number.
    for doc_type, doc in result.documents.items():
        if doc_type == "substitution" or "sku" not in doc.frame.columns:
            continue
        assert OLD not in set(doc.frame["sku"].dropna().astype(str)), doc_type

    inv = result.frame("inventory")
    assert float(inv.loc[inv["sku"] == NEW, "qty_on_hand"].sum()) == 550.0

    sales = result.frame("sales_history")
    assert len(sales[sales["sku"] == NEW]) == 18


def test_the_merge_is_recorded_per_document(exports):
    result = Intake(verbose=False).load_files(sorted(exports.glob("*.csv")))
    frame = result.supersessions.to_frame()

    assert set(frame["old_sku"]) == {OLD}
    assert set(frame["new_sku"]) == {NEW}
    # Stock, demand and inbound each contributed, and the file says how much.
    assert {"inventory", "sales_history", "open_po"} <= set(frame["doc_type"])
    stock = frame.loc[frame["doc_type"] == "inventory", "quantity_merged"].iloc[0]
    assert float(stock) == 150.0


def test_the_merge_leaves_every_document_at_its_declared_grain(exports):
    result = Intake(verbose=False).load_files(sorted(exports.glob("*.csv")))
    assert not result.supersessions.problems

    for doc_type, doc in result.documents.items():
        contract = CONTRACTS.get(doc_type)
        key = [k for k in contract.natural_key if k in doc.frame.columns]
        if not key:
            continue
        assert not doc.frame.duplicated(subset=key).any(), doc_type


def test_the_substitution_list_reaches_the_summary(exports):
    result = Intake(verbose=False).load_files(sorted(exports.glob("*.csv")))
    text = result.summary()
    assert "Material supersessions" in text
    assert f"{OLD} -> {NEW}" in text
    assert result.plan.has("substitution_signal")


def test_a_precompiled_time_series_merges_period_by_period(tmp_path):
    """
    The planner-supplied wide series is the one input that arrives already bucketed,
    so a renumbering has to be summed inside each period rather than across the file.
    """
    pd.DataFrame({
        "Old Material": [OLD], "New Material": [NEW],
    }).to_csv(tmp_path / "substitutions.csv", index=False)

    months = [d.strftime("%Y-%m") for d in pd.date_range("2025-01-01", periods=14, freq="MS")]
    wide = pd.DataFrame(
        [[OLD] + [50.0] * 14, [NEW] + [30.0] * 14, ["OTHER-1"] + [8.0] * 14],
        columns=["Material"] + months,
    )
    wide.to_csv(tmp_path / "demand.csv", index=False)
    pd.DataFrame({
        "Material": [OLD, NEW], "Plant": ["DC-01"] * 2, "Unrestricted Stock": [10, 20],
    }).to_csv(tmp_path / "stock.csv", index=False)

    result = Intake(verbose=False).load_files(sorted(tmp_path.glob("*.csv")))
    series = result.frame("demand_timeseries")

    assert OLD not in set(series["sku"].astype(str))
    merged = series[series["sku"] == NEW]
    assert len(merged) == 14                       # one row per period, not two
    assert set(merged["qty"].astype(float)) == {80.0}


def test_a_transfer_document_is_not_a_substitution_list(tmp_path):
    """
    SAP movement 309 moves stock from one material to another and is headed `From
    Material` / `To Material`. It routed here at 83% while those were aliases, and one
    posting would have permanently merged two live materials. The aliases went; this
    test is why they must not come back.
    """
    pd.DataFrame({
        "From Material": [OLD], "To Material": [NEW], "Quantity": [50],
        "Movement Date": ["2026-05-01"], "Movement Type": [309],
    }).to_csv(tmp_path / "transfer.csv", index=False)
    pd.DataFrame({
        "Material": [OLD, NEW], "Plant": ["DC-01"] * 2,
        "Unrestricted Stock": [150, 400],
    }).to_csv(tmp_path / "stock.csv", index=False)

    result = Intake(verbose=False).load_files(sorted(tmp_path.glob("*.csv")))
    assert "substitution" not in result.documents
    assert not result.supersessions.applied
    inv = result.frame("inventory")
    assert set(inv["sku"].astype(str)) >= {OLD, NEW}


def test_an_item_master_alone_merges_nothing_end_to_end(exports):
    """The whole run, with a follow-up-material column and no substitution list."""
    paths = [p for p in sorted(exports.glob("*.csv")) if "substitution" not in p.name]
    master = exports / "item_master.csv"
    pd.DataFrame({
        "Material": [OLD, NEW],
        "Vendor": ["ACME", "ACME"],
        "Planned Deliv. Time": [90, 90],
        "MOQ": [50, 50],
        "Follow-up Material": [NEW, None],
        "Discontinuation Date": ["2026-03-01", None],
    }).to_csv(master, index=False)

    result = Intake(verbose=False).load_files(paths + [master])
    assert result.documents["item_master"].doc_type == "item_master"

    assert not result.supersessions.applied
    assert [(p.old_sku, p.new_sku) for p in result.supersessions.proposals] == [(OLD, NEW)]
    assert OLD in set(result.frame("inventory")["sku"].astype(str))

    text = result.summary()
    assert "NOT applied" in text
    assert "old_sku,new_sku,relation" in text       # the list to hand back

    # And the plan must not report the signal as satisfied — nothing was merged.
    assert not result.plan.has("substitution_signal")


def test_a_thin_margin_route_reports_the_pairs_and_merges_nothing(exports, monkeypatch):
    """Identity is the one thing not rewritten on a guess."""
    from inventory_planning.ingest.intake import LoadedDocument
    monkeypatch.setattr(
        LoadedDocument, "route_uncertain",
        property(lambda self: self.doc_type == "substitution"),
    )
    result = Intake(verbose=False).load_files(sorted(exports.glob("*.csv")))

    assert not result.supersessions.applied
    assert [(p.old_sku, p.new_sku) for p in result.supersessions.pairs] == [(OLD, NEW)]
    assert OLD in set(result.frame("inventory")["sku"].astype(str))
    assert "NOTHING WAS MERGED" in result.summary()


def test_without_the_list_the_two_numbers_stay_apart(exports):
    """The control: the same four files, minus the declaration."""
    paths = [p for p in sorted(exports.glob("*.csv")) if "substitution" not in p.name]
    result = Intake(verbose=False).load_files(paths)

    inv = result.frame("inventory")
    assert set(inv["sku"]) >= {OLD, NEW}
    assert not result.supersessions.applied
    assert not result.plan.has("substitution_signal")
