"""
A maintenance plan pinned to the batches it was read against.

INTERFACE.md §7's third seam, the half that bites today. `store restate` printed a plan
and then, on `--apply`, **selected again from scratch** — so a predicate that matched
1,188 batches when it was read could match 1,191 when it was carried out, and the three
nobody approved were changed in silence. The store is written to continuously by shadow
write, so this is not a theoretical window.

The central test is `test_a_batch_that_landed_after_the_plan_is_not_touched`. Everything
else is a way of getting the same property wrong:

  - pinning the *selector* instead of the selection, which digests identically at both
    moments and checks nothing
  - pinning the whole ledger, which no plan would survive being read
  - re-selecting and comparing, which still rests on the selector behaving twice
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

pytest.importorskip("pyarrow", reason="the store writes parquet")

import pandas as pd                                                  # noqa: E402

from inventory_planning.store.__main__ import main                   # noqa: E402
from inventory_planning.store.fact_store import FactStore            # noqa: E402
from inventory_planning.store.ledger import (                        # noqa: E402
    LAYER_CANONICAL, LAYER_PREPARED, LAYER_UNKNOWN,
)
from inventory_planning.store.migration import (                     # noqa: E402
    MigrationPlan, OP_RESTATE, OP_VOID, PlanRefused, plan_basis,
)


@pytest.fixture
def store(tmp_path):
    return FactStore(tmp_path / "store")


@pytest.fixture
def frame():
    return pd.DataFrame({"sku": ["A-1", "A-2"], "location_id": ["01", "01"],
                         "qty_on_hand": [1, 2]})


def _land(store, frame, n=5):
    return [store.write_batch("inventory", frame, valid_time=f"2026-08-0{i + 1}")
            for i in range(n)]


def _run(store, *args):
    return main(["--store", str(store.root), *args])


class TestWhatThePlanIsPinnedBy:

    def test_the_selection_decides_the_basis_not_the_selector(self, store, frame):
        """
        The failure this seam closes, stated as a test. Two plans from the same selector
        over different batches must not share a basis — pinning the selector arguments
        would give them one, and would be a safety check that checks nothing.
        """
        _land(store, frame, 4)
        entries = store.ledger.batches(include_void=True)
        assert len(entries) == 4

        a = MigrationPlan.build(OP_RESTATE, entries[:3], to=LAYER_PREPARED,
                                selector="--doc-type inventory")
        b = MigrationPlan.build(OP_RESTATE, entries, to=LAYER_PREPARED,
                                selector="--doc-type inventory")
        assert a.selector == b.selector          # the same selector, both times
        assert a.basis != b.basis                # and a different thing approved

    def test_order_does_not_change_the_basis(self, store, frame):
        """The set is the content. A selector returning them reversed changed nothing."""
        _land(store, frame, 4)
        entries = store.ledger.batches(include_void=True)
        forward = MigrationPlan.build(OP_RESTATE, entries, to=LAYER_PREPARED)
        reverse = MigrationPlan.build(OP_RESTATE, list(reversed(entries)),
                                      to=LAYER_PREPARED)
        assert forward.basis == reverse.basis

    def test_the_operation_and_target_are_in_it(self):
        """
        One approved selection must not be carried out as a different operation over it.
        """
        ids = ["b1", "b2"]
        assert plan_basis(OP_RESTATE, ids, LAYER_PREPARED) \
            != plan_basis(OP_RESTATE, ids, LAYER_CANONICAL)
        assert plan_basis(OP_RESTATE, ids, LAYER_PREPARED) != plan_basis(OP_VOID, ids)

    def test_an_edited_plan_file_is_refused_by_its_own_basis(self, store, frame,
                                                             tmp_path):
        _land(store, frame, 3)
        plan = MigrationPlan.build(OP_RESTATE, store.ledger.batches(include_void=True),
                                   to=LAYER_PREPARED)
        path = plan.save(tmp_path / "p.json")
        body = json.loads(path.read_text(encoding="utf-8"))
        body["batch_ids"].append("smuggled-in")
        path.write_text(json.dumps(body), encoding="utf-8")

        with pytest.raises(PlanRefused, match="edited since it was written"):
            MigrationPlan.load(path)


class TestApplyingReplaysThePlan:
    """Never a fresh selection. That is what makes "approve this change" literal."""

    def test_a_batch_that_landed_after_the_plan_is_not_touched(self, store, frame,
                                                              tmp_path):
        """
        The defect, end to end. Before the plan file, `--apply` re-ran the selector and
        swept up whatever had arrived in between.
        """
        _land(store, frame, 5)
        plan_path = tmp_path / "plan.json"
        assert _run(store, "restate", "--doc-type", "inventory",
                    "--layer-to", LAYER_PREPARED, "--plan-out", str(plan_path)) == 0

        latecomer = store.write_batch("inventory", frame, valid_time="2026-08-09")

        assert _run(store, "restate", "--layer-to", LAYER_PREPARED,
                    "--plan", str(plan_path), "--apply",
                    "--by", "tests", "--reason", "the writer's meaning changed") == 0

        layers = {b["batch_id"]: b["frame_layer"]
                  for b in store.ledger.batches(include_void=True)}
        assert layers[latecomer.batch_id] == LAYER_CANONICAL
        assert sum(1 for v in layers.values() if v == LAYER_PREPARED) == 5

    def test_a_batch_restated_by_someone_else_is_a_refusal(self, store, frame, tmp_path):
        """Applying would overwrite a correction nobody in this plan knew about."""
        _land(store, frame, 3)
        plan_path = tmp_path / "plan.json"
        _run(store, "restate", "--doc-type", "inventory", "--layer-to", LAYER_PREPARED,
             "--plan-out", str(plan_path))

        target = MigrationPlan.load(plan_path).batch_ids[0]
        store.ledger.restate(target, LAYER_UNKNOWN, reason="got there first",
                             by="someone-else")

        assert _run(store, "restate", "--layer-to", LAYER_PREPARED,
                    "--plan", str(plan_path), "--apply",
                    "--by", "tests", "--reason", "r") == 1
        layers = {b["batch_id"]: b["frame_layer"]
                  for b in store.ledger.batches(include_void=True)}
        assert layers[target] == LAYER_UNKNOWN          # nothing was written

    def test_a_plan_from_another_store_is_a_refusal(self, store, frame, tmp_path):
        plan = MigrationPlan(op=OP_RESTATE, batch_ids=["not-from-here"],
                             to=LAYER_PREPARED,
                             assumed={"not-from-here": {"frame_layer": LAYER_CANONICAL,
                                                        "status": "active"}})
        plan.basis = plan_basis(plan.op, plan.batch_ids, plan.to)
        path = plan.save(tmp_path / "p.json")
        _land(store, frame, 2)

        assert _run(store, "restate", "--layer-to", LAYER_PREPARED, "--plan", str(path),
                    "--apply", "--by", "tests", "--reason", "r") == 1

    def test_a_plan_cannot_be_carried_out_as_a_different_operation(self, store, frame,
                                                                   tmp_path):
        _land(store, frame, 2)
        plan_path = tmp_path / "plan.json"
        _run(store, "restate", "--doc-type", "inventory", "--layer-to", LAYER_PREPARED,
             "--plan-out", str(plan_path))

        with pytest.raises(SystemExit):
            _run(store, "void", "--plan", str(plan_path), "--apply",
                 "--by", "tests", "--reason", "r")

    def test_a_plan_cannot_be_carried_out_to_a_different_layer(self, store, frame,
                                                               tmp_path):
        _land(store, frame, 2)
        plan_path = tmp_path / "plan.json"
        _run(store, "restate", "--doc-type", "inventory", "--layer-to", LAYER_PREPARED,
             "--plan-out", str(plan_path))

        with pytest.raises(SystemExit):
            _run(store, "restate", "--layer-to", LAYER_UNKNOWN, "--plan", str(plan_path),
                 "--apply", "--by", "tests", "--reason", "r")


class TestWhatStillWorksWithoutAPlan:
    """
    The rule, and the one exemption. A predicate over a moving store needs the plan; a
    list of ids does not, because it names the batches rather than describing them.
    """

    def test_a_selector_with_apply_and_no_plan_is_refused(self, store, frame):
        _land(store, frame, 3)
        with pytest.raises(SystemExit):
            _run(store, "restate", "--doc-type", "inventory",
                 "--layer-to", LAYER_PREPARED, "--apply",
                 "--by", "tests", "--reason", "r")
        assert all(b["frame_layer"] == LAYER_CANONICAL
                   for b in store.ledger.batches(include_void=True))

    def test_selecting_by_batch_id_alone_may_still_apply_directly(self, store, frame):
        batches = _land(store, frame, 3)
        assert _run(store, "restate", "--batch", batches[0].batch_id,
                    "--layer-to", LAYER_PREPARED, "--apply",
                    "--by", "tests", "--reason", "named, not described") == 0
        layers = {b["batch_id"]: b["frame_layer"]
                  for b in store.ledger.batches(include_void=True)}
        assert layers[batches[0].batch_id] == LAYER_PREPARED
        assert layers[batches[1].batch_id] == LAYER_CANONICAL

    def test_an_id_mixed_with_a_predicate_is_not_exempt(self, store, frame):
        """The predicate is what can drift, and mixing one in does not make it safe."""
        batches = _land(store, frame, 3)
        with pytest.raises(SystemExit):
            _run(store, "restate", "--batch", batches[0].batch_id,
                 "--doc-type", "inventory", "--layer-to", LAYER_PREPARED, "--apply",
                 "--by", "tests", "--reason", "r")

    def test_a_plan_still_needs_a_name_and_a_reason(self, store, frame, tmp_path):
        _land(store, frame, 2)
        plan_path = tmp_path / "plan.json"
        _run(store, "restate", "--doc-type", "inventory", "--layer-to", LAYER_PREPARED,
             "--plan-out", str(plan_path))
        with pytest.raises(SystemExit):
            _run(store, "restate", "--layer-to", LAYER_PREPARED, "--plan",
                 str(plan_path), "--apply", "--reason", "r")


class TestItSaysHowTheStoreMoved:
    """
    A refusal a person cannot act on sends them to re-run the same command and hope.
    Each difference names what moved and what to do about it.
    """

    def test_the_drift_names_the_batches(self, store, frame, tmp_path):
        _land(store, frame, 3)
        plan = MigrationPlan.build(OP_RESTATE, store.ledger.batches(include_void=True),
                                   to=LAYER_PREPARED)
        target = plan.batch_ids[0]
        store.ledger.restate(target, LAYER_UNKNOWN, reason="r", by="other")

        drift = plan.drift(store.ledger)
        assert len(drift) == 1
        assert target in drift[0]
        assert "overwrite somebody else's correction" in drift[0]

    def test_an_unmoved_store_drifts_not_at_all(self, store, frame):
        _land(store, frame, 3)
        plan = MigrationPlan.build(OP_RESTATE, store.ledger.batches(include_void=True),
                                   to=LAYER_PREPARED)
        assert plan.drift(store.ledger) == []

    def test_a_widened_selector_is_reported_and_not_acted_on(self, store, frame):
        """
        The plan's own list is carried out either way. The person is told because it
        usually means the plan is older than they think.
        """
        _land(store, frame, 2)
        plan = MigrationPlan.build(OP_RESTATE, store.ledger.batches(include_void=True),
                                   to=LAYER_PREPARED)
        store.write_batch("inventory", frame, valid_time="2026-08-09")

        note = plan.rescan(store.ledger.batches(include_void=True))
        assert note is not None
        assert "Only the 2 in the plan will be changed" in note
