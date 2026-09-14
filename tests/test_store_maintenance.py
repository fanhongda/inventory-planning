"""
Restating and voiding — correcting what a batch says about itself, without rewriting it.

The case: a writer's meaning changed while the store went on accumulating, so batches
written before the change describe a layer they never named. Their frames are right and
their marking is wrong, and the marking is what a reader uses to decide whether two
batches may be added together.
"""

import json

import pandas as pd
import pytest

from inventory_planning.store import __main__ as maintenance
from inventory_planning.store.fact_store import FactStore
from inventory_planning.store.ledger import (
    LAYER_CANONICAL, LAYER_PREPARED, LAYER_UNKNOWN, STATUS_VOID,
)


@pytest.fixture
def store(tmp_path):
    st = FactStore(tmp_path / "store")
    for i, source in enumerate(("real.xlsx", "real.xlsx", "sample_data.csv")):
        st.write_batch(doc_type="inventory",
                       frame=pd.DataFrame([{"sku": f"A{i}", "location_id": "DC-01",
                                            "qty_on_hand": i}]),
                       valid_time=f"2024-0{i + 1}-01", source_name=source,
                       source_sha=f"sha{i}", written_by="tests")
    return st


def _run(store, *argv):
    return maintenance.main(["--store", str(store.root), *argv])


class TestARestatementIsAppendedNotApplied:

    def test_it_changes_what_batches_reports(self, store):
        ledger = store.ledger
        before = {b["batch_id"]: b["frame_layer"] for b in ledger.batches()}
        assert set(before.values()) == {LAYER_CANONICAL}

        target = next(iter(before))
        ledger.restate(target, LAYER_PREPARED, reason="written by the old writer",
                       by="tests")
        after = {b["batch_id"]: b["frame_layer"] for b in ledger.batches()}
        assert after[target] == LAYER_PREPARED
        assert [b for b in ledger.batches() if b["batch_id"] == target][0][
            "frame_layer_restated"] is True

    def test_the_original_line_still_stands(self, store):
        ledger = store.ledger
        target = ledger.batches()[0]["batch_id"]
        ledger.restate(target, LAYER_PREPARED, reason="why", by="tests")

        lines = [json.loads(l) for l in
                 ledger.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        originals = [l for l in lines if l.get("batch_id") == target
                     and l.get("op") is None]
        assert len(originals) == 1
        assert originals[0]["frame_layer"] == LAYER_CANONICAL
        restatement = [l for l in lines if l.get("op") == "restate"][0]
        assert restatement["restated_by"] == "tests" and restatement["reason"] == "why"

    def test_a_restatement_can_itself_be_restated(self, store):
        ledger = store.ledger
        target = ledger.batches()[0]["batch_id"]
        ledger.restate(target, LAYER_PREPARED, reason="first", by="tests")
        ledger.restate(target, LAYER_CANONICAL, reason="wrong, undo", by="tests")
        assert {b["batch_id"]: b["frame_layer"] for b in ledger.batches()}[target] \
            == LAYER_CANONICAL

    def test_it_does_not_touch_the_parquet(self, store):
        target = store.ledger.batches()[0]
        path = store.root / target["path"]
        before = path.read_bytes()
        store.ledger.restate(target["batch_id"], LAYER_PREPARED, reason="r", by="t")
        assert path.read_bytes() == before

    def test_restatements_do_not_appear_as_batches(self, store):
        ledger = store.ledger
        count = len(ledger.batches())
        ledger.restate(ledger.batches()[0]["batch_id"], LAYER_PREPARED, reason="r",
                       by="t")
        assert len(ledger.batches()) == count


class TestNothingIsWrittenWithoutApply:

    def test_a_plan_leaves_the_ledger_alone(self, store, capsys):
        before = store.ledger.path.read_text(encoding="utf-8")
        _run(store, "restate", "--source", "real.xlsx", "--layer-to", LAYER_PREPARED)
        assert store.ledger.path.read_text(encoding="utf-8") == before
        assert "Nothing was written" in capsys.readouterr().out

    def test_the_plan_states_the_counts_that_are_the_diagnostic(self, store, capsys):
        _run(store, "restate", "--source", "real.xlsx", "--layer-to", LAYER_PREPARED)
        out = capsys.readouterr().out
        assert "2 batch(es) selected, to restate" in out
        assert "inventory 2" in out and "real.xlsx (2)" in out

    def test_apply_needs_an_account_of_who_and_why(self, store):
        with pytest.raises(SystemExit):
            _run(store, "void", "--source", "sample_data", "--apply")
        with pytest.raises(SystemExit):
            _run(store, "void", "--source", "sample_data", "--apply", "--by", "t")

    def test_no_selector_is_refused_rather_than_meaning_everything(self, store):
        """
        A maintenance command that defaults to the whole store is how the command
        becomes the thing that needed maintenance.
        """
        with pytest.raises(SystemExit):
            _run(store, "void", "--apply", "--by", "t", "--reason", "r")


class TestApplying:

    def test_restating_by_load_time_marks_only_the_older_batches(self, store):
        cutoff = store.ledger.batches()[1]["transaction_time"]
        _run(store, "restate", "--loaded-before", cutoff, "--layer-to", LAYER_PREPARED,
             "--apply", "--by", "tests", "--reason", "written by the old writer")
        layers = [b["frame_layer"] for b in store.ledger.batches()]
        assert layers.count(LAYER_PREPARED) == 1
        assert layers.count(LAYER_CANONICAL) == 2

    def test_voiding_by_source_withdraws_only_those(self, store):
        _run(store, "void", "--source", "sample_data", "--apply", "--by", "tests",
             "--reason", "synthetic data in a store of real facts")
        assert len(store.ledger.batches()) == 2
        withdrawn = [b for b in store.ledger.batches(include_void=True)
                     if b["status"] == STATUS_VOID]
        assert len(withdrawn) == 1
        assert withdrawn[0]["source_name"] == "sample_data.csv"

    def test_a_restated_layer_is_what_a_reading_sees(self, store):
        """The point of the whole exercise: one layer, so a reading stops refusing."""
        from inventory_planning.store.query import FactQuery, MixedLayers

        query = FactQuery(store.root)
        cutoff = store.ledger.batches()[1]["transaction_time"]
        _run(store, "restate", "--loaded-before", cutoff, "--layer-to", LAYER_PREPARED,
             "--apply", "--by", "tests", "--reason", "r")
        with pytest.raises(MixedLayers):
            query.current("inventory")

        _run(store, "restate", "--layer", LAYER_CANONICAL, "--layer-to", LAYER_PREPARED,
             "--apply", "--by", "tests", "--reason", "all of it predates the change")
        assert query.select("inventory").layers == [LAYER_PREPARED]
        assert len(query.current("inventory")) == 3


class TestShow:

    def test_it_reports_layers_and_restatements(self, store, capsys):
        store.ledger.restate(store.ledger.batches()[0]["batch_id"], LAYER_PREPARED,
                             reason="r", by="t")
        _run(store, "show")
        out = capsys.readouterr().out
        assert "inventory" in out and "prepared 1" in out and "1 restated" in out
