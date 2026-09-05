"""
Reading the store — that a position is a position, and an as-of question gets an as-of
answer.

The defect these guard is not an exception. A read with no as-of predicate returns one
purchase order once per load, the inbound quantity comes out multiplied by the number of
times the file was imported, nothing raises, and the number merely looks high.
"""

import json

import pandas as pd
import pytest

from inventory_planning.store.fact_store import FactStore
from inventory_planning.store.ledger import (
    LAYER_CANONICAL, LAYER_PREPARED, LAYER_UNKNOWN,
)
from inventory_planning.store.query import (
    BATCH_COLUMN, VALID_COLUMN, FactQuery, KeyIncomplete, MixedLayers,
)

pytest.importorskip("duckdb", reason="the read path is an optional extra")


@pytest.fixture
def store(tmp_path):
    return FactStore(tmp_path / "store")


@pytest.fixture
def query(store):
    return FactQuery(store.root)


def _write(store, valid_time, rows, doc_type="inventory"):
    return store.write_batch(
        doc_type=doc_type, frame=pd.DataFrame(rows), valid_time=valid_time,
        source_name=f"stock_{valid_time}.csv", source_sha=f"sha-{valid_time}",
        written_by="tests")


@pytest.fixture
def two_snapshots(store):
    _write(store, "2024-06-01", [{"sku": "A", "location_id": "DC-01", "qty_on_hand": 10},
                                 {"sku": "B", "location_id": "DC-01", "qty_on_hand": 20}])
    _write(store, "2024-07-01", [{"sku": "A", "location_id": "DC-01", "qty_on_hand": 15},
                                 {"sku": "C", "location_id": "DC-01", "qty_on_hand": 5}])
    return store


class TestHistoryIsEveryObservation:

    def test_it_returns_every_row_of_every_batch(self, query, two_snapshots):
        frame = query.history("inventory")
        assert len(frame) == 4
        assert frame[BATCH_COLUMN].nunique() == 2

    def test_summing_it_double_counts_and_that_is_the_point(self, query, two_snapshots):
        """
        A is observed twice, at 10 and 15. History says 50, the position is 40. This is
        the failure the two methods exist to keep apart, so it is asserted rather than
        avoided.
        """
        assert query.history("inventory")["qty_on_hand"].sum() == 50
        assert query.current("inventory")["qty_on_hand"].sum() == 40


class TestCurrentIsOneRowPerKey:

    def test_the_newest_observation_of_each_key_wins(self, query, two_snapshots):
        frame = query.current("inventory").set_index("sku")
        assert sorted(frame.index) == ["A", "B", "C"]
        assert frame.loc["A", "qty_on_hand"] == 15

    def test_every_row_says_when_it_was_observed(self, query, two_snapshots):
        """A value carried forward from an older batch has to be visible as one."""
        frame = query.current("inventory").set_index("sku")
        assert frame.loc["B", VALID_COLUMN] == "2024-06-01"
        assert frame.loc["A", VALID_COLUMN] == "2024-07-01"

    def test_an_as_of_date_hides_what_was_not_known_then(self, query, two_snapshots):
        frame = query.current("inventory", as_of="2024-06-15").set_index("sku")
        assert sorted(frame.index) == ["A", "B"]
        assert frame.loc["A", "qty_on_hand"] == 10

    def test_it_refuses_rather_than_reducing_on_a_partial_key(self, store, query):
        """
        Reducing on a partial key collapses rows that are genuinely different — an
        understatement that looks like a clean answer, and as wrong as not reducing.
        """
        _write(store, "2024-07-01", [{"sku": "A", "qty_on_hand": 1}])
        with pytest.raises(KeyIncomplete) as raised:
            query.current("inventory")
        assert "location_id" in str(raised.value)

    def test_a_document_with_no_contract_says_so(self, store, query):
        _write(store, "2024-07-01", [{"sku": "A"}], doc_type="warehouse_layout")
        with pytest.raises(KeyIncomplete):
            query.current("warehouse_layout")


class TestLatestIsTheNewestBatchOnly:

    def test_a_key_missing_from_the_newest_file_is_gone(self, query, two_snapshots):
        frame = query.latest("inventory")
        assert sorted(frame["sku"]) == ["A", "C"]
        assert frame["qty_on_hand"].sum() == 20

    def test_it_needs_no_natural_key(self, store, query):
        """One batch is one observation by construction."""
        _write(store, "2024-07-01", [{"sku": "A", "qty_on_hand": 1}])
        assert len(query.latest("inventory")) == 1

    def test_the_disagreement_with_current_is_reported_as_a_number(self, query,
                                                                  two_snapshots):
        carried = query.carried_forward("inventory")
        assert carried["newest_valid_time"] == "2024-07-01"
        assert carried["rows"] == 3 and carried["carried"] == 1
        assert carried["carried_share"] == pytest.approx(1 / 3)


class TestSelectionAndFilters:

    def test_a_voided_batch_is_not_read(self, store, query, two_snapshots):
        newest = query.select("inventory").batches[-1]["batch_id"]
        store.ledger.void(newest, reason="wrong month", by="tests")
        frame = query.current("inventory").set_index("sku")
        assert sorted(frame.index) == ["A", "B"]
        assert frame.loc["A", "qty_on_hand"] == 10

    def test_known_at_reconstructs_what_was_believed_then(self, query, two_snapshots):
        """
        The second timestamp. An extract loaded today may describe last week, and only
        transaction time can answer what was believed before it arrived.

        Rests on the two batches having distinguishable load times, which is why
        `transaction_time` is recorded to the microsecond: at millisecond resolution
        this passed alone and failed inside a full run.
        """
        batches = query.select("inventory").batches
        before_second = batches[0]["transaction_time"]
        frame = query.current("inventory", known_at=before_second)
        assert sorted(frame["sku"]) == ["A", "B"]

    def test_filters_are_by_value_and_by_membership(self, query, two_snapshots):
        assert sorted(query.current("inventory", where={"sku": "A"})["sku"]) == ["A"]
        assert sorted(query.current("inventory",
                                    where={"sku": ["A", "C"]})["sku"]) == ["A", "C"]

    def test_an_empty_store_returns_an_empty_frame_not_an_error(self, query):
        assert query.doc_types() == []
        assert query.current("inventory").empty
        assert query.select("inventory").describe().endswith("no batch matches")


class TestSchemaDrift:

    def test_a_batch_that_gained_a_column_unions_by_name(self, store, query):
        """
        Positional union would shear the two schemas together and land every value of
        the new column in the wrong field.
        """
        _write(store, "2024-06-01", [{"sku": "A", "location_id": "DC-01",
                                      "qty_on_hand": 10}])
        _write(store, "2024-07-01", [{"sku": "A", "location_id": "DC-01",
                                      "qty_on_hand": 15, "qty_in_transit": 7}])
        frame = query.current("inventory").set_index("sku")
        assert frame.loc["A", "qty_on_hand"] == 15
        assert frame.loc["A", "qty_in_transit"] == 7


class TestLayersAreNotBlended:
    """
    Two writers put two different things in one store before anyone noticed. The shadow
    write stored frames the bridge had already converted into the reporting currency;
    the interface stored the canonical frame the adapter produced. The money columns of
    the two are not the same measure, and nothing said so.
    """

    def test_a_batch_records_which_layer_it_holds(self, store, query):
        _write(store, "2024-07-01", [{"sku": "A", "location_id": "DC-01",
                                      "qty_on_hand": 1}])
        assert query.select("inventory").layers == [LAYER_CANONICAL]
        assert query.select("inventory").mixed is False

    def test_a_batch_written_before_the_distinction_reads_back_as_unknown(self, store,
                                                                          query):
        """Not as a guess about what it holds — the store has 389 of these."""
        _write(store, "2024-07-01", [{"sku": "A", "location_id": "DC-01",
                                      "qty_on_hand": 1}])
        lines = store.ledger.path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[0])
        del entry["frame_layer"]
        store.ledger.path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        assert query.select("inventory").layers == [LAYER_UNKNOWN]

    def test_a_reading_across_two_layers_is_refused_with_the_cutoff(self, store, query):
        store.write_batch(doc_type="inventory",
                          frame=pd.DataFrame([{"sku": "A", "location_id": "DC-01",
                                               "qty_on_hand": 10}]),
                          valid_time="2024-06-01", source_name="june.csv",
                          source_sha="june", written_by="tests",
                          frame_layer=LAYER_PREPARED)
        _write(store, "2024-07-01", [{"sku": "A", "location_id": "DC-01",
                                      "qty_on_hand": 15}])

        with pytest.raises(MixedLayers) as raised:
            query.current("inventory")
        message = str(raised.value)
        assert "canonical" in message and "prepared" in message
        assert "2024-07-01" in message          # the cutoff that stays inside one layer

    def test_narrowing_to_one_layer_reads_normally(self, store, query):
        store.write_batch(doc_type="inventory",
                          frame=pd.DataFrame([{"sku": "A", "location_id": "DC-01",
                                               "qty_on_hand": 10}]),
                          valid_time="2024-06-01", source_name="june.csv",
                          source_sha="june", written_by="tests",
                          frame_layer=LAYER_PREPARED)
        _write(store, "2024-07-01", [{"sku": "A", "location_id": "DC-01",
                                      "qty_on_hand": 15}])

        older = query.current("inventory", as_of="2024-06-15")
        assert older["qty_on_hand"].tolist() == [10]
        assert query.select("inventory", as_of="2024-06-15").layers == [LAYER_PREPARED]
