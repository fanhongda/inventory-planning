"""
The fact store.

Two things are being pinned. That the store keeps what a CSV would destroy — the reason
it is parquet and not the obvious thing — and that it cannot take the run down with it.
Phase one writes and nothing reads, so every failure here has to degrade to a warning
and a plan that still comes out.

The third thing, less obvious: that "editing a fact" is always a batch-level operation.
A void appends a line rather than rewriting one, and re-loading the same bytes is a
no-op rather than a second copy.
"""

import json
from datetime import datetime

import pandas as pd
import pytest

from inventory_planning.feedback.snapshot import SnapshotSaver
from inventory_planning.store import (
    ENV_VAR, BatchLedger, FactStore, SCHEMA_VERSION, StoreSchemaError,
    default_store_root, resolve_store_root,
)
from inventory_planning.store.location import inside_repo, warn_if_inside_repo


@pytest.fixture
def store(tmp_path):
    return FactStore(tmp_path / "store")


@pytest.fixture
def frame():
    """Leading zeros and a real date — the two things a CSV round trip loses."""
    return pd.DataFrame({
        "sku": ["000000000000123456", "0012345"],
        "qty_on_hand": [100.0, -50.0],
        "snapshot_date": pd.to_datetime(["2026-08-25", "2026-08-25"]),
    })


class TestWhereItLives:
    def test_an_argument_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR, str(tmp_path / "from_env"))
        root, source = resolve_store_root(tmp_path / "explicit")
        assert root == tmp_path / "explicit"
        assert source == "argument"

    def test_the_env_var_beats_the_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_VAR, str(tmp_path / "from_env"))
        root, source = resolve_store_root()
        assert root == tmp_path / "from_env"
        assert ENV_VAR in source

    def test_it_falls_back_outside_the_repository(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        root, source = resolve_store_root()
        assert root == default_store_root()
        assert source == "default"
        assert not inside_repo(root)

    def test_a_store_under_the_working_tree_is_reported(self):
        """Allowed — someone may be running a throwaway — but never silently."""
        from inventory_planning.store.location import repo_root
        root = repo_root()
        if root is None:
            pytest.skip("not a git checkout")
        warning = warn_if_inside_repo(root / "output" / "store")
        assert warning is not None
        assert "git clean" in warning

    def test_a_store_outside_it_is_not(self, tmp_path):
        assert warn_if_inside_repo(tmp_path / "store") is None


class TestSchemaVersion:
    def test_a_new_store_is_stamped(self, store):
        assert json.loads(store.schema_path.read_text(encoding="utf-8"))["schema_version"] == SCHEMA_VERSION

    def test_a_future_schema_is_refused_rather_than_guessed(self, tmp_path):
        root = tmp_path / "store"
        FactStore(root)
        (root / "_schema_version.json").write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
        with pytest.raises(StoreSchemaError, match="v99"):
            FactStore(root)

    def test_an_unreadable_stamp_is_refused(self, tmp_path):
        root = tmp_path / "store"
        FactStore(root)
        (root / "_schema_version.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(StoreSchemaError):
            FactStore(root)


class TestWriting:
    def test_the_types_survive(self, store, frame):
        """The whole reason this is not a directory of CSVs."""
        batch = store.write_batch("inventory", frame, valid_time="2026-08-25")
        back = store.read_batch(batch.batch_id)
        assert list(back["sku"]) == ["000000000000123456", "0012345"]
        assert str(back["snapshot_date"].dtype).startswith("datetime64")

    def test_valid_time_is_required_and_never_guessed(self, store, frame):
        with pytest.raises(ValueError, match="valid_time"):
            store.write_batch("inventory", frame, valid_time=None)

    def test_the_two_timestamps_are_recorded_apart(self, store, frame):
        """An extract loaded today routinely describes a date long past."""
        batch = store.write_batch("inventory", frame, valid_time="2024-01-15")
        assert batch.valid_time == "2024-01-15"
        assert batch.transaction_time.startswith(datetime.now().strftime("%Y-%m-%d"))

    def test_the_same_bytes_are_not_stored_twice(self, store, frame):
        first = store.write_batch("inventory", frame, valid_time="2026-08-25",
                                  source_sha="abc123")
        again = store.write_batch("inventory", frame, valid_time="2026-08-25",
                                  source_sha="abc123")
        assert first is not None
        assert again is None
        assert len(store.batches("inventory")) == 1
        assert any("already stored" in n for n in store.notes)

    def test_the_same_file_under_a_different_document_type_is_a_different_batch(
        self, store, frame
    ):
        store.write_batch("inventory", frame, valid_time="2026-08-25", source_sha="abc")
        other = store.write_batch("open_po", frame, valid_time="2026-08-25", source_sha="abc")
        assert other is not None

    def test_an_empty_frame_writes_nothing(self, store):
        assert store.write_batch("inventory", pd.DataFrame(), valid_time="2026-08-25") is None

    def test_the_key_verdict_travels_with_the_batch(self, store, frame):
        batch = store.write_batch("inventory", frame, valid_time="2026-08-25",
                                  key_verdict="degraded", storable=False)
        assert store.batches("inventory")[0]["storable"] is False
        assert batch.key_verdict == "degraded"


class TestTheLedgerIsAppendOnly:
    def test_voiding_appends_rather_than_rewrites(self, store, frame):
        batch = store.write_batch("inventory", frame, valid_time="2026-08-25")
        before = store.ledger.path.read_text(encoding="utf-8")
        store.ledger.void(batch.batch_id, reason="wrong extract", by="tester")
        after = store.ledger.path.read_text(encoding="utf-8")
        assert after.startswith(before)

    def test_a_voided_batch_drops_out_of_the_listing(self, store, frame):
        batch = store.write_batch("inventory", frame, valid_time="2026-08-25")
        store.ledger.void(batch.batch_id, reason="wrong extract")
        assert store.batches("inventory") == []
        assert len(store.ledger.batches("inventory", include_void=True)) == 1

    def test_the_data_of_a_voided_batch_is_still_there(self, store, frame):
        """Void is a statement about the batch, not an erasure of what it said."""
        batch = store.write_batch("inventory", frame, valid_time="2026-08-25")
        store.ledger.void(batch.batch_id)
        assert store.read_batch(batch.batch_id) is not None

    def test_a_truncated_final_line_does_not_lose_the_rest(self, tmp_path, frame):
        store = FactStore(tmp_path / "store")
        store.write_batch("inventory", frame, valid_time="2026-08-25")
        with open(store.ledger.path, "a", encoding="utf-8") as fh:
            fh.write('{"batch_id": "half\n')
        assert len(store.batches("inventory")) == 1

    def test_it_recognises_bytes_it_has_seen(self, store, frame):
        store.write_batch("inventory", frame, valid_time="2026-08-25", source_sha="deadbeef")
        assert BatchLedger(store.root).has_source("inventory", "deadbeef") is not None
        assert BatchLedger(store.root).has_source("inventory", "other") is None
        assert BatchLedger(store.root).has_source("inventory", "") is None


class TestHistoryHasOneHome:
    def test_it_hangs_off_the_store_not_the_output_directory(self, store):
        assert store.history_dir == store.root / "history"

    def test_the_saver_writes_where_it_is_told(self, tmp_path):
        out = tmp_path / "out"; out.mkdir()
        history = tmp_path / "elsewhere"
        path = SnapshotSaver().save({}, {}, out, history_root=history)
        assert history in path.parents

    def test_a_caller_that_passes_no_root_lands_where_it_used_to(self, tmp_path):
        """`feedback.loss` reads a snapshot by path; the old default has to hold."""
        out = tmp_path / "runs" / "out"; out.mkdir(parents=True)
        path = SnapshotSaver().save({}, {}, out)
        assert (tmp_path / "runs" / "history") in path.parents
