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
import pathlib
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

    @pytest.mark.parametrize("stamp", ['{"schema_version": "v2"}',
                                       '{"schema_version": null}',
                                       '{"schema_version": {"a": 1}}'])
    def test_a_version_that_is_not_a_number_is_a_schema_error(self, tmp_path, stamp):
        """
        `int(found)` sat outside the guard, so these raised ValueError or TypeError
        past it — invisible to anything catching StoreSchemaError, a migration tool
        first of all. The existing unreadable-stamp test missed it by using text that
        fails JSON parsing one line earlier.
        """
        root = tmp_path / "store"
        FactStore(root)
        (root / "_schema_version.json").write_text(stamp, encoding="utf-8")
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


class TestBatchIdentity:
    """
    Deduping on the source bytes alone was wrong in both directions: it dropped a
    re-export of unchanged data at a later as-of date, and it dropped a re-run of the
    same file after a parameter change. The frame stored is the canonical one, and the
    source bytes do not determine it.
    """

    def test_the_same_file_at_a_later_as_of_date_is_a_new_observation(self, store, frame):
        """Open POs unchanged for a week is evidence, not a duplicate."""
        first = store.write_batch("open_po", frame, valid_time="2026-08-25",
                                  source_sha="abc", config_fingerprint="cfg1")
        second = store.write_batch("open_po", frame, valid_time="2026-09-01",
                                   source_sha="abc", config_fingerprint="cfg1")
        assert first is not None and second is not None
        assert [b["valid_time"] for b in store.batches("open_po")] == \
            ["2026-08-25", "2026-09-01"]

    def test_the_same_file_under_changed_parameters_is_different_content(self, store, frame):
        """A corrected FX rate makes the canonical frame different bytes."""
        first = store.write_batch("open_po", frame, valid_time="2026-08-25",
                                  source_sha="abc", config_fingerprint="fx_old")
        second = store.write_batch("open_po", frame, valid_time="2026-08-25",
                                   source_sha="abc", config_fingerprint="fx_new")
        assert first is not None and second is not None
        assert len(store.batches("open_po")) == 2

    def test_the_parameter_set_is_recorded_readably_not_only_hashed(self, store, frame):
        """So a batch can be joined back to the run manifest carrying the same value."""
        store.write_batch("open_po", frame, valid_time="2026-08-25",
                          source_sha="abc", config_fingerprint="cfg1")
        assert store.batches("open_po")[0]["config_fingerprint"] == "cfg1"

    def test_all_three_the_same_is_a_duplicate(self, store, frame):
        store.write_batch("open_po", frame, valid_time="2026-08-25",
                          source_sha="abc", config_fingerprint="cfg1")
        again = store.write_batch("open_po", frame, valid_time="2026-08-25",
                                  source_sha="abc", config_fingerprint="cfg1")
        assert again is None
        assert len(store.batches("open_po")) == 1


class TestFrameHash:
    def test_it_reads_past_the_first_fifty_rows(self):
        """It hashed `head(50)`, so a correction at row 200 was dropped as a duplicate."""
        import pandas as pd
        from inventory_planning.store.fact_store import _sha256_frame

        a = pd.DataFrame({"sku": [f"S{i}" for i in range(300)], "qty": [1.0] * 300})
        b = a.copy()
        b.loc[200, "qty"] = 99.0
        assert _sha256_frame(a) != _sha256_frame(b)

    def test_an_identical_frame_hashes_the_same(self):
        import pandas as pd
        from inventory_planning.store.fact_store import _sha256_frame

        a = pd.DataFrame({"sku": ["A", "B"], "qty": [1.0, 2.0]})
        assert _sha256_frame(a) == _sha256_frame(a.copy())

    def test_a_corrected_row_is_stored_rather_than_swallowed(self, store):
        """The end of the path: no source_sha, so the frame hash is the identity."""
        import pandas as pd

        a = pd.DataFrame({"sku": [f"S{i}" for i in range(300)], "qty": [1.0] * 300})
        b = a.copy()
        b.loc[200, "qty"] = 99.0
        store.write_batch("inventory", a, valid_time="2026-08-25")
        assert store.write_batch("inventory", b, valid_time="2026-08-25") is not None


class TestABatchIsWholeOrAbsent:
    def test_a_failed_write_leaves_nothing_behind(self, store, frame, monkeypatch):
        """
        The file used to be written under its final name before the ledger line was
        appended, so a write that failed partway left a truncated parquet no line
        referred to — invisible to `batches()` and picked up as data by anything that
        globs the facts directory instead.
        """
        def boom(*args, **kwargs):
            raise OSError("No space left on device")

        monkeypatch.setattr(type(frame), "to_parquet", boom, raising=False)
        with pytest.raises(OSError):
            store.write_batch("inventory", frame, valid_time="2026-08-25")

        assert store.batches("inventory") == []
        assert list((store.facts_dir / "inventory").glob("*")) == []

    def test_a_successful_write_leaves_no_temporary(self, store, frame):
        batch = store.write_batch("inventory", frame, valid_time="2026-08-25")
        written = sorted((store.facts_dir / "inventory").iterdir())
        assert [f.name for f in written] == [f"{batch.batch_id}.parquet"]


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

    def test_it_resolves_without_opening_the_store(self, tmp_path):
        """
        Path resolution cannot fail; only the schema check can. A store this code
        refuses to open used to send snapshots back to `output_dir.parent / history`,
        reopening the split across exactly the runs where the store was broken.
        """
        from inventory_planning.store.fact_store import history_root

        root = tmp_path / "store"
        FactStore(root)
        (root / "_schema_version.json").write_text(json.dumps({"schema_version": 99}),
                                                   encoding="utf-8")
        with pytest.raises(StoreSchemaError):
            FactStore(root)
        assert history_root(root) == root / "history"

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


class TestWhatTheRunRetains:
    """
    The store has to hold what the source said, not what planning needed.

    `run_planning` collapses the inventory to one row per SKU before it does anything
    else — correctly, since every join downstream is on `sku` alone. Retaining that
    collapsed frame put 140 units in location `01` when 100 were sellable in `01` and
    40 were quarantined in `02`, against a contract whose natural key is
    `[sku, location_id]`. Nothing later can undo it: the quantities are summed and only
    the location *codes* survive, as a string.
    """

    def test_the_inventory_is_retained_before_its_locations_are_collapsed(self, tmp_path):
        from inventory_planning.orchestrator import InventoryPlanner

        captured = {}
        planner = InventoryPlanner(output_dir=tmp_path / "out", interactive=False,
                                   store_root=tmp_path / "store")
        planner._shadow_write = lambda frames, valid_time: captured.update(frames)

        root = pathlib.Path(__file__).resolve().parents[1]
        sample = root / "sample_data"
        sales, _ = planner.load_sales_history(sample / "sales_history.csv")
        po_hist, _ = planner.load_po_history(sample / "po_history.csv")
        open_so, _ = planner.load_open_so(sample / "open_so.csv")
        open_po, _ = planner.load_open_po(sample / "open_po.csv")
        inventory, _ = planner.load_inventory(sample / "inventory.csv")

        # Same stock, split across a sellable location and a quarantine one.
        quarantined = inventory.copy()
        quarantined["location_id"] = "02"
        quarantined["qty_on_hand"] = 7.0
        inventory = inventory.assign(location_id="01")
        two_location = pd.concat([inventory, quarantined], ignore_index=True)

        planner.run_planning(sales, po_hist, open_so, open_po, two_location)

        retained = captured["inventory"]
        assert len(retained) == len(two_location)
        assert set(retained["location_id"]) == {"01", "02"}
        assert (retained[retained["location_id"] == "02"]["qty_on_hand"] == 7.0).all()
