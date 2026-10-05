"""
Starting a run from the interface, and the three things that stop one.

The screen could change everything a run depends on and could not start one. A planner
signing a change to `transit_share_of_lt` watched it land in `config_changes.jsonl` and
then had to open a terminal to find out what it did — the screen that knew most about
the change was the one place its consequence could not be seen.

Most of what is asserted here is the refusals, because the refusals are the design. A
run that cannot be started is a message; a run started on documents that do not add up,
or past a gate, is a confident wrong answer with a workbook attached.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

pytest.importorskip("fastapi", reason="the HTTP surface is an optional extra")

from inventory_planning.api.runner import RunRefused, Runner  # noqa: E402
from inventory_planning.ingest.intake import Intake  # noqa: E402

SAMPLE = Path(__file__).parents[1] / "sample_data"


class TestReadingTheStoreMatchesReadingTheFolder:
    """
    `load_frames` exists so a run driven from the store goes through the same intake as
    one driven from a folder — the cross-document checks especially. SKU agreement, key
    shape and the capability plan decide whether a run may happen at all, and a second
    path into the pipeline that skipped them would be the one with no gate on it.
    """

    @staticmethod
    def _frames():
        import pandas as pd

        return [(pd.read_csv(p), p.name) for p in sorted(SAMPLE.glob("*.csv"))]

    def test_the_same_documents_are_found(self):
        by_file = Intake(verbose=False).load_files(sorted(SAMPLE.glob("*.csv")))
        by_frame = Intake(verbose=False).load_frames(self._frames())
        assert sorted(by_frame.documents) == sorted(by_file.documents)
        assert by_frame.can_run == by_file.can_run

    def test_the_cross_document_checks_run(self):
        """
        The capability plan is the one that decides, and it is built in the shared half.
        A frames path that returned documents and an empty plan would report `can_run`
        on a workspace holding one file.
        """
        one = Intake(verbose=False).load_frames(self._frames()[:1])
        assert one.can_run is False
        assert one.plan.missing_required


class TestTheThreeRefusals:

    @pytest.fixture
    def runner(self, tmp_path):
        from inventory_planning.api.app import Service

        return Runner(Service(config_dir=tmp_path / "config",
                              store_root=tmp_path / "store",
                              output_dir=tmp_path / "out"))

    def test_an_empty_store_refuses_before_anything_else(self, runner):
        with pytest.raises(RunRefused) as refused:
            runner.start()
        assert "nothing is landed" in refused.value.reason

    def test_a_refusal_carries_what_to_do_about_it(self, runner):
        """
        A reason with no detail is a dead end on a screen. The page renders the missing
        capabilities and the gate findings, so they have to arrive as data.
        """
        with pytest.raises(RunRefused) as refused:
            runner.start()
        assert isinstance(refused.value.detail, dict)


class TestTheRunIsAnswerableForItsInputs:
    """
    A store-driven input has no path and no bytes to hash, and `name` is the export it
    came from — which every re-export of that report shares. Without the batch id every
    store-driven run would carry the same input fingerprint as every other, and
    `input_fingerprint` claims the opposite: that two runs sharing it read the same
    bytes.
    """

    def test_a_batch_identifies_an_input_that_has_no_file(self):
        from inventory_planning.provenance import InputRecord

        batched = InputRecord(doc_type="inventory", name="stock.csv",
                              batch_id="20261005_120000-abc123")
        plain = InputRecord(doc_type="inventory", name="stock.csv")
        assert batched.identity != plain.identity
        assert "20261005_120000-abc123" in batched.identity

    def test_two_runs_over_different_batches_do_not_look_alike(self):
        from inventory_planning.provenance import RunManifest

        def fingerprint(batch_id):
            run = RunManifest.begin()
            run.record_input(None, doc_type="inventory", name="stock.csv",
                             batch_id=batch_id)
            return run.input_fingerprint

        assert fingerprint("batch-a") != fingerprint("batch-b")

    def test_a_hashed_file_still_identifies_by_its_bytes(self, tmp_path):
        """The batch id is a fallback, not a replacement: a path still wins."""
        from inventory_planning.provenance import RunManifest

        path = tmp_path / "stock.csv"
        path.write_text("sku,qty\nA,1\n", encoding="utf-8")
        run = RunManifest.begin()
        record = run.record_input(path, doc_type="inventory", batch_id="irrelevant")
        assert record.identity == record.sha256
