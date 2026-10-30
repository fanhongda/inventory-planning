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


class TestARunStartedFromTheScreenDrawsTheCharts:
    """
    The two ways into this pipeline wrote different sets of files.

    `run_kpi_review` renders the only output that is a picture. The CLI reaches it;
    `Runner._run` — the path behind the button on the Policy screen — called
    `run_planning` and `run_policy_analysis` and stopped. So a run started from the
    interface produced a workbook, three CSVs and a note, and the results screen, which
    renders a run's own outputs and deliberately computes nothing, had no picture to
    render. It looked like a screen that could not show charts. It was a run that never
    drew them.

    These are about the wiring, as the CLI's are: that the path reaches it with the
    frames it needs, that the html lands on the manifest the results screen reads, and
    that a failure drawing it cannot take down a run whose plan is already on disk.
    """

    class _Planner:
        def __init__(self, raises=False):
            self.reviewed = None
            self.raises = raises
            self.run = type("R", (), {"run_id": "20261030_000000-aaaaaa",
                                      "outputs": []})()

        def absorb_intake(self, inputs, batches=None):
            pass

        def run_planning(self, **kwargs):
            return {"results": True}

        def run_policy_analysis(self, *_, **__):
            return {"policy": True}

        def run_kpi_review(self, policy, **frames):
            if self.raises:
                raise RuntimeError("the chart library fell over")
            self.reviewed = (policy, frames)
            self.run.outputs = list(self.run.outputs) + [
                type("O", (), {"name": "kpi_review_20261030_000000-aaaaaa.html"})()]

    @staticmethod
    def _runner(tmp_path, planner, monkeypatch):
        from inventory_planning.api.app import Service

        runner = Runner(Service(config_dir=tmp_path / "config",
                                store_root=tmp_path / "store",
                                output_dir=tmp_path / "out"))
        from inventory_planning.api.runner import RunState

        # `_run` is the half after `start` has minted the run id and set the state, so
        # these tests hand it the state `start` would have left.
        runner._state = RunState(run_id=planner.run.run_id)
        monkeypatch.setattr("inventory_planning.ingest_bridge.IngestBridge.adapt",
                            lambda self, intake: {
                                "sales_df": "S", "po_history_df": "P",
                                "open_so_df": "O", "open_po_df": "Q",
                                "inventory_df": "I"})
        return runner

    def test_the_run_reaches_the_review_with_every_frame(self, tmp_path, monkeypatch):
        planner = self._Planner()
        runner = self._runner(tmp_path, planner, monkeypatch)
        runner._run(planner, intake=None, records={}, target_value=None, deadline=None)

        assert planner.reviewed is not None, "a run produced no visual review"
        _, frames = planner.reviewed
        assert set(frames) == {"sales_df", "open_so_df", "open_po_df",
                               "inventory_df", "po_history_df"}
        assert all(value is not None for value in frames.values())

    def test_the_picture_lands_on_the_manifest_the_screen_reads(
            self, tmp_path, monkeypatch):
        """
        The results screen reads the manifest, not the directory. An html written after
        the outputs were last read into the run state would be on disk and absent from
        the list the screen renders.
        """
        planner = self._Planner()
        runner = self._runner(tmp_path, planner, monkeypatch)
        runner._run(planner, intake=None, records={}, target_value=None, deadline=None)

        assert any(name.endswith(".html") for name in runner.state.outputs)
        assert runner.state.status == "done"

    def test_a_failure_drawing_it_does_not_fail_the_run(self, tmp_path, monkeypatch):
        planner = self._Planner(raises=True)
        runner = self._runner(tmp_path, planner, monkeypatch)
        runner._run(planner, intake=None, records={}, target_value=None, deadline=None)

        assert runner.state.status == "done"
        assert not runner.state.error
        assert "the chart library fell over" in runner.state.review_error
