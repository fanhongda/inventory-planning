"""
Starting a run from the store, and the three things that stop one.

Until this existed the interface could change everything a run depends on and could not
start one. A planner editing `transit_share_of_lt` on the policy screen saw the diff,
signed it, and watched it land in `config_changes.jsonl` — and then had to open a
terminal to find out what it did. The screen that knew the most about the change was the
one place the consequence could not be seen.

Four decisions shape this module.

**The run reads the fact store, not a folder.** Everything the interface does lands
batches, so a folder-driven run would be a second input path that the review screen,
the declarations and the quality gate had never seen. Reading the store also makes the
run sayable afterwards: `InputRecord.batch_id` records which batches were read, and two
runs that read the same ones carry the same input fingerprint.

**The gate decides, and it is the gate the run already uses.** `gate_intake` is called
here over the same documents, with waivers applied through `Declarations.waive` — the
same call the pipeline makes. A refusal names the failing checks and points at the
waiver route, and there is deliberately no "run anyway" box beside the button: that
route already exists, requires a reason, an owner and an expiry, and is narrower by
construction. A checkbox would be `allow_degraded=True` with better manners.

**One run at a time.** Two runs write the same output directory and the same run
registry, and the second would interleave its workbook with the first. The refusal
names the run in flight rather than queueing, because a queued run would be planned
against config that may have changed by the time it starts, and nothing would say so.

**The thread is a detail; the state is the product.** What a caller needs is the run id
before the work starts, and an honest account of where it got to — including that it
failed and why. A run that dies leaves `failed` with its message, never `running`
forever, because a progress bar that never ends is worse than an error.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from typing import Any, Dict, List, Optional

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

# The stages a caller sees. Named for what is happening rather than for the method,
# because the method names are an implementation and these are shown to a person.
STAGES = ("reading the store", "checking the documents against each other",
          "planning", "policy and targets", "writing the workbook")


class RunRefused(Exception):
    """The run did not start, and the reason is actionable."""

    def __init__(self, reason: str, detail: Dict[str, Any] = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


@dataclass
class RunState:
    """Where one run got to."""

    run_id: str = ""
    status: str = QUEUED
    stage: str = STAGES[0]
    started_at: str = ""
    finished_at: str = ""
    batch_ids: List[str] = dc_field(default_factory=list)
    documents: List[str] = dc_field(default_factory=list)
    target_value: Optional[float] = None
    error: str = ""
    outputs: List[str] = dc_field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "status": self.status, "stage": self.stage,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "batch_ids": list(self.batch_ids), "documents": list(self.documents),
            "target_value": self.target_value, "error": self.error,
            "outputs": list(self.outputs),
            "running": self.status in (QUEUED, RUNNING),
        }


class Runner:
    """One run at a time, for one workspace."""

    def __init__(self, service):
        self.service = service
        self._lock = threading.Lock()
        self._state: Optional[RunState] = None
        self._thread: Optional[threading.Thread] = None

    # ── State ────────────────────────────────────────────────────────────────

    @property
    def state(self) -> Optional[RunState]:
        with self._lock:
            return self._state

    @property
    def in_flight(self) -> bool:
        with self._lock:
            return self._state is not None and self._state.status in (QUEUED, RUNNING)

    def _set(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self._state, key, value)

    # ── Starting ─────────────────────────────────────────────────────────────

    def start(self, target_value: Optional[float] = None,
              deadline=None) -> RunState:
        """
        Check everything that can refuse, then hand the work to a thread.

        Everything that can say no says it here, on the caller's own request, so a
        caller that got a run id knows the run began. A refusal discovered inside the
        thread would arrive as a failed run minutes later and read as a defect in the
        planning rather than as a document that was never fit to plan from.
        """
        with self._lock:
            if self._state is not None and self._state.status in (QUEUED, RUNNING):
                raise RunRefused(
                    f"run {self._state.run_id} is already in flight "
                    f"({self._state.stage}). Two runs write the same workbook and the "
                    f"same registry, and the second would interleave with the first.",
                    {"run_id": self._state.run_id, "stage": self._state.stage})

        intake, records = self._read_store()
        self._gate(intake)

        # The planner is built here, not in the thread, because building it is what
        # mints the run id — and a caller handed an empty id cannot poll for the run it
        # just started, nor be told which run is in its way. It also pins the config to
        # the moment the run was asked for rather than to whenever the thread is
        # scheduled, which is the honest reading of "this run used these rules".
        from ..orchestrator import InventoryPlanner

        planner = InventoryPlanner(output_dir=self.service.output_dir,
                                   config_dir=self.service.config_dir)
        state = RunState(
            run_id=planner.run.run_id,
            status=QUEUED,
            started_at=datetime.now().isoformat(timespec="seconds"),
            batch_ids=[r["batch_id"] for r in records.values()],
            documents=sorted(intake.documents),
            target_value=target_value,
        )
        with self._lock:
            self._state = state
            self._thread = threading.Thread(
                target=self._run, args=(planner, intake, records, target_value, deadline),
                name="inventory-planning-run", daemon=True)
            self._thread.start()
        return state

    # ── The three refusals ───────────────────────────────────────────────────

    def _read_store(self):
        """
        Every landed batch, routed again rather than taken at the type it landed under.

        Re-routing matters more here than on the review screen: a declaration written
        since the batch landed is exactly the thing a planner clicked "run" to see the
        effect of, and a batch read back at its landed type would ignore it.
        """
        from ..ingest.intake import Intake

        frames = []
        records: Dict[str, Dict[str, Any]] = {}
        seen = set()
        for record in self.service.landing.batches():
            batch_id = record["batch_id"]
            doc_type = record.get("doc_type") or ""
            if doc_type in seen or self.service.status_of(batch_id) == "void":
                continue
            seen.add(doc_type)
            records[doc_type] = record
            frames.append((self.service.landed_frame(record),
                           record.get("source_name", batch_id)))

        if not frames:
            raise RunRefused(
                "nothing is landed, so there is nothing to plan from. Upload an export "
                "on the import review screen first.")

        intake = Intake(verbose=False,
                        declarations=self.service.declarations()).load_frames(frames)
        if not intake.can_run:
            # Named by what the run cannot do without them, not by the contract that
            # would have supplied them. "sales_history is missing" is a fact about the
            # upload; "there is nothing to forecast from" is the consequence, and the
            # requirements checklist is already phrased that way.
            missing = intake.plan.missing_required
            raise RunRefused(
                "what is landed does not add up to a run.",
                {"missing": [{"capability": c.name, "why": c.description}
                              for c in missing],
                 "landed": sorted(intake.documents)})
        return intake, records

    def _gate(self, intake) -> None:
        """The intake gate, waivers and all — the same call the pipeline makes."""
        from ..quality import GateThresholds
        from ..quality.checks import gate_intake

        report = gate_intake(intake, intake.plan,
                             GateThresholds.load(self.service.config_dir))
        report = self.service.declarations().waive(report)
        if report.passed:
            return
        raise RunRefused(
            "the intake gate does not pass, and a gate that can be clicked past is "
            "not a gate. A finding you have judged a false positive is waived on the "
            "quality screen, where the waiver carries a reason, an owner and an "
            "expiry — and the run then honours it.",
            {"findings": [
                {"check": f.check, "what": f.what, "why": f.why, "fix": f.fix,
                 "severity": getattr(f, "severity", "")}
                for f in list(report.blocking) + list(report.severe)]})

    # ── The work ─────────────────────────────────────────────────────────────

    def _run(self, planner, intake, records, target_value, deadline) -> None:
        try:
            self._set(status=RUNNING, stage=STAGES[2])
            from ..ingest_bridge import IngestBridge

            bridge = IngestBridge(config_dir=self.service.config_dir, verbose=False)
            inputs = bridge.adapt(intake)
            # Which batches were read goes on the manifest, not only in this object:
            # the run has to be answerable for its inputs after this process is gone.
            planner.absorb_intake(
                inputs, batches={d: r["batch_id"] for d, r in records.items()})

            results = planner.run_planning(**inputs)
            self._set(stage=STAGES[3])
            planner.run_policy_analysis(
                results,
                inventory_df=inputs.get("inventory_df"),
                open_po_df=inputs.get("open_po_df"),
                item_master_df=inputs.get("item_master_df"),
                planning_master_df=inputs.get("planning_master_df"),
                target_value=target_value, deadline=deadline,
            )
            self._set(stage=STAGES[4],
                      outputs=[o.name for o in planner.run.outputs])
            self._set(status=DONE, stage="done",
                      finished_at=datetime.now().isoformat(timespec="seconds"))
        except Exception as exc:  # noqa: BLE001 — the message is the product here
            self._set(status=FAILED, error=f"{type(exc).__name__}: {exc}",
                      finished_at=datetime.now().isoformat(timespec="seconds"))
            traceback.print_exc()
