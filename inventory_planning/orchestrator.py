"""
Inventory Planning Orchestrator.
Coordinates all readers and analytics modules in the correct sequence.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd

from .feedback.snapshot import SnapshotSaver
from .provenance import RunManifest, RunRegistry
from .readers.sales_history_reader import SalesHistoryReader
from .readers.po_history_reader import POHistoryReader
from .readers.open_so_reader import OpenSOReader
from .readers.open_po_reader import OpenPOReader
from .readers.inventory_reader import InventoryReader, consolidate_to_planning_grain
from .readers.timeseries_reader import TimeSeriesReader
from .analytics.backlog_realization import BacklogRealizationEstimator, RealizationResult
from .analytics.demand_classifier import DemandClassifier
from .analytics.safety_stock import SafetyStockCalculator
from .analytics.inventory_projector import InventoryProjector
from .analytics.forecaster import Forecaster
from .analytics.purchase_recommender import PurchaseRecommender
from .ingest.encoding import write_csv


class InventoryPlanner:
    """
    Single-stage DC inventory planning pipeline.
    multi-echelon ready: all outputs carry location_id.
    """

    def __init__(self, config_dir: Union[str, Path] = None, output_dir: Union[str, Path] = None,
                 interactive: bool = True, store_root: Union[str, Path] = None):
        base = Path(__file__).parents[1]
        self.config_dir = Path(config_dir) if config_dir else base / "config"
        self.output_dir = Path(output_dir) if output_dir else base / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interactive = interactive
        self._quality_log: list = []   # accumulates quality reports across all loads
        self._intake = None            # set by load_all(); carries adapter provenance
        self._intake_plan = None       # set by load_all(); what this run can answer
        self._fx = None                # set by load_all(); which money was restated, and what could not be
        # Identity for this run: what it read, what it resolved, what code ran. Started
        # here rather than at save time so the config is fingerprinted before anything
        # has had a chance to be edited mid-run.
        self.run = RunManifest.begin(config_dir=self.config_dir, output_dir=self.output_dir)
        self._store = None             # built on first use; see `store`
        self.store_root = store_root   # None -> $INVENTORY_PLANNING_STORE, then XDG

        # Readers
        self.sales_reader   = SalesHistoryReader(self.config_dir)
        self.po_reader      = POHistoryReader(self.config_dir)
        self.open_so_reader = OpenSOReader(self.config_dir)
        self.open_po_reader = OpenPOReader(self.config_dir)
        self.inv_reader     = InventoryReader(self.config_dir)
        self.ts_reader      = TimeSeriesReader(self.config_dir)

        # Analytics
        policy_cfg = json.loads((self.config_dir / "stocking_policy.json").read_text(encoding="utf-8"))
        self.policy_cfg    = policy_cfg
        self.backlog_horizon_days = int(policy_cfg.get("backlog_horizon_days", 30))
        self.classifier    = DemandClassifier(self.config_dir)
        self.ss_calc       = SafetyStockCalculator(self.config_dir)
        self.projector     = InventoryProjector(self.config_dir,
                                                horizon_days=self.backlog_horizon_days)
        self.forecaster    = Forecaster(horizon=policy_cfg["forecast_horizon_months"])
        self.recommender   = PurchaseRecommender(
            demand_basis=policy_cfg.get("demand_basis", "forecast_consumption"),
            horizon_days=self.backlog_horizon_days,
        )
        self.realization_estimator = BacklogRealizationEstimator(
            floor=float(policy_cfg.get("backlog_realization_floor", 0.25))
        )

    # ------------------------------------------------------------------
    # Phase 1: Data Ingestion (run in any order; returns raw clean dfs)
    # ------------------------------------------------------------------

    def load_sales_history(self, path: Union[str, Path]):
        self.run.record_input(path, doc_type="sales_history")
        df, report = self.sales_reader.read(path, interactive=self.interactive)
        self._check_quality(report)
        return df, report

    def load_po_history(self, path: Union[str, Path]):
        self.run.record_input(path, doc_type="po_history")
        df, report = self.po_reader.read(path, interactive=self.interactive)
        self._check_quality(report)
        return df, report

    def load_open_so(self, path: Union[str, Path]):
        self.run.record_input(path, doc_type="open_so")
        df, report = self.open_so_reader.read(path, interactive=self.interactive)
        self._check_quality(report)
        return df, report

    def load_open_po(self, path: Union[str, Path]):
        self.run.record_input(path, doc_type="open_po")
        df, report = self.open_po_reader.read(path, interactive=self.interactive)
        self._check_quality(report)
        return df, report

    def load_inventory(self, path: Union[str, Path]):
        self.run.record_input(path, doc_type="inventory")
        df, report = self.inv_reader.read(path, interactive=self.interactive)
        self._check_quality(report)
        return df, report

    def load_timeseries(self, path: Union[str, Path], rolling_months: int = 36):
        self.run.record_input(path, doc_type="demand_timeseries")
        """Load a pre-compiled wide-format time series file."""
        pivot, meta, report = self.ts_reader.read(
            path, rolling_months=rolling_months, interactive=self.interactive
        )
        return pivot, meta, report

    # ------------------------------------------------------------------
    # Phase 7: Policy analysis — should-be, levers, target
    # ------------------------------------------------------------------

    def run_policy_analysis(
        self,
        results: dict,
        inventory_df=None,
        open_po_df=None,
        target_value: float = None,
        deadline=None,
        parameters_file=None,
        item_master_df=None,
        planning_master_df=None,
    ) -> dict:
        """
        Answer the planner's actual questions on top of a completed planning run.

        Produces, in order:
          should_be   what stock ought to be under the policy in force, split into
                      cycle / safety / buyer-owned pipeline so each part points at a
                      different lever
          levers      what each available action is worth, ranked on *net* annual
                      benefit — a lever that costs more in ordering than it frees in
                      stock ranks below one that does not
          frontier    when `target_value` is given, the ordered set of moves that
                      reaches it, what they cost, and what was deliberately excluded

        `results` is the dict returned by `run_planning`.
        """
        from .policy.levers import LeverAnalyzer
        from .policy.should_be import ShouldBeCalculator
        from .policy.suggestions import SuggestionBuilder
        from .policy.target import TargetPlanner

        print("\n[8/8] Policy analysis — should-be, levers, target...")

        inventory_df = consolidate_to_planning_grain(
            inventory_df, planning_location=self.inv_reader.location_id, verbose=False
        )
        # The planning run already resolved these against the same file. Reusing them
        # is not an optimisation — rebuilding would give the policy layer its own copy
        # of every parameter, and the first time one of them drifted the report would
        # disagree with the recommendation CSV a planner is holding.
        attributes = results.get("sku_attributes")
        resolved = results.get("parameters")
        crosscheck = results.get("crosscheck")
        if parameters_file is not None or attributes is None or resolved is None:
            attributes, crosscheck, resolved, _ = self._resolve_policy(
                classified_demand=results["classified_demand"],
                supplier_lt=results.get("supplier_lt"),
                inventory=inventory_df,
                forecast_summary=results.get("forecast_summary"),
                timeseries_meta=results.get("timeseries_meta"),
                item_master=(item_master_df if item_master_df is not None
                             else results.get("item_master")),
                planning_master=(planning_master_df if planning_master_df is not None
                                 else results.get("planning_master")),
                po_history=results.get("po_history"),
                parameters_file=parameters_file,
            )

        calculator = ShouldBeCalculator(self.config_dir)
        should_be = calculator.calculate(resolved, actual=inventory_df)
        print()
        print(should_be.summary())

        levers = LeverAnalyzer(calculator).analyze(resolved, actual=inventory_df,
                                                   baseline=should_be)
        print()
        print(levers.summary())

        # What the data says the parameters should be, next to what they are. Written
        # out, never applied — a parameter that changed because a script decided it
        # should is one nobody can defend in a review.
        suggestions = SuggestionBuilder(self.config_dir).build(
            resolved, recommendations=results.get("recommendations"))
        print()
        print(suggestions.summary())
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = suggestions.to_csv(self.output_dir / f"parameter_suggestions_{stamp}.csv")
        md_path = self.output_dir / f"suggested_rules_{stamp}.md"
        suggestions.to_rules_markdown(md_path)
        print(f"\n    Per-SKU suggestions : {csv_path}")
        print(f"    Paste-able rules    : {md_path}")

        if len(crosscheck.all_disagreements):
            xc_path = self.output_dir / f"source_crosscheck_{stamp}.csv"
            write_csv(crosscheck.frame(), xc_path)
            print(f"    Source disagreements: {xc_path}")

        out = {
            "sku_attributes": attributes,
            "parameters": resolved,
            "should_be": should_be,
            "levers": levers,
            "suggestions": suggestions,
            # Carried forward so the KPI review's work list uses the same
            # recommendations the planning run produced, rather than deriving a second
            # set that would eventually disagree with the CSV the planner works from.
            "recommendations": results.get("recommendations"),
            "crosscheck": crosscheck,
            "frontier": None,
        }

        if target_value is not None:
            holding_rate = float(resolved.defaults.get("holding_cost_rate", 0.22))
            frontier = TargetPlanner(holding_rate).plan(
                should_be, target_value=target_value, deadline=deadline,
                levers=levers, open_po=open_po_df,
            )
            print()
            print(frontier.summary())
            out["frontier"] = frontier

        self._quality_log.append({
            "doc_type": "policy",
            "file": str((parameters_file
                         or (self.config_dir / "planning_parameters.md")).name),
            "rows_loaded": len(attributes),
            "issues": [str(h) for h in resolved.conflicts],
            "status": "WARNINGS" if resolved.conflicts else "OK",
        })
        return out

    # ------------------------------------------------------------------
    # Phase 8: KPI attribution — what happened, and what is coming
    # ------------------------------------------------------------------

    def run_kpi_review(
        self,
        policy: dict,
        sales_df=None,
        open_so_df=None,
        open_po_df=None,
        inventory_df=None,
        po_history_df=None,
        target_value: float = None,
        deadline=None,
        as_of=None,
        title: str = "Inventory & Service Review",
    ) -> dict:
        """
        Attribute the KPIs to the SKUs that moved them, then project forward.

        Chapter 1 answers "what happened and who caused it" — OTD misses, excess
        capital, and what the buying pattern cost. Chapter 2 answers "what is coming"
        — which SKUs run out, which will never move, and which inbound POs to change.

        The service split is the part worth reading carefully. A line past its request
        date with stock on the shelf is **not** an OTD failure — the goods were there.
        It is counted separately, because calling it a supply miss blames planning for
        a collection problem and simultaneously hides that the stock is sitting idle.
        """
        from datetime import date as _date

        from .policy.cadence import CadenceAnalyzer
        from .policy.diagnostics import DiagnosticsAnalyzer
        from .policy.service import ServiceAnalyzer, latest_observed_date
        from .reporting.kpi_report import KPIReport

        print("\n[8/8] KPI attribution — service, ordering, forward risk...")

        inventory_df = consolidate_to_planning_grain(
            inventory_df, planning_location=self.inv_reader.location_id, verbose=False
        )

        # One anchor for the whole review, taken from the data. Mixing a data-derived
        # date in one section with today's date in another makes the two halves
        # disagree about what "past due" and "days to stockout" mean.
        if as_of is None:
            derived, stale = latest_observed_date(sales_df, open_so_df, po_history_df)
            as_of = derived or _date.today()
            if stale > 45:
                print(f"      Anchoring to the newest date in the data ({as_of}), "
                      f"{stale} days ago — not today. Scoring this extract against "
                      f"today would mark every open order past due.")

        attributes = policy["sku_attributes"]
        should_be = policy["should_be"]

        service = ServiceAnalyzer().analyze(
            sales_history=sales_df, open_so=open_so_df,
            inventory=inventory_df, as_of=as_of,
        )
        if len(service.lines):
            print()
            print(service.summary())

        diagnostics = DiagnosticsAnalyzer()
        ordering = diagnostics.ordering(po_history_df, attributes, as_of=as_of)
        print()
        print(ordering.summary())

        # Order-size CV says whether the lots were consistent. It cannot say whether
        # they arrived at the right time or in the right direction, and consistent lots
        # bought at the wrong moments cost exactly what erratic ones do.
        params = policy.get("parameters")
        cadence = CadenceAnalyzer(
            order_cost=float(params.defaults.get("order_cost_usd", 350))
            if params is not None else 350.0,
        ).analyze(
            po_history=po_history_df,
            sales_history=sales_df,
            sku_attributes=attributes,
            parameters=getattr(params, "frame", None),
            as_of=as_of,
        )
        print()
        print(cadence.summary())

        forward = diagnostics.forward(
            sku_attributes=attributes, inventory=inventory_df,
            open_po=open_po_df, open_so=open_so_df,
            safety_stock=should_be.frame.set_index("sku")["safety_qty"],
            as_of=as_of,
        )
        print()
        print(forward.summary())

        frontier = policy.get("frontier")
        if frontier is None and target_value is not None:
            from .policy.target import TargetPlanner
            holding = float(policy["parameters"].defaults.get("holding_cost_rate", 0.22))
            frontier = TargetPlanner(holding).plan(
                should_be, target_value=target_value, deadline=deadline,
                levers=policy.get("levers"), open_po=open_po_df, as_of=as_of,
            )

        # The service level most of the catalogue actually runs at — the line the OTD
        # trend is scored against. Modal rather than mean: the parameter takes a few
        # discrete values from the policy tiers, and averaging them produces a target
        # no SKU is actually held to.
        service_target = None
        frame = getattr(params, "frame", None)
        if frame is not None and "service_level" in frame.columns:
            levels = pd.to_numeric(frame["service_level"], errors="coerce").dropna()
            if len(levels):
                service_target = float(levels.mode().iloc[0])

        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = self.output_dir / f"kpi_review_{stamp}.html"
        KPIReport(title).render(
            service=service, should_be=should_be, ordering=ordering,
            cadence=cadence, forward=forward, levers=policy.get("levers"),
            frontier=frontier, fx=self._fx, service_target=service_target,
            attributes=attributes, recommendations=policy.get("recommendations"),
            open_po=open_po_df, suggestions=policy.get("suggestions"),
            as_of=as_of, output_path=path,
        )
        print(f"\n  KPI review saved: {path}")
        print(f"  Open report: open {path}")

        return {
            "service": service,
            "ordering": ordering,
            "cadence": cadence,
            "forward": forward,
            "frontier": frontier,
            "report_path": path,
        }

    # ------------------------------------------------------------------
    # Phase 1b: Contract-driven intake (preferred)
    # ------------------------------------------------------------------

    @property
    def store(self):
        """
        The fact store, or None when it cannot be opened.

        Built lazily and never allowed to raise into the pipeline. A store on a schema
        this code does not understand, an unwritable directory, a missing parquet
        library: each is a warning and a run that still produces a plan. The store is
        written by this phase and read by nothing, so its absence costs history and
        costs no correctness.
        """
        if self._store is not None:
            return self._store or None
        try:
            from .store import FactStore
            self._store = FactStore(self.store_root)
            for note in self._store.notes:
                print(f"  Note: {note}")
        except Exception as e:
            print(f"  Warning: fact store unavailable ({e}) — planning continues, "
                  f"nothing is being retained")
            self._store = False
        return self._store or None

    def _shadow_write(self, frames: dict, valid_time) -> None:
        """
        Write this run's facts to the store, which nothing reads yet.

        Phase one of three. The pipeline goes on reading its files; only when a run's
        outputs can be shown identical from either source does anything start reading
        the store. Writing starts first regardless, because history not collected
        cannot be recovered later — every week spent waiting for the read path is a
        week of positions that can never be reconstructed.

        `valid_time` is the run's anchor: the newest date the data itself carries, from
        `latest_observed_date`. That is a property of the content, unlike a file's
        mtime, which a copy or a re-download rewrites while the data keeps describing
        whatever it described. Per-document valid times — an inventory snapshot date
        that differs from the sales anchor — belong with the merge interface, which
        needs a reviewed plan in front of a human anyway.
        """
        store = self.store
        if store is None:
            return
        if valid_time is None:
            print("  Note: no date anchor in the data — nothing retained, because a "
                  "batch with a guessed valid_time is worse than no batch")
            return

        by_source = {i.doc_type: i for i in self.run.inputs}
        seen_notes = len(store.notes)
        written = 0
        for doc_type, frame in frames.items():
            if frame is None or not len(frame):
                continue
            record = by_source.get(doc_type)
            try:
                batch = store.write_batch(
                    doc_type=doc_type,
                    frame=frame,
                    valid_time=valid_time,
                    source_name=record.name if record else "",
                    source_sha=record.sha256 if record else None,
                    config_fingerprint=self.run.config_fingerprint,
                    run_id=self.run.run_id,
                    key_verdict=record.key_verdict if record else None,
                    storable=record.storable if record else None,
                    written_by="pipeline",
                )
            except Exception as e:
                print(f"  Warning: {doc_type} not retained ({e})")
                continue
            if batch is not None:
                written += 1
        if written:
            print(f"  Retained: {written} batch(es) as of {valid_time} -> {store.root}")
        # A run that retained nothing is the normal case for a re-run of the same
        # extract, and saying so is the difference between "already have this" and a
        # store that has quietly stopped working.
        for note in store.notes[seen_notes:]:
            print(f"  Note: {note}")

    def load_all(
        self,
        paths: list,
        hints: dict = None,
        tenant: str = "default",
    ) -> dict:
        """
        Load every input file in one call, identifying each one automatically.

        Preferred over the individual `load_*` methods: the caller does not have to
        know which file is which, nor which loader a pre-aggregated demand series
        needs. Files are profiled, routed to a contract, transformed by an adapter and
        verified before any of them reach the analytics.

        Returns the keyword arguments for `run_planning`, so the whole pipeline is:

            planner = InventoryPlanner(output_dir=...)
            inputs  = planner.load_all(glob("exports/*"))
            results = planner.run_planning(**inputs)
        """
        from .ingest_bridge import IngestBridge

        bridge = IngestBridge(config_dir=self.config_dir, verbose=True)
        inputs = bridge.load(
            paths,
            hints=hints,
            tenant=tenant,
            baseline_path=self.output_dir.parent / "ingest_baselines.json",
        )

        plan = inputs["_intake_plan"]
        self._intake = inputs.pop("_intake", None)
        self._intake_plan = inputs.pop("_intake_plan", None)
        self._fx = inputs.pop("_fx", None)
        if self._intake is not None:
            self.run.record_intake(self._intake)

        self._write_supersession_record()

        if not plan.can_run:
            missing = ", ".join(c.name for c in plan.missing_required)
            raise ValueError(
                f"Cannot run planning — missing required input(s): {missing}.\n"
                f"{plan.summary()}"
            )

        # A capability can be satisfied by a document that is structurally broken: the
        # file is present and routed, so the plan reads green, but a field the
        # analytics index by name never arrived. Nothing downstream guards against
        # that, so the run does not degrade — it raises KeyError several layers from
        # the cause, naming a column rather than a file. Stop here, where the file and
        # the missing field can both be named.
        unusable = self._intake.unusable() if self._intake is not None else []
        if unusable:
            detail = "\n".join(
                f"    {doc.doc_type:<16} {doc.source_name}\n"
                f"      missing: {', '.join(fields)}"
                for doc, fields in unusable
            )
            raise ValueError(
                "Cannot run planning — a required field never arrived. The document "
                "routed correctly and its other columns mapped, but this one matched "
                "no alias and could not be derived:\n"
                f"{detail}\n\n"
                "    Run `python -m inventory_planning.explain <file>` to see which "
                "source columns went unmatched, then add the right one as an alias to "
                "the contract."
            )
        # Degradations are recorded rather than raised: the run is still meaningful,
        # but the report must be able to say which numbers rest on a fallback.
        self._quality_log.append({
            "doc_type": "intake",
            "file": "<multiple>",
            "rows_loaded": sum(len(f) for f in inputs.values() if isinstance(f, pd.DataFrame)),
            "issues": list(plan.degradations),
            "status": "OK" if not plan.degradations else "WARNINGS",
        })
        return inputs

    def _write_supersession_record(self) -> Optional[Path]:
        """
        Write down what the merge did, because the merge is invisible in every other
        output by design.

        After it, the run reports on 7100015 and says nothing about 7100014 — which is
        correct, they are one material. But the stock on the shelf is still labelled
        with the old number, the open POs at the supplier are still raised against it,
        and a buyer told to order 400 of the new one needs to know that 150 of the
        cover already counted is sitting under the old label. That question has one
        answer and this file is it.
        """
        if self._intake is None:
            return None
        report = self._intake.supersessions
        if not report.records:
            return None

        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        path = self.output_dir / f"supersessions_{stamp}.csv"
        write_csv(report.to_frame(), path)
        print(f"    Merged item numbers : {path}")
        return path

    # ------------------------------------------------------------------
    # Phase 2: Analytics Pipeline (sequential)
    # ------------------------------------------------------------------

    @staticmethod
    def _report_policy_suggestion(classified: pd.DataFrame,
                                  planning_master_df: pd.DataFrame) -> None:
        """
        What the demand says the stocking policy should be, against what the ERP holds.

        Printed rather than applied. The policy decides whether stock is expected to sit
        on the shelf at all, and a pipeline that rewrote it from twelve months of history
        would be overruling a decision on evidence the person who made it may already
        have weighed. The disagreement is the finding; acting on it is a person's call.
        """
        if classified is None or "suggested_stocking_policy" not in classified.columns:
            return
        suggested = classified.dropna(subset=["suggested_stocking_policy"])
        if not len(suggested):
            return

        counts = suggested["suggested_stocking_policy"].value_counts()
        print(f"      Suggested from demand: "
              f"{counts.get('MTS', 0)} MTS, {counts.get('MTO', 0)} MTO "
              f"(over {len(suggested)} SKUs with a demand series)")

        if planning_master_df is None or "stocking_policy" not in getattr(
                planning_master_df, "columns", []):
            print("      No ERP policy to compare against — the suggestion stands alone")
            return

        erp = planning_master_df[["sku", "stocking_policy"]].dropna(subset=["stocking_policy"])
        both = suggested.merge(erp, on="sku", how="inner")
        if not len(both):
            return
        disagree = both[both["stocking_policy"] != both["suggested_stocking_policy"]]
        agree_rate = 1 - len(disagree) / len(both)
        print(f"      Against the ERP policy on {len(both)} of them: "
              f"{agree_rate:.0%} agree")
        for direction, label in (
            (("MTO", "MTS"), "held as MTO, demand recurs — a candidate to stock"),
            (("MTS", "MTO"), "held as MTS, demand is sporadic — stock may be sitting"),
        ):
            rows = disagree[(disagree["stocking_policy"] == direction[0])
                            & (disagree["suggested_stocking_policy"] == direction[1])]
            if len(rows):
                names = ", ".join(rows["sku"].astype(str).head(4))
                more = f" … +{len(rows) - 4}" if len(rows) > 4 else ""
                print(f"        {len(rows):>4} {label}: {names}{more}")

    def run_planning(
        self,
        sales_df: pd.DataFrame,
        po_history_df: pd.DataFrame,
        open_so_df: pd.DataFrame,
        open_po_df: pd.DataFrame,
        inventory_df: pd.DataFrame,
        timeseries_pivot: pd.DataFrame = None,   # pre-compiled TS overrides sales_df TS
        timeseries_meta: pd.DataFrame = None,
        item_master_df: pd.DataFrame = None,     # ERP master: supplier, LT, MOQ, cost
        planning_master_df: pd.DataFrame = None, # planner's worksheet: SS, min/max, LT
    ) -> dict:
        print("\n" + "="*60)
        print("  INVENTORY PLANNING PIPELINE")
        print("="*60)

        # Step 0: One row per SKU, one "now" for the whole run.
        #
        # The inventory export is one row per SKU x storage location. Planning here is
        # single-node and every join downstream is on `sku` alone, so a SKU held in two
        # locations would fan out into two planning rows that each match the same open
        # PO and the same backlog — one saying pull in, the other saying push out.
        # Kept before the collapse. `consolidate_to_planning_grain` sums storage
        # locations into one row per SKU, which is right for planning and destroys the
        # only record of where the stock actually was: the surviving `location_id` is
        # the first code seen, so 100 sellable in `01` plus 40 quarantined in `02`
        # becomes 140 in `01`. Retaining that would put a fabricated location on every
        # stored row — against a contract whose natural key is `[sku, location_id]` —
        # and would discard exactly the per-location detail the topology work in
        # TODO.md is waiting on. History not collected cannot be recovered later, which
        # is the whole argument for writing the store before anything reads it.
        inventory_as_read = inventory_df
        inventory_df = consolidate_to_planning_grain(
            inventory_df, planning_location=self.inv_reader.location_id
        )
        from .policy.service import latest_observed_date
        as_of, stale = latest_observed_date(sales_df, open_so_df, po_history_df)
        if as_of and stale > 45:
            print(f"      Anchored to the newest date in the data ({as_of}), {stale} days "
                  f"ago — not today. Scoring this extract against today would mark every "
                  f"open order past due.")

        # The shape of what was loaded, before anything is computed on it. Printed
        # first because every silent mapping failure this pipeline has produced was
        # visible in these totals and invisible everywhere else.
        from .reporting.intake_summary import summarise_intake
        intake = summarise_intake({
            "sales_history": sales_df, "po_history": po_history_df,
            "open_so": open_so_df, "open_po": open_po_df, "inventory": inventory_df,
        }, anchor=as_of)
        print()
        print(intake.summary())

        self._shadow_write({
            "sales_history": sales_df, "po_history": po_history_df,
            "open_so": open_so_df, "open_po": open_po_df,
            "inventory": inventory_as_read,
        }, valid_time=as_of)

        # Step 1: Demand time series + summary
        print("\n[1/7] Building demand time series...")
        if timeseries_pivot is not None:
            ts = timeseries_pivot
            # Build demand summary from the pre-compiled pivot
            demand_summary = self._summarize_from_pivot(ts, timeseries_meta)
            print(f"      Source: pre-compiled time series")
        else:
            ts = self.sales_reader.to_time_series(sales_df)
            demand_summary = self.sales_reader.summarize(sales_df)
        print(f"      {len(ts.columns)} SKUs × {len(ts)} periods ({ts.index[0]} → {ts.index[-1]})")

        # Step 2: Supplier lead times
        print("\n[2/7] Computing supplier lead times...")
        supplier_lt = (
            self.po_reader.compute_supplier_lt(po_history_df)
            if po_history_df is not None and len(po_history_df)
            else pd.DataFrame(columns=["sku", "supplier", "wma_lead_time_days",
                                       "lt_std_days", "sample_count", "incoterm"])
        )
        print(f"      {len(supplier_lt)} SKU×supplier LT records measured from receipts")
        supplier_lt = self._fill_lead_time_from_masters(
            supplier_lt, ts.columns, item_master_df, planning_master_df
        )

        # Step 3: Demand classification
        print("\n[3/7] Classifying demand (stocking policy + CV pattern)...")
        classified = self.classifier.classify(demand_summary, ts)
        counts = classified["stocking_class"].value_counts().to_dict()
        pattern_counts = classified["demand_pattern"].value_counts().to_dict() if "demand_pattern" in classified.columns else {}
        print(f"      Stocking classes: {counts}")
        if pattern_counts:
            print(f"      Demand patterns: {pattern_counts}")
        self._report_policy_suggestion(classified, planning_master_df)

        # Step 4: Forecast — must run before safety stock to supply forecast RMSE
        print("\n[4/7] Forecasting demand (6 months)...")
        # MTS competes its models; MTO goes straight to Croston. The policy is the
        # ERP's, from the planner worksheet — not the pipeline's inferred class.
        policy = (planning_master_df[["sku", "stocking_policy"]]
                  if planning_master_df is not None
                  and "stocking_policy" in planning_master_df.columns else None)
        forecast_detail = self.forecaster.forecast_all(ts, classified=classified,
                                                       policy=policy)
        forecast_summary_df = self.forecaster.summary(forecast_detail)
        if not forecast_detail.empty:
            model_counts = forecast_detail.drop_duplicates("sku")["model_used"].value_counts().to_dict()
            # Counts of SKUs, not one choice for the run. Each SKU is scored on its
            # own history and keeps its own winner; `model_used` on every row of
            # forecast_detail and forecast_<ts>.csv says which.
            print(f"      Model chosen per SKU — {model_counts}")

        # Step 5: Planning parameters and the policy each SKU is on.
        #
        # This runs before safety stock rather than after the plan because the review
        # period is an input to both halves of the arithmetic. Resolving it afterwards
        # is what left the recommender sizing orders on a hardcoded thirty days while
        # the policy layer sized stock on R + LT.
        print("\n[5/7] Resolving planning parameters & replenishment policy...")
        open_so_summary = (
            self.open_so_reader.backlog_summary(
                open_so_df, as_of=as_of, horizon_days=self.backlog_horizon_days
            )
            if open_so_df is not None else None
        )
        attributes, crosscheck, resolved, profile = self._resolve_policy(
            classified_demand=classified,
            supplier_lt=supplier_lt,
            inventory=inventory_df,
            forecast_summary=forecast_summary_df if not forecast_detail.empty else None,
            timeseries_meta=timeseries_meta,
            item_master=item_master_df,
            planning_master=planning_master_df,
            po_history=po_history_df,
            backlog=open_so_summary,
        )
        print()
        print(profile.summary())

        # Step 6: Safety stock — uses forecast RMSE from step 4 as σDL, over the
        # exposure window the resolved review period implies.
        print("\n[6/7] Calculating safety stock...")
        ss_df = self.ss_calc.calculate(
            classified, supplier_lt,
            forecast_summary=forecast_summary_df if not forecast_detail.empty else None,
            review_period_days=resolved.frame[["sku", "review_period_days"]]
            if "review_period_days" in resolved.frame.columns else None,
            exposure=str(resolved.convention("safety_stock_exposure", "review_plus_lt")),
        )
        sigma_sources = ss_df["sigma_source"].value_counts().to_dict() if "sigma_source" in ss_df.columns else {}
        if sigma_sources:
            print(f"      σ source: {sigma_sources}")

        # Step 6: Effective inventory + projection + recommendations
        print("\n[7/7] Projecting inventory & generating recommendations...")
        if open_po_df is not None:
            open_po_df = self.open_po_reader.fill_estimated_delivery(open_po_df, supplier_lt)
        open_po_summary = (
            self.open_po_reader.inbound_schedule(
                open_po_df, as_of=as_of, horizon_days=self.recommender.horizon_days)
            if open_po_df is not None else None
        )
        eff_inv = self.inv_reader.effective_inventory(inventory_df, open_po_summary, supplier_lt)
        projection = self.projector.project(ss_df, eff_inv, open_po_summary)

        excess_count = (projection["inventory_status"] == "EXCESS").sum()
        shortage_count = (projection["inventory_status"] == "SHORTAGE-RISK").sum()
        print(f"      EXCESS: {excess_count} SKUs | SHORTAGE-RISK: {shortage_count} SKUs")

        # How much of the open order book actually ships. Measured, not assumed — the
        # recommender needs it to net backlog against the forecast rather than add the
        # two together and buy the same demand twice.
        # When each order-on-demand SKU's purchase order has to be *placed* to meet the
        # dates on the book. Needs the measured lead time and the resolved review
        # period, so it is built here rather than beside the backlog summary.
        mto_schedule = (
            self.open_so_reader.order_by_schedule(
                open_so_df, lead_times=resolved.frame, as_of=as_of,
                review_period_days=float(
                    pd.to_numeric(resolved.frame.get("review_period_days"),
                                  errors="coerce").median()
                ) if "review_period_days" in resolved.frame.columns
                else self.recommender.horizon_days,
            )
            if open_so_df is not None else None
        )
        if mto_schedule is not None and len(mto_schedule):
            late = int((mto_schedule["mto_order_past_due_qty"].fillna(0) > 0).sum())
            if late:
                print(f"      {late} SKUs have order lines whose order-by date has "
                      f"already passed — lead time no longer recoverable")

        realization = self._estimate_realization(open_so_df, inventory_df, as_of)
        print()
        print(realization.summary())

        # Purchase recommendations — uses forecast_next_period (t+1), not 6-month avg
        forecast_summary = forecast_summary_df
        recommendations = self.recommender.recommend(
            projection, forecast_summary, open_so_summary, open_po_summary,
            realization=realization, parameters=resolved,
            mto_schedule=mto_schedule,
        )

        results = {
            "time_series": ts,
            "demand_summary": demand_summary,
            "supplier_lt": supplier_lt,
            "classified_demand": classified,
            "safety_stock": ss_df,
            "effective_inventory": eff_inv,
            "projection": projection,
            "forecast_detail": forecast_detail,
            "forecast_summary": forecast_summary,
            "recommendations": recommendations,
            "sku_attributes": attributes,
            "parameters": resolved,
            "crosscheck": crosscheck,
            "policy_profile": profile,
            "backlog_realization": realization,
            "intake_summary": intake,
            "inventory_consolidated": inventory_df,
            "item_master": item_master_df,
            "planning_master": planning_master_df,
            # Carried for the policy layer's benefit: it is the only source of a
            # measured unit cost on an extract whose inventory export is quantity-only.
            "po_history": po_history_df,
            "as_of": as_of,
            "_quality_reports": self._quality_log,
        }

        self._save_outputs(results)
        self._print_summary(recommendations)
        return results

    def _resolve_policy(self, classified_demand, supplier_lt, inventory,
                        forecast_summary=None, timeseries_meta=None,
                        item_master=None, planning_master=None, po_history=None,
                        backlog=None, parameters_file=None):
        """
        Build the per-SKU attribute frame, apply the rule engine, and profile the
        replenishment policy each SKU ends up on.

        One place, called once per run. The planning stage needs the review period to
        size an order and the policy stage needs the same frame to say what stock ought
        to be; deriving it twice is how the two come to disagree about a SKU nobody is
        looking at.
        """
        from .policy.assemble import build_sku_attributes
        from .policy.parameters import PlanningParameters
        from .policy.profile import build_policy_profile

        params_file = parameters_file or (self.config_dir / "planning_parameters.md")
        planning_params = PlanningParameters(params_file)
        self.run.record_rules(r.rule_id for r in planning_params.rules)

        attributes, crosscheck = build_sku_attributes(
            classified_demand=classified_demand,
            supplier_lt=supplier_lt,
            inventory=inventory,
            forecast_summary=forecast_summary,
            timeseries_meta=timeseries_meta,
            params=planning_params,
            item_master=item_master,
            planning_master=planning_master,
            po_history=po_history,
        )
        if crosscheck.resolutions:
            print()
            print(crosscheck.summary())

        resolved = planning_params.resolve(attributes)
        print()
        print(resolved.summary())

        profile = build_policy_profile(resolved.frame, backlog=backlog,
                                       inventory=inventory)
        return attributes, crosscheck, resolved, profile

    @staticmethod
    def _fill_lead_time_from_masters(supplier_lt, skus, item_master_df, planning_master_df):
        """
        Give SKUs with no receipt history a lead time from master data.

        Without this a SKU that has simply never been bought through this ERP export
        gets a lead time of zero, and therefore a safety stock of zero — the pipeline
        confidently recommends holding nothing for an item with a three-month lead
        time. A stated value from the item master or the planner's worksheet is not a
        measurement, and it is labelled as such, but it is enormously better than that.

        The variability term is the honest casualty: a stated lead time carries no
        sigma, so safety stock for these SKUs covers demand variability only and is
        understated. The count is printed rather than buried.
        """
        sources = [
            (item_master_df, "lead_time_days", "item_master"),
            (planning_master_df, "planner_lead_time_days", "planning_master"),
        ]
        have = set(supplier_lt["sku"]) if len(supplier_lt) else set()
        rows = []
        for master, column, label in sources:
            if master is None or len(master) == 0 or column not in master.columns:
                continue
            stated = (
                master[["sku", column]]
                .assign(**{column: pd.to_numeric(master[column], errors="coerce")})
                .dropna(subset=["sku", column])
                .drop_duplicates("sku")
            )
            for _, row in stated.iterrows():
                sku = row["sku"]
                if sku in have or sku not in set(skus) or float(row[column]) <= 0:
                    continue
                have.add(sku)
                rows.append({
                    "sku": sku,
                    "supplier": (master.set_index("sku")["supplier"].get(sku)
                                 if "supplier" in master.columns else None),
                    "wma_lead_time_days": round(float(row[column]), 1),
                    "lt_std_days": 0.0,
                    "sample_count": 0,
                    "lt_source": label,
                })

        if not rows:
            if len(supplier_lt) and "lt_source" not in supplier_lt.columns:
                supplier_lt = supplier_lt.assign(lt_source="measured")
            return supplier_lt

        if len(supplier_lt) and "lt_source" not in supplier_lt.columns:
            supplier_lt = supplier_lt.assign(lt_source="measured")
        filled = pd.DataFrame(rows)
        by_source = filled["lt_source"].value_counts().to_dict()
        print(f"      {len(filled)} SKUs had no receipt history — lead time taken from "
              f"master data {by_source}")
        print(f"      Those carry no lead-time variability, so their safety stock covers "
              f"demand variability only and is understated.")
        return pd.concat([supplier_lt, filled], ignore_index=True)

    def _estimate_realization(self, open_so_df, inventory_df, as_of):
        """
        Backlog realization, honouring the configured override.

        `backlog_realization` may be `measured` or a fixed rate. A fixed rate is a
        legitimate choice — a planner who knows the collection problem is being fixed
        should be able to say so — but it is a stated assumption, not a measurement,
        and the report labels it as one.
        """
        setting = self.policy_cfg.get("backlog_realization", "measured")
        if setting == "measured":
            return self.realization_estimator.estimate(
                open_so=open_so_df, inventory=inventory_df, as_of=as_of
            )

        rate = float(np.clip(float(setting), 0.0, 1.0))
        return RealizationResult(
            per_sku=pd.DataFrame(columns=["sku", "open_qty", "uncollected_qty",
                                          "raw_rate", "realization_rate", "open_lines"]),
            global_rate=rate,
            measured=False,
            reason=f"fixed at {rate:.0%} by config (backlog_realization), not measured",
            as_of=as_of,
        )

    def _save_outputs(self, results: dict) -> None:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M")
        out = self.output_dir

        # ── CSV outputs ───────────────────────────────────────────────────────
        write_csv(results["supplier_lt"], out / "supplier_params.csv")
        write_csv(results["classified_demand"], out / "sku_planning_params.csv")
        write_csv(results["projection"], out / f"inventory_projection_{ts_str}.csv")
        write_csv(results["forecast_detail"], out / f"forecast_detail_{ts_str}.csv")
        sheet = self.forecaster.history_and_forecast(
            results["time_series"], results["forecast_detail"])
        if len(sheet):
            write_csv(sheet, out / f"forecast_{ts_str}.csv")
        write_csv(results["recommendations"], out / f"purchase_recommendations_{ts_str}.csv")

        profile = results.get("policy_profile")
        if profile is not None and len(profile.frame):
            write_csv(profile.frame, out / f"policy_profile_{ts_str}.csv")

        realization = results.get("backlog_realization")
        if realization is not None and len(realization.per_sku):
            write_csv(realization.per_sku, out / f"backlog_realization_{ts_str}.csv")

        # Save planning snapshot for next-month feedback comparison
        try:
            policy_cfg = json.loads((self.config_dir / "stocking_policy.json").read_text(encoding="utf-8"))
            from .store.fact_store import history_root
            snapshot_path = SnapshotSaver().save(
                results, policy_cfg, out,
                history_root=history_root(self.store_root),
            )
            print(f"  Snapshot saved:   {snapshot_path.name}")
        except Exception as e:
            print(f"  Warning: snapshot save failed ({e})")

        self._record_run(results, ts_str)

        print(f"\n  Outputs saved to: {out}")

    def _record_run(self, results: dict, ts_str: str) -> None:
        """
        Bind this run's outputs to the facts, parameters and code behind them.

        Wrapped because provenance must never be able to fail the run it documents —
        an unwritable registry is worth a warning, not a lost plan.
        """
        try:
            for path in sorted(self.output_dir.glob(f"*_{ts_str}.*")):
                self.run.record_output(path)
            for name in ("supplier_params.csv", "sku_planning_params.csv"):
                path = self.output_dir / name
                if path.exists():
                    self.run.record_output(path)
            manifest_path = RunRegistry(self.output_dir).save(self.run)
            print()
            print(self.run.summary())
            print(f"    manifest {manifest_path.name}")
        except Exception as e:
            print(f"  Warning: run manifest not written ({e})")

    def _print_summary(self, recommendations: pd.DataFrame) -> None:
        print("\n" + "="*60)
        print("  PURCHASE RECOMMENDATIONS SUMMARY")
        print("="*60)
        action_counts = recommendations["recommended_action"].value_counts()
        for action, count in action_counts.items():
            print(f"  {action:<30} {count:>5} SKUs")

        # Named at the top of the summary, ranked by how long the shelf is bare, and
        # with the ask spelled out. A supply gap is not a quantity to order — the order
        # exists — it is a delivery date to go and get, and the planner has to know
        # which supplier to ring before anything else in this report matters.
        gap = recommendations[recommendations.get("supply_gap", False) == True]
        if len(gap):
            gap = gap.sort_values("supply_gap_days", ascending=False)
            print()
            print(f"  \033[91m*** SUPPLY GAP — {len(gap)} SKU(s) run dry before the next "
                  f"delivery ***\033[0m")
            print(f"  Confirm the ship date with the supplier daily and price the air freight; "
                  f"expediting is the action, not a purchase order.")
            for _, r in gap.head(10).iterrows():
                late = (f", {r['inbound_past_due_qty']:,.0f} already past due"
                        if r.get("inbound_past_due_qty", 0) > 0 else "")
                nxt = (f"next arrival in {r['days_to_next_arrival']:.0f}d"
                       if pd.notna(r.get("days_to_next_arrival")) else "no arrival scheduled")
                print(f"    {str(r['sku']):<14} {r['on_hand_cover_days']:>6.1f}d on the shelf, "
                      f"bare for {r['supply_gap_days']:>5.1f}d — {nxt}{late}")
            if len(gap) > 10:
                print(f"    … and {len(gap) - 10} more, in the recommendations CSV")

        purchase_skus = recommendations[recommendations["recommended_action"] == "PURCHASE-REQUEST"]
        if len(purchase_skus):
            total_qty = purchase_skus["suggested_po_qty"].sum()
            print(f"\n  Total suggested purchase qty : {total_qty:,.0f}")

        if "demand_driver" in recommendations.columns:
            basis = recommendations["demand_basis"].iloc[0] if len(recommendations) else "n/a"
            drivers = recommendations["demand_driver"].value_counts().to_dict()
            print(f"\n  Demand basis  : {basis}  {drivers}")
            if basis == "forecast_consumption":
                print("                  Forecast and backlog are two estimates of one "
                      "demand, so the requirement is the larger — not the sum.")

        pushout_skus = recommendations[recommendations["recommended_action"] == "PUSH-OUT-OPEN-PO"]
        if len(pushout_skus):
            pushout_qty = pushout_skus["pushout_open_po_qty"].sum()
            print(f"  Total push-out candidate qty : {pushout_qty:,.0f}")

    def _summarize_from_pivot(self, ts: pd.DataFrame, meta: pd.DataFrame = None) -> pd.DataFrame:
        """Build a demand_summary DataFrame from a pre-compiled pivot."""
        import numpy as np
        location_id = self.inv_reader.location_id
        rows = []
        for sku in ts.columns:
            series = ts[sku]
            active = int((series > 0).sum())
            total = int(len(series))
            mean_d = float(series[series > 0].mean()) if active > 0 else 0.0
            std_d = float(series.std())
            rows.append({
                "sku": sku,
                "location_id": location_id,
                "demand_mean": round(mean_d, 2),
                "demand_std": round(std_d, 2),
                "demand_cv": round(std_d / mean_d, 3) if mean_d > 0 else np.nan,
                "active_cycles": active,
                "total_cycles": total,
                "total_qty": float(series.sum()),
                "total_amount": np.nan,
                "first_sale": ts.index[series > 0][0].to_timestamp() if active > 0 else pd.NaT,
                "last_sale":  ts.index[series > 0][-1].to_timestamp() if active > 0 else pd.NaT,
                # Enrich with metadata if available
                "description": meta.loc[sku, "description"] if (meta is not None and sku in meta.index and "description" in meta.columns) else "",
                "sopc_class":  meta.loc[sku, "sopc_classification"] if (meta is not None and sku in meta.index and "sopc_classification" in meta.columns) else "",
            })
        return pd.DataFrame(rows)

    def _check_quality(self, report: dict) -> None:
        self._quality_log.append(report)
        if report.get("issues"):
            if self.interactive:
                print(f"\n  ⚠  Quality issues found in {report['doc_type']}. Continue? [Y/n]: ", end="")
                if input().strip().lower() in ("n", "no"):
                    raise ValueError(f"User aborted after quality check on {report['doc_type']}")
