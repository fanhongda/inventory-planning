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
from .analytics.sop import SOPWorksheet
from .analytics.siop import build_siop_plan
from .analytics.inventory_health import build_inventory_health
from .analytics.forecast_accuracy import build_forecast_accuracy
from .analytics.sales_plan import apply_sales_plan, read_sales_plan
from .ingest.encoding import write_csv
from .quality import DataQualityError, GateReport, GateThresholds
from .quality import assess as assess_run_health
from .quality import checks as quality_checks
from .store.declarations import Declarations


def _price_lookup(sop):
    """
    Selling price per SKU from the S&OP worksheet's price basis, or nothing.

    Accuracy is weighted by what a SKU is worth, and worth here means revenue: the
    forecast is the demand plan, and demand is the thing sales committed to. The basis
    already carries its own provenance label, so a SKU priced from a fallback is
    weighted the same as one priced from realised revenue — which is right, because the
    alternative is dropping it from the measurement altogether.
    """
    price = getattr(sop, "price", None)
    frame = getattr(price, "frame", None)
    if frame is None or not len(frame) or "asp" not in frame.columns:
        return None
    priced = frame.dropna(subset=["asp"]).drop_duplicates("sku")
    if not len(priced):
        return None
    return pd.Series(
        pd.to_numeric(priced["asp"], errors="coerce").values,
        index=priced["sku"].astype(str).tolist(),
    )


class InventoryPlanner:
    """
    Single-stage DC inventory planning pipeline.
    multi-echelon ready: all outputs carry location_id.
    """

    def __init__(self, config_dir: Union[str, Path] = None, output_dir: Union[str, Path] = None,
                 interactive: bool = True, store_root: Union[str, Path] = None,
                 parameters_file: Union[str, Path] = None,
                 allow_degraded: bool = False):
        base = Path(__file__).parents[1]
        self.config_dir = Path(config_dir) if config_dir else base / "config"
        self.output_dir = Path(output_dir) if output_dir else base / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interactive = interactive
        # The rule set this planner plans under. A scenario is a planner built with a
        # different one: the rules decide the review period that sizes an order and the
        # service level that sizes safety stock, so they have to reach `run_planning`
        # rather than only the policy report, or a scenario changes nothing a buyer acts
        # on. One planner, one rule set, one run identity.
        self.parameters_file = (Path(parameters_file) if parameters_file
                                else self.config_dir / "planning_parameters.md")
        self._quality_log: list = []   # accumulates quality reports across all loads
        # Quality gates. `allow_degraded` proceeds past a BLOCK finding, and is not a
        # convenience: a threshold is a judgement and judgements are occasionally wrong
        # on data nobody anticipated. Every override is recorded on the gate report and
        # travels into the manifest, so an output produced under one says so.
        self.allow_degraded = bool(allow_degraded)
        self.gate_thresholds = GateThresholds.load(self.config_dir)
        # Per-check, per-document waivers, and the mapping corrections that reach
        # intake. Loaded here rather than passed in so a planner built any of the four
        # ways honours the same declared file.
        self.declarations = Declarations.load(self.config_dir)
        self._gate_reports: list = []
        self._intake = None            # set by load_all(); carries adapter provenance
        self._intake_plan = None       # set by load_all(); what this run can answer
        self._fx = None                # set by load_all(); which money was restated, and what could not be
        # Identity for this run: what it read, what it resolved, what code ran. Started
        # here rather than at save time so the config is fingerprinted before anything
        # has had a chance to be edited mid-run.
        self.run = RunManifest.begin(config_dir=self.config_dir, output_dir=self.output_dir,
                                     policy_file=self.parameters_file)
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
        self.sop           = SOPWorksheet(horizon=policy_cfg["forecast_horizon_months"])
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

    def load_sales_plan(self, path: Union[str, Path]):
        """
        Read a reviewed S&OP worksheet — the forecast after sales have adjusted it.

        Loaded by name and not through `load_all`, because this is the one input that
        is a decision rather than an extract. Overwriting the demand plan is an act
        that should be named: a spreadsheet that happened to be sitting in an input
        folder must not be able to do it by being profiled.
        """
        plan = read_sales_plan(path)
        self.run.record_input(path, doc_type="sales_plan", rows=len(plan.adjustments))
        print()
        print(plan.summary())
        return plan

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
        should_be = calculator.calculate(resolved, actual=inventory_df,
                                         committed=results.get("mto_schedule"))
        print()
        print(should_be.summary())

        # Days on hand rolled up to the product line, with the two tails pulled out.
        # Built off `should_be` because that is where the position is already valued;
        # the demand series supplies when each item last sold.
        health = build_inventory_health(
            should_be=should_be,
            time_series=results.get("time_series"),
            attributes=attributes,
            inventory=inventory_df,
            currency=str(getattr(self._fx, "reporting_currency", "USD") or "USD"),
        )
        print()
        print(health.summary())

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
        stamp = self.run.run_id
        md_path = self.output_dir / f"suggested_rules_{stamp}.md"
        suggestions.to_rules_markdown(md_path)
        print(f"\n    Paste-able rules    : {md_path}")

        if len(crosscheck.all_disagreements):
            xc_path = self.output_dir / f"source_crosscheck_{stamp}.csv"
            write_csv(crosscheck.frame(), xc_path)
            print(f"    Source disagreements: {xc_path}")

        out = {
            "sku_attributes": attributes,
            "parameters": resolved,
            "should_be": should_be,
            "inventory_health": health,
            # Forwarded from the planning stage so the KPI review, which is handed the
            # policy dict and not the planning one, can render them without a second
            # argument nobody would remember to pass.
            "siop": results.get("siop"),
            "forecast_accuracy": results.get("forecast_accuracy"),
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

        # Rewritten now that the policy stage has produced the half of the workbook the
        # planning stage could not: should-be, the suggestions, the S&IOP projection.
        workbook = self._save_workbook(results, out)
        if workbook is not None:
            print(f"\n    Planning workbook   : {workbook}")

        self._quality_log.append({
            "doc_type": "policy",
            "file": str(Path(parameters_file).name if parameters_file
                        else self.parameters_file.name),
            "rows_loaded": len(attributes),
            "issues": [str(h) for h in resolved.conflicts],
            "status": "WARNINGS" if resolved.conflicts else "OK",
        })
        # This stage writes outputs of its own after `run_planning` recorded the
        # manifest, so the outputs are collected again and the manifest rewritten. The
        # fingerprints do not move — the rule set was fixed when the run began.
        self._record_run_state()
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

        stamp = self.run.run_id
        path = self.output_dir / f"kpi_review_{stamp}.html"
        KPIReport(title).render(
            service=service, should_be=should_be, ordering=ordering,
            cadence=cadence, forward=forward, levers=policy.get("levers"),
            frontier=frontier, fx=self._fx, service_target=service_target,
            attributes=attributes, recommendations=policy.get("recommendations"),
            open_po=open_po_df, suggestions=policy.get("suggestions"),
            health=self.run_health(),
            siop=policy.get("siop"),
            accuracy=policy.get("forecast_accuracy"),
            inventory_health=policy.get("inventory_health"),
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
                "source columns went unmatched.\n\n"
                "    Check config/declarations.yaml first: a `scope: mapping` override "
                "there is reviewed and attributable, and it outranks a frozen adapter "
                "without you having to hand-edit one — see the worked examples already "
                "in that file. Add an alias to the contract only when the right column "
                "will never be ambiguous for anyone else's export of this document."
            )
        # Everything above is about a document being absent or unreadable. This is
        # about the documents being present, readable, and not describing the same
        # business — the failure that produces a full report of zeroes.
        self._run_gate(quality_checks.gate_intake(
            self._intake, plan, self.gate_thresholds))

        # The gate reports the spelling collisions; this applies the fold. Both halves
        # are needed and neither substitutes for the other — folding without reporting
        # merges two of somebody's business units on the pipeline's own authority, and
        # reporting without folding leaves every rollup split while the console says it
        # noticed. Done after the gate so what is reported is what arrived.
        self._normalise_dimensions(inputs)

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

    # Columns a rollup, a forecast segment or a review sheet groups by. Only these:
    # folding a free-text column nobody groups by is churn, and folding an identifier
    # would merge two real things.
    _DIMENSION_COLUMNS = ("business_unit", "product_family", "country", "region")

    def _normalise_dimensions(self, inputs: dict) -> None:
        """
        Fold each dimension column onto one spelling per thing, everywhere at once.

        "Everywhere at once" for the same reason the supersession rewrite is: the sales
        history and the master both carry a business unit, and folding `Hydraulics Segment`
        into `Hydraulics` in one of them and not the other produces a segment that has
        history and no items, beside one that has items and no history.
        """
        from .quality.dimensions import normalise_frame

        for key, frame in list(inputs.items()):
            if not isinstance(frame, pd.DataFrame) or not len(frame):
                continue
            folded, collisions = normalise_frame(frame, self._DIMENSION_COLUMNS)
            if collisions:
                inputs[key] = folded

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

        stamp = self.run.run_id
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
        sales_plan=None,                         # reviewed forecast from load_sales_plan
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
        self._run_gate(quality_checks.gate_demand(
            ts, as_of, stale if as_of else None, self.gate_thresholds))

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
        # Sales have the last word on the quantity, and no word at all on the error.
        # Applied here, before `summary()` builds the frame that safety stock and the
        # recommender read, so there is exactly one forecast downstream rather than a
        # statistical one and a reviewed one that some steps use and others do not.
        override = apply_sales_plan(forecast_detail, sales_plan)
        forecast_detail = override.forecast_detail
        if override.applied or sales_plan is not None:
            print()
            print(override.summary())

        forecast_summary_df = self.forecaster.summary(forecast_detail)
        if not forecast_detail.empty:
            model_counts = forecast_detail.drop_duplicates("sku")["model_used"].value_counts().to_dict()
            # Counts of SKUs, not one choice for the run. Each SKU is scored on its
            # own history and keeps its own winner; `model_used` on every row of
            # forecast_detail and forecast_<ts>.csv says which.
            print(f"      Model chosen per SKU — {model_counts}")
        self._run_gate(quality_checks.gate_forecast(
            forecast_detail, ts, self.gate_thresholds))

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

        # The same forecast, written for the other audience. Built here rather than
        # beside the forecast itself because the segmentation it is grouped by comes
        # from `attributes`, which the policy step above is what resolves.
        sop = self.sop.build(
            time_series=ts,
            forecast_detail=forecast_detail,
            sales_df=sales_df,
            attributes=attributes,
            unit_cost=attributes if "unit_cost" in getattr(attributes, "columns", [])
            else None,
        )
        print()
        print(sop.price.summary())
        print()
        print(sop.summary())

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
        self._run_gate(quality_checks.gate_plan(
            forecast_summary_df if not forecast_detail.empty else None,
            inventory_df, self.gate_thresholds,
            # So the check can tell an item that should be on the shelf from one the
            # policy never intended to hold. Without them every make-to-order item
            # counts as a missing position, which is the ordinary state of the data.
            attributes=attributes, classified=classified))
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

        # The period balance, in money. Built here because it is the first point at
        # which all four of its inputs exist: the bucketed forecast, the open order
        # book with delivery dates filled in, the opening position, and the safety
        # stock the gap is measured against.
        siop = build_siop_plan(
            forecast_detail=forecast_detail,
            inventory=inventory_df,
            open_po=open_po_df,
            attributes=attributes,
            safety_stock=ss_df,
            currency=str(getattr(self._fx, "reporting_currency", "USD") or "USD"),
        )
        print()
        print(siop.summary())

        # Whether the plan this business published last time turned out to be right —
        # the published number against what then sold, not the model's own backtest.
        # Needs history, so it says nothing on a first run rather than substituting a
        # statistic that would look like an answer. Built after the override has been
        # applied so `statistical_qty` and `forecast_qty` are both on the frame.
        from .store.fact_store import history_root
        accuracy = build_forecast_accuracy(
            time_series=ts,
            history_root=history_root(self.store_root),
            forecast_detail=forecast_detail,
            attributes=attributes,
            price=_price_lookup(sop),
            currency=str(getattr(self._fx, "reporting_currency", "USD") or "USD"),
            # This run's own snapshot is written after the analytics, but a re-run into
            # the same store would otherwise score a plan against the actuals it was
            # built from and report a perfect forecast.
            exclude_run=self.run.run_id,
        )
        print()
        print(accuracy.summary())

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
            # Carried so the policy layer sizes an order-on-demand item against the same
            # commitment the recommender buys for. Rebuilding it there would give the
            # two stages their own copy of one number, and the first time they drifted
            # the report would call excess the stock the recommendation had just asked
            # for.
            "mto_schedule": mto_schedule,
            "sop": sop,
            "siop": siop,
            "forecast_accuracy": accuracy,
            "sales_plan_override": override,
            "intake_summary": intake,
            "inventory_consolidated": inventory_df,
            "item_master": item_master_df,
            "planning_master": planning_master_df,
            # Carried for the policy layer's benefit: it is the only source of a
            # measured unit cost on an extract whose inventory export is quantity-only.
            "po_history": po_history_df,
            "as_of": as_of,
            "_quality_reports": self._quality_log,
            "gate_reports": self._gate_reports,
            "run_health": self.run_health(),
        }

        self._save_outputs(results)
        self._print_summary(recommendations)
        # Last, deliberately. A caveat printed before the numbers it qualifies is read
        # as preamble; printed after them it is read as it is meant to be — the answer
        # to "how much of that do I believe".
        print(results["run_health"].console())
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

        params_file = Path(parameters_file) if parameters_file else self.parameters_file
        planning_params = PlanningParameters(params_file)
        self.run.note_policy_override(params_file)
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

    def _save_workbook(self, results: dict, policy: dict = None):
        """
        The run as one file. Never allowed to fail the run it reports on.

        Called twice — once with what planning produced, once with the policy stage's
        additions on top — because a run that stops at the gate in between should still
        leave something readable rather than a folder of nothing.
        """
        from .reporting.workbook import build_workbook

        path = self.output_dir / f"planning_{self.run.run_id}.xlsx"
        try:
            written = build_workbook(
                path, results, policy,
                currency=str(getattr(self._fx, "reporting_currency", "USD") or "USD"))
        except Exception as e:                     # pragma: no cover - reported, not raised
            print(f"  Warning: workbook not written ({e})")
            return None
        if written is not None:
            self.run.record_output(written)
        return written

    def _save_outputs(self, results: dict) -> None:
        ts_str = self.run.run_id
        out = self.output_dir

        # ── The workbook ──────────────────────────────────────────────────────
        #
        # Sixteen CSVs named for the stage that produced them, where the question a
        # planner arrives with is answered by joining four of them. Written here with
        # whatever the planning stage has, and again by the policy stage with the rest —
        # same path, so a run that stops early still leaves a readable file.
        results["forecast_sheet"] = self.forecaster.history_and_forecast(
            results["time_series"], results["forecast_detail"])
        self._save_workbook(results)

        # Kept as its own CSV: SKU x supplier, which no per-SKU sheet can hold without
        # either dropping a supplier or repeating an item.
        write_csv(results["supplier_lt"], out / "supplier_params.csv")

        sop = results.get("sop")
        if sop is not None and len(sop.sheet):
            write_csv(sop.sheet, out / f"sop_worksheet_{ts_str}.csv")
            try:
                xlsx = out / f"sop_worksheet_{ts_str}.xlsx"
                self._write_sop_workbook(xlsx, sop)
                self.run.record_output(xlsx, rows=len(sop.sheet))
            except Exception as e:
                print(f"  Warning: S&OP workbook not written ({e})")

        # What each checkpoint found, including the ones that passed. Written every run
        # rather than only on failure: "the gates found nothing" is a statement about
        # this extract, and a file that appears only when something is wrong cannot
        # make it. It is also the record of an override — an output produced under
        # allow_degraded has to be able to say so after the console has scrolled away.
        health = results.get("run_health")
        if health is not None:
            gate_path = out / f"quality_gates_{ts_str}.json"
            gate_path.write_text(json.dumps(health.to_dict(), indent=2,
                                            ensure_ascii=False), encoding="utf-8")
            self.run.record_output(gate_path)
            # Also as prose. The JSON is for the report builder; this is for the person
            # who opens the output folder a week later and has to decide whether the
            # numbers in it were ever safe to use.
            health_path = out / f"run_health_{ts_str}.md"
            health_path.write_text(health.markdown(), encoding="utf-8")
            self.run.record_output(health_path)

        # Save planning snapshot for next-month feedback comparison
        try:
            policy_cfg = json.loads((self.config_dir / "stocking_policy.json").read_text(encoding="utf-8"))
            from .store.fact_store import history_root
            snapshot_path = SnapshotSaver().save(
                results, policy_cfg, out,
                history_root=history_root(self.store_root),
                stamp=self.run.run_id,
            )
            print(f"  Snapshot saved:   {snapshot_path.name}")
        except Exception as e:
            print(f"  Warning: snapshot save failed ({e})")

        self._record_run()

        print(f"\n  Outputs saved to: {out}")

    def _collect_outputs(self) -> None:
        """
        Record every file this run has written so far. Safe to run again.

        Run again is the point: `_save_outputs` collects at the end of `run_planning`,
        and the policy stage writes `parameter_suggestions`, `suggested_rules` and the
        source cross-check afterwards. Collecting only once left those out of the
        manifest — including the suggested rules, which is the output a planner acts on
        when tuning a scenario and so the one that least deserves to be the one with no
        provenance.
        """
        for path in sorted(self.output_dir.glob(f"*_{self.run.run_id}.*")):
            if path.is_file():
                self.run.record_output(path)
        for name in ("supplier_params.csv", "sku_planning_params.csv"):
            path = self.output_dir / name
            if path.exists():
                self.run.record_output(path)

    def _write_sop_workbook(self, path: Path, sop) -> None:
        """
        The review sheet as a workbook, with somewhere to write the answer.

        Three things make this different from dumping the frame to xlsx. The
        **adjustment columns are empty and adjacent** — a reviewer types beside the
        month they are arguing with, not on a second tab. The **statistical columns are
        left alone**, so what the model said survives next to what sales decided and
        the two can be compared afterwards; overwriting the forecast in place would
        destroy the only evidence of whether the review helped. And the **rollup goes
        on its own sheet**, because the argument happens at product-line level and the
        SKU detail is what gets consulted when somebody disputes the rollup.

        Written with pandas rather than by hand: cell formatting is not worth a
        dependency on openpyxl's styling API, and a reviewer will reformat it anyway.
        """
        sheet = sop.sheet.copy()

        # One editable column per forecast month, plus a place to say why. Empty on
        # purpose — pre-filling them with the statistical number means a reviewer who
        # changes nothing has silently endorsed everything, and one who changes one
        # month leaves no signal that the others were considered.
        for period in sop.horizon_periods:
            sheet[f"REVIEWED qty {period}"] = np.nan
        sheet["REVIEWED by"] = ""
        sheet["REVIEWED reason"] = ""

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            sheet.to_excel(writer, sheet_name="Review by SKU", index=False)

            group_cols = [c for c in ("business_unit", "product_family", "country")
                          if c in sheet.columns and sheet[c].notna().any()]
            if group_cols:
                measures = [c for c in sheet.columns
                            if c.startswith(("hist amt ", "fcst amt "))
                            or c in ("history_amount_total", "forecast_amount_total",
                                     "history_qty_total", "forecast_qty_total")]
                rollup = (sheet.groupby(group_cols, dropna=False)[measures]
                          .sum().reset_index())
                rollup.to_excel(writer, sheet_name="Rollup", index=False)

            # What the numbers rest on, in the file rather than in a chat message. A
            # worksheet that circulates for three weeks outlives every explanation
            # given when it was sent.
            notes = pd.DataFrame({"note": [
                "History is what was invoiced. Amounts are as booked at the time, not "
                "restated at today's price.",
                "Forecast amount = forecast quantity x average selling price. "
                "`price_basis` on each row says where that price came from; "
                "`standard cost (not a price)` means there is no margin in that row.",
                "`vs_naive` is the model's error over the error of simply repeating "
                "last month. 1.00 means the model added nothing.",
                "Blank amount means no price could be established. It is blank rather "
                "than zero so it cannot sum into a total unnoticed.",
                "Type into the REVIEWED qty columns. Leave the others alone — keeping "
                "the statistical forecast beside your number is how we find out, next "
                "quarter, which of the two was closer.",
                "A blank REVIEWED cell means 'no change', and the statistical forecast "
                "stands for that month.",
            ]})
            notes.to_excel(writer, sheet_name="How to read this", index=False)

        print(f"  S&OP worksheet:   {path.name}")

    def _record_run_state(self) -> None:
        """Write the manifest as it now stands. Safe to call more than once per run."""
        try:
            self._collect_outputs()
            RunRegistry(self.output_dir).save(self.run)
        except Exception as e:
            print(f"  Warning: run manifest not updated ({e})")

    def _record_run(self) -> None:
        """
        Bind this run's outputs to the facts, parameters and code behind them.

        Wrapped because provenance must never be able to fail the run it documents —
        an unwritable registry is worth a warning, not a lost plan.
        """
        try:
            self._collect_outputs()
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
                print(f"    … and {len(gap) - 10} more, on the Purchase sheet")

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

    def _run_gate(self, report: GateReport) -> GateReport:
        """
        Print a checkpoint's findings and stop the run if any of them block.

        Every gate goes through here so that the override, the recording and the
        message are the same wherever the check lives. A gate that raised on its own
        would be a gate that could be added without an override path or without leaving
        a trace, and the first time a threshold turned out to be wrong on somebody's
        data the answer would be to delete the check.
        """
        # Declared waivers first. A waiver is narrower than `allow_degraded` by
        # construction — one check on one document — so waiving the SKU-agreement
        # finding on a whole-warehouse stock snapshot does not also wave through an
        # open PO quantity that was mapped to a money column. The finding is not
        # hidden: it is downgraded, still printed, and still travels into the manifest
        # carrying who waived it and until when.
        declarations = getattr(self, "declarations", None)
        if declarations is not None:
            report = declarations.waive(report)

        self._gate_reports.append(report)
        if report.findings:
            print()
            print(report.summary())
        if report.passed:
            return report
        if not self.allow_degraded:
            raise DataQualityError(report)
        report.overridden = True
        print(f"\n  ⚠ Continuing past {len(report.blocking)} blocking "
              f"{'finding' if len(report.blocking) == 1 else 'findings'} at the "
              f"'{report.stage}' gate — allow_degraded is set. Every figure that "
              f"follows rests on data that did not pass.")
        return report

    @property
    def gate_reports(self) -> list:
        """Every checkpoint this run passed through, in order, with what it found."""
        return list(self._gate_reports)

    def run_health(self):
        """
        Every reservation attached to this run, collected and ranked.

        The findings were all printed as they occurred, which is the wrong moment for
        all but the first: by the time a planner reaches the recommendations, the one
        sentence that would have changed how they read the revenue table has scrolled
        past. This restates them where the outputs are.
        """
        return assess_run_health(
            run_id=self.run.run_id, gates=self._gate_reports,
            plan=self._intake_plan, allow_degraded=self.allow_degraded,
        )

    def _check_quality(self, report: dict) -> None:
        self._quality_log.append(report)
        if report.get("issues"):
            if self.interactive:
                print(f"\n  ⚠  Quality issues found in {report['doc_type']}. Continue? [Y/n]: ", end="")
                if input().strip().lower() in ("n", "no"):
                    raise ValueError(f"User aborted after quality check on {report['doc_type']}")
