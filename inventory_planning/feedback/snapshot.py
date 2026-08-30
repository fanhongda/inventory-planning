"""
Snapshot: saves a planning run's key outputs alongside the parameters used.
Called automatically at the end of each run_planning() call.

Snapshot is stored as:
  output/history/YYYY-MM/snapshot_YYYYMMDD_HHMM.json

The JSON contains:
  - run metadata (date, model versions, config params)
  - per-SKU: forecast_next_period, suggested_po_qty, safety_stock, days_of_supply,
    demand_pattern, and the lead time the run planned on
  - path to the full CSV outputs for that run

Next month, the user fills in actuals by calling FeedbackCollector.record_actuals().

## Why lead time is in here

Everything else recorded per SKU is an *output* to be scored against what happened.
Lead time is different: it is an input, and it is the input most likely to have moved
without anyone noticing. A supplier drifting from 45 to 62 days changes the reorder
point, the safety stock and the exposure period all at once, and each monthly run
simply recomputes it and carries on — the drift is invisible unless successive runs
are compared.

So the lead time actually used is recorded here alongside its sigma, its sample count
and its source. `feedback.drift` reads them back. The number stored is the one the
safety stock calculation consumed, not a re-derivation from PO history: a value the
snapshot computes for itself can disagree with the plan it claims to describe.
"""

import json
from datetime import datetime
from pathlib import Path
import pandas as pd


class SnapshotSaver:

    @staticmethod
    def _published_plan(results: dict) -> list:
        """
        Every period of the horizon as it was planned, with both numbers where a sales
        review moved one.

        Kept flat and small — sku, period, two quantities and a source — because it is
        written every run and read by every later one. The rest of the forecast detail
        (model, backtest scores) is already in `forecast_detail_<run>.csv`; what is
        needed here is only what a future actual can be compared against.
        """
        detail = results.get("forecast_detail")
        if detail is None or not len(detail):
            return []
        columns = {"sku", "period", "forecast_qty"} & set(detail.columns)
        if len(columns) < 3:
            return []
        rows = []
        for _, row in detail.iterrows():
            entry = {
                "sku": str(row["sku"]),
                "period": str(row["period"]),
                "forecast_qty": float(row["forecast_qty"]),
            }
            if "statistical_qty" in detail.columns and pd.notna(row.get("statistical_qty")):
                entry["statistical_qty"] = float(row["statistical_qty"])
            if "forecast_source" in detail.columns:
                entry["forecast_source"] = str(row.get("forecast_source") or "statistical")
            rows.append(entry)
        return rows

    def save(self, results: dict, config: dict, output_dir: Path,
             history_root: Path = None, stamp: str = None) -> Path:
        """
        results:      output of InventoryPlanner.run_planning()
        config:       stocking_policy.json contents
        output_dir:   the run's output directory (where CSVs were saved)
        history_root: where snapshots accumulate. Defaults to the old location.

        Returns path to the saved snapshot JSON.

        `history_root` exists because deriving it from `output_dir.parent` made the
        location depend on the shape of an unrelated argument: `--output output` wrote
        to a git-*tracked* `history/` at the repository root while `--output output/real`
        wrote to the ignored `output/history/`. Two homes for one series, and the
        default was the one that commits planning data. The caller now resolves it —
        to the fact store, which is outside the working tree — and the fallback is kept
        so a direct caller passing only three arguments still lands where it used to.
        """
        run_dt = datetime.now()
        month_label = run_dt.strftime("%Y-%m")
        # The run's id when the caller has one, so a snapshot names the run that wrote
        # it and two runs a few seconds apart do not land on the same filename.
        ts_str = stamp or run_dt.strftime("%Y%m%d_%H%M")

        root = Path(history_root) if history_root else output_dir.parent / "history"
        history_dir = root / month_label
        history_dir.mkdir(parents=True, exist_ok=True)

        rec = results.get("recommendations", pd.DataFrame())
        proj = results.get("projection", pd.DataFrame())
        classified = results.get("classified_demand", pd.DataFrame())

        skus = {}
        for _, row in rec.iterrows():
            sku = row["sku"]
            skus[sku] = {
                "recommended_action": row.get("recommended_action"),
                "suggested_po_qty": float(row.get("suggested_po_qty", 0)),
                "forecast_next_period": float(row.get("forecast_next_period", 0)),
                "forecast_avg_monthly": float(row.get("forecast_avg_monthly", 0)),
                # Both numbers, so next period can score the review rather than argue
                # about it. `forecast_next_period` is what was planned on — the
                # reviewed figure where sales changed it — and this is what the model
                # said before they did. Storing only the first makes the question
                # "did the adjustment help?" permanently unanswerable, and a review
                # process that cannot be scored is one that never improves.
                "statistical_next_period": float(
                    row.get("statistical_next_period",
                            row.get("forecast_next_period", 0)) or 0),
                "forecast_source": row.get("forecast_source", "statistical"),
                "net_requirement": float(row.get("net_requirement", 0)),
                "available_supply": float(row.get("available_supply", 0)),
                "safety_stock": float(row.get("safety_stock", 0)),
            }

        # Enrich with projection fields
        for _, row in proj.iterrows():
            sku = row["sku"]
            if sku in skus:
                skus[sku].update({
                    "days_of_supply": float(row.get("days_of_supply", 0)) if pd.notna(row.get("days_of_supply")) else None,
                    "inventory_status": row.get("inventory_status"),
                })

        # Enrich with demand pattern
        for _, row in classified.iterrows():
            sku = row["sku"]
            if sku in skus:
                skus[sku].update({
                    "demand_pattern": row.get("demand_pattern"),
                    "demand_cv": float(row.get("demand_cv", 0)) if pd.notna(row.get("demand_cv")) else None,
                    "stocking_class": row.get("stocking_class"),
                })

        # Enrich with the lead time the run planned on
        for sku, params in self._lead_time_by_sku(results).items():
            if sku in skus:
                skus[sku].update(params)

        snapshot = {
            "run_at": run_dt.isoformat(),
            "planning_month": month_label,
            # The plan as published, across the whole horizon rather than only t+1.
            #
            # Forecast accuracy that matters to an S&IOP meeting is not how a model
            # scored on held-out history — it is whether the number the business
            # committed to turned out to be right. Answering that needs the plan for
            # every period, kept, so a later run can put the actual beside it. t+1
            # alone can only ever score the month that was already nearly certain.
            "as_of": str(results.get("as_of") or ""),
            "plan": self._published_plan(results),
            "config_snapshot": {
                "cv_intermittent_threshold": config.get("cv_intermittent_threshold"),
                "cv_erratic_threshold": config.get("cv_erratic_threshold"),
                "excess_dos_threshold_days": config.get("excess_dos_threshold_days"),
                "service_level_metric": config.get("service_level_metric"),
                "stocking_tiers": config.get("stocking_tiers"),
            },
            "output_dir": str(output_dir),
            "sku_count": len(skus),
            "skus": skus,
            # Actuals filled in next month by FeedbackCollector
            "actuals": {},
            "loss": {},
        }

        path = history_dir / f"snapshot_{ts_str}.json"
        path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    @staticmethod
    def _lead_time_by_sku(results: dict) -> dict:
        """
        The lead time each SKU was planned on, with the evidence behind it.

        Taken from the safety-stock frame rather than recomputed from PO history,
        because that frame holds the value the calculation actually consumed —
        including a lead time that came from an item master because no receipts
        existed. Recomputing here would let the record disagree with the plan.

        `sample_count`, `supplier` and `lt_source` are not on that frame, so they come
        from the supplier lead-time table using the same choice the planning layer
        makes: the supplier with the most history, not the fastest one.
        """
        ss = results.get("safety_stock")
        if ss is None or len(ss) == 0 or "wma_lead_time_days" not in ss.columns:
            return {}

        evidence = {}
        supplier_lt = results.get("supplier_lt")
        if supplier_lt is not None and len(supplier_lt) and "sku" in supplier_lt.columns:
            sort_col = next((c for c in ("order_count", "sample_count")
                             if c in supplier_lt.columns), None)
            chosen = (
                supplier_lt.sort_values(sort_col, ascending=False)
                if sort_col else supplier_lt
            ).groupby("sku", as_index=False).first()
            for _, row in chosen.iterrows():
                evidence[row["sku"]] = {
                    "supplier": row.get("supplier"),
                    "lt_samples": _int_or_none(row.get("sample_count")),
                    # `lt_source` is absent on runs with no master data; those lead
                    # times are measured by construction.
                    "lt_source": row.get("lt_source") or "measured",
                }

        out = {}
        for _, row in ss.iterrows():
            sku = row["sku"]
            lead_time = _float_or_none(row.get("wma_lead_time_days"))
            if lead_time is None:
                continue
            entry = {
                "lead_time_days": lead_time,
                "lt_sigma_days": _float_or_none(row.get("lt_std_days")),
            }
            entry.update(evidence.get(sku, {"supplier": None, "lt_samples": None,
                                            "lt_source": "measured"}))
            out[sku] = entry
        return out


def _float_or_none(value):
    return float(value) if pd.notna(value) else None


def _int_or_none(value):
    return int(value) if pd.notna(value) else None
