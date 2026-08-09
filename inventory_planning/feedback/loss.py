"""
Loss function & deviation pattern analyzer.

Core design principle:
  Not all deviations between recommendations and actuals indicate model errors.
  Deviations must first be attributed before triggering any parameter change.

Attribution logic:
  If recommended to buy X but actual receipt < X, for N consecutive months
  → classified as OPERATOR_DEVIATION: model is correct, operator is not executing.
  → System maintains same recommendation next month. No parameter adjustment.

  If actual_demand > available_supply for N consecutive months
  → classified as CUMULATIVE_SUPPLY_GAP: real shortage risk building up.
  → Escalate as supply chain risk signal. Does NOT imply model is wrong.

  Only when deviations are NOT explained by operator pattern or supply gap
  → investigate model logic (forecast bias, wrong safety stock, etc.)

Metrics computed (per MIT CTL §2):
  MD   = mean(forecast_next_period − actual_demand)   [bias: + = over-forecast]
  MAD  = mean(|forecast_error|)
  MAPE = mean(|forecast_error / actual_demand|)        [skips zero-demand periods]

  supply_gap_monthly   = suggested_po_qty − actual_receipt_qty  [+ = under-bought]
  demand_gap_monthly   = actual_demand − available_supply        [+ = went short]

Cumulative gap tracking requires reading all snapshots in history/, not just one.
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


CONSECUTIVE_MONTHS_THRESHOLD = 3   # months of persistent gap to trigger classification


class LossCalculator:

    def __init__(self, snapshot_path: str | Path):
        self.path = Path(snapshot_path)
        with open(self.path) as f:
            self.snapshot = json.load(f)
        # history dir = parent of this snapshot's month folder
        self.history_dir = self.path.parent.parent

    # ── Public entry point ────────────────────────────────────────────────────

    def compute(self) -> dict:
        """
        Compute loss metrics and deviation attribution for the current snapshot.
        Reads prior snapshots to assess cumulative gaps.
        Returns dict with aggregate metrics, per-SKU detail, and attribution.
        """
        actuals = self.snapshot.get("actuals", {})
        if not actuals:
            raise ValueError("No actuals recorded — call FeedbackCollector.record_actuals() first.")

        # ── Per-SKU loss for this month ───────────────────────────────────────
        detail_df = self._compute_monthly_loss(self.snapshot)

        # ── Cumulative gap analysis across all completed snapshots ────────────
        history = self._load_history()   # list of (month, sku_loss_df) sorted oldest→newest
        cumulative = self._cumulative_gaps(history, detail_df)

        # ── Aggregate metrics (stocking items only) ───────────────────────────
        stocking = detail_df[detail_df["stocking_class"] != "non-stocking"].copy()
        valid_mape = stocking["mape_contrib"].dropna()

        agg = {
            "planning_month": self.snapshot.get("planning_month"),
            "sku_count_with_actuals": int(detail_df["actual_demand"].notna().sum()),
            "forecast": {
                "MD":   round(float(stocking["forecast_error"].mean()), 2),
                "MAD":  round(float(stocking["abs_forecast_error"].mean()), 2),
                "MAPE": round(float(valid_mape.mean()), 4) if len(valid_mape) else None,
                "bias_direction": self._bias_direction(stocking["forecast_error"]),
            },
            "inventory": {
                "avg_dos_realized": round(float(detail_df["dos_realized"].dropna().mean()), 1),
                "excess_rate": round(float((detail_df["dos_realized"].dropna() > 90).mean()), 3),
            },
            "cumulative": cumulative,
        }

        # Lead-time drift needs no actuals — it compares what successive runs planned
        # on — but it belongs in this report because it explains a class of gap the
        # loss numbers can only show as unexplained error.
        drift = self.lead_time_drift()
        agg["lead_time_drift"] = {
            "months": drift.months,
            "skus_tracked": drift.skus_tracked,
            "moves": drift.frame().to_dict(orient="records"),
            "source_changes": len(drift.source_changes),
            "reason": drift.reason,
        }

        # ── Attribution & suggestions ─────────────────────────────────────────
        attribution = self._attribute(detail_df, cumulative)
        agg["attribution"] = attribution
        agg["suggestions"] = self._suggest(agg, attribution)

        # Write back
        self.snapshot["loss"] = agg
        self.snapshot["loss_detail"] = detail_df.to_dict(orient="records")
        self.path.write_text(json.dumps(self.snapshot, indent=2, ensure_ascii=False))

        self._print_report(agg)
        return {"aggregate": agg, "detail": detail_df}

    def lead_time_drift(self, relative_threshold: float = None,
                        absolute_threshold: float = None):
        """
        How the lead times this pipeline plans on have moved across snapshots.

        Callable on its own — it reads plan-time parameters, so unlike `compute` it
        does not need actuals recorded, and it is worth a look the moment a second
        month exists.
        """
        from .drift import (
            DEFAULT_ABSOLUTE_THRESHOLD,
            DEFAULT_RELATIVE_THRESHOLD,
            LeadTimeDriftTracker,
        )

        tracker = LeadTimeDriftTracker(
            relative_threshold=(relative_threshold if relative_threshold is not None
                                else DEFAULT_RELATIVE_THRESHOLD),
            absolute_threshold=(absolute_threshold if absolute_threshold is not None
                                else DEFAULT_ABSOLUTE_THRESHOLD),
        )
        return tracker.track(self.history_dir)

    # ── Monthly loss computation ──────────────────────────────────────────────

    def _compute_monthly_loss(self, snapshot: dict) -> pd.DataFrame:
        skus_data = snapshot.get("skus", {})
        actuals = snapshot.get("actuals", {})
        rows = []

        for sku, planned in skus_data.items():
            actual = actuals.get(sku, {})
            actual_demand = actual.get("actual_demand")
            actual_eom_inv = actual.get("actual_eom_inv")
            actual_receipts = actual.get("actual_receipt_qty") or 0

            if actual_demand is None:
                continue

            forecast = planned.get("forecast_next_period") or 0
            suggested_po = planned.get("suggested_po_qty") or 0
            available = planned.get("available_supply") or 0

            forecast_error = forecast - actual_demand
            # supply_gap: how much less was received vs recommended
            supply_gap = suggested_po - actual_receipts if suggested_po > 0 else 0
            # demand_gap: how much demand exceeded available supply
            demand_gap = max(0, actual_demand - available)

            mape = abs(forecast_error / actual_demand) if actual_demand > 0 else None
            dos_realized = (actual_eom_inv / actual_demand * 30) if (actual_eom_inv and actual_demand > 0) else None

            rows.append({
                "sku": sku,
                "stocking_class": planned.get("stocking_class"),
                "demand_pattern":  planned.get("demand_pattern"),
                "forecast_next_period": forecast,
                "actual_demand":    actual_demand,
                "forecast_error":   round(forecast_error, 2),
                "abs_forecast_error": round(abs(forecast_error), 2),
                "mape_contrib":     round(mape, 4) if mape is not None else None,
                "suggested_po_qty": suggested_po,
                "actual_receipt_qty": actual_receipts,
                "supply_gap":       round(supply_gap, 1),   # + means under-bought
                "available_supply": available,
                "demand_gap":       round(demand_gap, 1),   # + means went short
                "actual_eom_inv":   actual_eom_inv,
                "dos_realized":     round(dos_realized, 1) if dos_realized else None,
            })

        return pd.DataFrame(rows)

    # ── History loading ───────────────────────────────────────────────────────

    def _load_history(self) -> list[tuple[str, pd.DataFrame]]:
        """
        Loads all completed snapshots (with actuals) from history/, sorted oldest first.
        Returns list of (month_label, loss_df).
        """
        results = []
        current_month = self.snapshot.get("planning_month")

        for month_dir in sorted(self.history_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            month_label = month_dir.name
            if month_label >= current_month:   # exclude current month
                continue
            for snap_file in sorted(month_dir.glob("snapshot_*.json")):
                with open(snap_file) as f:
                    snap = json.load(f)
                if snap.get("actuals"):   # only snapshots with actuals recorded
                    df = self._compute_monthly_loss(snap)
                    if not df.empty:
                        results.append((month_label, df))
                        break   # one snapshot per month is enough

        # Append current month
        results.append((current_month, self._compute_monthly_loss(self.snapshot)))
        return results

    # ── Cumulative gap analysis ───────────────────────────────────────────────

    def _cumulative_gaps(self, history: list, current_df: pd.DataFrame) -> dict:
        """
        For each SKU that has a recommendation to buy this month,
        count consecutive months of:
          - supply_gap > 0  (recommended but not executed)
          - demand_gap > 0  (went short — actual demand exceeded available supply)

        A run of N consecutive months triggers classification.
        """
        # Build month × SKU matrices
        months = [m for m, _ in history]
        all_skus = set(current_df["sku"])

        supply_gap_history: dict[str, list] = {sku: [] for sku in all_skus}
        demand_gap_history: dict[str, list] = {sku: [] for sku in all_skus}

        for _, df in history:
            for sku in all_skus:
                row = df[df["sku"] == sku]
                if row.empty:
                    supply_gap_history[sku].append(None)
                    demand_gap_history[sku].append(None)
                else:
                    supply_gap_history[sku].append(float(row.iloc[0]["supply_gap"]))
                    demand_gap_history[sku].append(float(row.iloc[0]["demand_gap"]))

        operator_deviations = []    # SKUs with persistent under-execution
        supply_risk_skus = []       # SKUs with persistent demand > supply

        for sku in all_skus:
            sg = supply_gap_history[sku]
            dg = demand_gap_history[sku]

            consec_supply_gap = self._consecutive_positive_tail(sg)
            consec_demand_gap = self._consecutive_positive_tail(dg)

            if consec_supply_gap >= CONSECUTIVE_MONTHS_THRESHOLD:
                operator_deviations.append({
                    "sku": sku,
                    "consecutive_months_under_executed": consec_supply_gap,
                    "cumulative_unexecuted_qty": round(sum(v for v in sg if v and v > 0), 1),
                    "verdict": "OPERATOR_DEVIATION — maintain system recommendation",
                })

            if consec_demand_gap >= CONSECUTIVE_MONTHS_THRESHOLD:
                supply_risk_skus.append({
                    "sku": sku,
                    "consecutive_months_short": consec_demand_gap,
                    "cumulative_demand_gap": round(sum(v for v in dg if v and v > 0), 1),
                    "verdict": "CUMULATIVE_SUPPLY_GAP — escalate for supply chain review",
                })

        return {
            "months_in_history": len(months),
            "consecutive_threshold": CONSECUTIVE_MONTHS_THRESHOLD,
            "operator_deviation_skus": operator_deviations,
            "supply_risk_skus": supply_risk_skus,
        }

    @staticmethod
    def _consecutive_positive_tail(values: list) -> int:
        """Count how many trailing values are > 0 (ignoring None gaps)."""
        count = 0
        for v in reversed(values):
            if v is None:
                continue
            if v > 0:
                count += 1
            else:
                break
        return count

    # ── Attribution ───────────────────────────────────────────────────────────

    def _attribute(self, detail_df: pd.DataFrame, cumulative: dict) -> dict:
        """
        Classify each deviation type so suggestions don't mix causes.

          OPERATOR_DEVIATION  → model is right, operator under-executed
          SUPPLY_RISK         → real shortage building, escalate
          MODEL_BIAS          → forecast systematically off, investigate model
          UNEXPLAINED         → one-off, no pattern yet
        """
        operator_skus = {d["sku"] for d in cumulative["operator_deviation_skus"]}
        risk_skus = {d["sku"] for d in cumulative["supply_risk_skus"]}

        stocking = detail_df[detail_df["stocking_class"] != "non-stocking"]
        bias = self._bias_direction(stocking["forecast_error"])

        return {
            "operator_deviation_count": len(operator_skus),
            "supply_risk_count": len(risk_skus),
            "overlap_count": len(operator_skus & risk_skus),   # both patterns — most critical
            "forecast_bias": bias,
            "model_adjustment_warranted": (
                bias != "neutral"
                and len(operator_skus) < len(detail_df) * 0.5   # not explained by ops
            ),
        }

    # ── Suggestions ───────────────────────────────────────────────────────────

    def _suggest(self, agg: dict, attribution: dict) -> list:
        suggestions = []
        cumulative = agg["cumulative"]

        # Operator deviation — maintain recommendation, flag for ops review
        for item in cumulative["operator_deviation_skus"]:
            suggestions.append({
                "type": "OPERATOR_DEVIATION",
                "sku": item["sku"],
                "evidence": f"{item['consecutive_months_under_executed']} consecutive months of under-execution, "
                            f"cumulative unexecuted qty = {item['cumulative_unexecuted_qty']}",
                "action": "Maintain system recommendation unchanged. "
                          "Review with operations: storage constraints, budget freeze, or manual override?",
                "model_change": False,
            })

        # Supply risk — escalate, not a model problem
        for item in cumulative["supply_risk_skus"]:
            overlap = item["sku"] in {d["sku"] for d in cumulative["operator_deviation_skus"]}
            suggestions.append({
                "type": "CUMULATIVE_SUPPLY_GAP" + (" + OPERATOR_DEVIATION" if overlap else ""),
                "sku": item["sku"],
                "evidence": f"{item['consecutive_months_short']} consecutive months demand > supply, "
                            f"cumulative gap = {item['cumulative_demand_gap']} units",
                "action": "Escalate to supply chain review. "
                          + ("Operator non-execution is compounding the shortage risk." if overlap else
                             "Check supplier capacity and lead time. Consider dual-sourcing."),
                "model_change": False,
            })

        # Model bias — only if not explained by operator pattern
        if attribution["model_adjustment_warranted"]:
            bias = attribution["forecast_bias"]
            fc = agg["forecast"]
            if bias == "over-forecast":
                suggestions.append({
                    "type": "MODEL_BIAS",
                    "evidence": f"Systematic over-forecast: MD={fc['MD']:+.1f}, MAD={fc['MAD']:.1f}",
                    "action": "Consider reducing z_score (e.g., 1.645 → 1.28) or reviewing demand classification.",
                    "config_param": "stocking_tiers[stocking-high].z_score",
                    "model_change": True,
                })
            elif bias == "under-forecast":
                suggestions.append({
                    "type": "MODEL_BIAS",
                    "evidence": f"Systematic under-forecast: MD={fc['MD']:+.1f}, MAD={fc['MAD']:.1f}",
                    "action": "Consider increasing safety factor or switching from CSL to IFR metric.",
                    "config_param": "service_level_metric or z_score",
                    "model_change": True,
                })

        if not suggestions:
            suggestions.append({
                "type": "NO_PATTERN",
                "action": "No persistent deviations detected. No model changes recommended.",
                "model_change": False,
            })

        return suggestions

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _bias_direction(self, errors: pd.Series) -> str:
        md = errors.mean()
        mad = errors.abs().mean()
        if mad == 0:
            return "neutral"
        ratio = md / mad
        if ratio > 0.2:
            return "over-forecast"
        if ratio < -0.2:
            return "under-forecast"
        return "neutral"

    # ── Print ─────────────────────────────────────────────────────────────────

    def _print_report(self, agg: dict) -> None:
        print("\n" + "="*60)
        print(f"  FEEDBACK REPORT — {agg['planning_month']}")
        print("="*60)

        fc = agg["forecast"]
        mape_str = f"  MAPE={fc['MAPE']*100:.1f}%" if fc.get("MAPE") else ""
        print(f"\n  Forecast:  MD={fc['MD']:+.1f}  MAD={fc['MAD']:.1f}{mape_str}  ({fc['bias_direction']})")

        inv = agg["inventory"]
        print(f"  Inventory: avg DOS realized={inv['avg_dos_realized']:.0f}d  excess rate={inv['excess_rate']*100:.0f}%")

        cum = agg["cumulative"]
        print(f"\n  Cumulative gap analysis ({cum['months_in_history']} months, threshold={cum['consecutive_threshold']}):")
        if cum["operator_deviation_skus"]:
            print(f"  ⚠  OPERATOR DEVIATION:   {len(cum['operator_deviation_skus'])} SKUs — model recommendation maintained")
        if cum["supply_risk_skus"]:
            print(f"  🔴 SUPPLY RISK:          {len(cum['supply_risk_skus'])} SKUs — escalate to supply chain review")
        overlap = agg["attribution"]["overlap_count"]
        if overlap:
            print(f"  ❗ Both patterns:        {overlap} SKUs — highest priority")

        drift = agg.get("lead_time_drift") or {}
        moves = drift.get("moves") or []
        if moves:
            longer = [m for m in moves if m["change_days"] > 0]
            print(f"\n  Lead-time drift ({len(drift['months'])} snapshots): "
                  f"{len(moves)} SKUs moved materially, {len(longer)} lengthening")
            for move in sorted(moves, key=lambda m: -abs(m["change_days"]))[:5]:
                supplier = f"  [{move['supplier']}]" if move.get("supplier") else ""
                print(f"     {move['sku']:<18}{move['first_lt_days']:>5.0f}d -> "
                      f"{move['latest_lt_days']:>5.0f}d  "
                      f"{move['change_days']:+.0f}d / {move['change_pct']:+.0%}{supplier}")
            print("     A supplier that moved is a supplier conversation — the plan "
                  "absorbed it silently.")
        elif drift.get("source_changes"):
            print(f"\n  Lead-time drift: none, but {drift['source_changes']} SKUs changed "
                  f"lead-time source (what the pipeline knew, not what the supplier did)")

        print("\n  Suggestions:")
        for s in agg["suggestions"]:
            tag = f"[{s['type']}]"
            sku_str = f" SKU={s['sku']}" if "sku" in s else ""
            print(f"  ► {tag}{sku_str}")
            print(f"    {s['action']}")
            if s.get("config_param"):
                print(f"    param: {s['config_param']}")
            if not s.get("model_change", True):
                print(f"    → No model parameter change.")

        print("="*60)
