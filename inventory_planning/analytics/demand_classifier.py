"""
Demand classification and stocking policy assignment.

Classification uses two axes (per MIT CTL principles):
  1. Frequency (active_cycles / total_cycles) → stocking tier
  2. CV = σ / μ (unconditional) → demand pattern within tier

Demand means:
  demand_mean_rolling     = unconditional mean (including zero periods) → used for ROP/SS
  demand_mean_conditional = mean over non-zero periods only → used for Croston magnitude
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


class DemandClassifier:

    def __init__(self, config_dir: Path):
        cfg = json.loads((config_dir / "stocking_policy.json").read_text(encoding="utf-8"))
        self.tiers = cfg["stocking_tiers"]
        self.cycles_per_year = cfg["cycles_per_year"]
        self.rolling_cycles = cfg["demand_rolling_cycles"]
        # CV threshold above which a stocking item is flagged as erratic/intermittent
        self.cv_erratic_threshold = cfg.get("cv_erratic_threshold", 1.0)
        self.cv_intermittent_threshold = cfg.get("cv_intermittent_threshold", 0.5)

    def classify(self, demand_summary: pd.DataFrame, time_series: pd.DataFrame) -> pd.DataFrame:
        """
        demand_summary: output of SalesHistoryReader.summarize()
        time_series: pivot (period × SKU) from SalesHistoryReader.to_time_series()

        Returns demand_summary enriched with:
          stocking_class, demand_pattern, service_level, z_score,
          demand_mean_rolling (unconditional), demand_mean_conditional,
          demand_std_rolling, demand_cv
        """
        df = demand_summary.copy()

        ts = time_series.tail(self.rolling_cycles)
        total_cycles = len(ts)

        rows = []
        for _, row in df.iterrows():
            sku = row["sku"]
            if sku in ts.columns:
                series = ts[sku].fillna(0)
                active = int((series > 0).sum())
                # Unconditional mean — used for ROP and SS formulas
                mean_unconditional = float(series.mean())
                # Conditional mean — magnitude when demand occurs (Croston numerator)
                nonzero = series[series > 0]
                mean_conditional = float(nonzero.mean()) if len(nonzero) > 0 else 0.0
                std_demand = float(series.std())
            else:
                active = 0
                mean_unconditional = float(row.get("demand_mean", 0))
                mean_conditional = mean_unconditional
                std_demand = float(row.get("demand_std", 0))

            # CV based on unconditional mean (handles zero periods correctly)
            cv = round(std_demand / mean_unconditional, 3) if mean_unconditional > 0 else np.nan

            tier = self._assign_tier(active, total_cycles)
            demand_pattern = self._demand_pattern(tier["name"], active, total_cycles, cv)

            rows.append({
                **row.to_dict(),
                "active_cycles_rolling": active,
                "total_cycles_evaluated": int(total_cycles),
                # Unconditional mean (including zero months) — use for ROP/SS
                "demand_mean_rolling": round(mean_unconditional, 2),
                # Conditional mean (non-zero months only) — use for Croston magnitude
                "demand_mean_conditional": round(mean_conditional, 2),
                "demand_std_rolling": round(std_demand, 2),
                "demand_cv": cv,
                "stocking_class": tier["name"],
                "stocking_label": tier["label"],
                "demand_pattern": demand_pattern,
                "service_level": tier["service_level"],
                "z_score": tier["z_score"],
            })
        return pd.DataFrame(rows)

    def _assign_tier(self, active_cycles: int, total_cycles: int) -> dict:
        for tier in self.tiers:
            if active_cycles >= tier["min_active_cycles"]:
                return tier
        return self.tiers[-1]  # non-stocking fallback

    def _demand_pattern(self, stocking_class: str, active: int, total: int, cv) -> str:
        """
        Classify demand pattern within a stocking tier.
        Used to route SKUs to the correct forecasting method.

          smooth      → ETS/Holt-Winters (low CV, high frequency)
          intermittent → Croston's method (moderate CV or moderate frequency)
          erratic     → Croston's method (high CV regardless of frequency)
          lumpy       → Croston's method (low frequency + high CV)
          non-stocking → no forecast needed
        """
        if stocking_class == "non-stocking":
            return "non-stocking"
        if pd.isna(cv):
            return "unknown"
        freq_ratio = active / total if total > 0 else 0
        if cv <= self.cv_intermittent_threshold and freq_ratio >= 0.75:
            return "smooth"
        if cv >= self.cv_erratic_threshold:
            return "erratic" if freq_ratio >= 0.5 else "lumpy"
        # moderate CV or moderate frequency
        return "intermittent"
