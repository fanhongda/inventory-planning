"""
Demand classification and stocking policy assignment.
Classifies each SKU as stocking-high / stocking-med / non-stocking
based on demand frequency over the rolling cycle window.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


class DemandClassifier:

    def __init__(self, config_dir: Path):
        cfg = json.loads((config_dir / "stocking_policy.json").read_text())
        self.tiers = cfg["stocking_tiers"]
        self.cycles_per_year = cfg["cycles_per_year"]
        self.rolling_cycles = cfg["demand_rolling_cycles"]

    def classify(self, demand_summary: pd.DataFrame, time_series: pd.DataFrame) -> pd.DataFrame:
        """
        demand_summary: output of SalesHistoryReader.summarize()
        time_series: pivot (period × SKU) from SalesHistoryReader.to_time_series()

        Returns demand_summary enriched with stocking_class, service_level, z_score.
        """
        df = demand_summary.copy()

        # Use only the last rolling_cycles periods
        ts = time_series.tail(self.rolling_cycles)
        total_cycles = len(ts)

        rows = []
        for _, row in df.iterrows():
            sku = row["sku"]
            if sku in ts.columns:
                active = (ts[sku] > 0).sum()
                mean_demand = ts[sku][ts[sku] > 0].mean() if active > 0 else 0
                std_demand = ts[sku].std()
            else:
                active = 0
                mean_demand = row.get("demand_mean", 0)
                std_demand = row.get("demand_std", 0)

            tier = self._assign_tier(int(active), total_cycles)
            rows.append({
                **row.to_dict(),
                "active_cycles_rolling": int(active),
                "total_cycles_evaluated": int(total_cycles),
                "demand_mean_rolling": round(float(mean_demand), 2) if pd.notna(mean_demand) else 0,
                "demand_std_rolling": round(float(std_demand), 2) if pd.notna(std_demand) else 0,
                "stocking_class": tier["name"],
                "stocking_label": tier["label"],
                "service_level": tier["service_level"],
                "z_score": tier["z_score"],
            })
        return pd.DataFrame(rows)

    def _assign_tier(self, active_cycles: int, total_cycles: int) -> dict:
        for tier in self.tiers:
            if active_cycles >= tier["min_active_cycles"]:
                return tier
        return self.tiers[-1]  # non-stocking fallback
