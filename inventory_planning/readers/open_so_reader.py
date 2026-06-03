"""Open sales order (backlog) reader."""

import pandas as pd
from .base_reader import BaseReader


class OpenSOReader(BaseReader):
    doc_type = "open_so"

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["sku", "open_qty"])
        df = df[df["open_qty"] > 0]
        return df

    def backlog_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Total open backlog qty and amount per SKU, with earliest request date."""
        grp = df.groupby("sku").agg(
            backlog_qty=("open_qty", "sum"),
            backlog_amount=("open_amount", "sum") if "open_amount" in df.columns else ("open_qty", "count"),
            earliest_request=("customer_request_date", "min"),
            latest_request=("customer_request_date", "max"),
            open_so_lines=("open_qty", "count"),
        ).reset_index()
        grp["location_id"] = df["location_id"].iloc[0] if "location_id" in df.columns else "DC-01"
        return grp
