"""
Generate realistic sample data files for testing.
Covers: mixed column name styles, some missing optional fields,
        different ERP formats.
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SKUS = ["SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005",
        "SKU-006", "SKU-007", "SKU-008", "SKU-009", "SKU-010"]

SUPPLIERS = {"SKU-001": "Sup-Alpha", "SKU-002": "Sup-Beta",  "SKU-003": "Sup-Alpha",
             "SKU-004": "Sup-Gamma", "SKU-005": "Sup-Beta",  "SKU-006": "Sup-Delta",
             "SKU-007": "Sup-Alpha", "SKU-008": "Sup-Gamma", "SKU-009": "Sup-Beta",
             "SKU-010": "Sup-Delta"}

INCOTERMS = {"Sup-Alpha": "FOB", "Sup-Beta": "EXW", "Sup-Gamma": "DDP", "Sup-Delta": "CIF"}


def random_date(start: datetime, end: datetime) -> datetime:
    return start + timedelta(days=random.randint(0, (end - start).days))


def gen_sales_history(out_path: Path, n_rows: int = 500):
    """Simulate 18 months of sales. SKU-009 is slow-moving (non-stocking)."""
    rows = []
    start = datetime(2023, 1, 1)
    end = datetime(2024, 6, 30)
    for _ in range(n_rows):
        sku = random.choice(SKUS)
        # SKU-009 appears rarely
        if sku == "SKU-009" and random.random() > 0.15:
            continue
        order_dt = random_date(start, end)
        ship_dt = order_dt + timedelta(days=random.randint(1, 5))
        rows.append({
            "Item Code": sku,
            "Order Date": order_dt.strftime("%Y-%m-%d"),
            "Ship Date": ship_dt.strftime("%Y-%m-%d"),
            "Customer Request Date": (order_dt + timedelta(days=random.randint(3, 14))).strftime("%Y-%m-%d"),
            "Sales Qty": round(random.uniform(10, 200), 0),
            "Net Amount": round(random.uniform(500, 10000), 2),
            "Customer": f"CUST-{random.randint(1,20):03d}",
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Generated: {out_path} ({len(rows)} rows)")


def gen_po_history(out_path: Path, n_rows: int = 200):
    rows = []
    start = datetime(2023, 1, 1)
    end = datetime(2024, 3, 31)
    for _ in range(n_rows):
        sku = random.choice(SKUS)
        sup = SUPPLIERS[sku]
        inco = INCOTERMS[sup]
        po_dt = random_date(start, end)
        lt_days = int(np.random.normal(45, 8))  # avg 45d LT
        lt_days = max(7, min(120, lt_days))
        receive_dt = po_dt + timedelta(days=lt_days)
        rows.append({
            "Part No": sku,
            "Vendor Name": sup,
            "PO Date": po_dt.strftime("%Y-%m-%d"),
            "Goods Receipt Date": receive_dt.strftime("%Y-%m-%d"),
            "PO Qty": round(random.uniform(50, 500), 0),
            "Net Value": round(random.uniform(2000, 50000), 2),
            "Inco Terms": inco,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Generated: {out_path} ({len(rows)} rows)")


def gen_open_so(out_path: Path, n_rows: int = 80):
    rows = []
    today = datetime(2024, 7, 1)
    for _ in range(n_rows):
        sku = random.choice(SKUS[:8])  # Not all SKUs have backlog
        req_dt = today + timedelta(days=random.randint(5, 60))
        rows.append({
            "Material": sku,
            "Sold To": f"CUST-{random.randint(1,20):03d}",
            "Requested Delivery Date": req_dt.strftime("%Y-%m-%d"),
            "Promise Date": (req_dt + timedelta(days=random.randint(0, 14))).strftime("%Y-%m-%d"),
            "Outstanding Qty": round(random.uniform(5, 100), 0),
            "Remaining Amount": round(random.uniform(200, 5000), 2),
            "Order Number": f"SO-{random.randint(10000, 99999)}",
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Generated: {out_path} ({len(rows)} rows)")


def gen_open_po(out_path: Path, n_rows: int = 60):
    rows = []
    today = datetime(2024, 7, 1)
    for _ in range(n_rows):
        sku = random.choice(SKUS)
        sup = SUPPLIERS[sku]
        inco = INCOTERMS[sup]
        eta = today + timedelta(days=random.randint(10, 90))
        rows.append({
            "Item": sku,
            "Supplier": sup,
            "PO Open Qty": round(random.uniform(50, 400), 0),
            "Outstanding Amount": round(random.uniform(2000, 40000), 2),
            "Scheduled Delivery": eta.strftime("%Y-%m-%d"),
            "Purchase Order": f"PO-{random.randint(10000, 99999)}",
            "Inco Term": inco,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Generated: {out_path} ({len(rows)} rows)")


def gen_inventory(out_path: Path):
    rows = []
    for sku in SKUS:
        sup = SUPPLIERS[sku]
        inco = INCOTERMS[sup]
        # EXW/FOB → has GIT; DDP → no GIT (seller owns transit)
        git = round(random.uniform(0, 200), 0) if inco in ("EXW", "FOB", "CIF") else 0
        rows.append({
            "Item Code": sku,
            "On Hand": round(random.uniform(50, 500), 0),
            "In Transit": git,
            "Base Unit": "EA",
            "Report Date": "2024-07-01",
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Generated: {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    out = Path(__file__).parents[1] / "sample_data"
    out.mkdir(exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    gen_sales_history(out / "sales_history.csv")
    gen_po_history(out / "po_history.csv")
    gen_open_so(out / "open_so.csv")
    gen_open_po(out / "open_po.csv")
    gen_inventory(out / "inventory.csv")
    print("\nAll sample files generated.")
