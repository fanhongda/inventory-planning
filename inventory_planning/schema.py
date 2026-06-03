"""
Canonical column schemas for all five input documents.
Each schema maps canonical field names to lists of known aliases.
Readers use these to auto-detect and remap columns regardless of ERP source format.
Aliases are normalized (lowercase, collapsed whitespace) at match-time.
"""

from typing import List

SALES_HISTORY_SCHEMA = {
    "sku":                  ["sku", "item", "item code", "item_code", "item no", "material",
                             "product_code", "part_no", "part number", "article"],
    "order_date":           ["order_date", "order date", "so date", "sales order date",
                             "created date", "created_date", "doc date"],
    "ship_date":            ["ship_date", "ship date", "shipped date", "actual ship date",
                             "delivered date", "delivery date", "goods issue date",
                             "delv date", "delivered  date"],
    "customer_request_date":["customer_request_date", "request date", "requested delivery date",
                             "crd", "customer request date", "req date"],
    "qty":                  ["qty", "quantity", "order qty", "shipped qty", "sales qty",
                             "demand qty", "volume", "delivered quantity", "delivered  quantity"],
    "amount":               ["amount", "revenue", "net amount", "net value", "sales amount",
                             "line amount", "extended amount", "invoice amount"],
    "customer":             ["customer", "customer name", "sold to", "sold-to", "account", "client"],
    "uom":                  ["uom", "unit", "unit of measure", "base unit"],
}

PO_HISTORY_SCHEMA = {
    "sku":                  ["sku", "item", "item code", "item_code", "material",
                             "product_code", "part_no", "part number"],
    "supplier":             ["supplier", "supplier name", "vendor", "vendor name", "creditor"],
    "po_date":              ["po_date", "po date", "order date", "purchase order date",
                             "doc date", "created date"],
    "receive_date":         ["receive_date", "receive date", "receipt date",
                             "goods receipt date", "gr date", "actual delivery",
                             "received date", "received  date"],
    "po_qty":               ["po_qty", "po quantity", "order qty", "ordered qty",
                             "ordered  qty", "quantity", "qty"],
    "po_amount":            ["po_amount", "po amount", "net amount", "net value",
                             "purchase amount", "receipt value"],
    "incoterm":             ["incoterm", "inco term", "inco terms", "delivery term", "trade term"],
    # Pre-computed LT (days) present in some ERP exports — captured as a bonus field
    "lt_days_precalc":      ["lt", "lead time", "lt days", "lead time days"],
}

OPEN_SO_SCHEMA = {
    "sku":                  ["sku", "item", "item code", "item_code", "material",
                             "product_code", "part_no"],
    "customer":             ["customer", "customer name", "sold to", "account"],
    "customer_request_date":["customer_request_date", "request date", "crd",
                             "customer request date", "req date",
                             "requested delivery date", "customer  req. date",
                             "customer req. date", "customer req date"],
    "promise_date":         ["promise_date", "promise date", "confirmed delivery date",
                             "confirmed date", "commit date", "committed date",
                             "committed  date"],
    "open_qty":             ["open_qty", "open quantity", "backlog qty", "remaining qty",
                             "outstanding qty", "balance quantity", "balance  quantity"],
    "open_amount":          ["open_amount", "open amount", "backlog amount",
                             "remaining amount", "outstanding amount"],
    "so_number":            ["so_number", "so number", "sales order", "order number",
                             "document number"],
}

OPEN_PO_SCHEMA = {
    "sku":                  ["sku", "item", "item code", "item_code", "material",
                             "product_code", "part_no"],
    "supplier":             ["supplier", "supplier name", "vendor", "vendor name"],
    "order_qty":            ["order_qty", "order quantity", "ordered qty", "ordered quantity",
                             "po qty", "po quantity"],
    "delivered_qty":        ["delivered_qty", "delivered quantity", "received qty",
                             "received quantity", "qty delivered"],
    "open_qty":             ["open_qty", "open quantity", "remaining qty", "outstanding qty",
                             "po open qty", "balance qty"],
    "open_amount":          ["open_amount", "open amount", "remaining amount",
                             "outstanding amount", "po value", "remaining value"],
    "committed_delivery":   ["committed_delivery", "committed delivery date", "delivery date",
                             "scheduled delivery", "eta", "expected delivery",
                             "planned delivery", "planned del date"],
    "po_number":            ["po_number", "po number", "purchase order", "document number",
                             "po no", "po no."],
    "incoterm":             ["incoterm", "inco term", "inco terms", "delivery term"],
    "closed_status":        ["closed_status", "closed status", "status", "po status"],
    "order_date":           ["order_date", "order date", "po date", "doc date"],
}

INVENTORY_SCHEMA = {
    "sku":                  ["sku", "item", "item code", "item_code", "material",
                             "product_code", "part_no"],
    "qty_on_hand":          ["qty_on_hand", "on hand", "on-hand", "stock qty",
                             "available qty", "unrestricted stock", "physical stock",
                             "inventory on hand", "inventory  on hand"],
    "qty_in_transit":       ["qty_in_transit", "in transit", "in-transit",
                             "goods in transit", "git qty", "intransit qty", "transit stock"],
    "open_po_qty_inv":      ["openpo quantity", "openpo qty", "openpo", "open po qty",
                             "openpoquantity", "open po quantity"],
    "uom":                  ["uom", "unit", "unit of measure", "base unit"],
    "location_id":          ["location_id", "location", "warehouse", "plant",
                             "storage location", "dc", "wrh", "wh"],
    "snapshot_date":        ["snapshot_date", "snapshot date", "report date",
                             "as of date", "date"],
}

ALL_SCHEMAS = {
    "sales_history": SALES_HISTORY_SCHEMA,
    "po_history":    PO_HISTORY_SCHEMA,
    "open_so":       OPEN_SO_SCHEMA,
    "open_po":       OPEN_PO_SCHEMA,
    "inventory":     INVENTORY_SCHEMA,
}

REQUIRED_FIELDS = {
    "sales_history": ["sku", "order_date", "qty"],
    "po_history":    ["sku", "supplier", "po_date", "po_qty"],   # receive_date optional if lt_days_precalc present
    "open_so":       ["sku", "open_qty"],
    "open_po":       ["sku"],                                     # open_qty computed; committed_delivery may be missing
    "inventory":     ["sku", "qty_on_hand"],
}
