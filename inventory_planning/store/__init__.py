"""
The fact store — append-only, typed, and not yet read by the planning path.

See `fact_store.FactStore` for the layout and `ledger.BatchLedger` for why every
operation that looks like editing a fact is a batch-level one instead.
"""

from .fact_store import (
    FactStore, SCHEMA_VERSION, StoreSchemaError, StoreUnavailable,
)
from .ledger import BatchLedger, BatchRecord, STATUS_ACTIVE, STATUS_VOID
from .location import ENV_VAR, default_store_root, resolve_store_root

__all__ = [
    "FactStore", "SCHEMA_VERSION", "StoreSchemaError", "StoreUnavailable",
    "BatchLedger", "BatchRecord", "STATUS_ACTIVE", "STATUS_VOID",
    "ENV_VAR", "default_store_root", "resolve_store_root",
]
