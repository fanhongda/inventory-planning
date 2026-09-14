"""
Reading the fact store — the half that did not exist.

`FactStore` writes batches and hands one back by id, which is enough to verify a write
and no use at all to a caller asking what the stock position is. This is the read path:
an as-of query across batches, and a single-grain view of what is currently believed.

Two things are load-bearing.

**Nothing is stored inside the query engine.** DuckDB reads the Parquet files in place
and holds no state, so every request opens its own connection over the same files: no
write lock, no single-writer bottleneck, and unlimited concurrent readers. It also means
the store is engine-independent — moving to Trino or Snowflake later is a connection
change, because there is nothing in DuckDB to migrate. That property is the whole reason
the substrate is Parquet plus a ledger rather than a database file, and it is easy to
lose by accident the first time something is cached inside the engine.

**The batch ledger decides which files are read, not a glob.** Batch identity, void
status and both timestamps live in the ledger; the Parquet holds the canonical frame and
nothing else. Duplicating the timestamps into the files would create two answers to
"when was this true" that can disagree, and the disagreement would surface as a quantity
that is quietly wrong rather than as an error.

## Why `current` is a separate method and not a default argument

A missing as-of predicate does not raise. One purchase order appears once per load, the
inbound quantity is multiplied by the number of times the file was imported, and the
result merely looks high. So the two readings are two named methods and the dangerous
one says what it is: `history` returns every observation and carries the batch each row
came from, `current` returns one row per natural key.

`current` refuses rather than guesses when the key is incomplete. Deduplicating on a
partial key collapses rows that are genuinely different — an understatement that looks
like a clean answer — and it is the same defect as not deduplicating at all, in the
other direction.

## `current` and `latest` are two readings, and no contract says which one applies

Take two stock snapshots, June and July. June lists A and B; July lists A and C. There
are two defensible answers to "what is the position", and they differ by B:

    current   the newest observation of every key ever seen  → A, B, C
    latest    the newest batch, and only it                  → A, C

Which is right is a property of the export, not of this code. A whole-population
snapshot replaces its predecessor, so a material missing from July is a material that
went to zero and `current` overstates it by carrying June forward. An incremental window
— three months of sales history at a time — is the opposite: `latest` would throw away
every period the newest file does not cover.

Nothing in the contracts declares which kind an export is, and inferring it from the
grain would be a guess with a plausible wrong number on the other side of it. So both
readings exist under their own names, neither is the silent default of the other, and
every row of `current` carries the `valid_time` it was observed at — which is what makes
a value carried forward from an older batch visible rather than merely present.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .fact_store import FactStore, StoreUnavailable
from .ledger import BatchLedger, LAYER_UNKNOWN
from .location import resolve_store_root

# Columns the reader adds. Named so they cannot collide with a canonical field, and
# stripped from `current` because a single-grain view has one batch by construction.
BATCH_COLUMN = "__batch_id"
VALID_COLUMN = "__valid_time"
RANK_COLUMN = "__batch_rank"


class QueryUnavailable(StoreUnavailable):
    """The query engine is not installed, or the store cannot be read."""


class KeyIncomplete(ValueError):
    """`current` was asked for on a document whose stored batches lack a key column."""


class MixedLayers(ValueError):
    """A reading would have blended batches holding different layers of the pipeline."""


class NoSuchColumn(ValueError):
    """A filter named a column the stored batches do not carry."""


def _connect():
    try:
        import duckdb
    except ImportError as exc:
        raise QueryUnavailable(
            "reading the store needs DuckDB. Install the optional extra:\n"
            "    pip install -e '.[store]'\n"
            "Writing the store does not need it; only reading across batches does."
        ) from exc
    return duckdb.connect()


@dataclass
class Selection:
    """The batches an as-of question selects, oldest first."""

    doc_type: str
    batches: List[Dict[str, Any]] = dc_field(default_factory=list)
    as_of: Optional[str] = None
    known_at: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.batches)

    @property
    def layers(self) -> List[str]:
        """Which layers of the pipeline the selected batches hold. See `ledger.LAYER_*`."""
        return sorted({str(b.get("frame_layer") or LAYER_UNKNOWN) for b in self.batches})

    @property
    def mixed(self) -> bool:
        return len(self.layers) > 1

    def layer_spans(self) -> Dict[str, Any]:
        """Each layer's first and last `valid_time`, so overlap can be reasoned about."""
        spans: Dict[str, Any] = {}
        for batch in self.batches:
            layer = str(batch.get("frame_layer") or LAYER_UNKNOWN)
            when = str(batch.get("valid_time") or "")
            first, last = spans.get(layer, (when, when))
            spans[layer] = (min(first, when), max(last, when))
        return spans

    def isolating_cutoff(self) -> Optional[tuple]:
        """
        An `as_of` that selects exactly one layer, and the layer it selects — or None.

        `as_of` is an upper bound, so it can only isolate the *earliest* layer, and only
        where that layer finishes before every other one starts. Where the layers
        interleave — which is what a store looks like when the writer changed while old
        batches were still being re-loaded — no cutoff exists, and offering one anyway
        sends the reader round a loop that ends in this same refusal.
        """
        spans = self.layer_spans()
        if len(spans) < 2:
            return None
        earliest = min(spans, key=lambda layer: spans[layer][1])
        finishes = spans[earliest][1]
        if all(start > finishes for layer, (start, _) in spans.items()
               if layer != earliest):
            return finishes, earliest
        return None

    def how_to_narrow(self) -> str:
        """What a caller can actually do about a mixed selection, in this store."""
        cutoff = self.isolating_cutoff()
        if cutoff:
            when, layer = cutoff
            return (f"Read `as_of={when}` for the {layer} batches alone, or void the "
                    f"batches of the layer you are not using.")
        spans = ", ".join(f"{layer} {first}\u2026{last}"
                          for layer, (first, last) in sorted(self.layer_spans().items()))
        return (f"The layers overlap in time ({spans}), so no `as_of` isolates either "
                f"one \u2014 it is an upper bound and the older layer runs past the start "
                f"of the newer. Void the batches of the layer you are not using, or "
                f"re-store them through one path.")

    def describe(self) -> str:
        if not self.batches:
            return f"{self.doc_type}: no batch matches"
        first, last = self.batches[0], self.batches[-1]
        layers = f", {'/'.join(self.layers)}" if self.layers != ["canonical"] else ""
        return (f"{self.doc_type}: {len(self.batches)} batch(es), "
                f"valid {first['valid_time']} → {last['valid_time']}, "
                f"{sum(int(b.get('rows') or 0) for b in self.batches):,} rows{layers}")


class FactQuery:
    """As-of reads over the fact store. Opens a connection per call and keeps none."""

    def __init__(self, root=None, contracts=None):
        self.root, self.source = resolve_store_root(root)
        self.ledger = BatchLedger(self.root)
        self._contracts = contracts

    @property
    def contracts(self):
        if self._contracts is None:
            from ..ingest.contract import default_registry
            self._contracts = default_registry()
        return self._contracts

    # ── What is there ────────────────────────────────────────────────────────

    def doc_types(self) -> List[str]:
        return sorted({b["doc_type"] for b in self.ledger.batches()})

    def select(self, doc_type: str, as_of: Any = None,
               known_at: Any = None) -> Selection:
        """
        The batches a bitemporal question selects, oldest first.

        `as_of` filters on `valid_time` — the moment the data describes. `known_at`
        filters on `transaction_time` — the moment it was loaded. They are routinely
        different, and only the second can reconstruct what was believed last week
        after a correction has been loaded since.
        """
        as_of = _as_text(as_of)
        known_at = _as_text(known_at)
        chosen = []
        for batch in self.ledger.batches(doc_type=doc_type):
            if as_of and str(batch.get("valid_time") or "") > as_of:
                continue
            if known_at and str(batch.get("transaction_time") or "") > known_at:
                continue
            if not (self.root / str(batch.get("path") or "")).exists():
                continue
            chosen.append(batch)
        chosen.sort(key=lambda b: (str(b.get("valid_time") or ""),
                                   str(b.get("transaction_time") or ""),
                                   str(b.get("batch_id") or "")))
        return Selection(doc_type=doc_type, batches=chosen,
                         as_of=as_of, known_at=known_at)

    # ── Reading ──────────────────────────────────────────────────────────────

    def history(self, doc_type: str, as_of: Any = None, known_at: Any = None,
                where: Dict[str, Any] = None, columns: Sequence[str] = None,
                limit: int = None) -> pd.DataFrame:
        """
        Every observation the selected batches hold, each row carrying its batch.

        This is not a position. The same purchase order appears once per load, so
        summing a quantity here multiplies it by the number of times the file was
        imported — which does not raise and merely looks high. Use `current` for
        anything that gets added up.
        """
        return self._read(doc_type, self.select(doc_type, as_of, known_at),
                          where=where, columns=columns, limit=limit, dedupe=False)

    def current(self, doc_type: str, as_of: Any = None, known_at: Any = None,
                where: Dict[str, Any] = None, columns: Sequence[str] = None,
                limit: int = None) -> pd.DataFrame:
        """
        One row per natural key: the newest observation of each, as of the cutoffs.

        Newest is by `valid_time` then `transaction_time` — the moment described before
        the moment loaded, because a correction to last week's file describes last week
        and must not outrank this week's data merely by having been loaded later.
        """
        selection = self.select(doc_type, as_of, known_at)
        return self._read(doc_type, selection, where=where, columns=columns,
                          limit=limit, dedupe=True)

    def latest(self, doc_type: str, as_of: Any = None, known_at: Any = None,
               where: Dict[str, Any] = None, columns: Sequence[str] = None,
               limit: int = None) -> pd.DataFrame:
        """
        The newest selected batch and only it — the reading for a whole-population
        export, where a key missing from the newest file is a key that went to zero.

        Needs no natural key, because one batch is one observation by construction.
        Where an export is an incremental window rather than a full snapshot this
        discards everything the newest file does not cover, which is why it is a
        separate method and not a mode of `current`.
        """
        selection = self.select(doc_type, as_of, known_at)
        if selection:
            selection = Selection(doc_type=doc_type, batches=selection.batches[-1:],
                                  as_of=selection.as_of, known_at=selection.known_at)
        return self._read(doc_type, selection, where=where, columns=columns,
                          limit=limit, dedupe=False)

    def carried_forward(self, doc_type: str, as_of: Any = None,
                        known_at: Any = None) -> Dict[str, Any]:
        """
        How much of `current` did not come from the newest batch.

        The size of the disagreement between the two readings above, as a number rather
        than as a caveat. Zero means they agree and the question does not arise.
        """
        selection = self.select(doc_type, as_of, known_at)
        if not selection:
            return {"newest_valid_time": None, "rows": 0, "carried": 0}
        newest = str(selection.batches[-1].get("valid_time") or "")
        frame = self.current(doc_type, as_of=as_of, known_at=known_at)
        older = frame[frame[VALID_COLUMN] < newest] if len(frame) else frame
        return {"newest_valid_time": newest, "rows": len(frame),
                "carried": len(older),
                "carried_share": (len(older) / len(frame)) if len(frame) else 0.0}

    def count(self, doc_type: str, as_of: Any = None, known_at: Any = None) -> int:
        return len(self.current(doc_type, as_of=as_of, known_at=known_at,
                                columns=self._key_columns(doc_type)))

    # ── Internals ────────────────────────────────────────────────────────────

    def _key_columns(self, doc_type: str) -> List[str]:
        try:
            contract = self.contracts.get(doc_type)
        except KeyError as exc:
            raise KeyIncomplete(
                f"no contract for {doc_type!r}, so there is no natural key to reduce "
                f"it by. Read it with `history` and reduce it yourself.") from exc
        key = [str(k) for k in (contract.natural_key or [])]
        if not key:
            raise KeyIncomplete(
                f"the {doc_type} contract declares no natural key, so a single-grain "
                f"view of it cannot be defined.")
        return key

    def _read(self, doc_type: str, selection: Selection, where=None, columns=None,
              limit=None, dedupe=False) -> pd.DataFrame:
        if not selection:
            return pd.DataFrame()

        # Blending layers is the one thing a reading must not do quietly. A `prepared`
        # batch holds money already converted into the reporting currency and a
        # `canonical` one holds the source currency; putting them in one frame produces
        # a column that is partly one and partly the other, sums cleanly, and is wrong
        # by whatever the rate was. Refusing names the boundary and the cutoff that
        # stays inside one layer, which is a one-time answer rather than a caveat
        # carried forever.
        if selection.mixed:
            raise MixedLayers(
                f"{doc_type} has batches from more than one layer of the pipeline "
                f"({', '.join(selection.layers)}), and their money columns are not the "
                f"same measure: a `prepared` batch was converted into the reporting "
                f"currency before it was stored and a `canonical` one was not. "
                + selection.how_to_narrow())

        paths = [str(self.root / b["path"]) for b in selection.batches]
        # `union_by_name` because an export that gained a column mid-history is the
        # ordinary case, and positional union would silently shear the two schemas
        # together — every value in the new column landing in the wrong field.
        source = ("read_parquet($paths, union_by_name=true, filename=true)")

        params: Dict[str, Any] = {"paths": paths}
        ranks = ", ".join(
            f"($p{i}, {i}, $b{i}, $v{i})" for i in range(len(selection.batches)))
        for i, batch in enumerate(selection.batches):
            params[f"p{i}"] = paths[i]
            params[f"b{i}"] = batch["batch_id"]
            params[f"v{i}"] = str(batch.get("valid_time") or "")

        # The union schema, read once. Needed to check a filter names a real column and
        # again by the dedupe branch below; both used to be able to reach DuckDB with a
        # column it has never heard of, and the binder error that follows is not one any
        # caller catches.
        available = self._columns(source, {"paths": paths}) if (where or dedupe) else []
        if where:
            unknown = [c for c in sorted(where) if c not in available]
            if unknown:
                # Refused rather than answered with nothing. An empty result reads as
                # "no rows match that location", which is a claim about the data; the
                # truth is that this document does not record a location at all, and
                # the two lead a reader to opposite conclusions.
                raise NoSuchColumn(
                    f"{doc_type} carries no {', '.join(unknown)}. Filtering on it cannot "
                    f"return rows, and an empty answer would read as 'none match' rather "
                    f"than 'this document does not record that'. Columns available: "
                    f"{', '.join(available) or 'none'}.")

        predicate, where_params = _where_clause(where)
        params.update(where_params)

        select_cols = "rows.* EXCLUDE (filename)"
        if columns:
            select_cols = ", ".join(f'rows."{c}"' for c in columns)

        base = f"""
            WITH batches(path, rank, batch_id, valid_time) AS (VALUES {ranks}),
            rows AS (SELECT * FROM {source})
            SELECT {select_cols},
                   batches.batch_id AS "{BATCH_COLUMN}",
                   batches.valid_time AS "{VALID_COLUMN}",
                   batches.rank AS "{RANK_COLUMN}"
            FROM rows JOIN batches ON rows.filename = batches.path
            {predicate}
        """

        if dedupe:
            key = self._key_columns(doc_type)
            missing = [k for k in key if k not in available]
            if missing:
                raise KeyIncomplete(
                    f"{doc_type} is keyed on {', '.join(key)} and the stored batches "
                    f"carry no {', '.join(missing)}. A single-grain view cannot be "
                    f"built: reducing on a partial key collapses rows that are "
                    f"genuinely different, which is as wrong as not reducing at all "
                    f"and looks cleaner. Use `history` and reduce it deliberately.")
            partition = ", ".join(f'"{k}"' for k in key)
            base = f"""
                WITH picked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY {partition}
                        ORDER BY "{RANK_COLUMN}" DESC) AS __rn
                    FROM ({base})
                )
                SELECT * EXCLUDE (__rn, "{RANK_COLUMN}") FROM picked WHERE __rn = 1
            """
        else:
            base = f'SELECT * EXCLUDE ("{RANK_COLUMN}") FROM ({base})'

        if limit:
            base = f"SELECT * FROM ({base}) LIMIT {int(limit)}"

        connection = _connect()
        try:
            return connection.execute(base, params).df()
        finally:
            connection.close()

    def _columns(self, source: str, params: Dict[str, Any]) -> List[str]:
        """The union schema of the selected batches, without reading a row."""
        connection = _connect()
        try:
            return [d[0] for d in connection.execute(
                f"SELECT * FROM {source} LIMIT 0", params).description]
        finally:
            connection.close()


def _as_text(value) -> Optional[str]:
    if value in (None, ""):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _where_clause(where: Dict[str, Any]) -> (str, Dict[str, Any]):
    """
    Equality and membership only, always parameterised.

    Deliberately not a SQL fragment. The caller with a filter to apply is a web request,
    and a store this cheap to re-read is not worth an injection surface.
    """
    if not where:
        return "", {}
    clauses, params = [], {}
    for i, (column, value) in enumerate(sorted(where.items())):
        name = f"w{i}"
        if isinstance(value, (list, tuple, set)):
            params[name] = list(value)
            clauses.append(f'rows."{column}" IN (SELECT unnest(${name}))')
        else:
            params[name] = value
            clauses.append(f'rows."{column}" = ${name}')
    return "WHERE " + " AND ".join(clauses), params
