"""
The identity layer — which codes name the same material.

A run of the real Controls SDC extract keyed six documents on two different numbering
systems without noticing: four of them on the SAP material number (`1003524`,
`419348`, `000000000005757504`), and two — the planning master and the purchase
history — on a commercial part number (`MS-FAC2513-0`, `V46AS-9301`). Every join was
on a bare `sku` string, so the two halves of the business never met. The quality gate
then blocked the one document whose key was right and passed one whose key was wrong,
because the two wrongly-keyed files agreed with *each other*.

The fix is not to pick a winner. Both numbering systems are real and both are correct
for their own report. What was missing is a place to say that `5391348` and
`MS-FAC2513-0` are the same material — and the answer to that is already in the data:
the planning master carries both on one row, and the open sales order carries
`Material` beside `Old Material Number`. The crosswalk does not have to be maintained
by anyone. It has to be *read*.

Two properties this module is built around.

**Two phases, one version.** Resolving a batch needs the alias table, and the alias
table is derived from the batches — a circular dependency that, resolved greedily,
makes the result depend on which file was loaded first. Loading the master before the
sales history would then give different answers from the reverse, and the promise that
one input yields one output would be quietly false. So: every batch lands first, the
alias table is built once from all of them and stamped with an `alias_version`, and a
resolution names the version it used.

**Ambiguity is recorded, never resolved by guessing.** One part number reaching two
SAP materials is a real thing that happens, and any automatic choice is a coin flip
that reports itself as a fact. Those components go to the conflict list intact and
their codes stay separate until someone rules.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd

from ..ingest.adapter import _normalize_material
from .landing import LandingStore
from .location import resolve_store_root

IDENTITY_DIRNAME = "identity"
CURRENT_NAME = "current.json"

# Code systems, recognised by the shape of the value rather than by the column it sits
# in. The column name is exactly the evidence that failed: `Material` held a local code
# in one export and the ERP number in another.
SYSTEM_MATNR = "sap_matnr"        # all digits once SAP's zero padding is removed
SYSTEM_PARTNO = "partno"          # letters and digits, usually with a separator
SYSTEM_UNKNOWN = "unknown"

_PARTNO = re.compile(r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9][A-Z0-9._/-]{2,31}$")

# A column needs this many distinct codes to be treated as an identifier at all, and
# its values must vary at least this much. Both guard the same failure: a status or a
# plant column that repeats across every row would otherwise co-occur with everything
# and chain the entire catalogue into one material.
MIN_DISTINCT = 10
MIN_DISTINCT_RATE = 0.005

# Headers that name a *document*, never a material. Used only to exclude, never to
# select — which column is the item key is still decided by shape and by corroboration,
# because the column name is exactly the evidence that failed. But a purchase order
# number is all digits and highly distinct, so on shape alone it is indistinguishable
# from a material number, and it sits on the same row as one. Left in, it makes every
# PO number an alias of the material it happened to be ordered against, and one
# material then reaches two "material numbers" and is quarantined. That is not a
# hypothesis: it is what a six-document run of the real extract produced — 120
# conflicts and not one resolvable item.
_DOCUMENT_KEYS = (
    "po no", "po number", "purchasing document", "purchase order", "sales order",
    "order no", "order number", "delivery", "invoice", "document no", "document number",
    "sales docu no", "customer po number", "batch", "serial",
    "sold to", "ship to", "customer", "vendor", "supplier", "account",
)

# How strongly a column's values must match some column in another batch before it is
# trusted as an item key. Compared as an overlap coefficient — |a ∩ b| / min(|a|, |b|) —
# rather than as a share of either side, because a whole-warehouse stock snapshot holds
# far more materials than any transaction extract and is *supposed* to. Measuring only
# "how much of mine appears elsewhere" is the direction that made the intake gate block
# the one document whose key was right.
MIN_CORROBORATION = 0.30

# No real material has this many aliases. A component this large means a bad edge, not
# a well-connected item, and merging it would destroy far more than it joined.
MAX_COMPONENT = 50


def classify_code(value: str) -> str:
    """
    Which numbering system a value belongs to, from its shape alone.

    Deliberately conservative. A value that is neither clearly a material number nor
    clearly a part number is `unknown` and takes no part in crosswalking — a wrong
    system assignment produces a wrong merge, and a wrong merge is worse than a
    missing one because it reports success.
    """
    if not value:
        return SYSTEM_UNKNOWN
    text = str(value).strip().upper()
    if text.isdigit():
        # Long enough to be an identifier rather than a quantity or a year.
        return SYSTEM_MATNR if 4 <= len(text) <= 18 else SYSTEM_UNKNOWN
    if _PARTNO.match(text):
        return SYSTEM_PARTNO
    return SYSTEM_UNKNOWN


def _looks_like_measure(values: Sequence[str]) -> bool:
    """Dates, decimals and money are never item keys however distinct they are."""
    sample = [v for v in values[:200] if v]
    if not sample:
        return True
    decimals = sum(1 for v in sample if re.fullmatch(r"-?\d+\.\d+", v))
    dated = sum(1 for v in sample if re.match(r"^\d{4}-\d{2}-\d{2}", v))
    return (decimals + dated) / len(sample) > 0.5


def names_a_document(header: str) -> bool:
    """Whether a header names a document or a party rather than a material."""
    from ..ingest.profiler import normalize_header
    text = normalize_header(header)
    tokens = frozenset(text.split())
    for banned in _DOCUMENT_KEYS:
        wanted = frozenset(banned.split())
        if wanted <= tokens:
            return True
    return False


def corroborated(
    per_batch: Dict[Any, Dict[str, Set[str]]], minimum: float = MIN_CORROBORATION,
) -> Dict[Any, Set[str]]:
    """
    Which candidate columns are backed by a column somewhere else in the run.

    This is the evidence the pipeline already computed and then only ever printed in a
    warning: whether a column's values meet another document's. It is the one signal
    that actually decides a join key, so here it decides.

    Compared as an overlap coefficient against each other column individually, not
    against the union of everything else. Against a union, a stock snapshot's material
    column is diluted by every unrelated code in the run; against `Material` in the
    purchase history it matches almost perfectly, which is the true statement.

    With a single batch there is no cross-document evidence by definition, and every
    candidate is kept — shape and cardinality are then all there is.
    """
    if len(per_batch) < 2:
        return {key: set(columns) for key, columns in per_batch.items()}

    kept: Dict[Any, Set[str]] = {}
    for key, columns in per_batch.items():
        survivors = set()
        for column, values in columns.items():
            if not values:
                continue
            # Only the anchoring system is filtered. Two reasons, and the second is the
            # one that cost a test:
            #
            # A duplicate in the anchor system is fatal — it leaves the item with no
            # identity — and the anchor system is all digits, which is exactly what a
            # purchase order number looks like. So that is where an uncorroborated
            # column does damage.
            #
            # A part number that appears in only one document is not a rogue column,
            # it is *the crosswalk*: the planning master is the one place carrying
            # `Product code` beside `Material`, and filtering it for lack of
            # corroboration deletes the very evidence this module exists to read.
            if not all(classify_code(v) == SYSTEM_MATNR for v in values):
                survivors.add(column)
                continue
            best = 0.0
            for other_key, other_columns in per_batch.items():
                if other_key == key:
                    continue
                for other_values in other_columns.values():
                    if not other_values:
                        continue
                    shared = len(values & other_values)
                    best = max(best, shared / min(len(values), len(other_values)))
            if best >= minimum:
                survivors.add(column)
        kept[key] = survivors
    return kept


def candidate_columns(rows: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """
    Columns of a landed batch that could hold an item code, as {column: {row_no: code}}.

    Values are normalised the same way the adapter normalises `sku`, so a padded
    `000000000007100017` and a bare `7100017` are one code here as they must be
    everywhere else.
    """
    out: Dict[str, Dict[str, str]] = {}
    for column in rows.columns:
        if column == "row_no" or names_a_document(str(column)):
            continue
        raw = rows[column].astype("string")
        values = _normalize_material(raw)
        present = values.dropna()
        if len(present) == 0:
            continue
        if _looks_like_measure(list(present.astype(str))):
            continue
        distinct = present.nunique()
        if distinct < MIN_DISTINCT or distinct / len(present) < MIN_DISTINCT_RATE:
            continue
        coded = {
            str(rows.loc[idx, "row_no"]): str(value)
            for idx, value in present.items()
            if classify_code(str(value)) != SYSTEM_UNKNOWN
        }
        if len(set(coded.values())) >= MIN_DISTINCT:
            out[str(column)] = coded
    return out


class _Union:
    """Union-find over (erp_id, system, code) nodes."""

    def __init__(self):
        self.parent: Dict[Tuple[str, str, str], Tuple[str, str, str]] = {}

    def find(self, node):
        self.parent.setdefault(node, node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Lower key wins, so the result does not depend on the order edges arrive.
            low, high = sorted((ra, rb))
            self.parent[high] = low

    def components(self) -> Dict[Any, List]:
        groups: Dict[Any, List] = defaultdict(list)
        for node in self.parent:
            groups[self.find(node)].append(node)
        return groups


def item_uid(nodes: Sequence[Tuple[str, str, str]]) -> str:
    """
    A component's identity: its lowest-sorted code in the most authoritative system.

    Not a hash of the whole component, which was the first instinct and is wrong —
    adding one alias would change the uid, so every fact written under the old one
    would be orphaned and no two alias versions could be compared. Anchoring on the
    material number instead keeps the uid stable while the component grows around it,
    and keeps it readable, which matters more than it sounds at three in the morning.
    """
    for system in (SYSTEM_MATNR, SYSTEM_PARTNO):
        members = sorted(n for n in nodes if n[1] == system)
        if members:
            erp, _, code = members[0]
            return f"{erp}:{system}:{code}" if erp else f"{system}:{code}"
    erp, system, code = sorted(nodes)[0]
    return f"{erp}:{system}:{code}" if erp else f"{system}:{code}"


class AliasVersion:
    """One built snapshot of the alias table, on disk and addressable by version."""

    def __init__(self, root: Path, version: str):
        self.root, self.version = Path(root), version

    @property
    def dir(self) -> Path:
        return self.root / IDENTITY_DIRNAME / self.version

    @property
    def aliases(self) -> pd.DataFrame:
        target = self.dir / "aliases.parquet"
        if not target.exists():
            return pd.DataFrame(columns=["erp_id", "system_id", "code", "item_uid",
                                         "confidence", "source", "first_seen_batch"])
        return pd.read_parquet(target)

    @property
    def conflicts(self) -> pd.DataFrame:
        target = self.dir / "conflicts.parquet"
        if not target.exists():
            return pd.DataFrame(columns=["erp_id", "system_id", "code", "reason",
                                         "candidates"])
        return pd.read_parquet(target)

    @property
    def substitution_candidates(self) -> pd.DataFrame:
        target = self.dir / "substitutions.parquet"
        if not target.exists():
            return pd.DataFrame(columns=["erp_id", "system_id", "code_a", "code_b",
                                         "batch_id", "row_no"])
        return pd.read_parquet(target)

    @property
    def manifest(self) -> Dict[str, Any]:
        target = self.dir / "manifest.json"
        return json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}

    def resolve(self, code: str, erp_id: str = "") -> Optional[str]:
        """The item a code names, whichever numbering system it belongs to."""
        normalized = _normalize_material(pd.Series([str(code)], dtype="string")).iloc[0]
        if pd.isna(normalized):
            return None
        frame = self.aliases
        hit = frame[(frame["code"] == str(normalized)) & (frame["erp_id"] == erp_id)]
        return None if hit.empty else str(hit.iloc[0]["item_uid"])

    def codes_for(self, uid: str) -> List[str]:
        frame = self.aliases
        return sorted(frame[frame["item_uid"] == uid]["code"].astype(str))


class IdentityBuilder:
    """
    Builds one versioned alias table from a set of landed batches.

    Nothing is built incrementally and nothing is built as a side effect of resolving.
    `build` takes every batch that will take part, reads them all, and writes one
    snapshot — which is what makes the version meaningful and the run reproducible.
    """

    def __init__(self, landing: LandingStore = None, root=None):
        self.root, _ = resolve_store_root(root)
        self.landing = landing or LandingStore(self.root)

    def build(self, inputs: Iterable[Tuple[str, str]], erp_of: Dict[str, str] = None,
              built_by: str = "") -> AliasVersion:
        """
        `inputs` is [(doc_type, batch_id), …]; `erp_of` maps batch_id to its ERP.

        Batches are read in sorted order and edges are accumulated as a set, so the
        version and every uid in it are a function of the inputs alone — not of the
        order they were handed over. That property is the entire reason this is a
        separate phase, so it is enforced here rather than assumed.
        """
        erp_of = erp_of or {}
        ordered = sorted(set(inputs))

        edges: Set[Tuple] = set()
        nodes: Set[Tuple[str, str, str]] = set()
        substitutions: List[Dict[str, Any]] = []
        first_seen: Dict[Tuple[str, str, str], str] = {}
        columns_used: Dict[str, List[str]] = {}

        # Phase one: read every batch and collect what *could* be an item key.
        collected: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = {}
        for doc_type, batch_id in ordered:
            rows = self.landing.rows(doc_type, batch_id, named=True)
            if rows.empty:
                continue
            collected[(doc_type, batch_id)] = candidate_columns(rows)

        # Phase two: keep only the columns another document recognises. Done across all
        # batches at once — a per-batch decision would depend on which batch came first,
        # which is the circularity this whole two-phase shape exists to break.
        keep = corroborated({
            key: {column: set(codes.values()) for column, codes in columns.items()}
            for key, columns in collected.items()
        })

        for (doc_type, batch_id), candidates in collected.items():
            erp = erp_of.get(batch_id, "")
            candidates = {c: v for c, v in candidates.items()
                          if c in keep.get((doc_type, batch_id), set())}
            columns_used[batch_id] = sorted(candidates)

            by_row: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
            for column, coded in candidates.items():
                for row_no, code in coded.items():
                    node = (erp, classify_code(code), code)
                    nodes.add(node)
                    first_seen.setdefault(node, batch_id)
                    by_row[row_no].append(node)

            for row_no, found in by_row.items():
                unique = sorted(set(found))
                for i, a in enumerate(unique):
                    for b in unique[i + 1:]:
                        if a[1] == b[1]:
                            # Two codes of the same system on one row are a renumbering,
                            # not an alias. Proposed here and applied nowhere — a run
                            # that merged them would plan one item as two, or two as one,
                            # on nobody's authority.
                            substitutions.append({
                                "erp_id": erp, "system_id": a[1],
                                "code_a": a[2], "code_b": b[2],
                                "batch_id": batch_id, "row_no": int(row_no),
                            })
                        else:
                            edges.add((a, b, batch_id))

        union = _Union()
        for node in nodes:
            union.find(node)
        for a, b, _batch in sorted(edges):
            union.union(a, b)

        alias_rows: List[Dict[str, Any]] = []
        conflict_rows: List[Dict[str, Any]] = []
        for _root, members in sorted(union.components().items()):
            members = sorted(members)
            reason = self._component_fault(members)
            if reason:
                for erp, system, code in members:
                    conflict_rows.append({
                        "erp_id": erp, "system_id": system, "code": code,
                        "reason": reason,
                        "candidates": json.dumps(
                            [f"{s}:{c}" for _e, s, c in members][:20],
                            ensure_ascii=False),
                    })
                continue
            uid = item_uid(members)
            for erp, system, code in members:
                alias_rows.append({
                    "erp_id": erp, "system_id": system, "code": code,
                    "item_uid": uid,
                    "confidence": 1.0 if len(members) == 1 else 0.9,
                    "source": "singleton" if len(members) == 1 else "co_occurrence",
                    "first_seen_batch": first_seen.get((erp, system, code), ""),
                })

        version = self._version(ordered, edges, erp_of)
        return self._write(version, alias_rows, conflict_rows, substitutions,
                           ordered, columns_used, built_by)

    @staticmethod
    def _component_fault(members: Sequence[Tuple[str, str, str]]) -> str:
        """
        Why this component must not be merged, or an empty string if it may be.

        Only the *anchoring* system is checked, not every system. The two cases look
        alike and are not:

          two material numbers reaching one part number — which material the part
          number names is unknown, so the item has no identity and nothing here may be
          merged

          one material number carrying two part numbers — a legacy code beside a
          current one. The item's identity is not in doubt for a moment, and refusing
          the whole component would make the *material number itself* unresolvable
          over an ambiguity that does not touch it

        Checking every system treated the second as the first, and quarantined items
        whose identity was never ambiguous.
        """
        if len(members) > MAX_COMPONENT:
            return f"component of {len(members)} codes — a bad edge, not one material"
        anchor = next((s for s in (SYSTEM_MATNR, SYSTEM_PARTNO)
                       if any(m[1] == s for m in members)), None)
        if anchor is None:
            return ""
        per_erp: Dict[str, int] = defaultdict(int)
        for erp, system, _code in members:
            if system == anchor:
                per_erp[erp] += 1
        for erp, count in sorted(per_erp.items()):
            if count > 1:
                return f"{count} distinct {anchor} codes reach one item"
        return ""

    @staticmethod
    def _version(ordered, edges, erp_of) -> str:
        """
        Content-addressed: the same batches with the same evidence give the same version.

        Includes the ERP assignment because the same code in two ERP instances is two
        materials, so changing that mapping changes the answer even when nothing else
        moved.
        """
        payload = json.dumps({
            "inputs": [list(pair) for pair in ordered],
            "erp_of": dict(sorted(erp_of.items())),
            "edges": sorted(f"{a}|{b}" for a, b, _batch in edges),
        }, ensure_ascii=False, sort_keys=True)
        return sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _write(self, version, alias_rows, conflict_rows, substitutions,
               ordered, columns_used, built_by) -> AliasVersion:
        result = AliasVersion(self.root, version)
        result.dir.mkdir(parents=True, exist_ok=True)

        def _put(name: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
            frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
            staged = result.dir / f"{name}.parquet.partial"
            frame.to_parquet(staged, index=False)
            staged.replace(result.dir / f"{name}.parquet")

        _put("aliases", alias_rows,
             ["erp_id", "system_id", "code", "item_uid", "confidence", "source",
              "first_seen_batch"])
        _put("conflicts", conflict_rows,
             ["erp_id", "system_id", "code", "reason", "candidates"])
        _put("substitutions", substitutions,
             ["erp_id", "system_id", "code_a", "code_b", "batch_id", "row_no"])

        (result.dir / "manifest.json").write_text(json.dumps({
            "alias_version": version,
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "built_by": built_by,
            "inputs": [{"doc_type": d, "batch_id": b} for d, b in ordered],
            "key_columns": columns_used,
            "items": len({row["item_uid"] for row in alias_rows}),
            "aliases": len(alias_rows),
            "conflicts": len(conflict_rows),
            "substitution_candidates": len(substitutions),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    # ── Current pointer ──────────────────────────────────────────────────────

    def promote(self, version: AliasVersion) -> None:
        """
        Name a built version as the one new resolutions should use.

        Separate from `build` on purpose: building is cheap and safe, promoting changes
        what the next run means by an item number. A version with conflicts can still be
        promoted — the conflicting codes simply stay unmerged — but the decision to do
        so is a person's.
        """
        (self.root / IDENTITY_DIRNAME).mkdir(parents=True, exist_ok=True)
        (self.root / IDENTITY_DIRNAME / CURRENT_NAME).write_text(json.dumps({
            "alias_version": version.version,
            "promoted_at": datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")

    def current(self) -> Optional[AliasVersion]:
        target = self.root / IDENTITY_DIRNAME / CURRENT_NAME
        if not target.exists():
            return None
        version = json.loads(target.read_text(encoding="utf-8"))["alias_version"]
        return AliasVersion(self.root, version)
