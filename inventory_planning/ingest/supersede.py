"""
Material renumbering — one material trading under two numbers, made into one again.

A part that was 7100014 and is now 7100015 is a single planning problem that arrives
as two. Nothing in the transactions says so, and nothing downstream notices: the old
number holds stock and open POs against a demand history that stops, so it reads as
dead stock with a year of cover; the new number carries three months of demand and no
receipts, so it is classified non-stocking, forecast flat, and given almost no safety
stock. Two confident, self-consistent, wrong answers.

So the correction has to happen before anything is computed, and it has to happen in
exactly one place. Every join in this pipeline is on `sku`; rewriting the number in
some documents and not others turns one silent error into a worse one — an inventory
position for a SKU whose demand history now lives under a different key.

Two things this module refuses to do:

**Infer the pairs.** A rule pairing a SKU whose demand goes to zero with one that ramps
up in the same month would be right often, and the two ways of being wrong do not cost
the same. A missed pair leaves the status quo. An invented pair adds two unrelated
materials' stock and history together and produces a complete report with no error in
it anywhere. Pairs are declared or they do not exist.

**Guess past a contradiction.** A map containing a cycle, or one number with two
different successors, cannot be resolved by choosing. Those pairs are dropped and
reported; the rest of the map still applies, because one mistyped row should not cost
the other forty merges.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from .contract import DocContract

# Units, as the contracts declare them, and what a conversion ratio does to each.
# A ratio of 10 means one old piece is ten new ones: the quantities multiply, the
# per-unit money divides, and the total money does not move at all — which is the
# check that the rule is the right one.
_QUANTITY_UNIT = "base_uom"
_MONEY_UNIT = "currency"
_PER_UNIT_MONEY = "currency_per_base_uom"


@dataclass(frozen=True)
class Supersession:
    """One declared renumbering, after chain resolution."""

    old_sku: str
    new_sku: str
    ratio: float = 1.0
    effective_date: Optional[pd.Timestamp] = None
    rationale: str = ""
    source: str = "substitution"
    # The successor as written on the row. Differs from `new_sku` only when the
    # declaration was a chain — A→B and B→C leave `declared_as` at B and `new_sku`
    # at C — and is kept because the planner will look for what they typed.
    declared_as: str = ""

    @property
    def chained(self) -> bool:
        return bool(self.declared_as) and self.declared_as != self.new_sku


@dataclass
class MergeRecord:
    """What one predecessor contributed to one document."""

    old_sku: str
    new_sku: str
    doc_type: str
    rows: int
    quantity: float = 0.0
    quantity_field: str = ""


@dataclass
class SupersessionReport:
    """Everything the run should be able to say about the merge afterwards."""

    pairs: List[Supersession] = dc_field(default_factory=list)
    records: List[MergeRecord] = dc_field(default_factory=list)
    phase_pairs: int = 0
    # Contradictions in the map itself. These dropped pairs, and why.
    problems: List[str] = dc_field(default_factory=list)
    # Declarations the transactions do not support. Not errors — questions.
    challenges: List[str] = dc_field(default_factory=list)
    # Pairs found in a document that was handed over for another purpose. Reported,
    # never applied — see `SupersessionMap.from_frames`.
    proposals: List[Supersession] = dc_field(default_factory=list)
    # Why a map that exists was not acted on at all.
    withheld: Optional[str] = None
    # True when the substitution list carried no `relation` column, so every pair on
    # it was read as a renumbering. Worth saying out loud: it is the reading that
    # merges data, and it was assumed rather than stated.
    relation_assumed: bool = False

    @property
    def applied(self) -> List[Supersession]:
        """Pairs that actually found something to merge."""
        touched = {r.old_sku for r in self.records}
        return [p for p in self.pairs if p.old_sku in touched]

    def proposal_block(self) -> str:
        """The proposals as a substitution list the planner can save and hand back."""
        if not self.proposals:
            return ""
        lines = ["old_sku,new_sku,relation,effective_date,rationale"]
        for p in sorted(self.proposals, key=lambda p: p.old_sku):
            date = f"{p.effective_date:%Y-%m-%d}" if p.effective_date is not None else ""
            lines.append(f"{p.old_sku},{p.new_sku},supersede,{date},")
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per predecessor per document — the durable record of the merge."""
        by_pair = {p.old_sku: p for p in self.pairs}
        rows = []
        for rec in self.records:
            pair = by_pair.get(rec.old_sku)
            rows.append({
                "old_sku": rec.old_sku,
                "new_sku": rec.new_sku,
                "doc_type": rec.doc_type,
                "rows_merged": rec.rows,
                "quantity_field": rec.quantity_field,
                "quantity_merged": rec.quantity,
                "ratio": pair.ratio if pair else 1.0,
                "effective_date": pair.effective_date if pair else pd.NaT,
                "declared_as": pair.declared_as if pair else "",
                "source": pair.source if pair else "",
                "rationale": pair.rationale if pair else "",
            })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        if not (self.pairs or self.problems or self.phase_pairs or self.proposals):
            return ""
        lines = ["  Material supersessions"]
        lines.append("  " + "-" * 58)

        if self.withheld:
            lines.append(f"    NOTHING WAS MERGED — {self.withheld}")
            lines.append("")

        applied = self.applied
        if applied:
            by_pair: Dict[str, List[MergeRecord]] = {}
            for rec in self.records:
                by_pair.setdefault(rec.old_sku, []).append(rec)
            for pair in sorted(applied, key=lambda p: p.old_sku):
                recs = by_pair[pair.old_sku]
                where = ", ".join(
                    f"{r.doc_type} {r.rows:,} row{'s' if r.rows != 1 else ''}"
                    for r in sorted(recs, key=lambda r: r.doc_type)
                )
                tail = f"  ×{pair.ratio:g}" if pair.ratio != 1.0 else ""
                lines.append(f"    {pair.old_sku} -> {pair.new_sku}{tail}   {where}")
                if pair.chained:
                    lines.append(f"      (declared as {pair.declared_as}; "
                                 f"resolved through the chain to {pair.new_sku})")

        inert = [p for p in self.pairs if p not in applied]
        if inert:
            lines.append(f"    {len(inert)} declared pair(s) matched nothing in this "
                         f"extract: {', '.join(p.old_sku for p in inert[:6])}"
                         + (" …" if len(inert) > 6 else ""))

        if self.phase_pairs:
            lines.append(f"    {self.phase_pairs} phase-in/phase-out pair(s) read and "
                         f"not merged — coexisting materials are annotated, never combined")

        if applied and self.relation_assumed:
            lines.append("")
            lines.append("    The list carried no `relation` column, so every pair on it "
                         "was read as a")
            lines.append("    renumbering and merged. Add relation=phase to any pair whose "
                         "two numbers")
            lines.append("    still trade side by side — those must not be combined.")

        if self.proposals:
            lines.append("")
            lines.append(f"  ○ {len(self.proposals)} renumbering(s) declared in the item "
                         f"master and NOT applied:")
            for p in sorted(self.proposals, key=lambda p: p.old_sku)[:8]:
                date = (f", effective {p.effective_date:%Y-%m-%d}"
                        if p.effective_date is not None else "")
                lines.append(f"      {p.old_sku} -> {p.new_sku}{date}")
            if len(self.proposals) > 8:
                lines.append(f"      … and {len(self.proposals) - 8} more")
            lines.append("      Merging item numbers restructures every document, so it "
                         "needs a decision")
            lines.append("      taken for this run rather than a column that came along "
                         "with the master.")
            lines.append("      To apply them, save this as a file and include it in the "
                         "next run:")
            lines.append("")
            lines.extend(f"        {line}" for line in self.proposal_block().splitlines())

        if self.challenges:
            lines.append("")
            lines.append("  ⚠ Declarations the transactions do not support:")
            lines.extend(f"      {c}" for c in self.challenges)

        if self.problems:
            lines.append("")
            lines.append("  ⚠ Dropped from the map — these pairs were NOT applied:")
            lines.extend(f"      {p}" for p in self.problems)

        return "\n".join(lines)


class SupersessionMap:
    """
    Declared renumberings, resolved to terminal successors and applied to a frame.

    Resolution is transitive: A→B and B→C means A→C, because a document written
    before either change still says A and has to arrive at the number that is alive
    now. Ratios compound along the chain for the same reason.
    """

    def __init__(self, pairs: Sequence[Supersession], report: SupersessionReport):
        self._pairs = {p.old_sku: p for p in pairs}
        self.report = report

    def __len__(self) -> int:
        return len(self._pairs)

    def __bool__(self) -> bool:
        return bool(self._pairs)

    @property
    def pairs(self) -> List[Supersession]:
        return list(self._pairs.values())

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_frames(
        cls,
        substitution_df: pd.DataFrame = None,
        item_master_df: pd.DataFrame = None,
        withheld: str = None,
    ) -> "SupersessionMap":
        """
        One document can cause a merge, and it is the one whose purpose is to declare
        identity: a substitution list.

        The gate is deliberate and it is the difference between this and every other
        input. Elsewhere a stated value is ranked, cross-checked and reported — a lead
        time in a master is an intention that measurement can outrank. Identity cannot
        be ranked against anything. Merging two numbers restructures every document in
        the run, no downstream figure carries a trace of it, and there is nothing left
        to disagree with afterwards. So it takes an act, not an attribute.

        An item master's follow-up-material column is therefore read and *proposed*,
        never applied. That column is maintained years before anyone plans on it and is
        routinely stale, and the master was handed over to supply lead times and MOQs —
        nobody chose it for this. The proposal comes out as a substitution list the
        planner can check and hand back, which turns it into the act the merge needs
        while still saving them from retyping it.

        `withheld` refuses to apply an otherwise valid map, for when the caller has
        decided the source cannot be trusted — a substitution list routed on a thin
        margin, say. The pairs are still reported.
        """
        report = SupersessionReport()

        declared = cls._read_substitution(substitution_df, report)
        report.proposals = cls._read_item_master(item_master_df, report)

        # A pair the planner has already declared is not also a proposal, and one they
        # declared *differently* is a disagreement worth naming — but the act wins over
        # the attribute either way, so neither drops the merge.
        by_old = {p.old_sku: p for p in declared}
        kept: List[Supersession] = []
        for proposal in report.proposals:
            match = by_old.get(proposal.old_sku)
            if match is None:
                kept.append(proposal)
            elif match.new_sku != proposal.new_sku:
                report.challenges.append(
                    f"{proposal.old_sku}: the substitution list says it became "
                    f"{match.new_sku}, the item master says {proposal.new_sku}. The "
                    f"list was acted on; the master's follow-up material may be stale."
                )
        report.proposals = kept

        if withheld:
            report.withheld = withheld
            report.pairs = cls._resolve(declared, report)
            return cls([], report)

        resolved = cls._resolve(declared, report)
        report.pairs = resolved
        return cls(resolved, report)

    @staticmethod
    def _read_substitution(
        df: Optional[pd.DataFrame], report: SupersessionReport
    ) -> List[Supersession]:
        if df is None or not len(df):
            return []
        if "old_sku" not in df.columns or "new_sku" not in df.columns:
            return []

        out: List[Supersession] = []
        # An absent relation column is a list of renumberings — that is what the
        # value domain defaults to, and the default only reaches values that are
        # present. A column that is not there never passes through it. Recorded,
        # because the assumed reading is the one that merges data and the planner
        # should be told it was assumed.
        report.relation_assumed = "relation" not in df.columns
        relation = (
            df["relation"] if "relation" in df.columns
            else pd.Series("supersede", index=df.index)
        )
        relation = relation.fillna("supersede")

        for idx, row in df.iterrows():
            old, new = _clean(row.get("old_sku")), _clean(row.get("new_sku"))
            if not old or not new:
                continue
            if relation.loc[idx] == "phase":
                report.phase_pairs += 1
                continue
            if old == new:
                report.problems.append(
                    f"{old} is declared as its own successor — dropped"
                )
                continue
            out.append(Supersession(
                old_sku=old,
                new_sku=new,
                ratio=_ratio(row.get("ratio")),
                effective_date=_date(row.get("effective_date")),
                rationale=str(row.get("rationale") or ""),
                source="substitution",
            ))
        return out

    @staticmethod
    def _read_item_master(
        df: Optional[pd.DataFrame], report: SupersessionReport
    ) -> List[Supersession]:
        if df is None or not len(df):
            return []
        if "sku" not in df.columns or "successor_sku" not in df.columns:
            return []

        out: List[Supersession] = []
        for _, row in df.iterrows():
            old, new = _clean(row.get("sku")), _clean(row.get("successor_sku"))
            if not old or not new or old == new:
                continue
            out.append(Supersession(
                old_sku=old,
                new_sku=new,
                ratio=1.0,
                effective_date=_date(row.get("discontinued_date")),
                rationale="follow-up material declared in the item master",
                source="item_master",
            ))
        return out

    @staticmethod
    def _resolve(
        declared: Sequence[Supersession], report: SupersessionReport
    ) -> List[Supersession]:
        """
        Collapse chains, and drop what cannot be collapsed.

        Many-to-one is ordinary — two obsolete numbers consolidated onto one current
        part. One-to-many is not a renumbering at all: a number with two successors
        is a split, and which of them a given transaction belongs to is a question
        about the transaction, not about the material. There is nothing to choose
        between, so both edges go.
        """
        edges: Dict[str, List[Supersession]] = {}
        for pair in declared:
            edges.setdefault(pair.old_sku, []).append(pair)

        # Two sources agreeing on the same successor is agreement, not conflict.
        graph: Dict[str, Supersession] = {}
        for old, candidates in edges.items():
            targets = {(p.new_sku, round(p.ratio, 9)) for p in candidates}
            if len(targets) > 1:
                shown = ", ".join(sorted(p.new_sku for p in candidates))
                report.problems.append(
                    f"{old} is declared to supersede into more than one number "
                    f"({shown}) — that is a split, not a renumbering, and no rule can "
                    f"decide which transaction belongs to which. Both edges dropped."
                )
                continue
            graph[old] = candidates[0]

        resolved: List[Supersession] = []
        for old, pair in graph.items():
            seen: Set[str] = {old}
            node, ratio = pair.new_sku, pair.ratio
            broken = False
            while node in graph:
                if node in seen:
                    report.problems.append(
                        f"the chain from {old} loops back on itself "
                        f"({' -> '.join(list(seen) + [node])}) — dropped"
                    )
                    broken = True
                    break
                seen.add(node)
                nxt = graph[node]
                node, ratio = nxt.new_sku, ratio * nxt.ratio
            if broken:
                continue
            resolved.append(Supersession(
                old_sku=old,
                new_sku=node,
                ratio=ratio,
                effective_date=pair.effective_date,
                rationale=pair.rationale,
                source=pair.source,
                declared_as=pair.new_sku,
            ))
        return sorted(resolved, key=lambda p: p.old_sku)

    # ── Application ──────────────────────────────────────────────────────────

    def apply(
        self, df: pd.DataFrame, contract: DocContract, doc_type: str = None
    ) -> Tuple[pd.DataFrame, List[MergeRecord]]:
        """
        Rewrite `sku`, scale by the conversion ratio, and recombine the rows that
        have just become duplicates of each other.

        The recombination is not optional bookkeeping. After the rewrite two rows can
        share the document's natural key — the same month in a demand series, the same
        SKU and location in a stock report — and the contract's own grain test would
        fail on a frame the pipeline itself just produced.
        """
        doc_type = doc_type or contract.doc_type
        if not self._pairs or "sku" not in df.columns:
            return df, []

        original = df["sku"].astype("string")
        mapped = original.map({p.old_sku: p.new_sku for p in self._pairs.values()})
        hit = mapped.notna()
        if not hit.any():
            return df, []

        df = df.copy()
        records = self._records(df, original[hit], doc_type, contract)

        df["sku"] = original.where(~hit, mapped)
        df = self._scale(df, contract, original, hit)

        key = [k for k in contract.natural_key if k in df.columns]
        if key and df.duplicated(subset=key).any():
            df = self._recombine(df, contract, key, hit)
        return df, records

    def _records(
        self, df: pd.DataFrame, moved: pd.Series, doc_type: str, contract: DocContract
    ) -> List[MergeRecord]:
        qty_col = _quantity_column(df, contract)
        out = []
        for old, idx in moved.groupby(moved).groups.items():
            pair = self._pairs[old]
            qty = float(pd.to_numeric(df.loc[idx, qty_col], errors="coerce").sum()) \
                if qty_col else 0.0
            out.append(MergeRecord(
                old_sku=str(old),
                new_sku=pair.new_sku,
                doc_type=doc_type,
                rows=len(idx),
                quantity=qty * pair.ratio,
                quantity_field=qty_col or "",
            ))
        return sorted(out, key=lambda r: r.old_sku)

    def _scale(
        self,
        df: pd.DataFrame,
        contract: DocContract,
        original: pd.Series,
        hit: pd.Series,
    ) -> pd.DataFrame:
        """Apply the conversion ratio to the rows that were rewritten, and only those."""
        ratios = original.map({p.old_sku: p.ratio for p in self._pairs.values()})
        if (ratios.dropna() == 1.0).all():
            return df

        factor = pd.to_numeric(ratios.where(hit, 1.0), errors="coerce").fillna(1.0)
        for name, spec in contract.fields.items():
            if name not in df.columns or spec.role != "measure":
                continue
            if spec.unit == _QUANTITY_UNIT:
                df[name] = pd.to_numeric(df[name], errors="coerce") * factor
            elif spec.unit == _PER_UNIT_MONEY:
                df[name] = pd.to_numeric(df[name], errors="coerce") / factor
        return df

    @staticmethod
    def _recombine(
        df: pd.DataFrame, contract: DocContract, key: List[str], hit: pd.Series
    ) -> pd.DataFrame:
        """
        Combine rows that now share a natural key, by what each column *is*.

        The rule follows from the key rather than from the document's name. A natural
        key of `[sku]` alone means one row per material of its standing attributes —
        a lead time, an MOQ, a nominated supplier — so two such rows are two
        descriptions of one thing, and adding them up would produce a 180-day lead
        time out of two 90-day ones. Every other key in the contract set carries an
        event or a period alongside the SKU, so two rows there are two things that
        happened and the quantities belong together.
        """
        df = df.copy()
        is_master = list(contract.natural_key) == ["sku"]
        # The successor's own row describes the successor. The predecessor's is a
        # fallback for what the successor's leaves blank — a master row created for a
        # new number is routinely thinner than the one it replaced.
        df["__successor_row"] = ~hit.reindex(df.index).fillna(False).to_numpy()
        df = df.sort_values("__successor_row", ascending=False, kind="stable")

        if is_master:
            out = df.groupby(key, as_index=False, dropna=False).first()
            return out.drop(columns="__successor_row")

        qty_col = _quantity_column(df, contract)
        per_unit = [
            name for name, spec in contract.fields.items()
            if name in df.columns and spec.role == "measure"
            and spec.unit == _PER_UNIT_MONEY
        ]
        weighted = []
        if qty_col:
            weight = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)
            for name in per_unit:
                df[f"__w_{name}"] = pd.to_numeric(df[name], errors="coerce") * weight
                weighted.append(name)
            df["__weight"] = weight

        agg: Dict[str, str] = {}
        for col in df.columns:
            if col in key:
                continue
            spec = contract.field(col)
            additive = (
                spec is not None and spec.role == "measure"
                and spec.unit in (_QUANTITY_UNIT, _MONEY_UNIT)
            )
            if additive or col.startswith("__w_") or col == "__weight":
                agg[col] = "sum"
            else:
                agg[col] = "first"

        out = df.groupby(key, as_index=False, dropna=False).agg(agg)
        # A per-unit price is not a thing two rows have two of; it is one number the
        # combined quantity was bought at. Summing it invents money, averaging it
        # ignores that one row may be a hundred times the other.
        # Two rows for one material both at zero stock is ordinary, not exceptional —
        # an item can appear in a stock report with nothing on the shelf. There is no
        # weighted price to compute there, so the successor's own price stands.
        total = pd.to_numeric(out["__weight"], errors="coerce").replace(0.0, float("nan")) \
            if "__weight" in out.columns else None
        for name in weighted:
            blended = pd.to_numeric(out[f"__w_{name}"], errors="coerce") / total
            out[name] = blended.fillna(pd.to_numeric(out[name], errors="coerce"))
        drop = [c for c in out.columns if c.startswith("__w_")] + ["__successor_row"]
        if "__weight" in out.columns:
            drop.append("__weight")
        return out.drop(columns=[c for c in drop if c in out.columns])

    # ── Verification ─────────────────────────────────────────────────────────

    def challenge(self, frames: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Check each declaration against the transactions that are supposed to support it.

        This is not the inference the module refuses to do — nothing here proposes a
        pair. It tests the premise of a pair somebody already asserted, which is the
        same treatment a stated lead time gets: an old number still shipping long after
        its own effective date has not been superseded, whatever the master says, and
        the merge has just folded a live material into another one.
        """
        out: List[str] = []
        for pair in self._pairs.values():
            if pair.effective_date is None:
                continue
            # One finding per pair, not per document. The same live material shows up
            # in all five extracts, and five near-identical lines about it read as
            # five problems rather than as the one that it is.
            where: List[str] = []
            for doc_type, df in sorted(frames.items()):
                if df is None or not len(df) or "sku" not in df.columns:
                    continue
                date_col = _event_date_column(df)
                if date_col is None:
                    continue
                rows = df[df["sku"].astype("string") == pair.old_sku]
                if not len(rows):
                    continue
                after = int((pd.to_datetime(rows[date_col], errors="coerce")
                             > pair.effective_date).sum())
                if after:
                    where.append(f"{doc_type} {after:,}")
            if where:
                out.append(
                    f"{pair.old_sku} -> {pair.new_sku}: the old number is still "
                    f"transacting after its effective date "
                    f"{pair.effective_date:%Y-%m-%d} ({', '.join(where)} row(s)). "
                    f"Either the switch has not happened yet, or the two numbers "
                    f"coexist — in which case this is a phase pair and the merge has "
                    f"just folded a live material into another one."
                )
        self.report.challenges.extend(out)
        return out


# ── Helpers ──────────────────────────────────────────────────────────────────


# Event dates, most specific first. A supersession is checked against when goods
# actually moved, not against when a document happens to have been raised.
_EVENT_DATE_FIELDS = ("ship_date", "receive_date", "period", "order_date",
                      "committed_delivery", "customer_request_date")

# What each document means by "how much of this material". Named per document rather
# than found by rule, because the rule would get two of them backwards: `open_qty` is
# the right answer for an open PO and close to zero on a receipt history, where the
# quantity that matters is what was ordered.
_HEADLINE_QUANTITY = {
    "inventory": "qty_on_hand",
    "sales_history": "qty",
    "demand_timeseries": "qty",
    "po_history": "po_qty",
    "open_po": "open_qty",
    "open_so": "open_qty",
}


def _quantity_column(df: pd.DataFrame, contract: DocContract = None) -> Optional[str]:
    """
    The column that stands for 'how much' in this document, if it has one.

    A master has none. Its base-unit measures are parameters — an MOQ, a pack size,
    a hand-set safety stock — and reporting one of them as the quantity that moved in
    a merge would be a number that looks like stock and is not.
    """
    if contract is None:
        return None
    if list(contract.natural_key) == ["sku"]:
        return None
    named = _HEADLINE_QUANTITY.get(contract.doc_type)
    if named and named in df.columns:
        return named
    for name, spec in contract.fields.items():
        if name in df.columns and spec.role == "measure" and spec.unit == _QUANTITY_UNIT:
            return name
    return None


def _event_date_column(df: pd.DataFrame) -> Optional[str]:
    for name in _EVENT_DATE_FIELDS:
        if name in df.columns:
            return name
    return None


def _clean(value) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "nat", "none", "<na>", "null"} else text


def _ratio(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 1.0
    # A missing ratio is 1. A zero or negative one is a maintenance error the contract
    # already reports; treating it as 1 here keeps it from multiplying a catalogue by
    # zero while the assertion carries the finding.
    return out if out > 0 else 1.0


def _date(value) -> Optional[pd.Timestamp]:
    out = pd.to_datetime(value, errors="coerce")
    return None if pd.isna(out) else out
