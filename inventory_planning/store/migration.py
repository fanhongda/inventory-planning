"""
A maintenance plan, pinned to the batches it was read against.

INTERFACE.md §7's third seam, the half that bites today. `store restate` and `store void`
print a plan and then, on `--apply`, **select again from scratch**. Between the two
commands the store keeps moving — shadow write lands batches continuously — so a
predicate that matched 1,188 batches when it was read can match 1,191 when it is carried
out, and the three nobody approved are changed in silence.

The macro and rule editors already solved this shape: a proposal reports the digest of
what the diff was computed against, and an apply that cannot quote it back is refused.
What is approved is a specific change to specific bytes, not a wish about a setting.

## What the "bytes" are here, and the answer that looks right and checks nothing

For a file it is obvious. For a plan over a store it is not, and there are three
candidates:

    the ledger's digest      too strict. The pipeline writes batches constantly, so any
                             unrelated arrival invalidates a plan while it is being
                             read, and nothing would ever reach `--apply`.

    the selector arguments   wrong, and wrong in the dangerous direction.
                             `--loaded-before X --layer canonical` digests identically
                             at both moments and selects differently — which is the
                             failure itself, wearing a safety check's clothes.

    the selection            right. The plan is "these 1,188 batch ids become prepared".

So the basis covers the sorted batch ids, the operation, and the target layer — the last
because a plan to restate to `prepared` must not be applicable as a plan to restate to
`canonical` over the same batches.

## Applying replays the plan; it does not select again

The stronger half, and the reason this is a file rather than a digest passed on the
command line. A 1,188-batch migration gets read, thought about and often shown to
somebody, which is not the few seconds a form takes. The plan is written out, and
`--apply` carries out **exactly the ids in it**.

Re-selecting and comparing would still rest on the selector behaving identically twice.
Replaying a fixed list rests on nothing. What is checked at apply time is therefore not
"does re-selecting agree" but "are these batches still in the state the plan assumed",
which is a question about the store rather than about the query.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field as dc_field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PLAN_VERSION = 1

OP_RESTATE = "restate"
OP_VOID = "void"


class PlanRefused(ValueError):
    """A plan that cannot be carried out, with what differs."""


def plan_basis(op: str, batch_ids: Sequence[str], to: Optional[str] = None) -> str:
    """
    The digest a plan is pinned by.

    Sorted, because the set is the content — a selector that returns the same batches in
    another order has not changed anything. The operation and target are in it so that
    one approved selection cannot be carried out as a different operation over it.
    """
    body = json.dumps({"op": op, "to": to or "", "batch_ids": sorted(set(batch_ids))},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass
class MigrationPlan:
    """What was read and approved, in the shape it will be carried out."""

    op: str
    batch_ids: List[str]
    to: Optional[str] = None
    # What each batch looked like when the plan was made. The apply checks against this
    # rather than against the selector — a batch voided or already restated since is a
    # difference worth refusing over, and it cannot be seen by re-running a query.
    assumed: Dict[str, Dict[str, Any]] = dc_field(default_factory=dict)
    selector: str = ""
    created_at: str = ""
    store: str = ""
    version: int = PLAN_VERSION
    basis: str = ""

    @classmethod
    def build(cls, op: str, batches: Sequence[Dict[str, Any]], to: str = None,
              selector: str = "", store: str = "") -> "MigrationPlan":
        ids = [str(b["batch_id"]) for b in batches]
        assumed = {str(b["batch_id"]): {
            "frame_layer": str(b.get("frame_layer") or ""),
            "status": str(b.get("status") or "active"),
            "doc_type": str(b.get("doc_type") or ""),
        } for b in batches}
        return cls(op=op, batch_ids=sorted(set(ids)), to=to, assumed=assumed,
                   selector=selector, store=str(store),
                   created_at=datetime.now().isoformat(timespec="seconds"),
                   basis=plan_basis(op, ids, to))

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "MigrationPlan":
        path = Path(path)
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PlanRefused(f"{path} is not a readable plan: {exc}") from exc
        if int(body.get("version") or 0) != PLAN_VERSION:
            raise PlanRefused(
                f"{path} is a v{body.get('version')} plan; this reads v{PLAN_VERSION}.")
        plan = cls(**{k: v for k, v in body.items() if k in cls.__dataclass_fields__})
        # The plan is pinned by its own contents, so a file edited between writing and
        # applying is caught here rather than by the store disagreeing later.
        recomputed = plan_basis(plan.op, plan.batch_ids, plan.to)
        if plan.basis != recomputed:
            raise PlanRefused(
                f"{path} has been edited since it was written — its basis does not "
                f"match its own contents. Generate the plan again.")
        return plan

    # ── Checking against the store as it stands now ──────────────────────────

    def drift(self, ledger) -> List[str]:
        """
        How the store differs from what this plan assumed. Empty means carry on.

        Stated per difference rather than as "stale": a refusal a person cannot act on
        sends them to re-run the same command and hope. Each line here names what moved
        and what to do about it.
        """
        current = {str(b["batch_id"]): b
                   for b in ledger.batches(include_void=True)}
        differences: List[str] = []

        missing = [b for b in self.batch_ids if b not in current]
        if missing:
            differences.append(
                f"{len(missing)} batch(es) in the plan are not in this store "
                f"({', '.join(missing[:3])}{' …' if len(missing) > 3 else ''}) — this "
                f"plan was made against a different store.")

        moved, voided = [], []
        for batch_id in self.batch_ids:
            entry = current.get(batch_id)
            if entry is None:
                continue
            assumed = self.assumed.get(batch_id) or {}
            if assumed.get("frame_layer") and \
                    str(entry.get("frame_layer") or "") != assumed["frame_layer"]:
                moved.append(batch_id)
            if assumed.get("status") != str(entry.get("status") or "active"):
                voided.append(batch_id)

        if moved:
            differences.append(
                f"{len(moved)} batch(es) have been restated since the plan was made "
                f"({', '.join(moved[:3])}{' …' if len(moved) > 3 else ''}) — applying "
                f"would overwrite somebody else's correction.")
        if voided:
            differences.append(
                f"{len(voided)} batch(es) have changed status since the plan was made "
                f"({', '.join(voided[:3])}{' …' if len(voided) > 3 else ''}).")
        return differences

    def rescan(self, selected: Sequence[Dict[str, Any]]) -> Optional[str]:
        """
        What the same selector matches now, against what the plan holds.

        Reported, never acted on. The apply carries out the plan's own list, so a
        selector that has since widened does not quietly grow the change — but the
        person should be told, because it usually means the plan is older than they
        think.
        """
        now = {str(b["batch_id"]) for b in selected}
        mine = set(self.batch_ids)
        if now == mine:
            return None
        extra, gone = now - mine, mine - now
        parts = []
        if extra:
            parts.append(f"{len(extra)} batch(es) now match that the plan does not")
        if gone:
            parts.append(f"{len(gone)} in the plan no longer match")
        return (f"The selector has moved since this plan was made: {', and '.join(parts)}. "
                f"Only the {len(mine)} in the plan will be changed. Re-plan if you meant "
                f"to include the rest.")

    def summary(self) -> str:
        what = f"restate to `{self.to}`" if self.op == OP_RESTATE else "void"
        return "\n".join([
            f"  Plan {self.basis[:12]} · {what}",
            f"    batches   {len(self.batch_ids):,}",
            f"    selector  {self.selector or '(none recorded)'}",
            f"    made      {self.created_at}",
            f"    store     {self.store or '(not recorded)'}",
        ])
