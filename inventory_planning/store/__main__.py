"""
Store maintenance: say what a batch is, or withdraw it.

    python -m inventory_planning.store show
    python -m inventory_planning.store restate --loaded-before 2026-09-05T21:00 \
        --layer prepared --reason "written by the pipeline before the writer changed"
    python -m inventory_planning.store void --source sample_data \
        --reason "synthetic data in a store of real facts"

**Nothing writes without `--apply`.** Every command prints the plan first, because the
counts are the diagnostic: expecting to restate a few hundred batches and being shown
twelve thousand means the selection is wrong, and that is worth learning before the
ledger has grown a line for each of them.

Neither operation touches a parquet file or an existing ledger line. A restatement
corrects what a batch says about *itself* — the layer of the pipeline its frame came
from, which is what a reader uses to decide whether two batches may be added together.
A void withdraws a batch from every reading and leaves its rows where they are. Both are
appended, both carry who and why, and both can themselves be undone by a later line.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .ledger import BatchLedger, LAYER_CANONICAL, LAYER_PREPARED, LAYER_UNKNOWN
from .location import resolve_store_root


def _select(ledger: BatchLedger, args) -> List[Dict[str, Any]]:
    """
    The batches a command applies to. Empty selectors select nothing, never everything.

    An operation over a whole store should have to be asked for one batch at a time or
    by an explicit filter; defaulting to "all of them" is how a maintenance command
    becomes the thing that needed maintenance.
    """
    chosen = []
    for batch in ledger.batches(doc_type=args.doc_type, include_void=True):
        if args.loaded_before and str(batch.get("transaction_time") or "") >= args.loaded_before:
            continue
        if args.loaded_after and str(batch.get("transaction_time") or "") <= args.loaded_after:
            continue
        if args.source and not any(s in str(batch.get("source_name") or "")
                                   for s in args.source):
            continue
        if args.layer and str(batch.get("frame_layer") or LAYER_UNKNOWN) not in args.layer:
            continue
        if args.batch and batch.get("batch_id") not in args.batch:
            continue
        chosen.append(batch)
    return chosen


def _describe(batches: List[Dict[str, Any]], verb: str) -> str:
    if not batches:
        return f"  Nothing selected — nothing to {verb}."
    by_type = Counter(b.get("doc_type", "?") for b in batches)
    by_layer = Counter(str(b.get("frame_layer") or LAYER_UNKNOWN) for b in batches)
    by_source = Counter(str(b.get("source_name") or "?") for b in batches)
    loaded = sorted(str(b.get("transaction_time") or "") for b in batches)
    lines = [
        f"  {len(batches):,} batch(es) selected, to {verb}:",
        "    by document : " + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())),
        "    by layer    : " + ", ".join(f"{k} {v}" for k, v in sorted(by_layer.items())),
        f"    loaded      : {loaded[0][:19]} … {loaded[-1][:19]}",
        "    sources     : " + ", ".join(
            f"{k} ({v})" for k, v in sorted(by_source.items(), key=lambda kv: -kv[1])[:6]),
    ]
    if len(by_source) > 6:
        lines.append(f"                  … and {len(by_source) - 6} more")
    return "\n".join(lines)


def _show(ledger: BatchLedger, root: Path) -> int:
    print(f"  {root}")
    batches = ledger.batches(include_void=True)
    if not batches:
        print("  empty")
        return 0
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for batch in batches:
        by_type.setdefault(batch.get("doc_type", "?"), []).append(batch)
    for doc_type, items in sorted(by_type.items()):
        layers = Counter(str(b.get("frame_layer") or LAYER_UNKNOWN) for b in items)
        restated = sum(1 for b in items if b.get("frame_layer_restated"))
        void = sum(1 for b in items if b.get("status") == "void")
        note = []
        if restated:
            note.append(f"{restated} restated")
        if void:
            note.append(f"{void} void")
        print(f"    {doc_type:<18} {len(items):>5} batch(es)   "
              + ", ".join(f"{k} {v}" for k, v in sorted(layers.items()))
              + (f"   ({'; '.join(note)})" if note else ""))
    return 0


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m inventory_planning.store",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=None,
                        help="Store root. Defaults to $INVENTORY_PLANNING_STORE or the "
                             "platform data directory.")
    parser.add_argument("--tenant", default=None,
                        help="Which tenant's store. Combines with --store rather than "
                             "replacing it: a tenant under an explicit root gets its "
                             "own subtree, so pointing --store at a dev store does not "
                             "merge every tenant's facts into it.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("show", help="What the store holds, by document and layer")

    for name, help_text in (("restate", "say what layer the selected batches hold"),
                            ("void", "withdraw the selected batches from every reading")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--doc-type", default=None)
        cmd.add_argument("--source", action="append", default=[],
                         help="Substring of the source file name. Repeatable.")
        cmd.add_argument("--layer", action="append", default=[],
                         choices=[LAYER_CANONICAL, LAYER_PREPARED, LAYER_UNKNOWN],
                         help="Only batches currently marked this. Repeatable.")
        cmd.add_argument("--batch", action="append", default=[], help="Batch id. Repeatable.")
        cmd.add_argument("--loaded-before", default=None,
                         help="ISO timestamp; batches loaded strictly before it.")
        cmd.add_argument("--loaded-after", default=None,
                         help="ISO timestamp; batches loaded strictly after it.")
        cmd.add_argument("--reason", default="", help="Required with --apply.")
        cmd.add_argument("--by", default="", help="Required with --apply.")
        cmd.add_argument("--apply", action="store_true",
                         help="Write. Without it the plan is printed and nothing changes.")
        if name == "restate":
            cmd.add_argument("--layer-to", dest="to",
                             choices=[LAYER_CANONICAL, LAYER_PREPARED, LAYER_UNKNOWN],
                             required=True, help="What the selected batches actually hold")

    args = parser.parse_args(argv)
    # Through the workspace resolver, so --tenant means the same thing here as it does
    # to the pipeline and the API. Maintenance that ran against a different store than
    # the run did is the failure this shares one resolver to avoid.
    from ..workspace import BadTenant, Workspace

    try:
        workspace = Workspace.resolve(args.tenant, store_root=args.store)
    except BadTenant as exc:
        parser.error(str(exc))
    root, source = workspace.store_root, workspace.origins["store_root"]
    ledger = BatchLedger(root)

    if args.command == "show":
        return _show(ledger, root)

    selectors = (args.doc_type, args.source, args.layer, args.batch,
                 args.loaded_before, args.loaded_after)
    if not any(selectors):
        parser.error("name at least one selector — --doc-type, --source, --layer, "
                     "--batch, --loaded-before or --loaded-after. A maintenance command "
                     "with no selector would apply to the whole store.")

    batches = _select(ledger, args)
    verb = "restate" if args.command == "restate" else "void"
    print(f"  {root}  (from {source})")
    if args.command == "restate":
        print(f"  Restating to `{args.to}`.")
    print(_describe(batches, verb))

    if not args.apply:
        print("\n  Plan only. Nothing was written. Re-run with --apply, --by and "
              "--reason to carry it out.")
        return 0
    if not batches:
        return 0
    if not args.reason.strip() or not args.by.strip():
        # The ledger refuses an unnamed change too, now. Kept here as well so the
        # message a person gets is about the flag they did not pass rather than about
        # a writer three frames down.
        parser.error("--apply needs --reason and --by. A change to the store with no "
                     "account of who made it or why is the thing this layer replaces.")

    for batch in batches:
        if args.command == "restate":
            ledger.restate(batch["batch_id"], args.to, reason=args.reason, by=args.by)
        else:
            ledger.void(batch["batch_id"], reason=args.reason, by=args.by)
    print(f"\n  Done: {len(batches):,} line(s) appended. The parquet is untouched and "
          f"every original ledger line still stands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
