"""
Store maintenance: say what a batch is, or withdraw it.

    python -m inventory_planning.store show

    # 1. read a plan out                                    (nothing is changed)
    python -m inventory_planning.store restate \
        --loaded-before 2026-09-05T21:00 --layer-to prepared

    # 2. carry out that file                                (exactly the batches in it)
    python -m inventory_planning.store restate --layer-to prepared \
        --plan <store>/plans/restate-7a5d9b502bef.json --apply \
        --by jfanhon --reason "written by the pipeline before the writer changed"

**Nothing writes without `--apply`.** Every command prints the plan first, because the
counts are the diagnostic: expecting to restate a few hundred batches and being shown
twelve thousand means the selection is wrong, and that is worth learning before the
ledger has grown a line for each of them.

**And `--apply` over a selector needs `--plan`.** It used to select again from scratch,
which meant the batches it changed were whatever matched *then* rather than what was
counted and approved — and the store keeps moving, because shadow write lands batches
while the plan is being read. The plan file is the approved artefact and `--apply`
replays it; it does not re-select. Selecting by `--batch` alone is exempt, because a
list of ids names the batches rather than describing them and cannot match something
else tomorrow.

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
from .migration import MigrationPlan, OP_RESTATE, OP_VOID, PlanRefused


def _add_workspace_flags(parser, suppress: bool = False) -> None:
    """
    `--store` / `--tenant` on the parent *and* on each subcommand.

    argparse puts a parent's options before the subcommand only, so
    `feedback runs --tenant prod` was an error while `feedback --tenant prod runs` was
    not — a distinction nobody should have to know. Defined in both places so either
    order works, and the subcommand's copies default to SUPPRESS: an ordinary default
    would be written over the parent's value, silently sending the command to the
    default tenant's store.
    """
    default = argparse.SUPPRESS if suppress else None
    parser.add_argument("--store", default=default, help="Store root.")
    parser.add_argument("--tenant", default=default,
                        help="Which tenant's workspace. May go before or after the "
                             "subcommand.")


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
    _add_workspace_flags(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    _add_workspace_flags(sub.add_parser(
        "show", help="What the store holds, by document and layer"), suppress=True)

    for name, help_text in (("restate", "say what layer the selected batches hold"),
                            ("void", "withdraw the selected batches from every reading")):
        cmd = sub.add_parser(name, help=help_text)
        _add_workspace_flags(cmd, suppress=True)
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
        cmd.add_argument("--plan-out", default=None,
                         help="Where to write the plan. Defaults beside the store. The "
                              "file is what --plan carries out later.")
        cmd.add_argument("--plan", default=None,
                         help="Carry out a saved plan: exactly the batches in it, not "
                              "whatever the selector matches now.")
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
        workspace = Workspace.resolve(getattr(args, "tenant", None),
                                      store_root=getattr(args, "store", None))
    except BadTenant as exc:
        parser.error(str(exc))
    root, source = workspace.store_root, workspace.origins["store_root"]
    ledger = BatchLedger(root)

    if args.command == "show":
        return _show(ledger, root)

    op = OP_RESTATE if args.command == OP_RESTATE else OP_VOID
    verb = "restate" if op == OP_RESTATE else "void"
    selectors = (args.doc_type, args.source, args.layer, args.batch,
                 args.loaded_before, args.loaded_after)
    print(f"  {root}  (from {source})")

    # ── Carrying out a saved plan ────────────────────────────────────────────
    if args.plan:
        try:
            plan = MigrationPlan.load(args.plan)
        except PlanRefused as exc:
            parser.error(str(exc))
        # `void` has no --layer-to, so the attribute is absent on that subcommand
        # rather than None — reading it directly crashed `void --plan`.
        wanted_layer = getattr(args, "to", None)
        if plan.op != op or (op == OP_RESTATE and plan.to != wanted_layer):
            parser.error(
                f"that plan is a `{plan.op}"
                f"{' to ' + str(plan.to) if plan.to else ''}`, and this command is a "
                f"`{verb}{' to ' + str(wanted_layer) if wanted_layer else ''}`. A selection "
                f"approved for one operation is not approved for another.")
        print(plan.summary())

        drift = plan.drift(ledger)
        if drift:
            print("\n  Refusing — the store has moved since this plan was read:")
            for line in drift:
                print(f"    · {line}")
            print("\n  Nothing was written. Generate the plan again against the store "
                  "as it stands.")
            return 1
        if any(selectors):
            # Reported, not acted on: the plan's own list is what gets carried out.
            moved = plan.rescan(_select(ledger, args))
            if moved:
                print(f"\n  Note: {moved}")
        if not args.apply:
            print("\n  Plan only. Nothing was written. Add --apply, --by and --reason "
                  "to carry it out.")
            return 0
        _require_attribution(parser, args)
        return _carry_out(ledger, plan, args)

    # ── Reading a plan out of a selection ────────────────────────────────────
    if not any(selectors):
        parser.error("name at least one selector — --doc-type, --source, --layer, "
                     "--batch, --loaded-before or --loaded-after. A maintenance command "
                     "with no selector would apply to the whole store.")

    batches = _select(ledger, args)
    if op == OP_RESTATE:
        print(f"  Restating to `{args.to}`.")
    print(_describe(batches, verb))
    if not batches:
        return 0

    plan = MigrationPlan.build(op, batches, to=getattr(args, "to", None),
                               selector=_selector_text(args), store=str(root))

    # An id-only selection is the one case that cannot drift: it names the batches
    # rather than describing them, so re-selecting tomorrow returns the same set by
    # construction. Everything else is a predicate over a store that keeps moving,
    # and a predicate re-evaluated at apply time is exactly the defect the plan file
    # exists to close.
    by_id_only = bool(args.batch) and not any(
        (args.doc_type, args.source, args.layer, args.loaded_before, args.loaded_after))

    if args.apply and by_id_only:
        _require_attribution(parser, args)
        return _carry_out(ledger, plan, args)

    if args.apply:
        parser.error(
            "--apply over a selector needs --plan. Write the plan out first, read it, "
            "then carry out that file:\n"
            "    ... (no --apply)                  → writes the plan\n"
            "    ... --plan <file> --apply --by … --reason …\n"
            "Applying the selector directly re-runs it against a store that has moved "
            "since you read the counts, and the batches that joined in between were "
            "never in the plan you approved. Selecting by --batch alone is exempt: it "
            "names the batches rather than describing them.")

    # Beside the store, not in the working directory. A plan belongs to the store it
    # was read against — writing it wherever the shell happened to be left a file that
    # names batches from a store nobody can tell it apart from.
    out = Path(args.plan_out) if args.plan_out else (
        root / "plans" / f"{op}-{plan.basis[:12]}.json")
    plan.save(out)
    print(f"\n  Plan written to {out}")
    print(f"  Nothing was changed. Carry it out with:\n"
          f"    python -m inventory_planning.store {args.command} "
          f"{'--layer-to ' + str(plan.to) + ' ' if plan.to else ''}"
          f"--plan {out} --apply --by <you> --reason <why>")
    return 0


def _require_attribution(parser, args) -> None:
    if not args.reason.strip() or not args.by.strip():
        # The ledger refuses an unnamed change too, now. Kept here as well so the
        # message a person gets is about the flag they did not pass rather than about
        # a writer three frames down.
        parser.error("--apply needs --reason and --by. A change to the store with no "
                     "account of who made it or why is the thing this layer replaces.")


def _carry_out(ledger: BatchLedger, plan: MigrationPlan, args) -> int:
    """
    Exactly the batches in the plan, in the order the plan holds them.

    Never a fresh selection. The plan is the approved artefact and this replays it,
    which is what makes "approve the diff, not the wish" true of a store the way it is
    already true of the two file editors.
    """
    for batch_id in plan.batch_ids:
        if plan.op == OP_RESTATE:
            ledger.restate(batch_id, plan.to, reason=args.reason, by=args.by)
        else:
            ledger.void(batch_id, reason=args.reason, by=args.by)
    print(f"\n  Done: {len(plan.batch_ids):,} line(s) appended under plan "
          f"{plan.basis[:12]}. The parquet is untouched and every original ledger line "
          f"still stands.")
    return 0


def _selector_text(args) -> str:
    """The selector as typed, recorded on the plan so it can be read months later."""
    parts = []
    for flag, value in (("--doc-type", args.doc_type),
                        ("--loaded-before", args.loaded_before),
                        ("--loaded-after", args.loaded_after)):
        if value:
            parts.append(f"{flag} {value}")
    for flag, values in (("--source", args.source), ("--layer", args.layer),
                         ("--batch", args.batch)):
        for value in values:
            parts.append(f"{flag} {value}")
    return " ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
