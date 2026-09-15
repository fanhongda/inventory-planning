"""
Score a plan against what happened.

    python -m inventory_planning.feedback runs
    python -m inventory_planning.feedback score --run 20260829_171813-6c1b09 \
        --sales <batch-id> --inventory <batch-id>

The half of the feedback loop that has never run. 2,430 decision snapshots had
accumulated and not one had ever been scored, because `FeedbackCollector` asked a person
to assemble two frames by hand a month later and point them at a snapshot path. The
store already holds both.

**Nothing is written back into the snapshot.** The actuals are read at scoring time and
the score is written as its own record, so the decision stays exactly as the run left
it. A decision record that can be edited afterwards cannot answer what was decided at
the time, which is the only question it is kept for.

**The batches are named, not searched for.** `FactQuery.current` refuses on sales history
— the stored batches carry no `so_line_number` and reducing on a partial key would
collapse rows that are genuinely different — and reading every batch would multiply-count
the same extract by the number of times it was imported. So a score names the extract it
was computed against, the way a run manifest names the facts it planned on, and `runs`
lists the candidates rather than picking one.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

SCORES_DIRNAME = "scores"


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


def _snapshots(history: Path) -> List[Path]:
    return sorted(history.glob("*/snapshot_*.json"))


def _run_id_of(path: Path) -> str:
    """`snapshot_<run_id>.json` — the run id is in the name, not in the body."""
    return path.name[len("snapshot_"):-len(".json")]


def _find(history: Path, run_id: str) -> Optional[Path]:
    for path in _snapshots(history):
        if _run_id_of(path) == run_id:
            return path
    return None


def _list_runs(history: Path, store, limit: int) -> int:
    from .actuals import ActualsUnavailable, candidates, scoring_period

    paths = _snapshots(history)
    if not paths:
        print(f"  No snapshots under {history}.")
        return 0

    print(f"  {len(paths):,} snapshot(s) under {history}. Newest {limit}:\n")
    for path in paths[-limit:][::-1]:
        body = json.loads(path.read_text(encoding="utf-8"))
        try:
            start, _ = scoring_period(body)
            period = f"{start:%Y-%m}"
        except ActualsUnavailable:
            period = "(no as_of)"
        scored = "scored" if _score_path(store.root, _run_id_of(path)).exists() else ""
        print(f"    {_run_id_of(path):<26} plans for {period}  "
              f"{body.get('sku_count', 0):>5} SKUs  {scored}")

    print("\n  Candidate extracts to score against, largest first — a store developed "
          "against\n  is mostly the same sample file re-imported, so size is the "
          "signal:\n")
    for doc_type in ("sales_history", "inventory"):
        print(f"    {doc_type}")
        for entry in candidates(store, doc_type, limit=4):
            print(f"      {entry['batch_id']:<26} {entry['rows'] or 0:>7,} rows  "
                  f"valid {entry['valid_time']}  {entry['source_name'] or ''}")
    return 0


def _score_path(store_root: Path, run_id: str) -> Path:
    return Path(store_root) / SCORES_DIRNAME / f"{run_id}.json"


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m inventory_planning.feedback",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_workspace_flags(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("runs", help="Snapshots, and the extracts to score against")
    _add_workspace_flags(listing, suppress=True)
    listing.add_argument("--limit", type=int, default=10)

    scoring = sub.add_parser("score", help="Score one run against an extract")
    _add_workspace_flags(scoring, suppress=True)
    scoring.add_argument("--run", required=True, help="Run id of the snapshot to score")
    scoring.add_argument("--sales", default=None,
                         help="sales_history batch supplying actual demand")
    scoring.add_argument("--inventory", default=None,
                         help="inventory batch supplying end-of-period on hand")
    scoring.add_argument("--write", action="store_true",
                         help="Write the score as its own record. The snapshot is "
                              "never touched either way.")

    args = parser.parse_args(argv)
    # SUPPRESS means the attribute is absent when the subcommand did not carry one.
    store_root = getattr(args, "store", None)
    tenant = getattr(args, "tenant", None)

    from ..store.fact_store import FactStore, history_root
    from ..workspace import BadTenant, Workspace

    try:
        workspace = Workspace.resolve(tenant, store_root=store_root)
    except BadTenant as exc:
        parser.error(str(exc))
    store = FactStore(workspace.store_root)
    history = history_root(workspace.store_root)
    print(f"  {workspace.store_root}  ({workspace.origins['store_root']})")

    if args.command == "runs":
        return _list_runs(history, store, args.limit)

    from .actuals import (
        ActualsUnavailable, actuals_for, candidates, scored_snapshot, scoring_period,
    )
    from .loss import LossCalculator

    path = _find(history, args.run)
    if path is None:
        parser.error(f"no snapshot for run {args.run!r} under {history}. "
                     f"`runs` lists what is there.")
    body = json.loads(path.read_text(encoding="utf-8"))

    if not args.sales:
        try:
            start, _ = scoring_period(body)
        except ActualsUnavailable as exc:
            parser.error(str(exc))
        print(f"\n  Run {args.run} plans for {start:%Y-%m}. Name the extract to score "
              f"it against:\n")
        for entry in candidates(store, "sales_history", covering=start, limit=5):
            print(f"    --sales {entry['batch_id']}   {entry['rows'] or 0:>7,} rows  "
                  f"valid {entry['valid_time']}  {entry['source_name'] or ''}")
        print("\n  Not chosen for you: which extract is the right one is a judgement "
              "about\n  what was pulled and when, and the batch id travels into the "
              "score so the\n  number can be reproduced.")
        return 2

    try:
        actuals, source = actuals_for(body, store, args.sales, args.inventory)
    except ActualsUnavailable as exc:
        print(f"\n  Cannot score: {exc}")
        return 1

    calculator = LossCalculator(path, snapshot=scored_snapshot(body, actuals))
    result = calculator.compute()

    record = {
        "run_id": args.run,
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "snapshot": str(path),
        "actuals_from": source.to_dict(),
        "result": result,
    }
    if args.write:
        out = _score_path(workspace.store_root, args.run)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
        print(f"\n  Score written to {out}")
        print("  The snapshot is unchanged — a decision record that could be edited "
              "afterwards\n  could not say what was decided at the time.")
    else:
        print("\n  Nothing written. Add --write to keep this score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
