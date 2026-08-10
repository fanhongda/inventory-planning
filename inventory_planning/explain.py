"""
Why did my file route where it did?

    python -m inventory_planning.explain "C:\\path\\to\\Regional PL30 Jul"
    python -m inventory_planning.explain sales.xlsx purchases.xlsx

Routing is deliberately automatic — the planner should not have to say which file is
which — and the cost of that is a failure mode with no obvious handle: a file lands on
the wrong contract, every number downstream is computed correctly from the wrong
premise, and the run reports `OK`. A misrouted PO history is the worst case, because
lead time then silently falls back to whatever else can supply it.

This prints the scoring the router actually did, per contract, and names the specific
reason each one lost:

  - which required fields could not be mapped, and to what the mapped ones went
  - which identifying fields are absent (a contract that finds none is disqualified)
  - whether the content discriminator could be evaluated at all
  - which source columns matched nothing

The last two lines are usually the answer. A document loses either because a column
it needs is named something the aliases do not know, or because the column exists but
holds something unexpected.

Fixing it is a data change, not a code change: add the alias to the contract, or
freeze an adapter for the export. Nothing here modifies anything.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pandas as pd

from .ingest.contract import DocContract
from .ingest.expressions import Expression, ExpressionError
from .ingest.intake import load_sheets
from .ingest.profiler import Profiler
from .ingest.registry import AdapterRegistry

READABLE = {".csv", ".xlsx", ".xls", ".xlsm"}


def explain_frame(df: pd.DataFrame, source_name: str, registry: AdapterRegistry = None) -> str:
    registry = registry or AdapterRegistry()
    profile = Profiler().profile(df, source_name=source_name)
    route = registry.route(df, source_name=source_name)

    out: List[str] = [
        "",
        "=" * 78,
        f"  {source_name}",
        "=" * 78,
        f"  {len(df):,} rows x {len(df.columns)} columns, shape={profile.shape}",
        "",
        "  Columns as read:",
    ]
    for col in df.columns:
        sample = next((str(v) for v in df[col].dropna().head(1)), "")
        out.append(f"    {str(col)[:38]:<40} e.g. {sample[:28]}")

    out += ["", f"  ROUTED TO: {route.doc_type}  ({route.confidence:.0%})",
            f"    {route.reason}", "", "  Per-contract scoring:"]

    contracts = registry.contracts.all()
    rows = []
    for doc_type, contract in sorted(contracts.items()):
        if doc_type == "demand_timeseries":
            continue
        hit = registry._assign_columns(profile, contract)
        hit_fields = set(hit)
        required = set(contract.required_fields)
        derivable = set(contract.derivable_fields)
        covered = {
            f for f in required
            if f in hit_fields or (f in derivable
                                   and registry._derivable_now(contract, f, hit_fields))
        }
        missing_identity = [g for g in contract.identifying_any if not (set(g) & hit_fields)]
        score = 0.0 if missing_identity else (
            0.75 * len(covered) / max(len(required), 1)
            + 0.25 * len(hit_fields) / max(len(contract.fields), 1)
        )
        rows.append((score, doc_type, contract, hit, required, covered, missing_identity))

    for score, doc_type, contract, hit, required, covered, missing_identity in sorted(
        rows, key=lambda r: -r[0]
    ):
        mark = "->" if doc_type == route.doc_type else "  "
        out.append(f"  {mark} {doc_type:<18} {score:>5.2f}   "
                   f"{len(covered)}/{len(required)} required, {len(hit)}/{len(contract.fields)} fields")
        for field in sorted(required):
            source_col = hit.get(field)
            if source_col:
                out.append(f"       required {field:<22} <- {source_col}")
            elif field in covered:
                out.append(f"       required {field:<22} (derived)")
            else:
                out.append(f"       required {field:<22} ** NOT MAPPED **")
        if missing_identity:
            out.append(f"       DISQUALIFIED — carries none of: {', '.join(missing_identity[0])}")
        out.append(f"       {_discriminator_note(contract, registry, profile, df)}")

    winner = contracts.get(route.doc_type)
    if winner is not None:
        mapped = registry._assign_columns(profile, winner)
        unmapped = [c for c in df.columns if c not in set(mapped.values())]
        out += ["", f"  Mapping under {route.doc_type}:"]
        for field, col in sorted(mapped.items()):
            out.append(f"    {field:<26} <- {col}")
        if unmapped:
            out += ["", "  Source columns that matched nothing "
                        "(add an alias to the contract if one of these matters):"]
            for col in unmapped:
                out.append(f"    {col}")
    return "\n".join(out)


def _discriminator_note(contract: DocContract, registry: AdapterRegistry,
                        profile, df: pd.DataFrame) -> str:
    """
    Whether the contract's content test could run, and what it said.

    Inapplicability is the interesting case and the easiest to miss: a discriminator
    referencing a column the file does not have is skipped entirely, so a document can
    lose a tie it would have won without anything saying so.
    """
    if not contract.discriminator:
        return "no content test"

    column_map = registry._assign_columns(profile, contract, include_empty=True)
    renamed = df.rename(columns={v: k for k, v in column_map.items()})
    available = set(column_map)
    renamed = registry._numeric_for_tests(renamed, contract, available)
    renamed, derived = registry._derive_for_discriminator(renamed, contract, available)
    available |= derived

    # Mirror the router exactly: every applicable test, strongest result stands.
    # Showing only the first one reported 52% beside a verdict reached on 85%, which
    # is worse than showing nothing — the tool exists to explain the decision.
    results, unusable = [], []
    for source in (contract.discriminators or [contract.discriminator]):
        expr = Expression(source)
        missing = expr.columns - available
        if missing:
            unusable.append(f"{source} (needs {sorted(missing)})")
            continue
        try:
            mask = expr.evaluate(renamed)
            results.append((float(pd.Series(mask).fillna(False).astype(bool).mean()), source))
        except (ExpressionError, TypeError, ValueError) as exc:
            unusable.append(f"{source} (errored: {exc})")

    if not results:
        return "content test NOT APPLICABLE — " + "; ".join(unusable)

    share, source = max(results)
    tail = f"; also tried {', '.join(unusable)}" if unusable else ""
    other = "".join(
        f", {s} for `{src}`" for s, src in sorted(results, reverse=True)[1:]
    )
    return f"content test `{source}` holds for {share:.0%} of rows{other}{tail}"


def explain_paths(paths: List[Path]) -> str:
    registry = AdapterRegistry()
    chunks = []
    for path in paths:
        try:
            sheets = load_sheets(path)
        except Exception as exc:
            chunks.append(f"\n  {path.name}: could not read — {exc}")
            continue
        for sheet_name, df in sheets:
            label = f"{path.name}" + (f" [{sheet_name}]" if sheet_name else "")
            chunks.append(explain_frame(df, label, registry))
    return "\n".join(chunks)


def main(argv: List[str] = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print(__doc__)
        return 2

    paths: List[Path] = []
    for arg in argv:
        p = Path(arg)
        if p.is_dir():
            paths += sorted(f for f in p.iterdir()
                            if f.is_file() and f.suffix.lower() in READABLE)
        elif p.is_file():
            paths.append(p)
        else:
            print(f"  not found: {arg}")

    if not paths:
        print("  nothing to explain")
        return 2

    print(explain_paths(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
