"""
Two-chapter KPI report.

Chapter 1 attributes what already happened to the SKUs that caused it. Chapter 2
projects what is about to happen and what to do about it. The split matters because
the two halves have different audiences and different actions: the first is a
post-mortem someone has to answer for, the second is a work list.

Design notes that are not arbitrary:

  Status colours, not categorical, for service states.  On-time / late / short is an
  ordered severity scale, so it takes the reserved status palette. That palette is
  documented as sub-3:1 on a light surface for `warning` by design, with icon + label
  as the mitigation — so every segment here carries an icon and a visible direct
  label, never colour alone.

  "Past due, stock available" is deliberately off the severity ramp.  It is not a
  supply failure, so it does not get a failure colour. It gets its own hue and its own
  section, because its fix belongs to whoever owns the customer relationship.

  Self-contained.  Inline CSS/SVG only, no CDN, no external fonts, and both light and
  dark are authored rather than auto-flipped.
"""

from __future__ import annotations

import html
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ── Palette (validated; see module docstring) ────────────────────────────────

SERVICE_STATE_STYLE = {
    "on_time": ("var(--status-good)", "✓", "On time"),
    "shipped_late": ("var(--status-warning)", "▲", "Shipped late"),
    "open_past_due_short": ("var(--status-critical)", "✕", "Past due — no stock"),
    "open_past_due_available": ("var(--accent-violet)", "◆", "Past due — stock available"),
    "open_not_yet_due": ("var(--muted-fill)", "·", "Open, not yet due"),
    "unjudgeable": ("var(--muted-fill)", "?", "No request date"),
}

COMPONENT_STYLE = [
    ("cycle_value", "var(--series-1)", "Cycle"),
    ("safety_value", "var(--series-2)", "Safety"),
    ("pipeline_value", "var(--series-3)", "Pipeline (owned)"),
]

CSS = """
*, *::before, *::after { box-sizing: border-box; }
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --plane: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
  --accent-violet: #4a3aa7;
  --status-good: #0ca30c; --status-warning: #fab219;
  --status-serious: #ec835a; --status-critical: #d03b3b;
  --muted-fill: #c3c2b7;
  --good-ink: #006300;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --plane: #0d0d0d;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --ring: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
    --accent-violet: #9085e9; --muted-fill: #383835; --good-ink: #0ca30c;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --plane: #0d0d0d;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #898781;
  --grid: #2c2c2a; --baseline: #383835; --ring: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  --accent-violet: #9085e9; --muted-fill: #383835; --good-ink: #0ca30c;
}

.viz-root {
  font-family: system-ui, -apple-system, "Segoe UI", "PingFang SC",
               "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  background: var(--plane); color: var(--text-primary);
  margin: 0; padding: 32px 24px 64px; line-height: 1.5;
}
.wrap { max-width: 1100px; margin: 0 auto; }

h1 { font-size: 26px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 13px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
     color: var(--text-secondary); margin: 48px 0 4px; }
h3 { font-size: 17px; font-weight: 600; margin: 28px 0 10px; }
.sub { color: var(--text-secondary); font-size: 14px; margin: 0 0 8px; }
.chapter-rule { height: 2px; background: var(--baseline); border: 0; margin: 6px 0 20px; }

.card { background: var(--surface-1); border: 1px solid var(--ring); border-radius: 10px;
        padding: 18px 20px; margin: 14px 0; }

/* KPI row */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
.kpi { background: var(--surface-1); border: 1px solid var(--ring); border-radius: 10px; padding: 16px 18px; }
.kpi .label { font-size: 12px; color: var(--text-secondary); letter-spacing: 0.03em; }
.kpi .value { font-size: 32px; font-weight: 600; margin-top: 6px; letter-spacing: -0.02em; }
.kpi .note { font-size: 12px; color: var(--text-muted); margin-top: 6px; }
.kpi .note strong { color: var(--text-secondary); font-weight: 600; }

/* stacked bar */
.stack { display: flex; width: 100%; height: 34px; border-radius: 5px; overflow: hidden;
         background: var(--plane); }
.stack .seg { position: relative; }
.stack .seg + .seg { box-shadow: -2px 0 0 0 var(--surface-1); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 20px; margin-top: 14px; }
.legend .item { display: flex; align-items: baseline; gap: 7px; font-size: 13px;
                color: var(--text-secondary); }
.legend .swatch { width: 11px; height: 11px; border-radius: 3px; flex: none;
                  transform: translateY(1px); }
.legend .n { color: var(--text-primary); font-weight: 600;
             font-variant-numeric: tabular-nums; }

/* horizontal bars */
.bars { display: grid; grid-template-columns: minmax(90px, auto) 1fr minmax(88px, auto);
        gap: 5px 12px; align-items: center; }
.bars .name { font-size: 13px; color: var(--text-secondary);
              white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bars .track { background: var(--plane); border-radius: 4px; height: 17px; position: relative; }
.bars .fill { height: 100%; border-radius: 4px; min-width: 2px; }
.bars .val { font-size: 13px; text-align: right; font-variant-numeric: tabular-nums;
             color: var(--text-primary); }

/* tables */
table { width: 100%; border-collapse: collapse; font-size: 13px; }
.scroll { overflow-x: auto; }
th { text-align: right; font-weight: 600; color: var(--text-secondary); font-size: 12px;
     padding: 7px 10px; border-bottom: 1px solid var(--baseline); white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
td { padding: 7px 10px; border-bottom: 1px solid var(--grid); text-align: right;
     font-variant-numeric: tabular-nums; white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
td.sku { font-weight: 600; }
/* Long labels wrap instead of forcing the numeric columns off the right edge. */
th.left, td.left { text-align: left; }
td.wrap { white-space: normal; min-width: 190px; font-variant-numeric: normal; }

.chip { display: inline-flex; align-items: center; gap: 5px; padding: 1px 8px;
        border-radius: 99px; font-size: 12px; font-weight: 600; }
.chip.good { color: var(--good-ink); background: color-mix(in srgb, var(--status-good) 14%, transparent); }
.chip.warn { color: var(--text-primary); background: color-mix(in srgb, var(--status-warning) 26%, transparent); }
.chip.crit { color: var(--status-critical); background: color-mix(in srgb, var(--status-critical) 14%, transparent); }

.callout { border-left: 3px solid var(--accent-violet); padding: 2px 0 2px 14px;
           margin: 14px 0; color: var(--text-secondary); font-size: 14px; }
.callout strong { color: var(--text-primary); }
.callout.warn { border-left-color: var(--status-warning); }
.callout.crit { border-left-color: var(--status-critical); }
.empty { color: var(--text-muted); font-size: 14px; font-style: italic; padding: 6px 0; }
.foot { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--grid);
        font-size: 12px; color: var(--text-muted); }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _money(value: float, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"${value:,.{decimals}f}"


def _num(value: float, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:,.{decimals}f}"


def _pct(value: float, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{value:.{decimals}%}"


class KPIReport:
    """Renders the two-chapter HTML report."""

    def __init__(self, title: str = "Inventory & Service Review"):
        self.title = title

    # ── Entry point ──────────────────────────────────────────────────────────

    def render(
        self,
        service=None,
        should_be=None,
        ordering=None,
        forward=None,
        levers=None,
        frontier=None,
        as_of: date = None,
        output_path: Path = None,
    ) -> str:
        as_of = as_of or date.today()
        body = [
            self._header(as_of, should_be, service),
            self._kpi_row(service, should_be, forward),
            self._chapter_past(service, should_be, ordering),
            self._chapter_future(forward, frontier),
            self._footer(as_of),
        ]
        page = (
            f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{_esc(self.title)}</title><style>{CSS}</style></head>"
            f"<body class='viz-root'><div class='wrap'>{''.join(body)}</div></body></html>"
        )
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(page, encoding="utf-8")
        return page

    # ── Header & KPIs ────────────────────────────────────────────────────────

    def _header(self, as_of: date, should_be, service) -> str:
        skus = len(should_be.frame) if should_be is not None else 0
        lines = len(service.lines) if service is not None else 0
        bits = [f"as of {as_of}"]
        if skus:
            bits.append(f"{skus:,} SKUs")
        if lines:
            bits.append(f"{lines:,} order lines")
        return (f"<h1>{_esc(self.title)}</h1>"
                f"<p class='sub'>{' · '.join(bits)}</p>")

    def _kpi_row(self, service, should_be, forward) -> str:
        tiles = []

        if service is not None and len(service.lines):
            fair, harsh = service.otd_line_rate, service.otd_line_rate_harsh
            if not np.isfinite(fair):
                # A blank tile reads as "fine". Say what is missing instead.
                tiles.append(self._tile(
                    "On-time delivery", "not measured",
                    "needs sales history with <strong>ship_date</strong> and "
                    "<strong>customer_request_date</strong>"))
            else:
                note = ("of <strong>completed</strong> deliveries met the request date · "
                        f"{len(service.completed):,} lines")
                if np.isfinite(harsh) and abs(fair - harsh) > 0.005:
                    note += (f"<br><strong>{_pct(harsh)}</strong> if uncollected stock were "
                             f"counted as a miss — that gap is not a supply failure")
                tiles.append(self._tile("On-time delivery", _pct(fair), note))

        if should_be is not None:
            actual, target = should_be.total_actual_value, should_be.total_should_be_value
            ratio = actual / target if target else float("nan")
            tiles.append(self._tile(
                "Inventory vs policy", _money(actual),
                f"should-be <strong>{_money(target)}</strong> · "
                f"{_pct(ratio, 0) if np.isfinite(ratio) else '—'} of policy"))
            # Gross, never net: overstock on one SKU does not offset a shortage on
            # another, so the two are reported side by side.
            over = should_be.frame["gap_value"] > 0
            tiles.append(self._tile(
                "Excess above policy", _money(should_be.excess_value),
                f"tied up across <strong>{int(over.sum()):,}</strong> SKUs · "
                f"{_money(should_be.shortfall_value)} short on "
                f"<strong>{int((should_be.frame['gap_value'] < 0).sum()):,}</strong> others"))

        if forward is not None and len(forward.stockout):
            urgent = len(forward.urgent(30))
            tiles.append(self._tile(
                "Demand at risk", _money(forward.value_at_risk),
                f"<strong>{urgent}</strong> SKUs run out within 30 days"))

        if not tiles:
            return ""
        return f"<div class='kpis'>{''.join(tiles)}</div>"

    @staticmethod
    def _tile(label: str, value: str, note: str = "") -> str:
        return (f"<div class='kpi'><div class='label'>{_esc(label)}</div>"
                f"<div class='value'>{_esc(value)}</div>"
                f"<div class='note'>{note}</div></div>")

    # ── Chapter 1: what happened ─────────────────────────────────────────────

    def _chapter_past(self, service, should_be, ordering) -> str:
        parts = ["<h2>Chapter 1 · What happened</h2><hr class='chapter-rule'>"]
        parts.append(self._service_section(service))
        parts.append(self._inventory_section(should_be))
        parts.append(self._ordering_section(ordering))
        return "".join(parts)

    def _service_section(self, service) -> str:
        if service is None or not len(service.lines):
            return ("<h3>Service — on-time delivery</h3>"
                    "<div class='card'><p class='empty'>No order lines available. "
                    "Supply sales history and open sales orders to measure OTD.</p></div>")

        counts = service.state_counts()
        counts = counts[counts["lines"] > 0]
        total = counts["lines"].sum()

        segs, legend = [], []
        for _, row in counts.iterrows():
            colour, icon, label = SERVICE_STATE_STYLE.get(
                row["service_state"], ("var(--muted-fill)", "·", row["service_state"])
            )
            share = row["lines"] / total if total else 0
            segs.append(
                f"<div class='seg' style='width:{share * 100:.3f}%;background:{colour}' "
                f"title='{_esc(label)}: {int(row['lines']):,} lines'></div>"
            )
            # Icon + label + count on every segment: the status palette is sub-3:1 for
            # `warning` on a light surface by design, so colour never carries meaning
            # alone here.
            legend.append(
                f"<div class='item'><span class='swatch' style='background:{colour}'></span>"
                f"<span>{icon} {_esc(label)} "
                f"<span class='n'>{int(row['lines']):,}</span> "
                f"({share:.0%}) · {_money(row['value'])}</span></div>"
            )

        callouts = []
        fair, harsh = service.otd_line_rate, service.otd_line_rate_harsh
        if np.isfinite(fair) and np.isfinite(harsh) and abs(fair - harsh) > 0.005:
            callouts.append(
                f"<div class='callout'><strong>OTD reads {_pct(fair)}, not {_pct(harsh)}.</strong> "
                f"The difference is lines past their request date where the stock was on the "
                f"shelf and the customer had not collected it. Counting those as supply "
                f"failures blames planning for a collection problem — and hides the fact that "
                f"the stock is sitting there.</div>"
            )
        clean = service.otd_line_rate_clean
        if np.isfinite(clean) and np.isfinite(fair) and abs(clean - fair) > 0.005:
            callouts.append(
                f"<div class='callout warn'>On lines whose request date represents a real "
                f"requirement, OTD is <strong>{_pct(clean)}</strong> ({clean - fair:+.1%}). "
                f"The rest is order-entry practice, not supply performance.</div>"
            )
        if not service.quality.is_trustworthy:
            q = service.quality
            callouts.append(
                f"<div class='callout crit'>Only <strong>{_pct(q.clean_rate, 0)}</strong> of "
                f"lines carry a request date that reflects a real requirement "
                f"({q.missing:,} missing, {q.same_as_order_date:,} same as order date, "
                f"{q.before_order_date:,} already past due when raised). Fix request-date "
                f"discipline before setting a target against this number.</div>"
            )

        out = [
            "<h3>Service — on-time delivery</h3>",
            "<div class='card'>",
            f"<div class='stack'>{''.join(segs)}</div>",
            f"<div class='legend'>{''.join(legend)}</div>",
            "".join(callouts),
            "</div>",
        ]

        failures = service.failures_by_sku(top=10)
        out.append(self._table(
            "Which SKUs caused the OTD misses",
            failures,
            [("sku", "SKU", "sku"), ("failed_lines", "Failed lines", "int"),
             ("failed_value", "Value", "money"), ("failure_rate", "Failure rate", "pct"),
             ("avg_days_late", "Avg days late", "num1"),
             ("max_days_late", "Worst", "num0"), ("customers", "Customers", "int")],
            note="Ranked by value, not line count — a Pareto by lines over-weights cheap, "
                 "frequently-ordered items.",
            empty="No OTD failures in the period.",
        ))

        stuck = service.uncollected_by_sku(top=10)
        out.append(self._table(
            "Past due, stock available — customer has not collected",
            stuck,
            [("sku", "SKU", "sku"), ("stuck_lines", "Lines", "int"),
             ("stuck_qty", "Qty", "num0"), ("stuck_value", "Value", "money"),
             ("avg_days_overdue", "Avg days overdue", "num1"),
             ("max_days_overdue", "Worst", "num0"), ("customers", "Customers", "int")],
            note="Not a supply failure — the goods were there. It is an inventory-efficiency "
                 "problem: committed, aged and immobile stock. The owner is whoever holds the "
                 "customer relationship, not planning.",
            empty="Nothing past due with stock available.",
        ))
        return "".join(out)

    def _inventory_section(self, should_be) -> str:
        if should_be is None:
            return ""
        f = should_be.frame
        components = [(label, float(f[col].sum()), colour)
                      for col, colour, label in COMPONENT_STYLE if col in f.columns]
        total = sum(v for _, v, _ in components)
        actual = should_be.total_actual_value
        scale = max(total, actual) or 1

        def bar(label: str, parts: Sequence) -> str:
            subtotal = sum(v for _, v, _ in parts)
            if subtotal <= 0:
                return ""
            # Two nested proportions: each segment is a share of *its own bar*, and the
            # bar is a share of the common scale. Sizing segments against the scale
            # instead collapses them into slivers with an empty remainder.
            segs = "".join(
                f"<div class='seg' style='width:{(v / subtotal) * 100:.3f}%;background:{c}' "
                f"title='{_esc(n)}: {_money(v)}'></div>"
                for n, v, c in parts if v > 0
            )
            return (f"<div class='name'>{_esc(label)}</div>"
                    f"<div class='track' style='height:34px;background:transparent'>"
                    f"<div class='stack' style='width:{(subtotal / scale) * 100:.3f}%;"
                    f"background:transparent'>{segs}</div></div>"
                    f"<div class='val'>{_money(subtotal)}</div>")

        legend = "".join(
            f"<div class='item'><span class='swatch' style='background:{c}'></span>"
            f"<span>{_esc(n)} <span class='n'>{_money(v)}</span></span></div>"
            for n, v, c in components
        )

        rows = [
            bar("Should-be", components),
            bar("Actual", [("Actual held", actual, "var(--muted-fill)")]),
        ]

        gap = should_be.gap_value
        verdict = (
            f"<div class='callout {'crit' if gap > 0 else ''}'>"
            f"<strong>{_money(abs(gap))}</strong> "
            f"{'above' if gap > 0 else 'below'} policy. "
            f"Pipeline counts only buyer-owned goods in transit, decided per SKU by incoterm; "
            f"actual is measured on the same boundary, so the gap is comparable.</div>"
        )

        top_gap = (
            f.nlargest(10, "gap_value")[
                ["sku", "actual_qty", "should_be_qty", "gap_qty", "gap_value",
                 "coverage_ratio", "actual_dioh", "should_be_dioh"]
            ]
            if "gap_value" in f.columns else pd.DataFrame()
        )

        return "".join([
            "<h3>Inventory — where the capital is</h3>",
            "<div class='card'>",
            f"<div class='bars' style='grid-template-columns:minmax(80px,auto) 1fr minmax(96px,auto)'>"
            f"{''.join(rows)}</div>",
            f"<div class='legend'>{legend}</div>",
            verdict,
            "</div>",
            self._table(
                "Which SKUs drive the excess",
                top_gap,
                [("sku", "SKU", "sku"), ("actual_qty", "On hand", "num0"),
                 ("should_be_qty", "Should be", "num0"), ("gap_qty", "Excess qty", "num0"),
                 ("gap_value", "Excess value", "money"),
                 ("coverage_ratio", "× policy", "ratio"),
                 ("actual_dioh", "DIOH actual", "num0"),
                 ("should_be_dioh", "DIOH policy", "num0")],
                note="DIOH is days of inventory on hand at the current run rate. "
                     "'× policy' above 1.0 is stock beyond what the policy calls for.",
                empty="No SKUs above policy.",
            ),
        ])

    def _ordering_section(self, ordering) -> str:
        if ordering is None:
            return ""
        out = ["<h3>Ordering behaviour — what the buying pattern cost</h3>"]

        out.append(self._table(
            "Over-ordered — bought more than demand can absorb",
            ordering.over_ordered.head(10) if len(ordering.over_ordered) else pd.DataFrame(),
            [("sku", "SKU", "sku"), ("ordered_qty", "Ordered", "num0"),
             ("demand_in_window", "Demand", "num0"),
             ("dos_purchased", "Days of supply bought", "num0"),
             ("excess_value", "Excess value", "money")],
            note="Shows up twice: as stock that will sit for years, and as freight paid to "
                 "move it early.",
            empty="No over-ordering detected.",
        ))
        out.append(self._table(
            "Chronic air freight — planning gap paid for in freight",
            ordering.chronic_air.head(10) if len(ordering.chronic_air) else pd.DataFrame(),
            [("sku", "SKU", "sku"), ("air_orders", "Air orders", "int"),
             ("total_orders", "Total orders", "int"), ("air_share", "Air share", "pct"),
             ("air_spend", "Air spend", "money")],
            note="Air on a genuine emergency is insurance. Air every month is a planning "
                 "failure being paid for at the freight desk — and the fix is upstream of it.",
            empty="No chronic air freight.",
        ))
        out.append(self._table(
            "Erratic lot sizing — no consistent order policy",
            ordering.erratic.head(8) if len(ordering.erratic) else pd.DataFrame(),
            [("sku", "SKU", "sku"), ("order_count", "Orders", "int"),
             ("min_qty", "Min", "num0"), ("max_qty", "Max", "num0"),
             ("mean_qty", "Mean", "num0"), ("lot_cv", "CV", "num2")],
            note="Every reorder made as a fresh decision. This is where surprise cycle stock "
                 "and expediting come from.",
            empty="Lot sizing is consistent.",
        ))
        return "".join(out)

    # ── Chapter 2: what's coming ─────────────────────────────────────────────

    def _chapter_future(self, forward, frontier) -> str:
        parts = ["<h2>Chapter 2 · What is coming</h2><hr class='chapter-rule'>"]

        if forward is None:
            parts.append("<div class='card'><p class='empty'>No forward projection "
                         "available.</p></div>")
            return "".join(parts)

        parts.append(self._stockout_section(forward))
        parts.append(self._slow_burn_section(forward))
        if frontier is not None:
            parts.append(self._frontier_section(frontier))
        return "".join(parts)

    def _stockout_section(self, forward) -> str:
        risk = forward.stockout
        out = ["<h3>Stockout risk</h3>"]
        if not len(risk):
            out.append("<div class='card'><p class='empty'>No SKU projected to run out "
                       "within the horizon.</p></div>")
            return "".join(out)

        top = risk.head(12)
        peak = float(top["value_at_risk"].max()) or 1.0
        rows = []
        for _, r in top.iterrows():
            days = r["days_to_stockout"]
            # Urgency is the reader's question, so it drives the colour — with the
            # days printed beside it, never colour alone.
            colour = ("var(--status-critical)" if days <= 14
                      else "var(--status-warning)" if days <= 45
                      else "var(--series-1)")
            width = (r["value_at_risk"] / peak) * 100
            rows.append(
                f"<div class='name'>{_esc(r['sku'])}</div>"
                f"<div class='track'><div class='fill' style='width:{width:.2f}%;"
                f"background:{colour}' title='{_money(r['value_at_risk'])} at risk'></div></div>"
                f"<div class='val'>{_money(r['value_at_risk'])} · {days:,.0f}d</div>"
            )
        out.append(f"<div class='card'><div class='bars'>{''.join(rows)}</div>"
                   f"<div class='legend'>"
                   f"<div class='item'><span class='swatch' style='background:var(--status-critical)'></span>"
                   f"<span>✕ within 14 days</span></div>"
                   f"<div class='item'><span class='swatch' style='background:var(--status-warning)'></span>"
                   f"<span>▲ within 45 days</span></div>"
                   f"<div class='item'><span class='swatch' style='background:var(--series-1)'></span>"
                   f"<span>· later in horizon</span></div></div>"
                   f"<div class='callout'>Value at risk is the <strong>demand</strong> that would "
                   f"go unserved, not the value of the stock — a stockout costs sales, not "
                   f"inventory.</div></div>")

        display = risk.head(12).copy()
        display["cover"] = np.where(
            display["inbound_day"].isna(), "none",
            np.where(display["inbound_day"] > display["days_to_stockout"],
                     "too late", "covered"),
        )
        out.append(self._table(
            "Detail",
            display,
            [("sku", "SKU", "sku"), ("days_to_stockout", "Days to out", "num0"),
             ("on_hand", "On hand", "num0"), ("backlog", "Backlog", "num0"),
             ("inbound_qty", "Inbound", "num0"), ("inbound_day", "Arrives day", "num0"),
             ("cover", "PO cover", "chip"), ("value_at_risk", "Value at risk", "money")],
            note="'PO cover' is 'too late' when the inbound receipt lands after the shelf is "
                 "already empty — the PO exists but does not prevent the stockout.",
        ))
        return "".join(out)

    def _slow_burn_section(self, forward) -> str:
        slow = forward.slow_burn
        out = ["<h3>Slow burn — stock that will not move</h3>"]
        if not len(slow):
            out.append("<div class='card'><p class='empty'>Nothing with implausibly long "
                       "cover.</p></div>")
            return "".join(out)

        display = slow.head(12).copy()
        display["years"] = display["days_of_cover"] / 365.0
        display["still_buying"] = np.where(display["inbound_qty"] > 0, "inbound PO", "—")
        still = int((display["inbound_qty"] > 0).sum())

        callout = ""
        if still:
            callout = (
                f"<div class='callout crit'><strong>{still}</strong> of these still have "
                f"inbound purchase orders. They are the first candidates to push out or "
                f"cancel — buying more of what already will not move.</div>"
            )
        out.append(self._table(
            "",
            display,
            [("sku", "SKU", "sku"), ("available_now", "Available", "num0"),
             ("years", "Years of cover", "num1"),
             ("stock_value", "Capital tied up", "money"),
             ("inbound_qty", "Inbound qty", "num0"),
             ("still_buying", "Status", "chip")],
            note="Cover beyond a year at the current run rate. Infinite cover means no demand "
                 "at all — those need liquidation, not patience.",
        ))
        if callout:
            out.append(callout)
        return "".join(out)

    def _frontier_section(self, frontier) -> str:
        if frontier.gap <= 0:
            return ""
        rows = []
        remaining = frontier.current_value
        reached = False
        for i, action in enumerate(frontier.actions, 1):
            remaining -= action.value_released
            hit = ""
            if not reached and remaining <= frontier.target_value:
                hit, reached = " ← target met here", True
            cost = ("unpriced" if not action.cost_known
                    else _money(action.annual_cost) + "/yr" if action.annual_cost else "free")
            rows.append({
                "step": i, "action": action.label, "skus": len(action.skus),
                "released": action.value_released, "cost": cost,
                "risk": action.service_risk, "balance": _money(remaining) + hit,
            })
        table = pd.DataFrame(rows)

        out = ["<h3>Reaching the inventory target</h3>",
               f"<div class='card'><div class='callout'>Current <strong>"
               f"{_money(frontier.current_value)}</strong> → target <strong>"
               f"{_money(frontier.target_value)}</strong>"
               + (f" by {frontier.deadline}" if frontier.deadline else "")
               + f" · gap <strong>{_money(frontier.gap)}</strong>. Actions are ordered by "
                 f"pain, not by size — free and reversible first.</div></div>"]

        out.append(self._table(
            "", table,
            [("step", "#", "int"), ("action", "Action", "text"), ("skus", "SKUs", "int"),
             ("released", "Frees", "money"), ("cost", "Cost", "chip"),
             ("risk", "Service risk", "chip"), ("balance", "Running balance", "chip")],
        ))
        if not frontier.reachable:
            short = frontier.gap - frontier.total_available
            out.append(f"<div class='callout crit'>Target <strong>not reachable</strong> — "
                       f"{_money(short)} short after every action above. The target needs "
                       f"renegotiating, or demand has to fall.</div>")
        for action in frontier.too_slow:
            out.append(f"<div class='callout warn'>⏳ <strong>{_esc(action.label)}</strong> "
                       f"({_money(action.value_theoretical)}) takes ~{action.time_to_effect_days}d "
                       f"to affect stock — right move, wrong horizon. Start it; it counts next "
                       f"year.</div>")
        for action in frontier.excluded:
            out.append(f"<div class='callout'>✗ <strong>Not recommended: "
                       f"{_esc(action.label)}</strong> — {_esc(action.rationale)}</div>")
        return "".join(out)

    # ── Table helper ─────────────────────────────────────────────────────────

    def _table(self, title: str, df: pd.DataFrame, columns: List, note: str = "",
               empty: str = "Nothing to show.") -> str:
        head = f"<h3>{_esc(title)}</h3>" if title else ""
        if df is None or not len(df):
            return f"{head}<div class='card'><p class='empty'>{_esc(empty)}</p></div>"

        available = [(key, label, kind) for key, label, kind in columns if key in df.columns]
        # Header alignment must follow the cells it labels, or a left-aligned text
        # column sits under a right-aligned heading.
        ths = "".join(
            f"<th class='{'left' if kind in ('sku', 'text') else ''}'>{_esc(label)}</th>"
            for _, label, kind in available
        )
        body = []
        for _, row in df.iterrows():
            cells = []
            for key, _, kind in available:
                cells.append(self._cell(row[key], kind))
            body.append(f"<tr>{''.join(cells)}</tr>")

        note_html = f"<p class='sub' style='margin-top:10px'>{note}</p>" if note else ""
        return (f"{head}<div class='card'><div class='scroll'><table><thead><tr>{ths}</tr>"
                f"</thead><tbody>{''.join(body)}</tbody></table></div>{note_html}</div>")

    @staticmethod
    def _cell(value: Any, kind: str) -> str:
        if kind == "sku":
            return f"<td class='sku left'>{_esc(value)}</td>"
        if kind == "text":
            return f"<td class='left wrap'>{_esc(value)}</td>"
        if kind == "money":
            return f"<td>{_money(_f(value))}</td>"
        if kind == "pct":
            return f"<td>{_pct(_f(value), 0)}</td>"
        if kind == "ratio":
            v = _f(value)
            return f"<td>{'—' if not np.isfinite(v) else f'{v:.2f}×'}</td>"
        if kind == "int":
            return f"<td>{_num(_f(value), 0)}</td>"
        if kind.startswith("num"):
            return f"<td>{_num(_f(value), int(kind[-1]))}</td>"
        if kind == "chip":
            text = str(value)
            cls = ("crit" if text in ("high", "too late", "inbound PO")
                   else "warn" if text in ("low", "medium", "unpriced")
                   else "good" if text in ("none", "free", "covered") else "")
            return f"<td><span class='chip {cls}'>{_esc(text)}</span></td>"
        return f"<td>{_esc(value)}</td>"

    @staticmethod
    def _footer(as_of: date) -> str:
        return (f"<p class='foot'>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                f"as of {as_of} · Conventions and segment rules come from "
                f"config/planning_parameters.md; every figure here follows them.</p>")


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out
