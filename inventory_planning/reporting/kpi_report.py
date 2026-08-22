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
import math
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

# Short forms for the small-multiple headers. The full sentence belongs in the table
# and the section note; repeated once per card it becomes wallpaper.
_VERDICT_SHORT = {
    "controlled": "in control",
    "chasing": "chasing demand",
    "losing_control": "positive bias",
    "oscillating": "over-correcting",
    "under_ordered": "under-ordered",
    "unearned_frequency": "over-ordering",
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

/* Charts. Inline SVG only — the report is a single self-contained file that has to
   open from a network share with no assets and no script, so a plotting library is
   not available and would not be appropriate if it were. */
.chart { width: 100%; height: auto; display: block; overflow: visible; }
.chart text { font-family: inherit; }
.chart .tick { fill: var(--text-muted); font-size: 11px; }
.chart .axis { stroke: var(--grid); stroke-width: 1; }
.chart .base { stroke: var(--baseline); stroke-width: 1; }
.chart .lbl { fill: var(--text-secondary); font-size: 11px; font-weight: 600; }
/* The mark carries the colour; the ring is surface-coloured so a dot stays readable
   where it sits on the line or on a neighbour. */
.chart .dot { stroke: var(--surface-1); stroke-width: 2; }
.chart .dot-thin { fill: var(--surface-1); stroke-width: 2; }
.chart .vol { fill: var(--muted-fill); }
.chart-note { color: var(--text-muted); font-size: 12px; margin: 8px 0 0; }

/* Small multiples. One shape per SKU, same scale rules, read by comparison. */
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 20px 26px; }
/* A grid item's default min-width is its content, which keeps auto-fit from ever
   reaching two columns once a long header sits inside it. */
.mini { min-width: 0; }
.mini-head { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.mini-head .sku { font-weight: 600; font-size: 14px; }
.mini-head .mini-val { margin-left: auto; font-size: 13px; font-variant-numeric: tabular-nums;
                       color: var(--text-secondary); }
.mini-sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
ul.tight { margin: 0; padding-left: 18px; }
ul.tight li { margin: 2px 0; }
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
        cadence=None,
        forward=None,
        levers=None,
        frontier=None,
        fx=None,
        service_target: float = None,
        attributes=None,
        recommendations=None,
        open_po=None,
        suggestions=None,
        as_of: date = None,
        output_path: Path = None,
    ) -> str:
        as_of = as_of or date.today()
        body = [
            self._header(as_of, should_be, service),
            self._kpi_row(service, should_be, forward),
            self._currency_note(fx),
            self._chapter_past(service, should_be, ordering, cadence, service_target,
                               attributes),
            self._chapter_future(forward, frontier, recommendations, open_po,
                                 attributes, suggestions),
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

    def _chapter_past(self, service, should_be, ordering, cadence=None,
                      service_target=None, attributes=None) -> str:
        parts = ["<h2>Chapter 1 · What happened</h2><hr class='chapter-rule'>"]
        parts.append(self._service_section(service, attributes))
        parts.append(self._otd_trend_section(service, service_target))
        parts.append(self._inventory_section(should_be))
        parts.append(self._ordering_section(ordering))
        parts.append(self._cadence_section(cadence))
        return "".join(parts)

    def _service_section(self, service, attributes=None) -> str:
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
            f"Which SKUs caused the OTD misses{self._window_suffix(service)}",
            self._with_policy(failures, attributes),
            [("sku", "SKU", "sku"), ("stocking_policy", "Policy", "policy"),
             ("failed_lines", "Failed lines", "int"),
             ("failed_value", "Value", "money"), ("failure_rate", "Failure rate", "pct"),
             ("avg_days_late", "Avg days late", "num1"),
             ("max_days_late", "Worst", "num0"), ("customers", "Customers", "int")],
            note="Ranked by value, not line count — a Pareto by lines over-weights cheap, "
                 "frequently-ordered items. Policy is the ERP's own make-to-stock / "
                 "make-to-order flag: a late MTS line is a stocking failure, a late MTO "
                 "line is a lead-time one, and they have different fixes.",
            empty="No OTD failures in the period.",
        ))

        stuck = service.uncollected_by_sku(top=10)
        out.append(self._table(
            f"Past due, stock available — customer has not collected "
            f"(open position at {service.as_of})",
            self._with_policy(stuck, attributes),
            [("sku", "SKU", "sku"), ("stocking_policy", "Policy", "policy"),
             ("stuck_lines", "Lines", "int"),
             ("stuck_qty", "Qty", "num0"), ("stuck_value", "Value", "money"),
             ("max_days_overdue", "Days overdue", "num0"),
             ("customers", "Customers", "int")],
            note="A snapshot, not a history: these are the lines open and past due right "
                 "now, so a line collected last month correctly does not appear — what it "
                 "cost has already been paid and nothing can be done about it. "
                 "Not a supply failure either; the goods were there. It is committed, aged, "
                 "immobile stock, owned by whoever holds the customer relationship.",
            empty="Nothing past due with stock available.",
        ))
        return "".join(out)

    @staticmethod
    def _window_suffix(service) -> str:
        window = getattr(service, "measured_window", None)
        if not window:
            return ""
        first, last = window
        return f" — {first} to {last}" if first != last else f" — {first}"

    @staticmethod
    def _with_policy(frame: pd.DataFrame, attributes: pd.DataFrame = None) -> pd.DataFrame:
        """
        Attach the ERP stocking policy to a per-SKU table.

        Left unset rather than guessed where the master does not reach the SKU. MTO and
        MTS carry opposite expectations about whether stock should have been on the
        shelf, so defaulting one of them rewrites the finding rather than completing it.
        """
        if frame is None or not len(frame) or "sku" not in frame.columns:
            return frame
        if attributes is None or not len(attributes) or "stocking_policy" not in attributes.columns:
            return frame
        lookup = attributes.drop_duplicates("sku").set_index("sku")["stocking_policy"]
        return frame.assign(stocking_policy=frame["sku"].map(lookup))

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

    def _cadence_section(self, cadence) -> str:
        """
        The buying rhythm: did the cumulative PO-minus-sales come back to zero, and how
        many orders were spent getting it there.

        Sits after the ordering section on purpose. That one reports what individual
        orders looked like; this one reports what the sequence of them added up to, and
        the sequence is where a planner who scores well on lot-size consistency can
        still be a month behind demand all year.
        """
        if cadence is None or not len(getattr(cadence, "frame", [])):
            return ""

        counts = cadence.counts()
        chips = "".join(
            f"<span class='chip {cls}'>{_esc(label)} {counts.get(key, 0)}</span> "
            for key, label, cls in (
                ("controlled", "In control", "good"),
                ("chasing", "Chasing demand", "crit"),
                ("under_ordered", "Under-ordered", "crit"),
                ("oscillating", "Over-correcting", "crit"),
                ("losing_control", "Positive bias", "warn"),
                ("unearned_frequency", "Over-ordering", "warn"),
            ) if counts.get(key)
        )

        out = [
            "<h3>Replenishment cadence — did buying keep pace with selling</h3>",
            f"<p class='sub'>Monthly PO quantity minus sales quantity, accumulated across "
            f"{cadence.window_months} months. A curve that returns to zero means the two run "
            f"rates match — a stronger result than any single month's accuracy, because errors "
            f"that cancel are errors the inventory never had to carry. "
            f"<strong>{cadence.control_rate:.0%}</strong> of measured SKUs did that.</p>",
            f"<p class='sub'>{chips}</p>" if chips else "",
        ]

        out.append(self._table(
            "Out of balance — ranked by the money behind the imbalance",
            cadence.out_of_control.head(12),
            [("sku", "SKU", "sku"), ("verdict_note", "Pattern", "text"),
             ("deficit_months", "Months short", "int"),
             ("surplus_months", "Months long", "int"),
             ("closing_balance", "Closing balance", "num0"),
             ("imbalance_pct", "vs demand", "pct"),
             ("order_count", "Orders", "int"),
             ("expected_orders", "Cadence allows", "int"),
             ("exposure_value", "Exposure", "money")],
            note="Months short is time spent buying behind what was being sold — cover "
                 "consumed rather than replaced. Months long is the opposite, and a long "
                 "run of it means nothing is pulling the orders back to the run rate. "
                 "Exposure adds up what the surplus tied up, what the shortfall left "
                 "uncovered, and what the extra orders cost to place.",
            empty="Every SKU's buying tracked its selling.",
        ))

        out.append(self._cadence_patterns(cadence))

        unearned = cadence.unearned_frequency
        if len(unearned):
            out.append(self._table(
                "Ordering frequency not earned — high cadence on non-critical items",
                unearned.head(10),
                [("sku", "SKU", "sku"), ("abc_class", "Class", "text"),
                 ("order_count", "Orders placed", "int"),
                 ("expected_orders", "Cadence allows", "int"),
                 ("orders_vs_expected", "Ratio", "ratio"),
                 ("review_period_days", "Review (days)", "int"),
                 ("avoidable_order_cost", "Avoidable/yr", "money")],
                note="Ordering more often than the review period requires is worth paying "
                     "for on a critical part and nothing else. On these items it buys no "
                     "service back, and the fix is a longer review period rather than a "
                     "better forecast. "
                     f"Total: {_money(cadence.total_avoidable_order_cost)} a year.",
            ))
        return "".join(out)

    # ── Service over time ────────────────────────────────────────────────────

    def _otd_trend_section(self, service, target: float = None) -> str:
        """
        On-time delivery month by month.

        A single OTD figure answers "how did we do" and hides "when did it change" —
        and the second is the question that decides whether anything needs fixing.
        82% flat for two years and 82% because the last four months collapsed are the
        same number describing different situations, and only one of them is urgent.

        The volume strip underneath is not decoration. A month at 100% on four lines
        looks identical to a month at 100% on four hundred, and without the counts the
        eye reads the thin months as the good ones.
        """
        if service is None:
            return ""
        monthly = service.monthly_otd()
        if not len(monthly):
            # Say why, rather than printing an empty axis. A missing metric shown as a
            # blank chart reads as a metric of zero.
            return (
                "<h3>On-time delivery over time</h3>"
                "<div class='card'><p class='empty'>Not measurable from this extract — "
                "no shipped line carries both a ship date and a customer request date, "
                "so no delivery has an outcome to score. The open order book is still "
                "judged below.</p></div>"
            )

        measured = monthly[monthly["lines"] > 0]
        worst = measured.loc[measured["otd_line_rate"].idxmin()] if len(measured) else None
        latest = measured.iloc[-1] if len(measured) else None
        thin_months = int(monthly["thin"].sum())

        note = (
            f"Each month is the {int(monthly['lines'].sum()):,} lines the customer asked "
            f"for in it — shipped early, shipped late, or still open — and the rate is "
            f"the share that went out on or before the requested date. A line still "
            f"sitting there past its date counts against the month it was due, which is "
            f"why this reads lower than the headline rate above: that one is measured "
            f"over settled deliveries only. Nothing past the latest date in the data is "
            f"drawn — the order book runs months ahead, and a rate over deliveries still "
            f"to come is not a measurement."
        )
        if monthly["partial"].any():
            note += (" The final month is drawn from an extract that stops part way "
                     "through it, so lines are still to fall due in it.")
        if thin_months:
            note += (f" {thin_months} month(s) carry fewer than five lines and are drawn "
                     f"hollow; a rate over that few deliveries is not a measurement.")

        return (
            "<h3>On-time delivery over time</h3>"
            f"<div class='card'>{self._otd_chart(monthly, target)}"
            f"<p class='chart-note'>{_esc(note)}</p></div>"
        )

    def _otd_chart(self, monthly: pd.DataFrame, target: float = None) -> str:
        """Rate panel over a volume panel, sharing one x axis and one scale each."""
        W, PAD_L, PAD_R = 760, 42, 16
        RATE_TOP, RATE_H = 14, 150
        VOL_TOP, VOL_H = 196, 46
        H = VOL_TOP + VOL_H + 22

        n = len(monthly)
        inner = W - PAD_L - PAD_R
        # A single month has no width to spread over; centre it rather than divide by nought.
        step = inner / max(n - 1, 1)
        x_of = (lambda i: PAD_L + inner / 2) if n == 1 else (lambda i: PAD_L + i * step)

        # A rate that never leaves the 85–100 band is invisible on a 0–100 axis: the
        # whole story sits in the top eighth of the plot. The axis is focused on the
        # range in play, floor labelled, so the zoom is stated rather than implied.
        #
        # This is a line encoding a rate, not a bar encoding a magnitude — the length
        # from zero carries no meaning here, so there is nothing for a zero baseline to
        # protect. The floor never rises above 80%, so the zoom stays bounded.
        rates = pd.to_numeric(monthly["otd_line_rate"], errors="coerce").dropna()
        floor = 0.0
        if len(rates):
            floor = math.floor(max(0.0, float(rates.min()) - 0.05) * 20) / 20
            if target and 0 < target <= 1:
                floor = min(floor, math.floor(max(0.0, target - 0.05) * 20) / 20)
            floor = min(floor, 0.80)
        span = max(1.0 - floor, 0.05)
        y_of = lambda rate: RATE_TOP + RATE_H * (1.0 - (rate - floor) / span)

        parts: List[str] = [
            f"<svg class='chart' viewBox='0 0 {W} {H}' role='img' "
            f"aria-label='On-time delivery by month'>"
        ]

        # Gridlines and y ticks — hairline, solid, recessive.
        for k in range(5):
            value = floor + span * k / 4
            y = y_of(value)
            parts.append(f"<line class='axis' x1='{PAD_L}' y1='{y:.1f}' x2='{W - PAD_R}' y2='{y:.1f}'/>")
            parts.append(f"<text class='tick' x='{PAD_L - 8}' y='{y + 4:.1f}' "
                         f"text-anchor='end'>{value:.0%}</text>")

        if target and 0 < target <= 1:
            ty = y_of(target)
            parts.append(f"<line class='base' x1='{PAD_L}' y1='{ty:.1f}' "
                         f"x2='{W - PAD_R}' y2='{ty:.1f}'/>")
            # Anchored left: the right edge belongs to the newest month's own label,
            # and the two collided there.
            parts.append(f"<text class='lbl' x='{PAD_L + 4}' y='{ty - 6:.1f}' "
                         f"text-anchor='start'>target {target:.0%}</text>")

        # The rate line. Months with no deliveries break it rather than being drawn
        # through — a straight segment across a silent month invents data.
        runs, current = [], []
        for i, (_, row) in enumerate(monthly.iterrows()):
            if pd.isna(row["otd_line_rate"]):
                if len(current) > 1:
                    runs.append(current)
                current = []
            else:
                current.append((x_of(i), y_of(float(row["otd_line_rate"]))))
        if len(current) > 1:
            runs.append(current)
        for run in runs:
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in run)
            parts.append(f"<polyline points='{pts}' fill='none' stroke='var(--series-1)' "
                         f"stroke-width='2' stroke-linejoin='round' stroke-linecap='round'/>")

        # Volume panel, on its own scale. Never a second y-axis on the rate plot.
        vol_max = float(monthly["lines"].max()) or 1.0
        band = inner / max(n, 1)
        bar_w = min(band * 0.62, 24)
        for i, (_, row) in enumerate(monthly.iterrows()):
            lines = float(row["lines"])
            if lines <= 0:
                continue
            h = max(VOL_H * (lines / vol_max), 1.5)
            parts.append(
                f"<rect class='vol' x='{x_of(i) - bar_w / 2:.1f}' y='{VOL_TOP + VOL_H - h:.1f}' "
                f"width='{bar_w:.1f}' height='{h:.1f}' rx='2'>"
                f"<title>{_esc(str(row['period']))} — {lines:,.0f} lines settled</title></rect>"
            )
        parts.append(f"<line class='base' x1='{PAD_L}' y1='{VOL_TOP + VOL_H}' "
                     f"x2='{W - PAD_R}' y2='{VOL_TOP + VOL_H}'/>")
        parts.append(f"<text class='tick' x='{PAD_L - 8}' y='{VOL_TOP + 12}' "
                     f"text-anchor='end'>{vol_max:,.0f}</text>")
        parts.append(f"<text class='tick' x='{PAD_L - 8}' y='{VOL_TOP + VOL_H + 4}' "
                     f"text-anchor='end'>0</text>")
        parts.append(f"<text class='lbl' x='{PAD_L}' y='{VOL_TOP - 8}'>lines settled</text>")

        # Markers last so they sit above the line, each with its own tooltip.
        for i, (_, row) in enumerate(monthly.iterrows()):
            if pd.isna(row["otd_line_rate"]):
                continue
            rate = float(row["otd_line_rate"])
            cls = "dot-thin" if bool(row["thin"]) else "dot"
            fill = "var(--surface-1)" if bool(row["thin"]) else "var(--series-1)"
            parts.append(
                f"<circle class='{cls}' cx='{x_of(i):.1f}' cy='{y_of(rate):.1f}' r='4.5' "
                f"fill='{fill}' stroke='{'var(--series-1)' if row['thin'] else 'var(--surface-1)'}'>"
                f"<title>{_esc(str(row['period']))} — {rate:.0%} on time "
                f"({row['on_time_lines']:,.0f} of {row['lines']:,.0f} lines)</title></circle>"
            )

        # Labelled selectively: the newest month and the worst one, never every point.
        labelled = set()
        measured = monthly[monthly["lines"] > 0]
        for idx, anchor in ((measured.index[-1], "end"), (measured["otd_line_rate"].idxmin(), "middle")):
            if idx in labelled:
                continue
            labelled.add(idx)
            i = monthly.index.get_loc(idx)
            rate = float(monthly.loc[idx, "otd_line_rate"])
            parts.append(f"<text class='lbl' x='{x_of(i):.1f}' y='{y_of(rate) - 10:.1f}' "
                         f"text-anchor='{anchor}'>{rate:.0%}</text>")

        # x labels thinned so they never collide.
        every = max(1, n // 8)
        for i, (_, row) in enumerate(monthly.iterrows()):
            if i % every and i != n - 1:
                continue
            parts.append(f"<text class='tick' x='{x_of(i):.1f}' y='{H - 4}' "
                         f"text-anchor='middle'>{_esc(str(row['period']))}</text>")

        parts.append("</svg>")
        return "".join(parts)

    # ── Cadence patterns ─────────────────────────────────────────────────────

    def _cadence_patterns(self, cadence, top: int = 6) -> str:
        """
        The curve each verdict was actually read from, for the SKUs with most at stake.

        Every column in the table above is a summary of this shape, and the shape says
        it faster: six flat months below the line then a vertical correction is one
        glance, where "deficit_months 6, longest_surplus_run 3, closing +775,728" is
        three numbers the reader has to reassemble into the same picture.

        Diverging around zero because that is what the data is — below the line is
        cover being consumed, above it is cash being tied up, and the midpoint is not a
        middling amount of anything, it is the target.
        """
        if getattr(cadence, "cumulative", None) is None or not len(cadence.frame):
            return ""
        worst = cadence.out_of_control.head(top)
        if not len(worst):
            return ""

        cards = []
        for n, (_, row) in enumerate(worst.iterrows()):
            curve = cadence.curve(row["sku"])
            if curve is None:
                continue
            cards.append(self._cadence_card(curve, row, n))
        if not cards:
            return ""

        return (
            "<h3>Replenishment pattern — the shape behind each verdict</h3>"
            f"<p class='sub'>The {len(cards)} SKUs with most money at stake. The line is "
            f"cumulative PO quantity minus sales quantity; the ticks underneath mark the "
            f"months an order was actually placed.</p>"
            f"<div class='card'><div class='grid2'>{''.join(cards)}</div>"
            "<p class='chart-note'>Below the line is cover being consumed rather than "
            "replaced. Above it is cash tied up. Zero is not a middling amount of "
            "anything — it is the target, and a curve that returns to it means the two "
            "run rates matched.</p></div>"
        )

    def _cadence_card(self, curve: pd.DataFrame, row: pd.Series, n: int) -> str:
        W, H = 380, 116
        PAD_L, PAD_R, TOP, PLOT_H = 8, 8, 22, 62
        TICK_Y = TOP + PLOT_H + 10

        cum = pd.to_numeric(curve["cumulative"], errors="coerce").fillna(0.0).to_numpy()
        n_pts = len(cum)
        inner = W - PAD_L - PAD_R
        step = inner / max(n_pts - 1, 1)
        x_of = lambda i: PAD_L + i * step

        # Symmetric around zero so the baseline sits mid-plot and the two directions
        # are visually comparable — an asymmetric scale makes a small surplus look like
        # a large one purely because the deficit was bigger.
        span = max(abs(cum.min()), abs(cum.max()), float(row.get("tolerance_qty") or 0), 1.0)
        y_of = lambda v: TOP + PLOT_H / 2 - (v / span) * (PLOT_H / 2)
        zero_y = y_of(0.0)

        uid = f"cad{n}"
        pts = [(x_of(i), y_of(v)) for i, v in enumerate(cum)]
        area = (f"{PAD_L},{zero_y:.1f} "
                + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
                + f" {x_of(n_pts - 1):.1f},{zero_y:.1f}")

        parts = [
            f"<svg class='chart' viewBox='0 0 {W} {H}' role='img' "
            f"aria-label='Cumulative balance for {_esc(row['sku'])}'>",
            "<defs>",
            f"<clipPath id='{uid}up'><rect x='0' y='{TOP}' width='{W}' "
            f"height='{max(zero_y - TOP, 0):.1f}'/></clipPath>",
            f"<clipPath id='{uid}dn'><rect x='0' y='{zero_y:.1f}' width='{W}' "
            f"height='{max(TOP + PLOT_H - zero_y, 0):.1f}'/></clipPath>",
            "</defs>",
        ]

        # The band inside which a periodic review cannot help but wander.
        tol = float(row.get("tolerance_qty") or 0)
        if tol > 0:
            parts.append(
                f"<rect x='{PAD_L}' y='{y_of(tol):.1f}' width='{inner:.1f}' "
                f"height='{abs(y_of(-tol) - y_of(tol)):.1f}' fill='var(--muted-fill)' "
                f"opacity='0.18'/>"
            )

        # Same polygon twice, clipped either side of zero: warm above, cool below.
        for clip, colour in ((f"{uid}up", "var(--series-2)"), (f"{uid}dn", "var(--series-1)")):
            parts.append(f"<polygon points='{area}' fill='{colour}' opacity='0.16' "
                         f"clip-path='url(#{clip})'/>")
        parts.append(f"<line class='base' x1='{PAD_L}' y1='{zero_y:.1f}' "
                     f"x2='{W - PAD_R}' y2='{zero_y:.1f}'/>")
        for clip, colour in ((f"{uid}up", "var(--series-2)"), (f"{uid}dn", "var(--series-1)")):
            line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f"<polyline points='{line}' fill='none' stroke='{colour}' "
                         f"stroke-width='2' stroke-linejoin='round' stroke-linecap='round' "
                         f"clip-path='url(#{clip})'/>")

        # When an order was placed — the cadence itself, on the time axis rather than
        # on a second scale.
        po = pd.to_numeric(curve["po_qty"], errors="coerce").fillna(0.0).to_numpy()
        for i, (qty, period) in enumerate(zip(po, curve["period"])):
            if qty <= 0:
                continue
            parts.append(
                f"<line x1='{x_of(i):.1f}' y1='{TICK_Y - 4}' x2='{x_of(i):.1f}' "
                f"y2='{TICK_Y + 4}' stroke='var(--text-muted)' stroke-width='2' "
                f"stroke-linecap='round'><title>{_esc(str(period))} — ordered "
                f"{qty:,.0f}</title></line>"
            )

        first, last = curve["period"].iloc[0], curve["period"].iloc[-1]
        parts.append(f"<text class='tick' x='{PAD_L}' y='{H - 3}'>{_esc(str(first))}</text>")
        parts.append(f"<text class='tick' x='{W - PAD_R}' y='{H - 3}' "
                     f"text-anchor='end'>{_esc(str(last))}</text>")

        # The endpoint labelled, never every point. Clamped inside the plot: a curve
        # ending near the floor put its label on the same baseline as the period
        # labels, and the two ran together.
        end_v = cum[-1]
        label_y = y_of(end_v) + (-8 if end_v >= 0 else 14)
        label_y = min(max(label_y, TOP + 10), TOP + PLOT_H - 2)
        parts.append(
            f"<text class='lbl' x='{W - PAD_R}' y='{label_y:.1f}' "
            f"text-anchor='end'>{end_v:+,.0f}</text>"
        )
        parts.append("</svg>")

        verdict_cls = {"controlled": "good", "losing_control": "warn",
                       "unearned_frequency": "warn"}.get(row["verdict"], "crit")
        # The order count lives in the header, not in the plot. Inside the SVG it sat
        # on the same baseline as the first period label and the two overlapped.
        orders = (f"{int(row['order_count'])} orders placed · cadence allows "
                  f"{int(row['expected_orders'])}")
        return (
            f"<div class='mini'><div class='mini-head'>"
            f"<span class='sku'>{_esc(row['sku'])}</span>"
            f"<span class='chip {verdict_cls}'>{_esc(_VERDICT_SHORT.get(row['verdict'], row['verdict']))}</span>"
            f"<span class='mini-val'>{_money(row['exposure_value'])}</span></div>"
            f"<div class='mini-sub'>{_esc(orders)}</div>"
            f"{''.join(parts)}</div>"
        )

    # ── Currency ─────────────────────────────────────────────────────────────

    def _currency_note(self, fx) -> str:
        """
        Says which currency the report is in, and names any money it could not restate.

        Shown next to the KPI tiles rather than buried, because every figure above and
        below it is a sum over lines that were raised in several currencies. A reader
        who does not know that a conversion happened cannot tell a genuine total from a
        mixture of scales, and both print with the same symbol.
        """
        if fx is None or not getattr(fx, "reports", None):
            return ""
        if not fx.multi_currency and fx.rates_configured:
            return ""

        parts = [
            f"<div class='callout'>Money is reported in <strong>{_esc(fx.reporting_currency)}</strong>. "
        ]
        converted = [r for r in fx.reports.values() if r.is_multi_currency]
        if converted:
            detail = "; ".join(
                f"{r.doc_type}: {r.rows_converted:,} of {r.rows_total:,} lines restated"
                for r in converted
            )
            parts.append(f"Source lines arrive in several currencies and were converted "
                         f"before any total was taken ({_esc(detail)}).")

        rejected = sum(r.rows_rate_rejected for r in fx.reports.values())
        if rejected:
            parts.append(
                f" {rejected:,} lines carried an exchange-rate column that was not a rate "
                f"— a placeholder 1.0 on a foreign line, or an inverted quote — and were "
                f"converted at the configured planning rate instead of the booked one."
            )

        gaps = fx.gaps
        if gaps:
            listed = ", ".join(f"{code} ({n:,} lines)" for code, n in sorted(gaps.items()))
            parts.append(
                f"</div><div class='callout crit'><strong>No exchange rate for "
                f"{_esc(listed)}.</strong> Those lines carry no value in "
                f"{_esc(fx.reporting_currency)} — they are excluded from every money "
                f"figure here rather than counted at face value, so totals below are "
                f"understated by whatever they were worth. Add the rates to "
                f"<code>config/fx_rates.json</code> and re-run."
            )
        elif not fx.rates_configured:
            parts.append(
                f"</div><div class='callout warn'><strong>No rate table configured.</strong> "
                f"Any line not already in {_esc(fx.reporting_currency)} is unvalued."
            )
        parts.append("</div>")
        return "".join(parts)

    # ── Chapter 2: what's coming ─────────────────────────────────────────────

    def _chapter_future(self, forward, frontier, recommendations=None,
                        open_po=None, attributes=None, suggestions=None) -> str:
        parts = ["<h2>Chapter 2 · What is coming</h2><hr class='chapter-rule'>"]

        # Ahead of the forward projection, and outside the branch that gives up
        # without one. Both come from what is on the shelf against what is on order,
        # so a run that could not project anything forward still has them to report —
        # and the supply gap is the most urgent thing in the run either way.
        parts.append(self._supply_gaps(recommendations, attributes))

        if forward is None:
            parts.append("<div class='card'><p class='empty'>No forward projection "
                         "available.</p></div>")
            parts.append(self._policy_disagreements(attributes))
            return "".join(parts)

        parts.append(self._stockout_section(forward))
        parts.append(self._slow_burn_section(forward))
        parts.append(self._actions_section(recommendations, open_po, forward, attributes))
        parts.append(self._policy_disagreements(attributes))
        parts.append(self._suggestions_section(suggestions, attributes))
        if frontier is not None:
            parts.append(self._frontier_section(frontier))
        return "".join(parts)

    # ── Parameter suggestions ────────────────────────────────────────────────

    def _policy_disagreements(self, attributes) -> str:
        """
        Items the ERP plans one way and the demand argues for the other.

        Both directions are shown, and the MTO-with-recurring-demand half is put first
        because it is the one with a customer on the end of it: the item is bought to
        order, so every request waits a full lead time that stock would have absorbed.
        The other half is money — held as make-to-stock against demand that turns up
        twice a year, which is stock sitting.

        Neither is applied. A policy is a decision someone made, often years before
        anyone planned on the item and for reasons no extract records — a sole-supplier
        risk, a shelf life, a customer contract. What the run can say is what the demand
        did, and where that and the policy point different ways.
        """
        if attributes is None or not len(attributes):
            return ""
        need = {"stocking_policy", "suggested_stocking_policy"}
        if not need <= set(attributes.columns):
            return ""

        work = attributes.dropna(subset=list(need)).copy()
        work = work[work["stocking_policy"] != work["suggested_stocking_policy"]]
        if not len(work):
            return ("<h3>Stocking policy to review</h3><div class='card'><p class='empty'>"
                    "Every item's policy agrees with what its demand did.</p></div>")

        # `annual_value` is already assembled upstream, against the demand basis the
        # rest of the report ranks on. Recomputing it here from unit cost would be a
        # second definition of the same number — and the first attempt at it silently
        # overwrote the real column with NaN, which showed up as a table of dashes.
        # Absent value is not zero; those rows sort last rather than being dropped.
        if "annual_value" not in work.columns:
            work["annual_value"] = np.nan
        work["annual_value"] = pd.to_numeric(work["annual_value"], errors="coerce")

        out = [f"<h3>Stocking policy to review — {len(work):,} items</h3>"]
        for held, suggested, heading, why in (
            ("MTO", "MTS", "Bought to order, but the demand recurs",
             "Every order waits a full lead time. These are the candidates to stock."),
            ("MTS", "MTO", "Held as stock, but the demand is sporadic",
             "Stock is sitting against demand that rarely comes. These are candidates "
             "to stop holding."),
        ):
            rows = work[(work["stocking_policy"] == held)
                        & (work["suggested_stocking_policy"] == suggested)]
            if not len(rows):
                continue
            rows = rows.sort_values("annual_value", ascending=False, na_position="last")
            value = rows["annual_value"].sum(skipna=True)
            sub = (f"<p class='sub'><strong>{len(rows):,}</strong> items — {why}"
                   + (f" Worth {_money(value)} a year." if value > 0 else "")
                   + "</p>")
            out.append(f"<h4>{_esc(heading)}</h4>{sub}")
            out.append(self._table(
                "", rows.head(12),
                [("sku", "SKU", "sku"), ("abc_class", "ABC", "text"),
                 ("stocking_policy", "In force", "text"),
                 ("suggested_stocking_policy", "Demand says", "text"),
                 ("policy_basis", "Why", "text"),
                 ("annual_value", "Annual value", "money")],
                note="Ranked by the annual value of the demand, so the list starts "
                     "where a change is worth the argument. Nothing is applied.",
            ))
        return "".join(out)

    def _suggestions_section(self, suggestions, attributes) -> str:
        """
        What the data supports, next to what is in force — never applied, only shown.

        Kept explicitly as a suggestion because the pipeline cannot see the reasons a
        parameter was set: a review period stretched to match a supplier's shipping
        window, a service level raised after a customer escalation. The arithmetic here
        knows none of that, so it proposes and a planner disposes.

        The safety-stock delta is the number to read first. It is capital, and its sign
        says which direction the whole catalogue is mis-set in.
        """
        if suggestions is None or not len(getattr(suggestions, "frame", [])):
            return ""
        changed = suggestions.changed
        if not len(changed):
            return ("<h3>Parameter suggestions</h3><div class='card'><p class='empty'>"
                    "Every parameter in force already matches what the data supports."
                    "</p></div>")

        delta = suggestions.safety_stock_delta_value
        direction = "more than" if delta > 0 else "less than"
        head = (
            f"<h3>Parameter suggestions — what the data supports</h3>"
            f"<p class='sub'><strong>{len(changed):,}</strong> of "
            f"{len(suggestions.frame):,} SKUs would change. Safety stock under the "
            f"suggested parameters is <strong>{_money(abs(delta))}</strong> {direction} "
            f"the policy in force. Nothing here is applied — these are proposals to "
            f"review, because the reason a parameter was set by hand is not visible in "
            f"the data.</p>"
        )

        work = self._with_policy(changed.copy(), attributes)
        if "ss_delta_value" in work.columns:
            work = work.reindex(
                work["ss_delta_value"].abs().sort_values(ascending=False).index
            )
        table = self._table(
            "",
            work.head(12),
            [("sku", "SKU", "sku"), ("stocking_policy", "Policy", "policy"),
             ("abc_class", "ABC", "text"),
             ("review_period_days", "Review now", "num0"),
             ("suggested_review_period_days", "Suggested", "num0"),
             ("service_level", "Service now", "pct"),
             ("suggested_service_level", "Suggested", "pct"),
             ("ss_delta_qty", "SS change", "num0"),
             ("ss_delta_value", "Capital", "money"),
             ("changes", "What changes", "text")],
            note="Ranked by the capital the safety-stock change moves. A positive "
                 "figure is stock the measured lead time and forecast error say is "
                 "missing; a negative one is capital the policy is holding without "
                 "cause.",
        )

        fired = [(r, suggestions.hits.get(r.rule_id, 0)) for r in suggestions.rules]
        fired = [(r, n) for r, n in fired if n]
        rules = ""
        if fired:
            rows = "".join(
                f"<tr><td class='left'>{_esc(r.rule_id)}</td>"
                f"<td class='left wrap'>{_esc(getattr(r, 'rationale', '') or getattr(r, 'name', ''))}</td>"
                f"<td>{n:,}</td></tr>"
                for r, n in sorted(fired, key=lambda x: -x[1])
            )
            rules = (
                "<div class='card'><div class='scroll'><table><thead><tr>"
                "<th class='left'>Rule</th><th class='left'>Why</th><th>SKUs</th>"
                f"</tr></thead><tbody>{rows}</tbody></table></div>"
                "<p class='sub' style='margin-top:10px'>Every rule that fired, and how "
                "many SKUs it caught. A rule matching almost everything is usually a "
                "scope that is too broad rather than a finding.</p></div>"
            )

        notes = ""
        if getattr(suggestions, "notes", None):
            items = "".join(f"<li>{_esc(n)}</li>" for n in suggestions.notes)
            notes = f"<div class='callout warn'><ul class='tight'>{items}</ul></div>"

        return head + table + rules + notes

    # ── The work list ────────────────────────────────────────────────────────

    def _actions_section(self, recommendations, open_po, forward, attributes) -> str:
        """
        The head of the work list: which materials to change, and which open POs.

        Everything above this point is diagnosis. This is the part a planner can act on
        before lunch, so it is ranked by money and cut to a head rather than printed in
        full — a list of nine hundred items is a list nobody starts.

        The PO half is derived from the forward risk rather than from a second
        recommendation pass. A separate engine would eventually disagree with the
        stockout and slow-burn sections sitting directly above it, and a report that
        contradicts itself two sections apart is worse than one that says less.
        """
        out = [self._materials_to_adjust(recommendations, attributes)]
        out.append(self._pos_to_adjust(open_po, forward, attributes))
        rendered = [p for p in out if p]
        if not rendered:
            return ""
        return "<h3>What to change — the head of the list</h3>" + "".join(rendered)

    def _supply_gaps(self, recommendations, attributes) -> str:
        """
        Items whose shelf runs dry before the next delivery lands. First, and loudly.

        These do not survive the work list below it, which ranks by the money an action
        moves — and an expedite moves none. The order is already placed; what is missing
        is a date, so `suggested_po_qty` is zero, the row values at zero, and all of them
        sort under everything else and fall off the end of the table. The most urgent
        thing in the run was the one thing invisible in it.

        Ranked by how long the shelf is bare rather than by value, because this is a
        service question before it is a money one: the customer waiting on an empty
        shelf does not care what the part costs. Value is shown, not sorted on.

        The ask is not a purchase order. It is a ship date confirmed with the supplier,
        chased daily, and a price for the air freight if the date will not move.
        """
        if recommendations is None or not len(recommendations):
            return ""
        if "supply_gap" not in recommendations.columns:
            return ""
        gaps = recommendations[recommendations["supply_gap"] == True].copy()
        if not len(gaps):
            return ""

        if attributes is not None and "annual_value" in getattr(attributes, "columns", []):
            value = attributes.drop_duplicates("sku").set_index("sku")["annual_value"]
            gaps["annual_value"] = gaps["sku"].map(value)
        gaps = self._with_policy(gaps, attributes)
        gaps = gaps.sort_values("supply_gap_days", ascending=False)

        late = gaps[gaps.get("inbound_past_due_qty", 0) > 0]
        head = (
            f"<h3>Supply gap — {len(gaps):,} items run dry before the next delivery</h3>"
            f"<p class='sub'>The shelf empties before anything arrives"
            + (f", and on <strong>{len(late):,}</strong> of them supply is already past "
               f"its committed date" if len(late) else "")
            + ". The order exists; what is missing is a delivery date. Confirm the ship "
              "date with the supplier daily and price the air freight — a purchase "
              "order is not the answer to any of these.</p>"
        )
        return head + self._table(
            "", gaps.head(12),
            [("sku", "SKU", "sku"), ("stocking_policy", "Policy", "policy"),
             ("on_hand_cover_days", "Cover left (d)", "num0"),
             ("supply_gap_days", "Days bare", "num0"),
             ("days_to_next_arrival", "Next arrival (d)", "num0"),
             ("inbound_past_due_qty", "Past due", "num0"),
             ("period_demand", "Demand this period", "num0"),
             ("annual_value", "Annual value", "money")],
            note="Ranked by how long the shelf is bare, not by value — an empty shelf "
                 "is a service failure whatever the part costs. Value is shown so the "
                 "call about air freight can be made against it.",
        )

    def _materials_to_adjust(self, recommendations, attributes) -> str:
        if recommendations is None or not len(recommendations):
            return ""
        df = recommendations.copy()
        if "recommended_action" not in df.columns:
            return ""

        # HOLD is the absence of an action; it is most of the catalogue and belongs in
        # the CSV, not in a work list.
        df = df[~df["recommended_action"].astype(str).str.upper().str.startswith("HOLD")]
        if df.empty:
            return ""

        cost = None
        if attributes is not None and "unit_cost" in attributes.columns:
            cost = attributes.drop_duplicates("sku").set_index("sku")["unit_cost"]
        unit = df["sku"].map(cost) if cost is not None else np.nan
        buy = pd.to_numeric(df.get("suggested_po_qty"), errors="coerce").fillna(0.0)
        push = pd.to_numeric(df.get("pushout_open_po_qty"), errors="coerce").fillna(0.0)
        df["action_qty"] = np.where(buy > 0, buy, push)
        df["action_value"] = df["action_qty"] * pd.to_numeric(unit, errors="coerce").fillna(0.0)
        df = self._with_policy(df, attributes)
        df = df.sort_values("action_value", ascending=False).head(12)

        return self._table(
            "Materials to act on",
            df,
            [("sku", "SKU", "sku"), ("stocking_policy", "Policy", "policy"),
             ("recommended_action", "Action", "text"),
             ("action_qty", "Qty", "num0"), ("action_value", "Value", "money"),
             ("days_of_supply", "Days of supply", "num0"),
             ("net_requirement", "Net requirement", "num0")],
            note="Ranked by the money the action moves, not by urgency alone — the "
                 "shortest cover on a cheap part is rarely the first call to make. "
                 "Items already in balance are held, and are in the recommendations CSV "
                 "rather than here.",
            empty="Nothing to change.",
        )

    def _pos_to_adjust(self, open_po, forward, attributes) -> str:
        """
        Open PO lines whose timing disagrees with the position they are arriving into.

        Two cases only, both read off the forward risk: an order landing into stock
        that already has years of cover, and an order landing after the shelf it was
        meant to fill is empty. Everything else is left alone — a report that proposes
        touching every PO gets ignored wholesale.
        """
        if open_po is None or not len(open_po) or "sku" not in open_po.columns:
            return ""
        qty_col = next((c for c in ("open_qty", "order_qty") if c in open_po.columns), None)
        if qty_col is None:
            return ""

        slow = forward.slow_burn if forward is not None else pd.DataFrame()
        risk = forward.stockout if forward is not None else pd.DataFrame()
        cover = (slow.set_index("sku")["days_of_cover"] if len(slow) else pd.Series(dtype=float))
        late = pd.DataFrame()
        if len(risk) and {"inbound_day", "days_to_stockout"} <= set(risk.columns):
            late = risk[risk["inbound_day"].notna()
                        & (risk["inbound_day"] > risk["days_to_stockout"])]
        late_by_sku = (late.set_index("sku")["days_to_stockout"]
                       if len(late) else pd.Series(dtype=float))

        df = open_po.copy()
        df["_qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(0.0)
        df = df[df["_qty"] > 0]
        if df.empty:
            return ""

        df["days_of_cover"] = df["sku"].map(cover)
        df["out_on_day"] = df["sku"].map(late_by_sku)
        df["action"] = np.where(
            df["days_of_cover"].notna(), "Push out or cancel",
            np.where(df["out_on_day"].notna(), "Pull in — arrives too late", ""),
        )
        df = df[df["action"] != ""]
        if df.empty:
            return ""

        value = None
        for col in ("open_amount", "line_value"):
            if col in df.columns:
                value = pd.to_numeric(df[col], errors="coerce")
                break
        if value is None and "unit_cost" in df.columns:
            value = df["_qty"] * pd.to_numeric(df["unit_cost"], errors="coerce")
        df["po_value"] = value.fillna(0.0) if value is not None else 0.0

        eta = next((c for c in ("committed_delivery", "estimated_delivery", "eta")
                    if c in df.columns), None)
        if eta:
            df["eta"] = pd.to_datetime(df[eta], errors="coerce").dt.date
        df = self._with_policy(df, attributes)
        df = df.sort_values("po_value", ascending=False).head(12)

        cols = [("po_number", "PO", "text"), ("sku", "SKU", "sku"),
                ("stocking_policy", "Policy", "policy"), ("action", "Action", "text"),
                ("_qty", "Open qty", "num0"), ("po_value", "Value", "money")]
        if "eta" in df.columns:
            cols.append(("eta", "ETA", "text"))
        cols.append(("days_of_cover", "Days of cover", "cover"))

        return self._table(
            "Open POs to change",
            df,
            cols,
            note="Push out or cancel where the stock it lands in already has more than "
                 "a year of cover; pull in where the SKU runs out before this order "
                 "arrives. Read off the same forward projection as the two sections "
                 "above, so the three cannot disagree.",
            empty="No open PO needs its timing changed.",
        )

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
        if kind == "cover":
            # Infinite cover is a real finding — stock against no demand at all — and
            # it arrives here as `inf`, which the numeric formatter renders as an
            # em dash indistinguishable from a missing value.
            v = _f(value)
            if not np.isfinite(v):
                return "<td>no demand</td>"
            return f"<td>{_num(v, 0)}</td>"
        if kind == "policy":
            # Text, not colour. MTO and MTS are not better and worse than one another,
            # and a status hue would say one of them is a problem.
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                return "<td class='left' style='color:var(--text-muted)'>—</td>"
            return f"<td class='left'>{_esc(value)}</td>"
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
