"""
HTML report generator.
Produces a single self-contained HTML dashboard with embedded charts + data tables.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from jinja2 import Template

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Inventory Planning Report — {{ run_date }}</title>
<style>
  :root {
    --blue:   #1A56DB; --green:  #057A55; --orange: #C27803;
    --red:    #C81E1E; --gray:   #6B7280; --purple: #7C3AED;
    --bg:     #F9FAFB; --card:   #FFFFFF; --border: #E5E7EB;
    --text:   #111827; --muted:  #6B7280;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         background: var(--bg); color: var(--text); font-size: 14px; }
  header { background: var(--blue); color: #fff; padding: 20px 32px;
           display:flex; align-items:center; justify-content:space-between; }
  header h1 { font-size: 20px; font-weight: 700; }
  header .meta { font-size: 12px; opacity: .8; text-align:right; line-height:1.6; }
  nav { background:#fff; border-bottom:1px solid var(--border);
        padding: 0 32px; display:flex; gap:4px; overflow-x:auto; }
  nav a { padding: 12px 16px; text-decoration:none; color:var(--muted);
          font-size:13px; font-weight:500; border-bottom:2px solid transparent;
          white-space:nowrap; transition:.15s; }
  nav a.active, nav a:hover { color:var(--blue); border-color:var(--blue); }
  .page { display:none; padding: 28px 32px; max-width:1400px; margin:0 auto; }
  .page.active { display:block; }
  h2 { font-size:18px; font-weight:700; margin-bottom:16px; color:var(--text); }
  h3 { font-size:14px; font-weight:600; margin-bottom:10px; color:var(--text); }
  .kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
              gap:16px; margin-bottom:28px; }
  .kpi { background:var(--card); border:1px solid var(--border); border-radius:10px;
         padding:18px 20px; }
  .kpi .val { font-size:26px; font-weight:700; line-height:1.1; }
  .kpi .lbl { font-size:11px; color:var(--muted); margin-top:4px; text-transform:uppercase;
              letter-spacing:.05em; }
  .kpi.red  .val { color:var(--red); }
  .kpi.green .val { color:var(--green); }
  .kpi.orange .val { color:var(--orange); }
  .kpi.purple .val { color:var(--purple); }
  .kpi.blue  .val { color:var(--blue); }
  .chart-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(440px,1fr));
                gap:20px; margin-bottom:28px; }
  .chart-card { background:var(--card); border:1px solid var(--border);
                border-radius:10px; padding:16px; }
  .chart-card img { width:100%; height:auto; display:block; border-radius:4px; }
  .chart-wide { background:var(--card); border:1px solid var(--border);
                border-radius:10px; padding:16px; margin-bottom:20px; }
  .chart-wide img { width:100%; height:auto; display:block; border-radius:4px; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  thead th { background:#F3F4F6; padding:8px 12px; text-align:left;
             font-weight:600; border-bottom:1px solid var(--border);
             font-size:11px; text-transform:uppercase; letter-spacing:.04em; white-space:nowrap; }
  tbody tr:nth-child(even) { background:#FAFAFA; }
  tbody td { padding:7px 12px; border-bottom:1px solid var(--border); white-space:nowrap; }
  tbody tr:hover { background:#EFF6FF; }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px;
           font-size:10px; font-weight:600; text-transform:uppercase; }
  .badge-red    { background:#FEE2E2; color:#991B1B; }
  .badge-orange { background:#FEF3C7; color:#92400E; }
  .badge-green  { background:#D1FAE5; color:#065F46; }
  .badge-purple { background:#EDE9FE; color:#5B21B6; }
  .badge-gray   { background:#F3F4F6; color:#374151; }
  .tbl-wrap { background:var(--card); border:1px solid var(--border);
              border-radius:10px; overflow:hidden; margin-bottom:20px; }
  .tbl-header { padding:14px 16px; border-bottom:1px solid var(--border);
                font-weight:600; font-size:14px; display:flex;
                justify-content:space-between; align-items:center; }
  .tbl-body { overflow-x:auto; max-height:420px; overflow-y:auto; }
  .section-title { font-size:16px; font-weight:700; margin:24px 0 12px;
                   padding-bottom:8px; border-bottom:1px solid var(--border); }
  .quality-item { padding:8px 14px; border-left:3px solid var(--orange);
                  background:#FFFBEB; margin-bottom:8px; border-radius:0 6px 6px 0;
                  font-size:12px; }
  .quality-ok   { border-color:var(--green); background:#ECFDF5; }
  .quality-warn { border-color:var(--orange); background:#FFFBEB; }
  .quality-err  { border-color:var(--red); background:#FEF2F2; }
  footer { text-align:center; padding:24px; color:var(--muted);
           font-size:11px; border-top:1px solid var(--border); margin-top:40px; }
</style>
</head>
<body>

<header>
  <div>
    <h1>📦 DC Inventory Planning Report</h1>
    <div style="font-size:13px;margin-top:4px;opacity:.9;">{{ location_id }} &nbsp;·&nbsp; Planning cycle: {{ run_date }}</div>
  </div>
  <div class="meta">
    Generated: {{ run_date }}<br>
    SKUs analysed: {{ total_skus }}<br>
    History: {{ history_range }}
  </div>
</header>

<nav>
  <a href="#" class="active" onclick="showPage('overview',this)">📊 Overview</a>
  <a href="#" onclick="showPage('demand',this)">📈 Demand</a>
  <a href="#" onclick="showPage('inventory',this)">🏭 Inventory</a>
  <a href="#" onclick="showPage('purchasing',this)">🛒 Purchasing</a>
  <a href="#" onclick="showPage('forecast',this)">🔮 Forecast</a>
  <a href="#" onclick="showPage('quality',this)">🔍 Data Quality</a>
</nav>

<!-- ── OVERVIEW ── -->
<div id="page-overview" class="page active">
  <h2>Executive Summary</h2>
  <div class="kpi-grid">
    <div class="kpi red">   <div class="val">{{ kpi.purchase_skus }}</div>   <div class="lbl">Purchase Requests</div></div>
    <div class="kpi purple"><div class="val">{{ kpi.backlog_skus }}</div>    <div class="lbl">Order for Backlog</div></div>
    <div class="kpi orange"><div class="val">{{ kpi.pushout_skus }}</div>    <div class="lbl">Push-Out POs</div></div>
    <div class="kpi red">  <div class="val">{{ kpi.shortage_skus }}</div>   <div class="lbl">Shortage-Risk SKUs</div></div>
    <div class="kpi orange"><div class="val">{{ kpi.excess_skus }}</div>     <div class="lbl">Excess SKUs</div></div>
    <div class="kpi green"> <div class="val">{{ kpi.hold_ok_skus }}</div>    <div class="lbl">On Target</div></div>
    <div class="kpi blue">  <div class="val">{{ kpi.total_buy_qty }}</div>   <div class="lbl">Total Buy Qty</div></div>
    <div class="kpi orange"><div class="val">{{ kpi.total_pushout_qty }}</div><div class="lbl">Total Push-Out Qty</div></div>
  </div>

  <div class="chart-grid">
    <div class="chart-card"><img src="data:image/png;base64,{{ charts.classification_donut }}" alt="Classification"></div>
    <div class="chart-card"><img src="data:image/png;base64,{{ charts.inventory_status }}" alt="Inventory Status"></div>
    <div class="chart-card"><img src="data:image/png;base64,{{ charts.recommendations }}" alt="Recommendations"></div>
    <div class="chart-card"><img src="data:image/png;base64,{{ charts.variability_scatter }}" alt="Variability"></div>
  </div>
</div>

<!-- ── DEMAND ── -->
<div id="page-demand" class="page">
  <h2>Demand Analysis</h2>
  <div class="chart-wide"><img src="data:image/png;base64,{{ charts.top_demand_ts }}" alt="Top Demand SKUs"></div>
  <div class="chart-wide"><img src="data:image/png;base64,{{ charts.lead_time }}" alt="Lead Time"></div>

  <div class="section-title">Top 30 SKUs by Total Demand (36 months)</div>
  <div class="tbl-wrap">
    <div class="tbl-body">
    <table>
      <thead><tr>
        <th>SKU</th><th>Description</th><th>S&OP Class</th>
        <th>Stocking Class</th><th>Avg Monthly Demand</th>
        <th>Std Dev</th><th>Active Months</th><th>Total Qty</th>
      </tr></thead>
      <tbody>{% for r in demand_table %}
      <tr>
        <td><b>{{ r.sku }}</b></td>
        <td>{{ r.description }}</td>
        <td>{{ r.sopc_class }}</td>
        <td>{{ stocking_badge(r.stocking_class) }}</td>
        <td style="text-align:right">{{ "{:,.1f}".format(r.demand_mean_rolling) }}</td>
        <td style="text-align:right">{{ "{:,.1f}".format(r.demand_std_rolling) }}</td>
        <td style="text-align:right">{{ r.active_cycles_rolling }} / {{ r.total_cycles_evaluated }}</td>
        <td style="text-align:right">{{ "{:,.0f}".format(r.total_qty) }}</td>
      </tr>{% endfor %}
      </tbody>
    </table>
    </div>
  </div>
</div>

<!-- ── INVENTORY ── -->
<div id="page-inventory" class="page">
  <h2>Inventory Position</h2>
  <div class="chart-grid">
    <div class="chart-card"><img src="data:image/png;base64,{{ charts.safety_stock_vs_onhand }}" alt="SS vs OnHand"></div>
    {% if charts.excess_pushout %}
    <div class="chart-card"><img src="data:image/png;base64,{{ charts.excess_pushout }}" alt="Excess Push-Out"></div>
    {% endif %}
  </div>

  <div class="section-title">Inventory Projection — All Stocking SKUs</div>
  <div class="tbl-wrap">
    <div class="tbl-body">
    <table>
      <thead><tr>
        <th>SKU</th><th>Class</th><th>Status</th>
        <th>On Hand</th><th>GIT Adj</th><th>Open PO</th>
        <th>Effective Pos.</th><th>Should-Be</th><th>Surplus / Deficit</th>
        <th>Safety Stock</th><th>Push-Out?</th>
      </tr></thead>
      <tbody>{% for r in inventory_table %}
      <tr>
        <td><b>{{ r.sku }}</b></td>
        <td>{{ stocking_badge(r.stocking_class) }}</td>
        <td>{{ status_badge(r.inventory_status) }}</td>
        <td style="text-align:right">{{ fmt_num(r.qty_on_hand) }}</td>
        <td style="text-align:right">{{ fmt_num(r.qty_in_transit_adj) }}</td>
        <td style="text-align:right">{{ fmt_num(r.total_open_po_qty) }}</td>
        <td style="text-align:right"><b>{{ fmt_num(r.effective_position) }}</b></td>
        <td style="text-align:right">{{ fmt_num(r.should_be_inventory) }}</td>
        <td style="text-align:right;color:{% if r.surplus_deficit < 0 %}#C81E1E{% else %}#057A55{% endif %}">
          {{ fmt_num(r.surplus_deficit) }}</td>
        <td style="text-align:right">{{ fmt_num(r.safety_stock) }}</td>
        <td>{% if r.pushout_candidate %}<span class="badge badge-orange">YES</span>{% endif %}</td>
      </tr>{% endfor %}
      </tbody>
    </table>
    </div>
  </div>
</div>

<!-- ── PURCHASING ── -->
<div id="page-purchasing" class="page">
  <h2>Purchase Recommendations</h2>

  <div class="section-title">Action Required — Purchase Requests &amp; Backlog Orders</div>
  <div class="tbl-wrap">
    <div class="tbl-body">
    <table>
      <thead><tr>
        <th>SKU</th><th>Action</th><th>Stocking Class</th>
        <th>Forecast/mo</th><th>Backlog Qty</th><th>Available Supply</th>
        <th>Safety Stock</th><th>Net Requirement</th><th>Suggested PO Qty</th>
      </tr></thead>
      <tbody>{% for r in purchase_table %}
      <tr>
        <td><b>{{ r.sku }}</b></td>
        <td>{{ action_badge(r.recommended_action) }}</td>
        <td>{{ stocking_badge(r.stocking_class) }}</td>
        <td style="text-align:right">{{ fmt_num(r.forecast_avg_monthly) }}</td>
        <td style="text-align:right">{{ fmt_num(r.backlog_qty) }}</td>
        <td style="text-align:right">{{ fmt_num(r.available_supply) }}</td>
        <td style="text-align:right">{{ fmt_num(r.safety_stock) }}</td>
        <td style="text-align:right;color:#C81E1E"><b>{{ fmt_num(r.net_requirement) }}</b></td>
        <td style="text-align:right;font-weight:600">{{ fmt_num(r.suggested_po_qty) }}</td>
      </tr>{% endfor %}
      </tbody>
    </table>
    </div>
  </div>

  <div class="section-title">Push-Out Candidates</div>
  <div class="tbl-wrap">
    <div class="tbl-body">
    <table>
      <thead><tr>
        <th>SKU</th><th>Should-Be Inventory</th><th>Current Position</th>
        <th>Surplus</th><th>Open PO Qty to Push</th>
      </tr></thead>
      <tbody>{% for r in pushout_table %}
      <tr>
        <td><b>{{ r.sku }}</b></td>
        <td style="text-align:right">{{ fmt_num(r.should_be_inventory) }}</td>
        <td style="text-align:right">{{ fmt_num(r.effective_position) }}</td>
        <td style="text-align:right;color:#C27803"><b>{{ fmt_num(r.surplus_deficit) }}</b></td>
        <td style="text-align:right;font-weight:600">{{ fmt_num(r.pushout_open_po_qty) }}</td>
      </tr>{% endfor %}
      </tbody>
    </table>
    </div>
  </div>
</div>

<!-- ── FORECAST ── -->
<div id="page-forecast" class="page">
  <h2>6-Month Demand Forecast</h2>
  <div class="chart-wide"><img src="data:image/png;base64,{{ charts.forecast }}" alt="Forecast"></div>

  <div class="section-title">Forecast Detail — Top SKUs</div>
  <div class="tbl-wrap">
    <div class="tbl-body">
    <table>
      <thead><tr>
        <th>SKU</th><th>Model</th><th>Trend</th><th>Seasonal</th>
        <th>6-Month Total</th><th>Avg Monthly</th>
      </tr></thead>
      <tbody>{% for r in forecast_table %}
      <tr>
        <td><b>{{ r.sku }}</b></td>
        <td><span class="badge badge-gray">{{ r.model_used }}</span></td>
        <td>{% if r.trend_detected %}✓{% else %}–{% endif %}</td>
        <td>{% if r.seasonal_detected %}✓{% else %}–{% endif %}</td>
        <td style="text-align:right">{{ fmt_num(r.forecast_6m_total) }}</td>
        <td style="text-align:right">{{ fmt_num(r.forecast_avg_monthly) }}</td>
      </tr>{% endfor %}
      </tbody>
    </table>
    </div>
  </div>
</div>

<!-- ── DATA QUALITY ── -->
<div id="page-quality" class="page">
  <h2>Data Quality &amp; Pipeline Notes</h2>
  {% for doc, issues in quality_notes.items() %}
  <div class="section-title">{{ doc }}</div>
  {% if issues %}
    {% for iss in issues %}
    <div class="quality-item quality-warn">⚠ {{ iss }}</div>
    {% endfor %}
  {% else %}
    <div class="quality-item quality-ok">✓ No issues detected</div>
  {% endif %}
  {% endfor %}

  <div class="section-title">Supplier Lead Time Summary</div>
  <div class="tbl-wrap">
    <div class="tbl-body">
    <table>
      <thead><tr>
        <th>SKU</th><th>Supplier</th><th>Incoterm</th>
        <th>WMA LT (days)</th><th>LT Std Dev</th><th>Sample Count</th>
      </tr></thead>
      <tbody>{% for r in lt_table %}
      <tr>
        <td><b>{{ r.sku }}</b></td>
        <td>{{ r.supplier }}</td>
        <td><span class="badge badge-gray">{{ r.incoterm or "–" }}</span></td>
        <td style="text-align:right">{{ "{:.1f}".format(r.wma_lead_time_days) }}</td>
        <td style="text-align:right">{{ "{:.1f}".format(r.lt_std_days) }}</td>
        <td style="text-align:right">{{ r.sample_count }}</td>
      </tr>{% endfor %}
      </tbody>
    </table>
    </div>
  </div>
</div>

<footer>
  Inventory Planning Report &nbsp;·&nbsp; {{ location_id }} &nbsp;·&nbsp;
  Generated {{ run_date }} &nbsp;·&nbsp; inventory-planning v0.2.0
</footer>

<script>
function showPage(id, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav a').forEach(a => a.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  el.classList.add('active');
  return false;
}
</script>
</body>
</html>
"""


def _stocking_badge(cls: str) -> str:
    m = {"stocking-high": ("green","High Svc"), "stocking-med": ("orange","Med Svc"),
         "non-stocking": ("gray","Non-Stock")}
    c, label = m.get(cls, ("gray", cls))
    return f'<span class="badge badge-{c}">{label}</span>'

def _status_badge(s: str) -> str:
    m = {"EXCESS": ("red","EXCESS"), "slightly-over": ("orange","OVER"),
         "OK": ("green","OK"), "SHORTAGE-RISK": ("purple","SHORTAGE"),
         "non-stocking": ("gray","N/S")}
    c, label = m.get(s, ("gray", s))
    return f'<span class="badge badge-{c}">{label}</span>'

def _action_badge(a: str) -> str:
    m = {"PURCHASE-REQUEST": ("red","Buy"), "ORDER-FOR-BACKLOG": ("purple","Backlog Buy"),
         "PUSH-OUT-OPEN-PO": ("orange","Push Out"), "HOLD-OK": ("green","Hold OK"),
         "HOLD-EXCESS": ("orange","Hold Excess"), "NO-ACTION": ("gray","No Action")}
    c, label = m.get(a, ("gray", a))
    return f'<span class="badge badge-{c}">{label}</span>'

def _fmt_num(v) -> str:
    try:
        f = float(v)
        if abs(f) >= 1000:
            return f"{f:,.0f}"
        return f"{f:,.1f}"
    except (TypeError, ValueError):
        return str(v) if v is not None else "—"


class HTMLReportGenerator:

    def generate(self, results: dict, charts: Dict[str, str],
                 quality_reports: list, output_path: Path,
                 location_id: str = "DC-01") -> Path:
        """
        Render the full HTML report.
        results: output dict from InventoryPlanner.run_planning()
        charts:  {name: base64_png} from ChartBuilder.build_all()
        quality_reports: list of quality report dicts from readers
        """
        run_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        ts       = results["time_series"]
        clf      = results["classified_demand"]
        rec      = results["recommendations"]
        proj     = results["projection"]
        fc_sum   = results["forecast_summary"]
        lt       = results["supplier_lt"]

        # ── KPIs ─────────────────────────────────────────────────────────────
        def _kpi(df, col, val): return int((df[col] == val).sum())
        kpi = {
            "purchase_skus":   f"{_kpi(rec,'recommended_action','PURCHASE-REQUEST'):,}",
            "backlog_skus":    f"{_kpi(rec,'recommended_action','ORDER-FOR-BACKLOG'):,}",
            "pushout_skus":    f"{_kpi(rec,'recommended_action','PUSH-OUT-OPEN-PO'):,}",
            "shortage_skus":   f"{_kpi(proj,'inventory_status','SHORTAGE-RISK'):,}",
            "excess_skus":     f"{_kpi(proj,'inventory_status','EXCESS'):,}",
            "hold_ok_skus":    f"{_kpi(rec,'recommended_action','HOLD-OK'):,}",
            "total_buy_qty":   f"{rec['suggested_po_qty'].sum():,.0f}",
            "total_pushout_qty": f"{proj[proj.get('pushout_candidate',False)==True]['pushout_open_po_qty'].sum() if 'pushout_open_po_qty' in proj.columns else 0:,.0f}",
        }

        # ── Tables ────────────────────────────────────────────────────────────
        demand_table = (
            clf.sort_values("total_qty", ascending=False)
               .head(30)
               .fillna("")
               .to_dict("records")
        )

        inv_stocking = proj[proj["stocking_class"] != "non-stocking"].copy()
        inventory_table = (
            inv_stocking.sort_values("surplus_deficit")
                        .head(80)
                        .fillna(0)
                        .to_dict("records")
        )

        purchase_table = (
            rec[rec["recommended_action"].isin(["PURCHASE-REQUEST","ORDER-FOR-BACKLOG"])]
               .sort_values("net_requirement", ascending=False)
               .drop_duplicates("sku")
               .head(60)
               .fillna(0)
               .to_dict("records")
        )

        pushout_table = (
            proj[proj.get("pushout_candidate", False) == True]
               .sort_values("surplus_deficit", ascending=False)
               .drop_duplicates("sku")
               .head(40)
               .fillna(0)
               .to_dict("records")
        ) if "pushout_candidate" in proj.columns else []

        forecast_table = (
            fc_sum.sort_values("forecast_6m_total", ascending=False)
                  .head(50)
                  .fillna("")
                  .to_dict("records")
        )

        lt_table = (
            lt.sort_values("wma_lead_time_days", ascending=False)
              .head(50)
              .fillna("")
              .to_dict("records")
        )

        # ── Quality notes ─────────────────────────────────────────────────────
        quality_notes = {}
        for qr in quality_reports:
            key = qr.get("doc_type", qr.get("file", "unknown")).replace("_", " ").title()
            quality_notes[key] = qr.get("issues", [])

        # ── Render ────────────────────────────────────────────────────────────
        tmpl = Template(_TEMPLATE)
        html = tmpl.render(
            run_date=run_date,
            location_id=location_id,
            total_skus=f"{len(clf):,}",
            history_range=f"{ts.index[0]} → {ts.index[-1]}",
            kpi=kpi,
            charts={k: v for k, v in charts.items() if v},
            demand_table=demand_table,
            inventory_table=inventory_table,
            purchase_table=purchase_table,
            pushout_table=pushout_table,
            forecast_table=forecast_table,
            lt_table=lt_table,
            quality_notes=quality_notes,
            stocking_badge=_stocking_badge,
            status_badge=_status_badge,
            action_badge=_action_badge,
            fmt_num=_fmt_num,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        print(f"  HTML report saved: {output_path}")
        return output_path
