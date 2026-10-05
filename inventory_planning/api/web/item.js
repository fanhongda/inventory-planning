// One item, as a run saw it.
//
// Under the results screen's constraint, one layer down: **it computes nothing**. Every
// figure is a cell out of the run's own workbook, and the chart plots those cells. A
// screen that re-forecast to draw a line would be a second forecast, and the first time
// the two disagreed the planner would have no way to tell which one was the plan.
//
// Two lines, not three. The intent was history-as-read, history-as-the-model-saw-it,
// and the forecast — but nothing in this pipeline cleanses demand history, so the
// middle line would be the first line drawn twice, implying a step that does not
// happen. Saying so is the point: the gap is worth more on screen than a tidy chart.
//
// The band around the forecast is `forecast_rmse`, which is measured, under a normal
// assumption, which is not. It is drawn because a forecast with no dispersion invites
// a confidence the backtest does not support, and it is labelled because none of these
// models produced an interval.

import { $, el, mount, num, api } from "/ui.js";

const PAD = { top: 16, right: 14, bottom: 30, left: 56 };

function chart(series, rmse) {
  const points = series.filter((p) => p.value !== null && p.value !== undefined
                                      && p.value !== "");
  if (points.length < 2) {
    return el("p", { class: "note" },
              "Not enough periods on this item to draw a line.");
  }
  const values = points.map((p) => Number(p.value)).filter((v) => !Number.isNaN(v));
  const band = Number(rmse) || 0;
  const top = Math.max(...values, ...(band ? values.map((v) => v + band) : []));
  const low = Math.min(0, ...values, ...(band ? values.map((v) => v - band) : []));
  const W = 920, H = 260;
  const span = (top - low) || 1;
  const x = (i) => PAD.left + i * (W - PAD.left - PAD.right) / (points.length - 1);
  const y = (v) => PAD.top + (top - v) * (H - PAD.top - PAD.bottom) / span;

  const path = (kind) => points.map((p, i) => [p, i])
    .filter(([p]) => p.kind === kind)
    .map(([p, i], n) => `${n ? "L" : "M"}${x(i).toFixed(1)},${y(Number(p.value)).toFixed(1)}`)
    .join(" ");

  // The join: the forecast line starts at the last history point, so the eye is not
  // asked to bridge a gap that is not in the data.
  const lastHistory = points.map((p, i) => [p, i]).filter(([p]) => p.kind === "history").pop();
  const firstForecast = points.map((p, i) => [p, i]).filter(([p]) => p.kind === "forecast")[0];

  const forecastPoints = points.map((p, i) => [p, i]).filter(([p]) => p.kind === "forecast");
  const bandPath = band && forecastPoints.length
    ? forecastPoints.map(([p, i], n) =>
        `${n ? "L" : "M"}${x(i).toFixed(1)},${y(Number(p.value) + band).toFixed(1)}`).join(" ")
      + " " + forecastPoints.slice().reverse().map(([p, i]) =>
        `L${x(i).toFixed(1)},${y(Math.max(low, Number(p.value) - band)).toFixed(1)}`).join(" ")
      + " Z"
    : "";

  const ticks = [top, low + span / 2, low].map((v) => el("g", {},
    el("line", { x1: PAD.left, x2: W - PAD.right, y1: y(v), y2: y(v),
                 class: "grid" }),
    el("text", { x: PAD.left - 8, y: y(v) + 4, class: "tick" }, num(v))));

  const labels = points.map((p, i) => (i === 0 || i === points.length - 1
      || (firstForecast && i === firstForecast[1]))
    ? el("text", { x: x(i), y: H - 10, class: "tick mid" }, p.period) : null);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "series",
                          role: "img", "aria-label": "demand history and forecast" },
    ticks,
    bandPath ? el("path", { d: bandPath, class: "band" }) : null,
    el("path", { d: path("history"), class: "line history" }),
    lastHistory && firstForecast
      ? el("path", { class: "line forecast",
                     d: `M${x(lastHistory[1])},${y(Number(lastHistory[0].value))} `
                        + `L${x(firstForecast[1])},${y(Number(firstForecast[0].value))}` })
      : null,
    el("path", { d: path("forecast"), class: "line forecast" }),
    labels);

  return el("div", {}, svg,
    el("p", { class: "legend" },
      el("span", { class: "key history" }, "history, as the run read it"),
      el("span", { class: "key forecast" }, "forecast"),
      band ? el("span", { class: "key band" }, `±${num(band)} (forecast_rmse)`) : null));
}

function pairs(title, values, { note = "" } = {}) {
  const entries = Object.entries(values || {}).filter(([, v]) =>
    v !== null && v !== undefined && v !== "");
  if (!entries.length) return null;
  return el("section", { class: "card" },
    el("h2", {}, title),
    note ? el("p", { class: "sub" }, note) : null,
    el("div", { class: "scroll" }, el("table", {}, el("tbody", {},
      entries.map(([k, v]) => se(k, v))))));
}

const se = (k, v) => el("tr", {},
  el("td", { class: "k" }, el("code", {}, k)),
  el("td", {}, typeof v === "number" ? num(v, Number.isInteger(v) ? 0 : 2) : String(v)));

// In force beside suggested, and only where they differ. A table of forty rows that
// agree buries the three that do not, and the three are the whole reason to look.
function comparison(inForce, suggested) {
  const names = Object.keys(suggested || {}).filter((k) =>
    suggested[k] !== null && suggested[k] !== undefined && suggested[k] !== "");
  if (!names.length) return null;
  const moved = names.filter((k) => String(inForce[k]) !== String(suggested[k]));
  return el("section", { class: "card" },
    el("h2", {}, "In force, and what the rules suggest",
       el("span", { class: "badge" }, `${moved.length} differ`)),
    el("p", { class: "sub" },
       "The suggestion is not applied. It is what this item's parameters would be "
       + "under the rules as they stand, beside what the run actually used."),
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {},
        ["parameter", "in force", "suggested"].map((h) => el("th", {}, h)))),
      el("tbody", {}, (moved.length ? moved : names).map((k) => el("tr",
        { class: String(inForce[k]) !== String(suggested[k]) ? "attention" : null },
        el("td", {}, el("code", {}, k)),
        el("td", {}, String(inForce[k] ?? "—")),
        el("td", {}, String(suggested[k]))))))));
}

function detail(body) {
  const rmse = body.model && body.model.forecast_rmse;
  return [
    el("section", { class: "card" },
      el("h2", {}, body.sku,
         el("span", { class: "badge" }, body.model.model_used || "no model")),
      el("p", { class: "sub" },
         `From ${body.workbook}, written by run ${body.run_id}`
         + (body.run_at ? ` on ${body.run_at.slice(0, 10)}` : "") + "."),
      chart(body.series, rmse)),
    pairs("How this forecast was arrived at", body.model, {
      note: "The model, how it was chosen, and what it scored against holding the "
            + "last value. A forecast whose vs_naive is 1.0 is worth reading "
            + "differently from one at 0.4.",
    }),
    comparison(body.in_force, body.suggested),
    pairs("What this run read for the item", body.in_force),
    el("section", { class: "card warn" },
      el("h2", {}, "What this screen cannot show"),
      el("ul", { class: "impacts" },
         (body.not_shown || []).map((line) => el("li", {}, line)))),
  ];
}

// ── The grid ────────────────────────────────────────────────────────────────
//
// Default view is the rows this run disagrees with the organisation about, not every
// item. A flat table of the whole catalogue asks the planner to find the exceptions in
// it, and a page nobody finishes reading produces the belief that it was read.
//
// The three groups are the run's own verdicts, restated: `policy_agrees`, the
// suggestion engine's sentence, and should-be against actual. Sorted by money, with no
// threshold — a cutoff would be a number invented by this screen.

const GROUPS = {
  policy: {
    title: "The policy in force is not the one the evidence implies",
    sub: "This run's own verdict, with the reason it gave. The most expensive "
       + "disagreement on the page, because it is not a parameter being off by a "
       + "margin — it is the wrong kind of item.",
    columns: ["sku", "policy_in_force", "policy_implied", "policy_implied_because",
              "gap_value"],
  },
  parameters: {
    title: "The rules would set different parameters",
    sub: "What this item's parameters would be under the rules as they stand, beside "
       + "what the run used. Nothing is applied.",
    columns: ["sku", "abc_class", "changes_suggested", "ss_delta_value_suggested",
              "gap_value"],
  },
  position: {
    title: "Stock away from should-be, by money",
    sub: "Not a filter: nearly every item is some distance from should-be, so this is "
       + "the ranking rather than a shortlist. Negative is short of should-be.",
    columns: ["sku", "abc_class", "actual_value", "should_be_value", "gap_value",
              "coverage_ratio"],
  },
};

const LABELS = {
  sku: "item", policy_in_force: "in force", policy_implied: "implied",
  policy_implied_because: "because", gap_value: "gap",
  changes_suggested: "would change", ss_delta_value_suggested: "safety stock Δ",
  abc_class: "class", actual_value: "actual", should_be_value: "should be",
  coverage_ratio: "coverage",
};

function cell(name, value) {
  if (value === null || value === undefined || value === "") return el("td", {}, "—");
  if (name === "gap_value") {
    const n = Number(String(value).replace(/,/g, ""));
    return el("td", { class: Number.isNaN(n) ? "" : (n < 0 ? "num short" : "num over") },
              num(n));
  }
  if (typeof value === "number") return el("td", { class: "num" }, num(value, 2));
  return el("td", {}, String(value));
}

function grid(name, rows, open) {
  const meta = GROUPS[name];
  if (!rows.length) return null;
  return el("details", { class: "group", open: name === "policy" },
    el("summary", {}, meta.title, el("span", { class: "k" }, ` · ${rows.length}`)),
    el("p", { class: "note" }, meta.sub),
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {},
        meta.columns.map((c) => el("th", {}, LABELS[c] || c)), el("th", {}, ""))),
      el("tbody", {}, rows.map((r) => el("tr", {},
        meta.columns.map((c) => cell(c, r[c])),
        el("td", {}, el("button", { class: "link", onclick: () => open(r.sku) },
                        "Open"))))))));
}

export async function mountItem() {
  const host = $("#item-body");
  const box = el("input", { placeholder: "item number, e.g. SKU-001", id: "sku-box" });
  const out = el("div", {});
  const gridHost = el("div", {});
  let runs = [];

  const picker = el("select", { onchange: () => { loadGrid(); mount(out); } });

  const open = async (sku) => {
    box.value = sku;
    mount(out, el("p", { class: "note" }, "Reading\u2026"));
    try {
      mount(out, detail(await api(
        `/runs/${encodeURIComponent(picker.value)}/skus/${encodeURIComponent(sku)}`)));
      out.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      mount(out, el("p", { class: "status bad" }, String(err.message)));
    }
  };

  const look = () => {
    const sku = box.value.trim();
    if (!sku) { mount(out, el("p", { class: "note" }, "Name an item.")); return; }
    if (!picker.value) {
      mount(out, el("p", { class: "note" },
        "No run to read. This screen renders a run's workbook, so there has to be "
        + "one \u2014 start a run on the policy screen."));
      return;
    }
    open(sku);
  };

  const loadGrid = async () => {
    if (!picker.value) { mount(gridHost); return; }
    mount(gridHost, el("p", { class: "note" }, "Reading the run\u2026"));
    try {
      const body = await api(`/runs/${encodeURIComponent(picker.value)}/items`);
      mount(gridHost, el("section", { class: "card" },
        el("h2", {}, "What this run disagrees with",
           el("span", { class: "badge" }, `${body.flagged} of ${body.total}`)),
        el("p", { class: "sub" },
           `From ${body.workbook}. Nothing here is recalculated \u2014 the groups `
           + `restate columns the run wrote, ordered by the money behind each row.`),
        Object.keys(GROUPS).map((name) =>
          grid(name, body.groups[name] || [], open))));
    } catch (err) {
      mount(gridHost, el("p", { class: "status bad" }, String(err.message)));
    }
  };

  mount(host, el("p", { class: "note" }, "Loading\u2026"));
  try {
    runs = (await api("/runs")).runs || [];
  } catch (err) {
    mount(host, el("p", { class: "status bad" }, String(err.message)));
    return;
  }
  mount(picker, runs.map((r) => el("option", { value: r.run_id }, r.run_id)));

  mount(host,
    el("section", { class: "card ask" },
      el("h2", {}, "Look up an item"),
      runs.length
        ? el("div", { class: "row2" },
            el("div", {}, el("label", {}, "Item number"), box),
            el("div", {}, el("label", {}, "As this run saw it"), picker))
        : el("p", { class: "note" },
             "No runs here yet. This screen renders a run's own workbook, so there "
             + "has to be one first."),
      runs.length ? el("div", {}, el("button", { class: "primary", onclick: look },
                                     "Look it up")) : null),
    gridHost,
    out);

  box.addEventListener("keydown", (e) => { if (e.key === "Enter") look(); });
  loadGrid();
}
