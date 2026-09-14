// The stored-facts screen.
//
// The request behind it was CRUD over the imported data, and the store is append-only.
// Both are satisfiable, because the four verbs mean something here that is not an
// UPDATE — and saying which is which is most of what this screen is for:
//
//   create   a new batch, through the import screen
//   read     an as-of query, and the landed rows beside the canonical ones
//   update   a re-resolve, or an override with a name on it — never a fact rewritten
//   delete   void a batch; the rows are untouched and the void is itself reversible
//
// The reading selector is not a preference. `history`, `current` and `latest` answer
// three different questions and one of them must never be summed, so the screen names
// the one in force and says what it means rather than defaulting quietly.

import { $, el, mount, num, api } from "/ui.js";

const MODES = {
  current: "the newest observation of every key ever seen",
  latest: "the newest batch, and only it — for an export that replaces its predecessor",
  history: "every observation. Do not add these up: one row appears once per import",
};

const state = { docType: null, mode: "current", asOf: "", sku: "", batch: null };

function table(columns, rows, { highlight = () => false } = {}) {
  return el("div", { class: "scroll" }, el("table", {},
    el("thead", {}, el("tr", {}, columns.map((c) => el("th", {}, c)))),
    el("tbody", {}, rows.map((row) =>
      el("tr", { class: highlight(row) ? "attention" : null },
         columns.map((c) => el("td", {
           class: typeof row[c] === "number" ? "num" : null,
         }, row[c] === null || row[c] === undefined ? "" : String(row[c]))))))));
}

// ── Panels ──────────────────────────────────────────────────────────────────

function picker(types, onChange) {
  const select = el("select", { onchange: (e) => onChange(e.target.value) },
    el("option", { value: "" }, "Choose a document —"),
    types.map((t) => el("option", { value: t.doc_type, selected: t.doc_type === state.docType },
                        `${t.doc_type} (${t.batches})`)));
  return el("section", { class: "card" },
    el("h2", {}, "What the store holds"),
    el("p", { class: "sub" },
       types.length
         ? "One row per document type, with the number of batches behind it."
         : "Nothing has been promoted yet. Review a file on the import screen, then "
           + "promote it with the date it describes — the date the data describes, "
           + "which is not the date you downloaded it."),
    types.length ? select : null);
}

function controls(onChange) {
  const mode = el("select", { onchange: (e) => { state.mode = e.target.value; onChange(); } },
    Object.keys(MODES).map((m) =>
      el("option", { value: m, selected: m === state.mode }, m)));
  const asOf = el("input", { type: "date", value: state.asOf,
    onchange: (e) => { state.asOf = e.target.value; onChange(); } });
  const sku = el("input", { placeholder: "filter by SKU", value: state.sku,
    onchange: (e) => { state.sku = e.target.value.trim(); onChange(); } });

  return el("section", { class: "card" },
    el("h2", {}, "Reading"),
    el("div", { class: "controls" },
      el("div", {}, el("label", {}, "as of"), asOf),
      el("div", {}, el("label", {}, "reading"), mode),
      el("div", {}, el("label", {}, "SKU"), sku)),
    el("p", { class: "note" }, MODES[state.mode]),
    state.asOf
      ? el("p", { class: "note" },
           "Only batches describing a moment at or before this date are read. Clearing "
           + "the date reads everything.")
      : null);
}

function factsCard(body) {
  const carried = body.carried_forward;
  return el("section", { class: "card" },
    el("h2", {}, `${body.doc_type} · ${body.mode}`,
       el("span", { class: "badge" }, `${body.rows.length} rows`)),
    el("p", { class: "sub" }, body.selection),
    carried && carried.carried
      ? el("p", { class: "note", style: "color:var(--warn)" },
           `${num(carried.carried)} of ${num(carried.rows)} rows `
           + `(${Math.round(carried.carried_share * 100)}%) come from a batch older than `
           + `${carried.newest_valid_time}. If this export replaces its predecessor each `
           + `time, those keys are gone rather than unchanged — read it as "latest" to `
           + `see that answer.`)
      : null,
    body.rows.length
      ? table(body.columns, body.rows,
              { highlight: (r) => carried && r.__valid_time
                                  && r.__valid_time < carried.newest_valid_time })
      : el("p", { class: "note" }, "No rows match."));
}

function batchesCard(body, refresh) {
  const rows = body.batches;
  if (!rows.length) return null;
  return el("section", { class: "card" },
    el("h2", {}, "Batches read"),
    el("p", { class: "sub" },
       "Two timestamps, and they are routinely different: what the data describes, and "
       + "when it was loaded. Only the second can reconstruct what was believed before "
       + "a correction arrived."),
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {},
        ["batch", "describes", "loaded", "rows", "source", ""].map((h) => el("th", {}, h)))),
      el("tbody", {}, rows.map((b) => el("tr", {},
        el("td", {}, el("code", {}, b.batch_id)),
        el("td", {}, b.valid_time),
        el("td", {}, String(b.transaction_time).replace("T", " ").slice(0, 19)),
        el("td", { class: "num" }, num(b.rows)),
        el("td", {}, b.source_name || ""),
        el("td", {},
          el("button", { class: "link", onclick: () => showBatch(b.batch_id) }, "rows"),
          " ",
          el("button", { class: "link", onclick: (e) =>
            e.target.closest("tr").after(voidRow(b, refresh)) }, "void"))))))));
}

function voidRow(batch, refresh) {
  const reason = el("input", { placeholder: "Why is this batch being withdrawn?" });
  const by = el("input", { placeholder: "Your name" });
  const out = el("span", { class: "note" });
  const row = el("tr", {}, el("td", { colspan: 6 },
    el("form", { class: "declare", onsubmit: async (e) => {
      e.preventDefault();
      try {
        await api(`/batches/${batch.batch_id}/void`, {
          method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ reason: reason.value, by: by.value }),
        });
        refresh();
      } catch (err) { out.textContent = String(err.message); }
    } },
      el("label", {},
         `Voiding ${batch.batch_id} withdraws all ${num(batch.rows)} of its rows from `
         + `every reading. The rows themselves are not touched and the void is appended, `
         + `so it can be read back and undone.`),
      el("div", { class: "row2" }, reason, by),
      el("div", {}, el("button", { class: "primary", type: "submit" }, "Void this batch")),
      out)));
  return row;
}

// One batch, before and after the adapter. The fastest way a person without the
// vocabulary to adjudicate a mapping still finds a mis-mapped column: the value is
// visibly the wrong kind of thing under a name that expects another.
async function showBatch(batchId) {
  const host = $("#batch-detail");
  mount(host, el("p", { class: "note" }, "Loading…"));
  try {
    const [raw, canon] = await Promise.all([
      api(`/batches/${batchId}/rows?limit=15`),
      api(`/batches/${batchId}/canonical?limit=15`),
    ]);
    mount(host, el("section", { class: "card" },
      el("h2", {}, "One batch, before and after the adapter",
         el("span", { class: "badge" }, batchId)),
      el("p", { class: "sub" },
         "Above, the rows as the file wrote them. Below, the same rows under canonical "
         + "field names. A column read as the wrong thing shows up as a value of the "
         + "wrong kind under a name that expects another."),
      el("h3", {}, "As the file wrote them"),
      table(raw.columns, raw.rows),
      el("h3", {}, `As the pipeline reads them · ${canon.doc_type}`),
      table(canon.columns, canon.rows),
      el("p", {}, el("button", { class: "link", onclick: () => mount(host) }, "Close"))));
  } catch (err) {
    mount(host, el("p", { class: "status bad" }, String(err.message)));
  }
}

// ── Assembly ────────────────────────────────────────────────────────────────

async function render() {
  const host = $("#browse-body");
  if (!state.docType) { mount(host); return; }
  const params = new URLSearchParams({ mode: state.mode, limit: "200" });
  if (state.asOf) params.set("as_of", state.asOf);
  if (state.sku) params.set("sku", state.sku);
  try {
    const body = await api(`/facts/${state.docType}?${params}`);
    mount(host, controls(render), factsCard(body), batchesCard(body, refresh));
  } catch (err) {
    mount(host, controls(render),
      el("section", { class: "card stop" },
        el("h2", {}, "This document cannot be read that way"),
        el("p", { class: "sub" }, String(err.message))));
  }
}

async function refresh() {
  mount($("#batch-detail"));
  const types = await api("/facts");
  if (state.docType && !types.some((t) => t.doc_type === state.docType)) {
    state.docType = null;
  }
  mount($("#browse-top"), picker(types, (value) => {
    state.docType = value || null;
    refresh();
  }));
  await render();
}

export function mountBrowse() {
  return refresh();
}
