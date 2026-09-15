// The import review screen.
//
// Written for someone who will not open an editor and cannot adjudicate a mapping. That
// constraint decides the order of everything below.
//
//   1. Ask a question they can answer. Not "is `sku <- Item Code` right" — nobody outside
//      this repository can answer that — but "this file says 2,125 units on hand across
//      10 materials; does that sound right". Whoever runs the warehouse knows.
//   2. Decide by default; escalate only the close calls. The router reports a margin
//      now, so a document that won by 56 points is stated and moved past, and one that
//      won by 3 is put in front of a person.
//   3. Give the consequence, not the evidence. Not "overlap coefficient 86.5%".
//   4. Refuse rather than default. Currency is a required question with a dropdown, not
//      a silent fallback to the reporting currency worth a multiple of the real number.

import { $, el, mount, num, pct, firstSentence, api } from "/ui.js";

// The checklist, grouped. Nine documents in one flat list is a wall, and translating
// nine contract descriptions would put a second copy of the schema's prose in here to
// go stale. So the plain-language explanation lives at the section — six of them, each
// a thing a planner already has a word for — and the documents keep the identifiers the
// contracts, the CLI and the workbook all use.
//
// `catch_all` is not decoration. A contract added later belongs to no section, and a
// checklist that quietly stopped listing it would be worse than an ugly one.
const SECTIONS = [
  { title: "Demand", gloss: "what you are planning for, and how much of it sold",
    docs: ["sales_history", "demand_timeseries"] },
  { title: "Stock", gloss: "what is on hand right now",
    docs: ["inventory"] },
  { title: "Master data", gloss: "the standing attributes and planning parameters per item",
    docs: ["item_master", "planning_master"] },
  { title: "On order and backlog", gloss: "placed but not received, promised but not shipped",
    docs: ["open_po", "open_so"] },
  { title: "Purchase history", gloss: "how long deliveries actually took, and how often you order",
    docs: ["po_history"] },
  { title: "Renumbering", gloss: "the same item under a new number, or old and new side by side",
    docs: ["substitution"] },
  { title: "Other", gloss: "documents that fit none of the sections above",
    docs: [], catch_all: true },
];

// ── Panels ──────────────────────────────────────────────────────────────────

// What a run still needs, and what has arrived. Standing on the page from the first
// load rather than appearing after an upload: the question "what else do you want from
// me" is the one a person has before they have uploaded anything, and answering it only
// afterwards is answering it too late.
//
// Capability-shaped underneath, document-shaped on the surface. The pipeline requires a
// demand signal; the person has files. Where two documents can satisfy one requirement
// they are shown as the alternatives they are, rather than as two missing items.
function requirementsCard(req) {
  const satisfiedBy = {};       // doc_type -> the sibling that already covers it
  const alternatives = {};      // doc_type -> other doc_types for the same requirement
  for (const cap of req.capabilities) {
    if (!cap.required || cap.suppliers.length < 2) continue;
    for (const doc of cap.suppliers) {
      alternatives[doc] = cap.suppliers.filter((d) => d !== doc);
      if (cap.satisfied && cap.supplied_by !== doc) satisfiedBy[doc] = cap.supplied_by;
    }
  }

  const state = (doc) => {
    if (doc.landed) {
      return doc.fields.every((f) => f.present)
        ? { mark: "\u2713", cls: "ok",
            note: `${num(doc.landed.rows)} rows \u00b7 ${doc.landed.source_name}` }
        : { mark: "!", cls: "part",
            note: `${num(doc.landed.rows)} rows, but a required field found no column` };
    }
    if (satisfiedBy[doc.doc_type]) {
      return { mark: "\u2013", cls: "spare",
               note: `not needed \u2014 ${satisfiedBy[doc.doc_type]} already supplies this` };
    }
    return { mark: "○", cls: "todo",
             note: alternatives[doc.doc_type]
               ? `not uploaded (either this or ${alternatives[doc.doc_type].join(", ")})`
               : "not uploaded" };
  };

  const line = (doc) => {
    const st = state(doc);
    return el("li", { class: `req ${st.cls}` },
      el("span", { class: "mark" }, st.mark),
      el("div", {},
        el("div", { class: "name" },
           el("code", { title: doc.description || "" }, doc.doc_type),
           el("span", { class: "k" }, st.note)),
        el("div", { class: "chips" }, doc.fields.map((f) =>
          el("span", { class: `chip ${f.present ? "on" : "off"}` },
             `${f.present ? "✓" : "○"} ${f.field}`)))));
  };

  const byType = new Map(req.documents.map((d) => [d.doc_type, d]));
  const claimed = new Set(SECTIONS.flatMap((s) => s.docs));
  const groups = SECTIONS.map((section) => {
    const docs = (section.catch_all
      ? req.documents.filter((d) => !claimed.has(d.doc_type))
      : section.docs.map((t) => byType.get(t)).filter(Boolean));
    return { ...section, docs };
  }).filter((g) => g.docs.length);

  // Done means every document in it has arrived, or is covered by the sibling that
  // satisfies the same requirement. Not "or is optional": that ticked every optional
  // section on an empty store, which is the one thing this panel exists not to do.
  const done = (g) => g.docs.every((d) => d.landed || satisfiedBy[d.doc_type]);
  const block = (g) => el("div", { class: `group ${done(g) ? "done" : "todo"}` },
    el("div", { class: "ghead" },
      el("span", { class: "gmark" }, done(g) ? "✓" : "○"),
      el("span", { class: "gtitle" }, g.title),
      el("span", { class: "gloss" }, g.gloss),
      g.docs.some((d) => d.required) ? null : el("span", { class: "tag" }, "optional")),
    el("ul", { class: "reqs" }, g.docs.map(line)));

  const outstanding = req.missing_required.length;
  const needed = groups.filter((g) => g.docs.some((d) => d.required));
  const extra = groups.filter((g) => !g.docs.some((d) => d.required));

  const extraBlock = el("div", { hidden: true },
    extra.map(block),
    el("ul", { class: "degrades" },
      req.degradations.slice(0, 6).map((d) => el("li", {}, d))));

  return el("section", { class: "card" },
    el("h2", {}, "What a run needs",
       el("span", { class: outstanding ? "badge todo" : "badge ok" },
          outstanding ? `${outstanding} still missing` : "all here")),
    el("p", { class: "sub" },
       outstanding
         ? "Greyed out means not uploaded yet. Each field ticks separately, so a column "
           + "this export does not carry shows up here without opening the mapping."
         : "Everything required is here. That is not a claim the numbers are right \u2014 "
           + "only that a run can start."),
    needed.map(block),
    extra.length
      ? el("p", {}, el("button", { class: "link", onclick: (e) => {
          extraBlock.hidden = !extraBlock.hidden;
          e.target.textContent = extraBlock.hidden
            ? `${extra.length} more sections are optional \u2014 see what a run loses without them`
            : "Hide the optional sections";
        } }, `${extra.length} more sections are optional \u2014 see what a run loses without them`))
      : null,
    extraBlock);
}


// The first thing on the page, because it is the only question the reader is qualified
// to answer without knowing anything about this pipeline.
function totalsCard(doc, batch, onDisagree) {
  const figures = [
    el("div", { class: "figure" }, el("div", { class: "n" }, num(doc.rows)),
       el("div", { class: "k" }, "rows")),
    el("div", { class: "figure" }, el("div", { class: "n" }, num(doc.skus)),
       el("div", { class: "k" }, "items")),
  ];
  for (const r of doc.readings) {
    const value = r.kind === "date"
      ? `${r.earliest ?? "—"} → ${r.latest ?? "—"}`
      : num(r.total, r.kind === "money" ? 2 : 0);
    figures.push(el("div", { class: "figure" },
      el("div", { class: "n", style: r.kind === "date" ? "font-size:15px" : null }, value),
      el("div", { class: "k" }, r.column),
      r.flags.map((f) => el("div", { class: "f" }, `⚠ ${f}`))));
  }

  return el("section", { class: "card ask" },
    el("h2", {}, `This file was read as ${batch.doc_type}`),
    el("p", { class: "sub" },
       "Here is what it says. If these do not look right, a column was read as the "
       + "wrong thing \u2014 far quicker to see here than in a mapping table."),
    el("div", { class: "figures" }, figures),
    el("div", { class: "answer" },
      el("button", { class: "primary", onclick: (e) => {
        mount(e.target.closest(".answer"),
          el("span", { class: "changed" },
             "Good. Nothing here has entered the planning run."));
      } }, "Looks right"),
      el("button", { onclick: onDisagree }, "Does not look right"),
      el("span", { class: "hint" }, "Opens the field table below, column by column.")));
}

// Shown only when the router did not win clearly. A document that won by 56 points is
// not worth a person's attention, and spending it there is why the close calls get
// skimmed.
function routingCard(res) {
  if (res.stated) {
    return el("section", { class: "card" },
      el("h2", {}, "The document type was stated, not judged"),
      el("p", { class: "sub" },
         `This file is handled as ${res.doc_type} because the template or the caller ` +
         `said so. Nothing was measured, and no independent check was made.`));
  }
  if (!res.close_call && !res.uncertain) return null;

  const rival = res.runner_up;
  // A margin under half a point rounds to "0% apart", which reads as a typo rather than
  // as the tie it is. A tie is the strongest form of this warning, not the weakest.
  const gap = res.margin < 0.005
    ? `${rival && rival.doc_type} scored exactly the same`
    : `it also looks like ${rival && rival.doc_type} \u2014 ${pct(res.margin)} apart`;
  return el("section", { class: "card warn" },
    el("h2", {}, "A close call \u2014 please confirm"),
    el("p", { class: "sub" },
       rival
         ? `On column names it looks most like ${res.doc_type}, but ${gap}. These two ` +
           `documents share nearly every column, so headers cannot separate them.`
         : `Judged on column names with only ${pct(res.confidence)} confidence.`),
    el("p", { class: "note" },
       "If this is the wrong one, every figure in the file is counted into the wrong "
       + "place and nothing raises an error. Confirm what this export actually is "
       + "before going on."));
}

// Every figure the run could not measure, largest exposure first. The currency case
// gets a dropdown rather than a default, because "what is this booked in" is business
// knowledge this reader has and a 7x error is not something they can spot afterwards.
function restingCard(resting, batch, refresh) {
  const items = resting.resting_on || [];
  if (!items.length) return null;

  const rows = items.map((item) => {
    const magnitude = item.money !== null && item.money !== undefined
      ? `${num(item.money)} ${resting.reporting_currency} (${item.money_field})`
      : item.priced_rows
        ? `${num(item.priced_rows)} rows priced \u2014 a rate, not a total`
        : item.qty !== null && item.qty !== undefined
          ? `${num(item.qty)} (${item.qty_field})`
          : "cannot be sized";
    return el("tr", { class: "attention" },
      el("td", {}, el("code", {}, item.field)),
      el("td", {}, String(item.value)),
      el("td", {}, item.basis === "declared"
        ? `declared by ${item.by || "nobody named"}` : "assumed"),
      el("td", { class: "num" }, `${item.upper_bound ? "\u2264 " : ""}${num(item.rows)} rows`),
      el("td", {}, magnitude));
  });

  const currency = items.find((i) => i.field === "currency" && i.basis === "assumed");
  return el("section", { class: "card warn" },
    el("h2", {}, "What these figures rest on"),
    el("p", { class: "sub" },
       "The export did not carry these, so a default was used. Where the default is "
       + "wrong, the money below is wrong by whatever the difference is."),
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {},
        ["Field", "Reading as", "Source", "Rows", "What rests on it"]
          .map((h) => el("th", {}, h)))),
      el("tbody", {}, rows))),
    currency ? currencyForm(batch, currency, refresh) : null);
}

function currencyForm(batch, item, refresh) {
  const select = el("select", { id: "cur" },
    el("option", { value: "" }, "Choose \u2014"),
    ["CNY", "USD", "EUR", "SGD", "JPY", "HKD", "GBP", "AUD"].map(
      (c) => el("option", { value: c }, c)));
  const reason = el("textarea", { placeholder:
    "Why this currency? e.g. the plant books at standard cost in CNY only, and the "
    + "export template carries no currency column." });
  const by = el("input", { placeholder: "Your name" });
  const out = el("p", { class: "note" });

  return el("form", { class: "declare", onsubmit: async (e) => {
    e.preventDefault();
    out.className = "note";
    if (!select.value) { out.textContent = "Choose a currency first."; return; }
    try {
      const body = await api(`/batches/${batch.batch_id}/declarations`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ scope: "value", field: "currency", value: select.value,
                               reason: reason.value, by: by.value }),
      });
      out.className = "changed";
      out.textContent = `Recorded as ${body.declaration.split("/").pop()} \u2014 `
        + `${body.changed.length} field(s) now read differently.`;
      refresh();
    } catch (err) { out.textContent = String(err.message); }
  } },
    el("label", {},
       `What currency is the money in this file booked in? Read as ${item.value} now.`),
    select,
    el("label", {},
       "Why (required \u2014 the only account of what this rested on)"), reason,
    el("div", { class: "row2" },
      el("div", {}, el("label", {}, "Who says so (required)"), by),
      el("div", {}, el("label", {}, " "),
         el("button", { class: "primary", type: "submit" }, "Record it"))),
    out);
}

// Collapsed to what needs attention. Opening the whole table is a click away and is
// what the "Does not look right" button does.
function fieldsCard(res, batch, refresh, openAll) {
  const needs = (f) => f.required || f.source === "absent" || f.empty;
  let showAll = openAll;

  const card = el("section", { class: "card" });
  const render = () => {
    const fields = showAll ? res.fields : res.fields.filter(needs);
    const rows = fields.map((f) => el("tr", { class: needs(f) && f.source !== "mapped" ? "attention" : null },
      el("td", {}, el("code", {}, f.field), f.required ? " *" : "",
         el("div", { class: "k", style: "color:var(--muted);font-size:12px" },
            f.description || "")),
      el("td", {}, el("span", { class: `tag ${f.source}` }, {
        mapped: "from a column", declared: "declared", default: "supplied default",
        derived: "computed", absent: "no column",
      }[f.source] || f.source)),
      el("td", {}, f.column || "—"),
      el("td", { class: "num" }, f.source === "absent" ? "—" : pct(f.fill_rate)),
      el("td", {}, el("button", { class: "link", onclick: () =>
        card.append(mappingForm(res, batch, f, refresh)) }, "use another column"))));

    mount(card,
      el("h2", {}, "Which column each field was read from"),
      el("p", { class: "sub" },
         showAll
           ? `All ${res.fields.length} fields. A * marks a required one.`
           : `Only the required ones and the ones that found no column. The other `
             + `${res.fields.length - fields.length} are fine.`),
      res.ignored_declarations.length
        ? el("p", { class: "note", style: "color:var(--stop)" },
             "\u26a0 A declaration names a column this file does not have, so it was "
             + "left unapplied: " +
             res.ignored_declarations.map((d) => `${d.field} ← ${d.column}`).join("、"))
        : null,
      el("div", { class: "scroll" }, el("table", {},
        el("thead", {}, el("tr", {},
          ["Field", "Source", "Column", "Filled", ""].map((h) => el("th", {}, h)))),
        el("tbody", {}, rows))),
      el("p", {}, el("button", { class: "link", onclick: () => { showAll = !showAll; render(); } },
        showAll ? "Show only what needs attention"
                : `Show all ${res.fields.length} fields`)),
      res.unmatched_columns.length
        ? el("p", { class: "note" },
             `${res.unmatched_columns.length} column(s) in the file matched no field: ` +
             res.unmatched_columns.join("、"))
        : null);
  };
  render();
  return card;
}

function mappingForm(res, batch, field, refresh) {
  const columns = [...new Set([...res.fields.map((f) => f.column).filter(Boolean),
                               ...res.unmatched_columns])].sort();
  const select = el("select", {}, el("option", { value: "" }, "Choose \u2014"),
    columns.map((c) => el("option", { value: c, selected: c === field.column }, c)));
  const reason = el("textarea", { placeholder: "Why is this the right column?" });
  const by = el("input", { placeholder: "Your name" });
  const out = el("p", { class: "note" });

  const form = el("form", { class: "declare", onsubmit: async (e) => {
    e.preventDefault();
    out.className = "note";
    if (!select.value) { out.textContent = "Choose a column first."; return; }
    try {
      const body = await api(`/batches/${batch.batch_id}/declarations`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ scope: "mapping", field: field.field, value: select.value,
                               reason: reason.value, by: by.value }),
      });
      out.className = "changed";
      out.textContent = body.changed.length
        ? `Recorded. Changed: ${body.changed.map((c) => c.field).join(", ")}`
        : "Recorded, and nothing changed \u2014 most likely the column is not in this "
          + "file. See the unapplied declaration noted above.";
      refresh();
    } catch (err) { out.textContent = String(err.message); }
  } },
    el("label", {}, `Which column should ${field.field} come from? `
       + `Currently ${field.column || "none"}.`),
    select,
    el("label", {}, "Why (required)"), reason,
    el("div", { class: "row2" },
      el("div", {}, el("label", {}, "Who says so (required)"), by),
      el("div", {}, el("label", {}, " "),
         el("button", { class: "primary", type: "submit" }, "Record it"))),
    out);
  return form;
}

function rowsCard(rows) {
  if (!rows.rows.length) return null;
  return el("section", { class: "card" },
    el("h2", {}, "The rows as the file wrote them"),
    el("p", { class: "sub" },
       `The first ${rows.rows.length} of ${num(rows.total)} rows, stored verbatim. `
       + `The headers are the file\u2019s own spelling.`),
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {}, rows.columns.map((c) => el("th", {}, c)))),
      el("tbody", {}, rows.rows.map((row) =>
        el("tr", {}, rows.columns.map((c) => el("td", {}, row[c] ?? ""))))))));
}

// ── Assembly ────────────────────────────────────────────────────────────────

async function showBatch(batch, container, openFields = false) {
  const [summary, resolution, rows] = await Promise.all([
    api(`/batches/${batch.batch_id}/summary`),
    api(`/batches/${batch.batch_id}/resolution`),
    api(`/batches/${batch.batch_id}/rows?limit=8`),
  ]);
  const refresh = () => { showRequirements(); return showBatch(batch, container, true); };
  const doc = summary.documents[0];

  mount(container,
    doc ? totalsCard(doc, batch, () => showBatch(batch, container, true)) : null,
    routingCard(resolution),
    restingCard(summary.resting_on, batch, refresh),
    fieldsCard(resolution, batch, refresh, openFields),
    rowsCard(rows));
}

// ── Quality gates ───────────────────────────────────────────────────────────
//
// The checkpoint that catches the failure this pipeline actually has: not a crash, but
// a complete report, correctly formatted, built on a join that matched nothing — and no
// error anywhere. That is visible here and invisible in the report it would go on to
// write, so it belongs on the screen of the person who just uploaded the file.
//
// Three severities, kept apart, because they ask for three different things. A BLOCK
// stops the run. A SEVERE does not, but means specific figures are not what they
// appear to be and the reader has to know which before acting on any of them. A WARN is
// context to note and move past. Collapsing them is how a page of eleven items gets
// skimmed and the one that mattered gets skimmed with it.
//
// And the page says what a clean result does *not* mean. This is one gate of four; the
// other three need a time series, a forecast and a position, none of which exist until
// the run has done the work. "Nothing at intake stops this" is the claim; "the run will
// pass" is not.

const SEVERITY = {
  block: { cls: "stop", label: "blocks the run", mark: "\u2717" },
  severe: { cls: "warn", label: "does not stop the run, but changes how to read it",
            mark: "!" },
  warn: { cls: "", label: "worth knowing", mark: "\u26a0" },
};

// A finding is waived on one check and one document, until a date. All three are
// required and the date is the one that matters: a waiver without an end is
// `allow_degraded` with extra steps, and it is how a checked pipeline becomes an
// unchecked one without anybody deciding to.
function waiverForm(finding, done) {
  const reason = el("input", { type: "text",
    placeholder: "why this check is a false positive on this document" });
  const by = el("input", { type: "text", placeholder: "who is declaring it" });
  const expires = el("input", { type: "date" });
  const out = el("div", {});

  const submit = async () => {
    try {
      const body = await api(`/gates/${encodeURIComponent(finding.check)}/waivers`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ doc_type: finding.doc_type, reason: reason.value,
                               by: by.value, expires: expires.value }),
      });
      done(body);
    } catch (err) {
      mount(out, el("p", { class: "status bad" }, String(err.message)));
    }
  };

  return el("form", { class: "declare",
                      onsubmit: (e) => { e.preventDefault(); submit(); } },
    el("p", { class: "note" },
       finding.doc_type
         ? `Waives ${finding.check} on ${finding.doc_type} only — every other document `
           + `is still checked.`
         : `This finding names no document, so the waiver covers the check everywhere. `
           + `It is the wide kind; prefer fixing the cause.`),
    el("div", { class: "row2" }, reason, by),
    el("label", {}, "review this by"), expires,
    el("div", {}, el("button", { type: "submit" }, "Declare it a false positive")),
    out);
}

function findingCard(finding, refresh) {
  const level = SEVERITY[finding.severity] || SEVERITY.warn;
  const drawer = el("div", { hidden: true });
  const waive = el("button", { class: "link", type: "button",
    onclick: () => {
      drawer.hidden = !drawer.hidden;
      mount(drawer, drawer.hidden ? [] : waiverForm(finding, refresh));
    } }, "Declare a false positive");

  return el("div", { class: "finding" },
    el("p", {},
       el("span", { class: `tag ${finding.severity === "block" ? "absent" : "default"}` },
          `${level.mark} ${finding.severity}`),
       " ", el("code", {}, finding.check),
       finding.doc_type ? el("span", { class: "k" }, ` \u00b7 ${finding.doc_type}`) : null),
    el("p", {}, finding.what),
    el("p", { class: "k" }, finding.why),
    (finding.impacts || []).length
      ? el("ul", { class: "impacts" },
           finding.impacts.map((i) => el("li", {}, i)))
      : null,
    el("p", { class: "k" }, `Fix: ${finding.fix}`),
    finding.waived
      ? el("p", { class: "note" },
           `Waived by ${finding.waived_by || "someone"} until ${finding.waived_until} `
           + `— still reported, no longer blocking.`)
      : (finding.severity === "block" ? waive : null),
    drawer);
}

function gatesCard(body, refresh) {
  const later = el("details", {},
    el("summary", {}, "Three more gates run during the plan"),
    el("ul", { class: "impacts" },
      (body.later_stages || []).map((g) =>
        el("li", {}, el("code", {}, g.stage), ` — ${g.checks}. Needs ${g.needs}, `
           + `which does not exist until the run builds it.`))));

  if (!body.ran) {
    return el("section", { class: "card" },
      el("h2", {}, "Quality gate"),
      el("p", { class: "sub" }, body.note),
      later);
  }

  const counts = body.counts || {};
  const worst = counts.block ? "stop" : (counts.severe ? "warn" : "");
  const head = counts.block
    ? `${counts.block} finding(s) would stop a run`
    : "Nothing at intake would stop a run";

  return el("section", { class: `card ${worst}` },
    el("h2", {}, "Quality gate",
       el("span", { class: `badge ${counts.block ? "todo" : "ok"}` },
          counts.block ? "blocked" : "clear")),
    el("p", { class: "sub" },
       `${head}, across ${(body.documents || []).length} document(s). `
       + `This is the intake checkpoint — it compares the documents against each other, `
       + `which is where a complete report of confident zeroes is caught. It is not a `
       + `promise about the run.`),
    body.findings.length
      ? el("div", {}, body.findings.map((f) => findingCard(f, refresh)))
      : el("p", { class: "note" },
           "No findings. The item numbers agree across the documents, the columns "
           + "carry what they claim, and the grouping dimensions are spelled one way."),
    later);
}

async function showGates() {
  const host = $("#gates");
  try {
    mount(host, gatesCard(await api("/gates"), () => { showGates(); }));
  } catch (err) {
    mount(host, el("p", { class: "status bad" }, String(err.message)));
  }
}

async function showRequirements() {
  const host = $("#requirements");
  try {
    mount(host, requirementsCard(await api("/requirements")));
  } catch (err) {
    mount(host, el("p", { class: "status bad" }, String(err.message)));
  }
}

async function upload(file) {
  const status = $("#status");
  const results = $("#results");
  status.hidden = false;
  status.className = "status";
  status.textContent = `Reading ${file.name} \u2026`;
  results.replaceChildren();

  const form = new FormData();
  form.append("file", file);
  try {
    const body = await api("/uploads", { method: "POST", body: form });
    status.textContent = `${body.source_name} \u2014 stored verbatim. `
      + `Nothing has entered the planning run.`;
    if (body.stale_template) {
      results.append(el("section", { class: "card warn" },
        el("h2", {}, "This template is out of date"),
        el("p", { class: "sub" }, body.stale_template)));
    }
    for (const landed of body.landed) {
      const section = el("div", {});
      results.append(section);
      await showBatch(landed, section);
    }
    await showRequirements();
    await showGates();
  } catch (err) {
    status.className = "status bad";
    status.textContent = String(err.message);
  }
}


export function mountReview() {
  showRequirements();
  showGates();
  const drop = $("#drop");
  $("#pick").addEventListener("click", () => $("#file").click());
  $("#file").addEventListener("change",
    (e) => e.target.files[0] && upload(e.target.files[0]));
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
  });
}
