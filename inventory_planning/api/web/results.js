// What a run produced.
//
// This screen exists because INTERFACE.md §7 revised the argument against it. The
// objection was that a page showing the same numbers becomes a second place they are
// formatted, rounded and subtly disagreed about. That holds only if the page
// *re-derives* them. It does not if the page renders the run's own workbook, located
// from the manifest that already records every output the run wrote. One artefact, two
// renderings.
//
// So the constraint, and it is the whole design: **this screen computes nothing.** No
// totals, no percentages, no re-sorting, no filling of blanks, no "top 10". Every figure
// on it is a cell out of the file, displayed under the file's own number format. The
// moment something here is calculated, the original objection returns in full and this
// page should be deleted rather than argued for.
//
// What it adds over opening the workbook is the two things a file cannot carry: which
// run it came from and what that run was resting on, and the rest of the run's outputs
// beside it — the health note and the gate findings that a planner opening only the
// xlsx never sees.

import { $, el, mount, api } from "/ui.js";

const KIND_NOTE = {
  workbook: "the five-sheet workbook — this is what gets handed round a meeting",
  text: "written by the run, shown verbatim",
  download: "not rendered here; open it in whatever reads it",
};

function runLine(runs, current, onPick) {
  const pick = el("select", {},
    runs.map((r) => el("option", { value: r.run_id }, r.run_id)));
  pick.value = current;
  pick.addEventListener("change", () => onPick(pick.value));
  const chosen = runs.find((r) => r.run_id === current) || {};

  return el("section", { class: "card" },
    el("h2", {}, "Run"),
    el("div", { class: "controls" },
      el("div", {}, el("label", {}, "showing"), pick)),
    el("p", { class: "sub" },
       `${String(chosen.run_at || "").replace("T", " ")} · code `
       + `${String(chosen.git_sha || "").slice(0, 8)} · facts `
       + `${String(chosen.input_fingerprint || "").slice(0, 8)} · parameters `
       + `${String(chosen.config_fingerprint || "").slice(0, 8)}/`
       + `${String(chosen.policy_fingerprint || "").slice(0, 8)}`),
    el("p", { class: "note" },
       "Which run these figures belong to is the thing the file itself cannot tell "
       + "you once it is in somebody's downloads folder. Two runs differ by facts, "
       + "parameters or code, and the Policy screen says which."));
}

// One sheet as a table. Paged, because these run to thousands of rows and a page that
// fetched all of them would be slower than opening the file in Excel — which is the
// thing it is meant to save, not replace.
function sheetView(runId, name, sheet, onPage) {
  const from = sheet.offset + 1;
  const to = sheet.offset + sheet.rows.length;
  const pager = (delta, label, enabled) =>
    el("button", { type: "button", disabled: !enabled || null,
                   onclick: () => onPage(Math.max(0, sheet.offset + delta)) }, label);

  return el("div", {},
    el("p", { class: "sub" },
       sheet.total
         ? `Rows ${from}–${to} of ${sheet.total}, ${sheet.columns.length} columns. `
           + `Values are the workbook's own, shown under its own number formats.`
         : "This sheet is empty."),
    sheet.unformatted.length
      ? el("p", { class: "note" },
           `Shown unformatted, because the workbook uses a number format this screen `
           + `does not render: ${sheet.unformatted.join(", ")}. The values are right; `
           + `only their display differs from the file.`)
      : null,
    el("div", { class: "scroll" }, el("table", { class: "sheet" },
      el("thead", {}, el("tr", {},
        sheet.columns.map((c) => el("th", {}, c)))),
      el("tbody", {}, sheet.rows.map((row) => el("tr", {},
        row.map((cell) => el("td", {},
          cell === null || cell === undefined ? "" : String(cell)))))))),
    sheet.total > sheet.rows.length
      ? el("div", { class: "controls" },
          el("div", {}, pager(-sheet.rows.length, "← earlier", sheet.offset > 0)),
          el("div", {}, pager(sheet.rows.length, "later →", to < sheet.total)))
      : null);
}

async function workbookCard(runId, output, host) {
  const body = await api(`/runs/${runId}/outputs/`
                         + `${encodeURIComponent(output.name)}/sheets`);
  const panel = el("div", {});
  const tabs = el("div", { class: "controls" });

  const load = async (sheet, offset = 0) => {
    for (const button of tabs.querySelectorAll("button")) {
      button.classList.toggle("on", button.textContent === sheet);
    }
    mount(panel, el("p", { class: "note" }, "Reading…"));
    try {
      const data = await api(
        `/runs/${runId}/outputs/${encodeURIComponent(output.name)}`
        + `/sheets/${encodeURIComponent(sheet)}?offset=${offset}&limit=100`);
      mount(panel, sheetView(runId, output.name, data,
                             (next) => load(sheet, next)));
    } catch (err) {
      mount(panel, el("p", { class: "status bad" }, String(err.message)));
    }
  };

  mount(tabs, body.sheets.map((s) =>
    el("button", { class: "tab", type: "button", onclick: () => load(s.name) },
       s.name)));

  mount(host,
    el("h2", {}, output.name,
       el("span", { class: "badge" }, `${body.sheets.length} sheets`)),
    el("p", { class: "sub" }, KIND_NOTE.workbook),
    el("p", { class: "note" },
       el("a", { href: `/runs/${runId}/outputs/`
                       + `${encodeURIComponent(output.name)}/download` },
          "Download the file")),
    tabs, panel);

  if (body.sheets.length) load(body.sheets[0].name);
}

async function textCard(runId, output, host) {
  const body = await api(`/runs/${runId}/outputs/`
                         + `${encodeURIComponent(output.name)}/text`);
  mount(host,
    el("h2", {}, output.name),
    el("p", { class: "sub" }, KIND_NOTE.text),
    el("pre", { class: "diff" }, body.text),
    body.truncated ? el("p", { class: "note" }, "Truncated — download it for the rest.")
                   : null,
    el("p", { class: "note" },
       el("a", { href: `/runs/${runId}/outputs/`
                       + `${encodeURIComponent(output.name)}/download` },
          "Download the file")));
}

function fileRow(runId, output) {
  return el("tr", { class: output.present ? null : "attention" },
    el("td", {}, el("code", {}, output.name)),
    el("td", {}, output.bytes ? `${Math.ceil(output.bytes / 1024)} kB` : ""),
    el("td", { class: "k" }, output.present ? KIND_NOTE[output.kind] || ""
                                            : "recorded by the run, no longer on disk"),
    el("td", {}, output.present
      ? el("a", { href: `/runs/${runId}/outputs/`
                        + `${encodeURIComponent(output.name)}/download` }, "Download")
      : ""));
}

async function showRun(runId, runs, onPick) {
  const host = $("#results-body");
  mount(host, el("p", { class: "note" }, "Reading…"));
  const body = await api(`/runs/${runId}/outputs`);

  const panels = [];
  for (const output of body.outputs) {
    if (!output.present) continue;
    if (output.kind === "workbook" || output.kind === "text") {
      panels.push({ output, host: el("section", { class: "card" }) });
    }
  }

  mount(host,
    runLine(runs, runId, onPick),
    el("section", { class: "card" },
      el("h2", {}, "Files this run wrote",
         el("span", { class: "badge" }, `${body.outputs.length}`)),
      el("p", { class: "sub" }, body.note),
      el("div", { class: "scroll" }, el("table", {},
        el("thead", {}, el("tr", {},
          ["file", "size", "", ""].map((h) => el("th", {}, h)))),
        el("tbody", {}, body.outputs.map((o) => fileRow(runId, o)))))),
    panels.map((p) => p.host));

  for (const { output, host: panel } of panels) {
    const render = output.kind === "workbook" ? workbookCard : textCard;
    render(runId, output, panel).catch((err) =>
      mount(panel, el("h2", {}, output.name),
            el("p", { class: "status bad" }, String(err.message))));
  }
}

export async function mountResults() {
  const host = $("#results-body");
  mount(host, el("p", { class: "note" }, "Loading…"));
  try {
    const { runs } = await api("/runs");
    if (!runs.length) {
      mount(host, el("section", { class: "card" },
        el("h2", {}, "No runs here"),
        el("p", { class: "sub" },
           "The run registry lives under the output directory the pipeline writes to. "
           + "Start the server with --output pointing at it, or run the pipeline.")));
      return;
    }
    const pick = (id) => showRun(id, runs, pick).catch((err) =>
      mount(host, el("p", { class: "status bad" }, String(err.message))));
    await pick(runs[0].run_id);
  } catch (err) {
    mount(host, el("p", { class: "status bad" }, String(err.message)));
  }
}
