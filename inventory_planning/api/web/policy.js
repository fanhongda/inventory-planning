// Policy and runs.
//
// There is no Save button on this screen and that is the design, not a stage of it.
// What a rule needs is review, a diff, a rationale and an owner, and markdown in git
// gives all four for nothing; a form writing rules into a database would have to rebuild
// every one of them. So the parameters are shown as they stand, with the file to edit
// named, and the screen spends its effort on the two things a text editor cannot show:
// which rule reached which SKUs, and what a change to one did to a run.
//
// The second of those is the whole point. A parameter change is not a state to inspect.
// It is the difference between two runs — and whether that difference is attributable to
// the change depends entirely on whether anything else moved at the same time, which is
// what `basis` answers.

import { $, el, mount, api } from "/ui.js";

const BASIS = {
  identical: "ok",
  scenario: "ok",
  new_data: "warn",
  mixed: "stop",
};

function kvTable(title, entries, { note = "" } = {}) {
  const rows = Object.entries(entries || {});
  if (!rows.length) return null;
  return el("section", { class: "card" },
    el("h2", {}, title),
    note ? el("p", { class: "sub" }, note) : null,
    el("div", { class: "scroll" }, el("table", {},
      el("tbody", {}, rows.map(([key, value]) =>
        el("tr", {},
          el("td", {}, el("code", {}, key)),
          el("td", {}, typeof value === "object"
            ? JSON.stringify(value) : String(value))))))));
}

function macroCard(body) {
  return el("section", { class: "card" },
    el("h2", {}, "Settings the engine reads"),
    el("p", { class: "sub" },
       `From ${body.config_dir}. Only what the pipeline actually consumes — a switch `
       + `nothing honours reads as a guarantee, which is worse than no switch.`),
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {},
        ["setting", "value", "from", ""].map((h) => el("th", {}, h)))),
      el("tbody", {}, body.settings.map((s) => el("tr", {},
        el("td", {}, el("code", {}, s.name)),
        el("td", {}, Array.isArray(s.value) ? s.value.join(", ") : String(s.value)),
        el("td", {}, s.source),
        el("td", { class: "k" }, s.note || "")))))));
}

function rulesCard(body) {
  return el("section", { class: "card" },
    el("h2", {}, "Rules in force",
       el("span", { class: "badge" }, `${body.rules.length}`)),
    el("p", { class: "sub" },
       `Read-only, from ${body.source}. Edit the file: a rule wants review, a diff, a `
       + `rationale and an owner, and version control gives all four.`),
    body.rules.length
      ? el("div", { class: "scroll" }, el("table", {},
          el("thead", {}, el("tr", {},
            ["rule", "applies to", "sets", "why", "owner"].map((h) => el("th", {}, h)))),
          el("tbody", {}, body.rules.map((r) => el("tr", {},
            el("td", {}, el("code", {}, r.rule_id),
               el("div", { class: "k" }, r.name)),
            el("td", {}, el("code", {}, r.scope)),
            el("td", {}, Object.entries(r.sets || {})
              .map(([k, v]) => `${k} = ${v}`).join("; ")),
            el("td", { class: "k" }, r.rationale),
            el("td", { class: "k" }, [r.owner, r.date].filter(Boolean).join(" · ")))))))
      : el("p", { class: "note" }, "No rules — every SKU takes the defaults below."),
    body.note ? el("p", { class: "note" }, body.note) : null);
}

function runsCard(runs, onDiff) {
  if (!runs.length) {
    return el("section", { class: "card" },
      el("h2", {}, "Runs"),
      el("p", { class: "sub" },
         "No runs recorded here. The registry lives under the output directory the "
         + "pipeline writes to; start the server with --output pointing at it."));
  }
  const pick = (id) => el("select", { id },
    el("option", { value: "" }, "—"),
    runs.map((r) => el("option", { value: r.run_id }, `${r.run_id}`)));
  const a = pick("run-a");
  const b = pick("run-b");
  if (runs[1]) { a.value = runs[1].run_id; }
  if (runs[0]) { b.value = runs[0].run_id; }

  return el("section", { class: "card" },
    el("h2", {}, "Runs", el("span", { class: "badge" }, `${runs.length}`)),
    el("p", { class: "sub" },
       "A parameter change is not a state to look at — it is the difference between two "
       + "runs, and whether that difference is the change depends on what else moved."),
    el("div", { class: "controls" },
      el("div", {}, el("label", {}, "baseline"), a),
      el("div", {}, el("label", {}, "compared with"), b),
      el("div", {}, el("label", {}, " "),
         el("button", { class: "primary",
                        onclick: () => a.value && b.value && onDiff(a.value, b.value) },
            "Compare"))),
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {},
        ["run", "at", "code", "facts", "parameters"].map((h) => el("th", {}, h)))),
      el("tbody", {}, runs.slice(0, 20).map((r) => el("tr", {},
        el("td", {}, el("code", {}, r.run_id)),
        el("td", {}, String(r.run_at || "").replace("T", " ")),
        el("td", {}, el("code", {}, String(r.git_sha || "").slice(0, 8))),
        el("td", {}, el("code", {}, String(r.input_fingerprint || "").slice(0, 8))),
        el("td", {}, el("code", {},
           `${String(r.config_fingerprint || "").slice(0, 8)}/`
           + `${String(r.policy_fingerprint || "").slice(0, 8)}`))))))));
}

function diffCard(body) {
  const cls = BASIS[body.basis] || "warn";
  return el("section", { class: `card ${cls === "ok" ? "" : cls}` },
    el("h2", {}, `${body.a} → ${body.b}`,
       el("span", { class: `badge ${cls === "ok" ? "ok" : "todo"}` }, body.basis)),
    el("p", { class: "sub" }, body.describe),
    el("p", { class: "note" },
       body.moved.length
         ? `Moved: ${body.moved.join(", ")}.`
         : "Nothing moved: same facts, same parameters, same code."));
}

// ── Assembly ────────────────────────────────────────────────────────────────

export async function mountPolicy() {
  const host = $("#policy-body");
  mount(host, el("p", { class: "note" }, "Loading…"));
  try {
    const [policy, macro, runs] = await Promise.all([
      api("/policy"), api("/policy/macro"), api("/runs"),
    ]);
    const diffHost = el("div", {});
    const onDiff = async (a, b) => {
      try {
        mount(diffHost, diffCard(await api(`/runs/${a}/diff/${b}`)));
      } catch (err) {
        mount(diffHost, el("p", { class: "status bad" }, String(err.message)));
      }
    };
    mount(host,
      macroCard(macro),
      kvTable("Conventions", policy.conventions, {
        note: "How the should-be inventory is defined. Changing one of these changes "
              + "every figure the run produces.",
      }),
      kvTable("Defaults", policy.defaults, {
        note: "What a SKU gets when no rule matches it.",
      }),
      kvTable("Segmentation", policy.segmentation, {
        note: "The band boundaries the rules are written against.",
      }),
      rulesCard(policy),
      runsCard(runs.runs, onDiff),
      diffHost);
  } catch (err) {
    mount(host, el("p", { class: "status bad" }, String(err.message)));
  }
}
