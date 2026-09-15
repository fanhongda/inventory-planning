// Policy and runs.
//
// There is no Save button on this screen yet. What a rule needs is review, a diff, a
// rationale and an owner, and markdown in git gives all four for nothing — which argued
// for storing rules in the file, and was then allowed to decide who may operate them,
// which it does not (INTERFACE.md §7). Editing is the next piece; it will edit the file
// and keep all four. For now the parameters are shown as they stand, with the file to
// edit named, and the screen spends its effort on the two things a text editor cannot
// show: which rule reached which SKUs, and what a change to one did to a run.
//
// The first of those comes from a run and is labelled with the run it came from. It is
// not a property of the rule — the same rule reaches a different number of SKUs next
// week — so a count with no run behind it would be a number pretending to be a fact.
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

// The field for one setting. A `choice` gets a dropdown rather than a text box
// because these are the branches the engine takes and a typo in one of them parses
// cleanly, loads cleanly, and quietly sends every SKU down the other path.
function valueField(s) {
  if (s.kind === "choice") {
    const select = el("select", {},
      s.choices.map((c) => el("option", { value: c }, c)));
    select.value = String(s.value);
    return select;
  }
  return el("input", { type: s.kind === "number" ? "number" : "text",
                       step: "any", value: String(s.value) });
}

// Edit in two steps, never one: propose, read the diff, then apply. The Apply button
// does not exist until a diff has been produced, and it carries that diff's `basis` —
// the digest of the file it was made against — so what gets approved is the change
// that was shown and not merely the thing it was shown for.
//
// One implementation for both editors. A scalar and a rule differ in what they send and
// in what is worth saying above the diff; they do not differ in the contract, and two
// copies of a propose-then-approve loop would be two places for the Apply button to
// stop carrying its basis.
function changeForm({ endpoint, change, fields, describe, done, submit = "Show the diff",
                      confirm = "Apply this change" }) {
  const reason = el("input", { type: "text",
                               placeholder: "why — recorded with the change" });
  const by = el("input", { type: "text", placeholder: "who is making it" });
  // Two panels, not one. The diff and the complaint about the form are different
  // things, and an earlier version wrote both to the same element — so "you forgot to
  // say who you are" wiped out the diff the person was about to approve and made them
  // produce it again.
  const panel = el("div", {});
  const problem = el("div", {});
  let shown = null;

  const say = (cls, text) => mount(problem, el("p", { class: cls }, text));
  const put = (extra) => api(endpoint, {
    method: "PUT", headers: { "content-type": "application/json" },
    body: JSON.stringify({ ...change(), ...extra }),
  });

  const applyIt = async () => {
    if (!shown) { say("status bad", "Show the diff first."); return; }
    if (!reason.value.trim() || !by.value.trim()) {
      say("status bad", "A reason and a name are required — they are the whole record "
                        + "of why this moved, and the file records what it says, never "
                        + "why it says it.");
      return;
    }
    try {
      done(await put({ apply: true, reason: reason.value, by: by.value,
                       basis: shown.basis }));
    } catch (err) { say("status bad", String(err.message)); }
  };

  const preview = async () => {
    // The Apply button belongs to one diff. Clearing it before asking for the next one
    // means there is never a button on screen that would apply something other than
    // what is displayed above it.
    shown = null;
    mount(panel);
    mount(problem);
    try {
      const body = await put({});
      if (body.unchanged) {
        say("note", "The file already says this — nothing to change.");
        return;
      }
      shown = body;
      mount(panel,
        describe ? describe(body) : null,
        el("pre", { class: "diff" }, body.diff),
        // `type: button` and not the default. A button inside a form submits it, so
        // Apply was also re-running the preview — which cleared the very message it
        // had just written about the missing reason.
        el("button", { class: "primary", type: "button", onclick: applyIt }, confirm));
    } catch (err) { say("status bad", String(err.message)); }
  };

  return el("form", { class: "declare",
                      onsubmit: (e) => { e.preventDefault(); preview(); } },
    fields,
    el("div", { class: "row2" }, reason, by),
    el("div", {}, el("button", { type: "submit" }, submit)),
    panel, problem);
}

function editor(s, done) {
  const value = valueField(s);
  return changeForm({
    endpoint: "/policy/macro",
    change: () => ({ name: s.name, value: value.value }),
    fields: [el("label", {}, `New value for ${s.name}`), value],
    describe: (body) => body.impact
      ? el("p", { class: "note" }, `What this moves: ${body.impact}`) : null,
    done: (body) => done(`${body.name}: ${body.from} \u2192 ${body.to}. `
                         + `Recorded in ${body.recorded_in}.`),
  });
}

function macroCard(body, reload) {
  const rows = body.settings.map((s) => {
    const row = el("tr", {},
      el("td", {}, el("code", {}, s.name)),
      el("td", {}, Array.isArray(s.value) ? s.value.join(", ") : String(s.value)),
      el("td", {}, s.source),
      el("td", { class: "k" }, s.note || ""),
      el("td", {}));
    if (!s.editable) return [row];
    const host = el("td", { colspan: "5" });
    const drawer = el("tr", { hidden: true }, host);
    row.lastChild.append(el("button", { class: "link",
      onclick: () => {
        drawer.hidden = !drawer.hidden;
        mount(host, drawer.hidden ? [] : editor(s, reload));
      } }, "Edit"));
    return [row, drawer];
  });

  return el("section", { class: "card" },
    el("h2", {}, "Settings the engine reads"),
    el("p", { class: "sub" },
       `From ${body.config_dir}. Only what the pipeline actually consumes — a switch `
       + `nothing honours reads as a guarantee, which is worse than no switch.`),
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {},
        ["setting", "value", "from", "", ""].map((h) => el("th", {}, h)))),
      el("tbody", {}, rows.flat()))),
    el("p", { class: "note" },
       "Editing writes the file, which stays the store — the form is a validator with "
       + "a nicer keyboard. A derived reading has no Edit because it is not a setting."),
    changesCard(body.changes, "Settings changed here"));
}

// Who moved what, and why. The diff is not kept here: it is in the file's own history,
// and the reason is the part that is nowhere else.
//
// One table for both editors, because the question is the same one. What differs is
// what "what" means — a setting's old and new value, or a rule and what was done to it
// — so that column is rendered from whichever the entry carries.
function changesCard(changes, title = "Changes made here") {
  if (!(changes || []).length) return null;
  const what = (c) => c.setting
    ? el("td", {}, el("code", {}, c.setting),
         el("div", { class: "k" }, `${JSON.stringify(c.from)} \u2192 `
            + `${JSON.stringify(c.to)}`))
    : el("td", {}, el("code", {}, c.rule_id),
         el("div", { class: "k" }, c.action || "edit"));

  return el("details", { class: "scroll" },
    el("summary", {}, `${title} (${changes.length})`),
    el("table", {},
      el("thead", {}, el("tr", {},
        ["when", "what", "who", "why"].map((h) => el("th", {}, h)))),
      el("tbody", {}, changes.map((c) => el("tr", {},
        el("td", {}, String(c.at || "").replace("T", " ")),
        what(c),
        el("td", {}, c.by),
        el("td", { class: "k" }, c.reason))))));
}

// What one rule reached, as a cell. The three silences are kept apart on purpose:
// a rule nobody has run, a rule whose scope names a column the run did not have, and a
// rule that ran and matched nothing are three different problems, and only the last one
// is an invitation to go and rewrite the scope.
function reachCell(hit) {
  if (!hit) return el("td", { class: "k" }, "—");
  if ((hit.unavailable_columns || []).length) {
    return el("td", { class: "k" },
      el("span", { class: "tag default" }, "skipped"),
      el("div", { class: "k" }, `scope needs ${hit.unavailable_columns.join(", ")}`));
  }
  if (!hit.matched) {
    return el("td", { class: "k" },
      el("span", { class: "tag default" }, "0 SKUs"),
      el("div", { class: "k" }, "scope may be wrong"));
  }
  const standing = hit.effective;
  const taken = standing !== null && standing !== undefined && standing < hit.matched;
  return el("td", {},
    `${hit.matched} SKUs`,
    (hit.sample_skus || []).length
      ? el("div", { class: "k" }, hit.sample_skus.slice(0, 3).join(", ")
          + (hit.matched > 3 ? ` +${hit.matched - hit.sample_skus.slice(0, 3).length}` : ""))
      : null,
    taken
      ? el("div", { class: "k" }, standing
          ? `${standing} still standing — later rules took the rest`
          : "every value later overridden — this rule decides nothing")
      : null);
}

// A rule that ran and changed nothing: either its scope selected no SKU, or every
// value it set was taken back by a later rule. Both are worth finding without reading
// down the table; a rule skipped for a missing column is not, since the file is fine
// and the run was short of a column.
function needsAttention(hit) {
  return Boolean(hit) && !(hit.unavailable_columns || []).length
    && (!hit.matched || hit.effective === 0);
}

// `set` as a form: one line per parameter, `name = value`. Not a JSON textarea, which
// would put the file's syntax in front of someone who was given a form precisely so
// they would not have to meet it — and not a fixed list of parameters either, because
// the rule engine takes whatever the pipeline reads and a fixed list goes stale.
function setField(sets) {
  const box = el("textarea", {});
  box.value = Object.entries(sets || {}).map(([k, v]) => `${k} = ${v}`).join("\n");
  return box;
}

function parseSet(text) {
  const out = {};
  for (const line of String(text).split("\n")) {
    if (!line.trim()) continue;
    const at = line.indexOf("=");
    if (at < 0) throw new Error(`"${line.trim()}" is not \`name = value\``);
    const value = line.slice(at + 1).trim();
    // Numbers as numbers: `0.98` written into YAML as the string "0.98" loads as a
    // string, and the arithmetic downstream would compare it to a float and never match.
    out[line.slice(0, at).trim()] = /^-?\d+(\.\d+)?$/.test(value)
      ? Number(value) : value;
  }
  return out;
}

function ruleEditor(rule, done) {
  const name = el("input", { type: "text", value: rule.name || "" });
  const scope = el("input", { type: "text", value: rule.scope || "" });
  const sets = setField(rule.sets);
  const rationale = el("textarea", {});
  rationale.value = rule.rationale || "";
  const owner = el("input", { type: "text", value: rule.owner || "" });

  return changeForm({
    endpoint: "/policy/rules",
    change: () => ({
      action: "edit", rule_id: rule.rule_id,
      changes: { name: name.value, scope: scope.value, set: parseSet(sets.value),
                 rationale: rationale.value, owner: owner.value },
    }),
    fields: [
      el("label", {}, "name"), name,
      el("label", {}, "applies to (scope)"), scope,
      el("label", {}, "sets — one `name = value` per line"), sets,
      el("label", {}, "why this rule exists"), rationale,
      el("label", {}, "owner"), owner,
    ],
    describe: ruleImpact,
    done: (body) => done(`${body.rule_id} updated. Recorded in ${body.recorded_in}.`),
  });
}

function ruleRemover(rule, done) {
  return changeForm({
    endpoint: "/policy/rules",
    change: () => ({ action: "remove", rule_id: rule.rule_id }),
    fields: el("p", { class: "note" },
       `Removes ${rule.rule_id} from the file. Its rationale goes with it, so the `
       + `reason below is the only record left of why it stopped applying.`),
    describe: ruleImpact,
    submit: "Show what removing it does",
    confirm: `Remove ${rule.rule_id}`,
    done: (body) => done(`${body.rule_id} removed. Recorded in ${body.recorded_in}.`),
  });
}

function ruleAdder(done) {
  const id = el("input", { type: "text", placeholder: "R-005" });
  const name = el("input", { type: "text", placeholder: "what this rule is for" });
  const scope = el("input", { type: "text", placeholder: 'abc_class == "A"' });
  const sets = el("textarea", { placeholder: "review_period_days = 7" });
  const rationale = el("textarea", {});
  const owner = el("input", { type: "text" });

  return changeForm({
    endpoint: "/policy/rules",
    change: () => ({
      action: "add",
      rule: { rule_id: id.value, name: name.value, scope: scope.value,
              set: parseSet(sets.value), rationale: rationale.value,
              owner: owner.value },
    }),
    fields: [
      el("div", { class: "row2" }, id, name),
      el("label", {}, "applies to (scope)"), scope,
      el("label", {}, "sets — one `name = value` per line"), sets,
      el("label", {}, "why this rule exists"), rationale,
      el("label", {}, "owner"), owner,
    ],
    describe: ruleImpact,
    submit: "Show the new rule",
    confirm: "Add this rule",
    done: (body) => done(`${body.rule_id} added. Recorded in ${body.recorded_in}.`),
  });
}

// What sits above the diff for a rule: where it lands in the order, and what the counts
// beside it are worth after this. Both are things a person cannot read off the diff.
function ruleImpact(body) {
  return el("div", {},
    body.note ? el("p", { class: "note" }, body.note) : null,
    (body.order || []).length
      ? el("p", { class: "note" },
           `Order after this — later wins: ${body.order.join(" → ")}`)
      : null);
}

function rulesCard(body, reload) {
  const hits = body.hits;
  const reach = (hits && hits.rules) || null;
  const drawers = [];

  const drawerFor = (build) => {
    const host = el("td", { colspan: "7" });
    const row = el("tr", { hidden: true }, host);
    drawers.push(row);
    return { host, row, open: () => {
      const wasHidden = row.hidden;
      // One drawer at a time. Two open editors on one table is two diffs on screen,
      // each with an Apply button, and the pair of them are not comparable — they were
      // made against the same bytes and only the first to land stays valid.
      for (const other of drawers) { other.hidden = true; other.firstChild.replaceChildren(); }
      row.hidden = !wasHidden;
      if (!row.hidden) mount(host, build());
    } };
  };

  const rows = body.rules.map((r) => {
    const edit = drawerFor(() => ruleEditor(r, reload));
    const remove = drawerFor(() => ruleRemover(r, reload));
    const row = el("tr",
      { class: needsAttention(reach && reach[r.rule_id]) ? "attention" : null },
      el("td", {}, el("code", {}, r.rule_id),
         el("div", { class: "k" }, r.name)),
      el("td", {}, el("code", {}, r.scope)),
      el("td", {}, Object.entries(r.sets || {})
        .map(([k, v]) => `${k} = ${v}`).join("; ")),
      reach ? reachCell(reach[r.rule_id]) : el("td", { class: "k" }, "—"),
      el("td", { class: "k" }, r.rationale),
      el("td", { class: "k" }, [r.owner, r.date].filter(Boolean).join(" · ")),
      el("td", {},
         el("button", { class: "link", type: "button", onclick: edit.open }, "Edit"),
         el("button", { class: "link", type: "button", onclick: remove.open },
            "Remove")));
    return [row, edit.row, remove.row];
  });

  const adder = el("div", {});
  const addButton = el("button", { type: "button",
    onclick: () => mount(adder, adder.firstChild ? [] : ruleAdder(reload)) },
    "Add a rule");

  return el("section", { class: "card" },
    el("h2", {}, "Rules in force",
       el("span", { class: "badge" }, `${body.rules.length}`)),
    el("p", { class: "sub" },
       `From ${body.source}, which stays the store. A change is proposed as a diff, `
       + `applied with a reason and a name, and written back to the file — the rule's `
       + `own rationale, owner and date move with it.`),
    body.rules.length
      ? el("div", { class: "scroll" }, el("table", {},
          el("thead", {}, el("tr", {},
            ["rule", "applies to", "sets", "reached", "why", "owner", ""]
              .map((h) => el("th", {}, h)))),
          el("tbody", {}, rows.flat())))
      : el("p", { class: "note" }, "No rules — every SKU takes the defaults below."),
    el("div", {}, addButton, adder),
    el("p", { class: "note" },
       "Rules apply top to bottom and a later one wins, so a new rule goes last — "
       + "which is where an exception to everything above it belongs."),
    body.note ? el("p", { class: "note" }, body.note) : null,
    changesCard(body.changes, "Rule changes made here"));
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

export async function mountPolicy(applied = "") {
  const host = $("#policy-body");
  mount(host, el("p", { class: "note" }, "Loading…"));
  try {
    const [policy, macro, runs] = await Promise.all([
      api("/policy"), api("/policy/macro"), api("/runs"),
    ]);
    // After a change, everything on this page is re-read from the files rather than
    // patched in place. The screen's whole claim is that it shows what the engine will
    // read, and a value updated in the DOM would be the interface telling itself what
    // it just did instead of asking.
    const reload = (what) => mountPolicy(what);
    const diffHost = el("div", {});
    const onDiff = async (a, b) => {
      try {
        mount(diffHost, diffCard(await api(`/runs/${a}/diff/${b}`)));
      } catch (err) {
        mount(diffHost, el("p", { class: "status bad" }, String(err.message)));
      }
    };
    mount(host,
      applied ? el("p", { class: "changed" }, applied) : null,
      macroCard(macro, reload),
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
      rulesCard(policy, reload),
      runsCard(runs.runs, onDiff),
      diffHost);
  } catch (err) {
    mount(host, el("p", { class: "status bad" }, String(err.message)));
  }
}
