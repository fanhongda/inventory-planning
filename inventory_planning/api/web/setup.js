// Setup: what this run needs from the planner.
//
// This screen exists because the settings it shows already existed and nobody filled
// them in. `/policy/macro` lists every scalar the engine reads, correctly grouped by
// how often each is a decision — and then renders `location_name`, which has a value
// and does not matter, in the same typeface as a target that has no value and does.
// Everything on the page looks configured, so nothing on it looks asked. A planner
// reading it has no way to tell the two states apart at a glance, and the two settings
// that most change what the run says were the ones sitting blank.
//
// Three things change here, and none of them is decoration:
//
//   **The question is the heading.** Not the key name. "What does each storage-location
//   code mean?" is answerable by someone who has never read this codebase;
//   `location_status.json` is not, and a form whose labels are identifiers is a form
//   only its author can fill in.
//
//   **What it costs to leave it unset is stated beside it**, because that is the whole
//   of why anyone would stop and answer. "No frontier is computed" is a consequence;
//   "target — changes what is recommended" is a category.
//
//   **Settled items shrink.** The opposite of the page this replaces: what is done
//   gets one line, and what is outstanding gets the room. A checklist whose completed
//   rows are as loud as its open ones is a checklist nobody finishes.
//
// What this screen deliberately does not do is block. One item blocks — the location
// map, because an undeclared code puts a wrong *number* in the position — and the run
// itself enforces that, here and in a headless run alike. The two targets do not:
// `targets.json` says in as many words that null is a legitimate state, and stopping a
// first-time planner at the door over a question whose honest answer is "we have not
// decided yet" teaches them that this screen is an obstacle rather than a tool.

import { $, el, mount, num, pct, api } from "/ui.js";
import { changeForm } from "/policy.js";

// The statuses a location code may be declared as, fetched with the rest rather than
// hardcoded here: the vocabulary is a decision in `node.py` and a copy of it in the
// browser would drift the first time one is added.
function statusSelect(vocabulary, current) {
  const select = el("select", {},
    el("option", { value: "" }, "— not declared —"),
    vocabulary.map((v) => el("option", { value: v.name },
                             `${v.name}${v.nets ? " (counts)" : ""}`)));
  select.value = current || "";
  return select;
}

// The codes the stock export actually carries, with the stock behind each.
//
// Asked this way round on purpose. "Declare your storage locations" is a chore with no
// obvious end; "02 holds 14,500 units and nothing says what it is" is a question with
// an answer, and the quantity is what makes it worth answering now rather than later.
function locationFields(detail) {
  const vocabulary = detail.vocabulary || [];
  const observed = detail.observed || [];
  const rows = [];
  const inputs = new Map();

  for (const o of observed) {
    const select = statusSelect(vocabulary, o.status);
    inputs.set(o.code, select);
    rows.push(el("tr", { class: o.status ? "" : "attention" },
      el("td", {}, el("code", {}, o.code)),
      el("td", { class: "num" }, num(o.qty_on_hand)),
      el("td", { class: "num" }, num(o.rows)),
      el("td", {}, select)));
  }

  // Somewhere to answer when nothing has been uploaded yet, and somewhere to add a
  // code that this extract happens not to carry — a quarantine location is empty most
  // of the time, and the run that finds it full is not the run to be declaring it on.
  const extraCode = el("input", { type: "text", placeholder: "code, e.g. 02" });
  const extraStatus = statusSelect(vocabulary, "");
  const wildcard = statusSelect(vocabulary, detail.wildcard);

  const table = observed.length
    ? el("div", { class: "scroll" }, el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "Code"), el("th", { class: "num" }, "On hand"),
          el("th", { class: "num" }, "Rows"), el("th", {}, "Means"))),
        el("tbody", {}, rows)))
    : el("p", { class: "note" },
         "No stock export has been uploaded yet, so there are no codes to show. "
         + "Declare them here, or come back once a file has landed and this will "
         + "list the codes it carries with the stock behind each.");

  // Codes declared earlier that this extract does not carry. Shown rather than
  // dropped: submitting the form rewrites the whole map, so a code left off the screen
  // would be silently withdrawn by a planner who only came to answer a new one.
  const seen = new Set(observed.map((o) => o.code));
  const carried = Object.entries(detail.statuses || {}).filter(([c]) => !seen.has(c));
  for (const [code, status] of carried) {
    const select = statusSelect(vocabulary, status);
    inputs.set(code, select);
  }

  const fields = [
    table,
    carried.length
      ? el("details", {}, el("summary", {},
          `${carried.length} code(s) declared earlier and not in this extract`),
          el("div", { class: "scroll" }, el("table", {}, el("tbody", {},
            carried.map(([code]) => el("tr", {},
              el("td", {}, el("code", {}, code)),
              el("td", {}, inputs.get(code))))))))
      : null,
    el("label", {}, "A code this extract does not carry"),
    el("div", { class: "row2" }, extraCode, extraStatus),
    el("label", {}, "Every other code (the “*” wildcard)"),
    wildcard,
    el("p", { class: "note" },
       "A code named above wins over the wildcard, so “everything sellable except "
       + "inspection” is the wildcard plus one row. Answering by wildcard is "
       + "reported on every run that uses it."),
  ];

  const collect = () => {
    const statuses = {};
    for (const [code, select] of inputs) {
      if (select.value) statuses[code] = select.value;
    }
    if (extraCode.value.trim() && extraStatus.value) {
      statuses[extraCode.value.trim().toUpperCase()] = extraStatus.value;
    }
    if (wildcard.value) statuses["*"] = wildcard.value;
    return statuses;
  };

  return { fields, collect };
}

function locationEditor(item, reload) {
  const { fields, collect } = locationFields(item.detail);
  return changeForm({
    endpoint: "/setup/locations",
    change: () => ({ statuses: collect() }),
    fields,
    describe: (body) => {
      const parts = [];
      if (body.added.length) parts.push(`declares ${body.added.join(", ")}`);
      if (body.changed.length) parts.push(`changes ${body.changed.join(", ")}`);
      if (body.removed.length) parts.push(`withdraws ${body.removed.join(", ")}`);
      return parts.length
        ? el("p", { class: "note" }, `This ${parts.join("; ")}.`) : null;
    },
    done: () => reload(),
    submit: "Show the diff",
    confirm: "Declare these locations",
  });
}

function macroEditor(item, reload) {
  const field = el("input", { type: "number", step: "any",
                              value: item.value === null ? "" : String(item.value) });
  return changeForm({
    endpoint: "/policy/macro",
    change: () => ({ name: item.setting, value: field.value }),
    fields: [
      el("label", {}, `${item.title} (${item.unit})`),
      field,
      // Folded away rather than printed. The consequence of leaving it unset is
      // already the card's subtitle; this is the longer note about what the setting
      // moves, and under the input it reads as a second argument for a decision the
      // planner has by then already made.
      item.impact
        ? el("details", {}, el("summary", {}, "what this setting moves"),
             el("p", { class: "note" }, item.impact))
        : null,
    ],
    done: () => reload(),
    confirm: "Set this target",
  });
}

// A settled item, in one line. The whole point of the layout: what is answered gets
// out of the way, so the eye lands on what is not.
function settled(item, reload) {
  const shown = item.key === "locations"
    ? `${Object.keys(item.detail.statuses || {}).length || "wildcard"} declared`
    : (item.unit === "fraction" ? pct(item.value) : `${num(item.value)} ${item.unit}`);
  const open = el("button", { class: "link", type: "button" }, "change");
  const body = el("div", {});
  open.addEventListener("click", () => {
    const editing = body.childElementCount > 0;
    mount(body, editing ? null
      : (item.editor === "locations" ? locationEditor(item, reload)
                                     : macroEditor(item, reload)));
    open.textContent = editing ? "change" : "cancel";
  });
  return el("section", { class: "card" },
    el("h2", {}, item.title,
       el("span", { class: "badge ok" }, shown)),
    el("p", { class: "sub" }, item.question, " ", open),
    body);
}

function outstanding(item, reload) {
  const stopped = item.blocking_now;
  return el("section", { class: `card ${stopped ? "stop" : "ask"}` },
    el("h2", {}, item.question,
       el("span", { class: "badge todo" },
          stopped ? "blocking this run" : "not set")),
    el("p", { class: "sub" }, item.why),
    item.blocking && !stopped
      ? el("p", { class: "note" },
           "A run refuses to read your stock while any code in it is undeclared. "
           + "One location, or none, needs nothing here — there is nothing to "
           + "separate — so this is not blocking anything yet.")
      : null,
    item.note ? el("p", { class: "note" }, item.note) : null,
    item.editor === "locations" ? locationEditor(item, reload)
                                : macroEditor(item, reload));
}

export async function mountSetup() {
  const body = $("#setup-body");
  mount(body, el("p", { class: "note" }, "Reading…"));
  let state;
  try {
    state = await api("/setup");
  } catch (err) {
    mount(body, el("section", { class: "card stop" },
      el("h2", {}, "Could not read the setup"),
      el("p", { class: "sub" }, String(err.message))));
    return;
  }

  const reload = () => mountSetup();
  const open = state.items.filter((i) => i.state !== "set");
  const done = state.items.filter((i) => i.state === "set");

  mount(body,
    // The count first. Not a progress bar: three is a number a person holds in their
    // head, and a bar that is two-thirds full reads as nearly finished when the third
    // left is the one that stops the run.
    el("section", { class: `card ${state.blocked ? "stop" : (open.length ? "warn" : "")}` },
      el("h2", {}, state.blocked
        ? "A run cannot read your stock correctly yet"
        : (open.length ? "Some of this is still unanswered" : "Everything here is set"),
        el("span", { class: `badge ${open.length ? "todo" : "ok"}` },
           `${state.set} of ${state.total} set`)),
      el("p", { class: "sub" }, state.blocked
        ? "Your stock export carries location codes whose meaning has not been "
          + "declared, so the run refuses rather than summing them into the position. "
          + "Answer the first card below and it will run."
        : (open.length
            ? "A run will go ahead without these. It will produce a plan and no "
              + "opinion about whether the plan is any good — the figures will be "
              + "there and nothing will say whether they are acceptable."
            : "The run has everything only you can tell it. These stay in force until "
              + "you change them.")),
      el("p", { class: "note" }, `Stored in ${state.config_dir}`)),
    open.map((i) => outstanding(i, reload)),
    done.map((i) => settled(i, reload)));
}
