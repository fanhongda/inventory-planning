// The shell: four screens, one client.
//
// The results screen was argued against and then built, and the reasoning is worth
// keeping here because it is what constrains it. A page showing the same numbers would
// become a second place they are formatted, rounded and subtly disagreed about — true
// only if the page re-derives them. It does not: it renders the run's own workbook,
// located from the manifest. One artefact, two renderings. If anything on that screen
// is ever calculated, the objection returns in full and the screen should go.

import { $, mount, el, api } from "/ui.js";
import { mountReview } from "/review.js";
import { mountBrowse } from "/browse.js";
import { mountPolicy } from "/policy.js";
import { mountResults } from "/results.js";

const SCREENS = {
  review: { label: "Import review", start: mountReview },
  browse: { label: "Stored facts", start: mountBrowse },
  policy: { label: "Policy & runs", start: mountPolicy },
  results: { label: "Results", start: mountResults },
};

let started = {};

function show(name) {
  for (const key of Object.keys(SCREENS)) {
    $(`#screen-${key}`).hidden = key !== name;
    $(`#tab-${key}`).classList.toggle("on", key === name);
  }
  // Started once, then left alone. Re-running the import screen’s setup would bind a
  // second listener to the drop zone and upload every file twice.
  if (!started[name]) {
    started[name] = true;
    SCREENS[name].start();
  } else if (name !== "review") {
    SCREENS[name].start();          // the store may have changed on another screen
  }
  location.hash = name;
}

mount($("#tabs"), Object.entries(SCREENS).map(([name, screen]) =>
  el("button", { id: `tab-${name}`, class: "tab", onclick: () => show(name) },
     screen.label)));

// A workspace nobody has set up, offered as one button rather than as four screens of
// errors. The tenant is the server's own and is never sent from here: a request that
// could name its own would be a request that writes a directory tree wherever the
// resolver resolves to.
function setupPanel(workspace, onDone) {
  const by = el("input", { type: "text", id: "setup-by",
                           placeholder: "who is setting this up" });
  const status = el("div", {});
  const go = el("button", { class: "primary", type: "button" }, "Set this workspace up");

  go.addEventListener("click", async () => {
    if (!by.value.trim()) {
      mount(status, el("p", { class: "status bad" }, "Name yourself first."));
      return;
    }
    go.disabled = true;
    mount(status, el("p", { class: "note" }, "Creating\u2026"));
    try {
      const body = await api("/workspace/setup", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ by: by.value }),
      });
      mount(status, el("ul", { class: "impacts" },
        (body.did || []).map((line) => el("li", {}, line))));
      onDone();
    } catch (err) {
      go.disabled = false;
      mount(status, el("p", { class: "status bad" }, String(err.message)));
    }
  });

  return el("section", { class: "card warn" },
    el("h2", {}, `Workspace "${workspace.tenant}" is not set up yet`),
    el("p", { class: "sub" },
       "There are no rules in it, so there is nothing to plan under and nothing to "
       + "show. Setting it up creates its three directories and copies a starting rule "
       + "set in. An existing rule set is never overwritten, so this is safe to repeat."),
    el("div", { class: "scroll" }, el("table", {},
      el("tbody", {}, ["config_dir", "store_root", "output_dir"].map((key) =>
        el("tr", {},
          el("td", {}, el("code", {}, key)),
          el("td", { class: "k" }, workspace[key])))))),
    el("form", { class: "declare", onsubmit: (e) => e.preventDefault() },
      el("label", {}, "who is setting this up"), by,
      el("div", {}, go), status),
    el("p", { class: "note" },
       workspace.tenant === "default"
         ? "This is the default workspace — its rules are the repository's own."
         : "A named workspace keeps its rules, facts and outputs outside any working "
           + "tree, so pulling the code cannot reach them."));
}

async function boot() {
  const first = location.hash.slice(1) in SCREENS ? location.hash.slice(1) : "review";
  try {
    const workspace = await api("/workspace");
    if (!workspace.ready) {
      for (const key of Object.keys(SCREENS)) $(`#screen-${key}`).hidden = true;
      $("#tabs").hidden = true;
      mount($("#setup"), setupPanel(workspace, () => location.reload()));
      return;
    }
  } catch (err) {
    // An older server has no /workspace. Carry on rather than refusing to start: the
    // screens have said what they cannot do for longer than this endpoint has existed.
  }
  $("#tabs").hidden = false;
  show(first);
}

boot();
