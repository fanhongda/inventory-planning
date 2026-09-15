// The shell: four screens, one client.
//
// The results screen was argued against and then built, and the reasoning is worth
// keeping here because it is what constrains it. A page showing the same numbers would
// become a second place they are formatted, rounded and subtly disagreed about — true
// only if the page re-derives them. It does not: it renders the run's own workbook,
// located from the manifest. One artefact, two renderings. If anything on that screen
// is ever calculated, the objection returns in full and the screen should go.

import { $, mount, el } from "/ui.js";
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

show(location.hash.slice(1) in SCREENS ? location.hash.slice(1) : "review");
