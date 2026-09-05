// The shell: two screens, one client.
//
// Planning and analysis have no screen and are not getting one — the run’s workbook is
// what gets handed round a meeting, and a page showing the same numbers would become a
// second place they are formatted, rounded and subtly disagreed about. These two are the
// surfaces a person needs before that: what was read, and what is stored.

import { $, mount, el } from "/ui.js";
import { mountReview } from "/review.js";
import { mountBrowse } from "/browse.js";
import { mountPolicy } from "/policy.js";

const SCREENS = {
  review: { label: "Import review", start: mountReview },
  browse: { label: "Stored facts", start: mountBrowse },
  policy: { label: "Policy & runs", start: mountPolicy },
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
