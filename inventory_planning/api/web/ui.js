// Shared helpers for every screen.
//
// No build step and no framework, deliberately. These talk to the same HTTP API any
// other client would, so replacing the whole directory costs nothing but the files in
// it — which is the property the API was built for, and the reason a UI framework would
// be buying something not yet needed at this size.

const $ = (sel, root = document) => root.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// `replaceChildren` renders a null child as the text "null", unlike `el` above, which
// drops them. Panels with nothing to say return null on purpose and several do so
// routinely, so every mount point goes through here rather than remembering to filter.
const mount = (node, ...children) => node.replaceChildren(...children.flat().filter(Boolean));

const num = (v, digits = 0) =>
  v === null || v === undefined || Number.isNaN(v)
    ? "—"
    : Number(v).toLocaleString("zh-CN", { minimumFractionDigits: digits,
                                          maximumFractionDigits: digits });
const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;

// Contract descriptions are written for someone reading the YAML and run to a
// paragraph. The first sentence is the label; the rest belongs in a tooltip, not in a
// checklist row where it buries the thing being checked.
const firstSentence = (text, cap = 96) => {
  const one = String(text || "").replace(/\s+/g, " ").split(/(?<=[.。])\s/)[0].trim();
  return one.length > cap ? `${one.slice(0, cap - 1)}…` : one;
};

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `${response.status} ${path}`);
  return body;
}

export { $, el, mount, num, pct, firstSentence, api };
