// The import review screen.
//
// Written for someone who will not open an editor and cannot adjudicate a mapping. That
// constraint decides the order of everything below.
//
//   1. Ask a question they can answer. Not "is `sku <- Material` right" — nobody outside
//      this repository can answer that — but "this file says 2,125 units on hand across
//      10 materials; does that sound right". Whoever runs the warehouse knows.
//   2. Decide by default; escalate only the close calls. The router reports a margin
//      now, so a document that won by 56 points is stated and moved past, and one that
//      won by 3 is put in front of a person.
//   3. Give the consequence, not the evidence. Not "overlap coefficient 86.5%".
//   4. Refuse rather than default. Currency is a required question with a dropdown, not
//      a silent fallback to the reporting currency worth a multiple of the real number.
//
// No build step and no framework, deliberately. This talks to the same HTTP API any
// other client would, so replacing it costs nothing but the files in this directory —
// which is the property the API was built for, and the reason a UI framework would be
// buying something not yet needed at one screen.

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

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `${response.status} ${path}`);
  return body;
}

// ── Panels ──────────────────────────────────────────────────────────────────

// The first thing on the page, because it is the only question the reader is qualified
// to answer without knowing anything about this pipeline.
function totalsCard(doc, batch, onDisagree) {
  const figures = [
    el("div", { class: "figure" }, el("div", { class: "n" }, num(doc.rows)),
       el("div", { class: "k" }, "行")),
    el("div", { class: "figure" }, el("div", { class: "n" }, num(doc.skus)),
       el("div", { class: "k" }, "个物料")),
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
    el("h2", {}, `这份文件被读成 ${batch.doc_type}`),
    el("p", { class: "sub" },
       "下面是它说的数。对不上，就说明有一列读错了 —— 这比看字段映射快得多。"),
    el("div", { class: "figures" }, figures),
    el("div", { class: "answer" },
      el("button", { class: "primary", onclick: (e) => {
        mount(e.target.closest(".answer"),
          el("span", { class: "changed" }, "好。这份文件可以用了 —— 它仍未进入计划。"));
      } }, "对得上"),
      el("button", { onclick: onDisagree }, "对不上 / 我要看字段"),
      el("span", { class: "hint" }, "对不上时打开下面的字段表，逐列核对。")));
}

// Shown only when the router did not win clearly. A document that won by 56 points is
// not worth a person's attention, and spending it there is why the close calls get
// skimmed.
function routingCard(res) {
  if (res.stated) {
    return el("section", { class: "card" },
      el("h2", {}, "文件类型是被指定的，不是判断出来的"),
      el("p", { class: "sub" },
         `这份文件按 ${res.doc_type} 处理，因为模板或调用方这么说的 —— ` +
         `不是比对出来的结果。系统没有对此做过独立判断。`));
  }
  if (!res.close_call && !res.uncertain) return null;

  const rival = res.runner_up;
  // A margin under half a point rounds to "只差 0%", which reads as a typo rather than
  // as the tie it is. A tie is the strongest form of this warning, not the weakest.
  const gap = res.margin < 0.005
    ? `${rival && rival.doc_type} 的得分与它完全相同`
    : `也很像 ${rival && rival.doc_type} —— 两者只差 ${pct(res.margin)}`;
  return el("section", { class: "card warn" },
    el("h2", {}, "这个判断不保险，请确认"),
    el("p", { class: "sub" },
       rival
         ? `按列名看它最像 ${res.doc_type}，但${gap}。` +
           `这两种单据的列几乎一样，光看表头分不开。`
         : `按列名判断的把握只有 ${pct(res.confidence)}。`),
    el("p", { class: "note" },
       "如果认错了，这份文件的每一个数都会被算进错误的地方，而且不会报错。" +
       "请确认这份导出到底是什么，再往下走。"));
}

// Every figure the run could not measure, largest exposure first. The currency case
// gets a dropdown rather than a default, because "这个文件记的是什么货币" is business
// knowledge this reader has and a 7x error is not something they can spot afterwards.
function restingCard(resting, batch, refresh) {
  const items = resting.resting_on || [];
  if (!items.length) return null;

  const rows = items.map((item) => {
    const magnitude = item.money !== null && item.money !== undefined
      ? `${num(item.money)} ${resting.reporting_currency} (${item.money_field})`
      : item.priced_rows
        ? `${num(item.priced_rows)} 行带单价 —— 是费率，不是总额`
        : item.qty !== null && item.qty !== undefined
          ? `${num(item.qty)} (${item.qty_field})`
          : "无法量化";
    return el("tr", { class: "attention" },
      el("td", {}, el("code", {}, item.field)),
      el("td", {}, String(item.value)),
      el("td", {}, item.basis === "declared"
        ? `${item.by || "未署名"} 声明` : "系统假设"),
      el("td", { class: "num" }, `${item.upper_bound ? "≤ " : ""}${num(item.rows)} 行`),
      el("td", {}, magnitude));
  });

  const currency = items.find((i) => i.field === "currency" && i.basis === "assumed");
  return el("section", { class: "card warn" },
    el("h2", {}, "这些数字压在没量到的东西上"),
    el("p", { class: "sub" },
       "导出文件没带这些信息，系统只能先按默认处理。默认错了，下面这些金额就是错的。"),
    el("div", { class: "scroll" }, el("table", {},
      el("thead", {}, el("tr", {},
        ["字段", "当前取值", "来源", "影响行数", "背后的量"].map((h) => el("th", {}, h)))),
      el("tbody", {}, rows))),
    currency ? currencyForm(batch, currency, refresh) : null);
}

function currencyForm(batch, item, refresh) {
  const select = el("select", { id: "cur" },
    el("option", { value: "" }, "请选择 —"),
    ["CNY", "USD", "EUR", "SGD", "JPY", "HKD", "GBP", "AUD"].map(
      (c) => el("option", { value: c }, c)));
  const reason = el("textarea", { placeholder: "为什么是这个币种？例如：本厂只按标准成本以人民币记账，导出模板不带币种列。" });
  const by = el("input", { placeholder: "你的名字 / 工号" });
  const out = el("p", { class: "note" });

  return el("form", { class: "declare", onsubmit: async (e) => {
    e.preventDefault();
    out.className = "note";
    if (!select.value) { out.textContent = "先选一个币种。"; return; }
    try {
      const body = await api(`/batches/${batch.batch_id}/declarations`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ scope: "value", field: "currency", value: select.value,
                               reason: reason.value, by: by.value }),
      });
      out.className = "changed";
      out.textContent = `已记录：${body.declaration.split("/").pop()} —— ${body.changed.length} 个字段的读法变了。`;
      refresh();
    } catch (err) { out.textContent = String(err.message); }
  } },
    el("label", {}, `这个文件里的金额记的是什么货币？现在按 ${item.value} 处理。`),
    select,
    el("label", {}, "为什么（必填 —— 这是日后唯一能说明当时依据的记录）"), reason,
    el("div", { class: "row2" },
      el("div", {}, el("label", {}, "谁说的（必填）"), by),
      el("div", {}, el("label", {}, " "),
         el("button", { class: "primary", type: "submit" }, "记下这条声明"))),
    out);
}

// Collapsed to what needs attention. Opening the whole table is a click away and is
// what the "对不上" button does.
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
        mapped: "取自列", declared: "有人指定", default: "补的默认值",
        derived: "算出来的", absent: "没有",
      }[f.source] || f.source)),
      el("td", {}, f.column || "—"),
      el("td", { class: "num" }, f.source === "absent" ? "—" : pct(f.fill_rate)),
      el("td", {}, el("button", { class: "link", onclick: () =>
        card.append(mappingForm(res, batch, f, refresh)) }, "换一列"))));

    mount(card,
      el("h2", {}, "字段是从哪几列读出来的"),
      el("p", { class: "sub" },
         showAll ? `全部 ${res.fields.length} 个字段。带 * 的是必需的。`
                 : `只列出必需的和没读到的。其余 ${res.fields.length - fields.length} 个字段正常。`),
      res.ignored_declarations.length
        ? el("p", { class: "note", style: "color:var(--stop)" },
             "⚠ 有声明指向了这个文件没有的列，未生效：" +
             res.ignored_declarations.map((d) => `${d.field} ← ${d.column}`).join("、"))
        : null,
      el("div", { class: "scroll" }, el("table", {},
        el("thead", {}, el("tr", {},
          ["字段", "来源", "对应的列", "有值的比例", ""].map((h) => el("th", {}, h)))),
        el("tbody", {}, rows))),
      el("p", {}, el("button", { class: "link", onclick: () => { showAll = !showAll; render(); } },
        showAll ? "只看需要注意的" : `显示全部 ${res.fields.length} 个字段`)),
      res.unmatched_columns.length
        ? el("p", { class: "note" },
             `文件里有 ${res.unmatched_columns.length} 列没有对应到任何字段：` +
             res.unmatched_columns.join("、"))
        : null);
  };
  render();
  return card;
}

function mappingForm(res, batch, field, refresh) {
  const columns = [...new Set([...res.fields.map((f) => f.column).filter(Boolean),
                               ...res.unmatched_columns])].sort();
  const select = el("select", {}, el("option", { value: "" }, "请选择 —"),
    columns.map((c) => el("option", { value: c, selected: c === field.column }, c)));
  const reason = el("textarea", { placeholder: "为什么这一列才对？" });
  const by = el("input", { placeholder: "你的名字 / 工号" });
  const out = el("p", { class: "note" });

  const form = el("form", { class: "declare", onsubmit: async (e) => {
    e.preventDefault();
    out.className = "note";
    if (!select.value) { out.textContent = "先选一列。"; return; }
    try {
      const body = await api(`/batches/${batch.batch_id}/declarations`, {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ scope: "mapping", field: field.field, value: select.value,
                               reason: reason.value, by: by.value }),
      });
      out.className = "changed";
      out.textContent = body.changed.length
        ? `已记录。变化：${body.changed.map((c) => c.field).join("、")}`
        : "已记录，但什么都没变 —— 多半是这一列在文件里不存在。往上看未生效的声明。";
      refresh();
    } catch (err) { out.textContent = String(err.message); }
  } },
    el("label", {}, `${field.field} 应该取自哪一列？现在取自 ${field.column || "（无）"}。`),
    select,
    el("label", {}, "为什么（必填）"), reason,
    el("div", { class: "row2" },
      el("div", {}, el("label", {}, "谁说的（必填）"), by),
      el("div", {}, el("label", {}, " "),
         el("button", { class: "primary", type: "submit" }, "记下这条声明"))),
    out);
  return form;
}

function rowsCard(rows) {
  if (!rows.rows.length) return null;
  return el("section", { class: "card" },
    el("h2", {}, "文件里的原始行"),
    el("p", { class: "sub" },
       `原样保存的前 ${rows.rows.length} 行，共 ${num(rows.total)} 行。表头是文件本来的写法。`),
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
  const refresh = () => showBatch(batch, container, true);
  const doc = summary.documents[0];

  mount(container,
    doc ? totalsCard(doc, batch, () => showBatch(batch, container, true)) : null,
    routingCard(resolution),
    restingCard(summary.resting_on, batch, refresh),
    fieldsCard(resolution, batch, refresh, openFields),
    rowsCard(rows));
}

async function upload(file) {
  const status = $("#status");
  const results = $("#results");
  status.hidden = false;
  status.className = "status";
  status.textContent = `正在读取 ${file.name} …`;
  results.replaceChildren();

  const form = new FormData();
  form.append("file", file);
  try {
    const body = await api("/uploads", { method: "POST", body: form });
    status.textContent = `${body.source_name} —— 已原样保存，尚未进入计划。`;
    if (body.stale_template) {
      results.append(el("section", { class: "card warn" },
        el("h2", {}, "这份模板不是最新的"),
        el("p", { class: "sub" }, body.stale_template)));
    }
    for (const landed of body.landed) {
      const section = el("div", {});
      results.append(section);
      await showBatch(landed, section);
    }
  } catch (err) {
    status.className = "status bad";
    status.textContent = String(err.message);
  }
}

const drop = $("#drop");
$("#pick").addEventListener("click", () => $("#file").click());
$("#file").addEventListener("change", (e) => e.target.files[0] && upload(e.target.files[0]));
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
});
