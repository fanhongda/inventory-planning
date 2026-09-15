"""
Rule editing — the same propose-then-approve contract, over a block instead of a value.

INTERFACE.md §7 revised the position this replaces. Read-only was argued as the design
rather than a stage of it, on the grounds that a rule wants review, a diff, a rationale
and an owner and markdown in git gives all four. That argument was about *storage*, and
it was allowed to decide *who may operate it*, which it does not. All four survive here
because the storage does not move: the form writes the same markdown a person would.

Three things a rule needs that a scalar did not:

**The rationale has a home in the file.** A rule already carries `rationale`, `owner`
and `date`, so an edit updates them rather than only logging them elsewhere — that is
where the next reader will look, three months on, when deciding whether the rule still
applies. `date` is stamped rather than asked for: it is the date the change was made,
which the form knows and a person would mistype.

**Order is meaning.** Rules apply in file order and later ones win, so the file reads
top to bottom as general policy and then exceptions. A new rule is appended last,
because that is where an exception to everything above it belongs, and the caller is
told so rather than left to discover it from a conflict report.

**Reach is not predictable from here.** A scope is a question about a frame of SKUs and
there is no frame in an interface, so this cannot say what an edited rule would match —
only what the rule matched in the last run under the rules as they were. That number is
attached to the rule the editor opened, and it goes stale the moment the change is
applied, which the screen says. Guessing would be worse: a wrong count next to a scope
is an invitation to write the scope around it.

What is deliberately not here is a second person's approval. Every `by` on this
interface is a form field nobody checks — INTERFACE.md §7 names that as the identity
seam, still uncut — so a second name would be a second unverified string, and a review
step that verifies nothing is worse than an honest absence of one: it reads as a
control.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .edits import (
    EditRefused, check_basis, collapse, config_root, digest, history as _history,
    record, require_attribution, unified,
)

RULES_FILE = "planning_parameters.md"

# The fields a form may write, and nothing else. `rule_id` is absent on purpose: it is
# the join key the run manifest records hits against, so renaming one silently detaches
# every count ever recorded for it from the rule it belongs to.
EDITABLE = ("name", "scope", "set", "rationale", "owner")


# The shared class under a local name, not a subclass. A subclass would have meant
# `except RuleEditError` quietly missing every refusal raised by the shared checks —
# the basis, the reason, the name — which are most of them. What tells a rule edit from
# a scalar one is the endpoint that was called, not the type of the refusal.
RuleEditError = EditRefused


@dataclass
class RuleProposal:
    """A change to the rules that has not been made."""

    action: str                      # edit | add | remove
    rule_id: str
    diff: str
    basis: str
    before: Dict[str, Any] = dc_field(default_factory=dict)
    after: Dict[str, Any] = dc_field(default_factory=dict)
    # What the file will say after this, in the order the run will apply it. An edit
    # that moves nothing still answers "where does this sit" — which for a rule is half
    # the question, since a later rule takes a parameter back from an earlier one.
    order: List[str] = dc_field(default_factory=list)
    note: str = ""
    unchanged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "rule_id": self.rule_id, "diff": self.diff,
                "basis": self.basis, "from": self.before, "to": self.after,
                "order": self.order, "note": self.note, "unchanged": self.unchanged}


def _read(config_dir) -> Tuple[Path, str]:
    path = config_root(config_dir) / RULES_FILE
    try:
        return path, path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuleEditError(f"{path} cannot be read: {exc}") from exc


# ── Finding one rule in the text ─────────────────────────────────────────────

def _heading_pattern(rule_id: str) -> re.Pattern:
    # Horizontal whitespace only at the ends. `\s*$` under MULTILINE consumes the
    # newlines that follow, so the match swallowed the blank line between a rule's
    # heading and its fence and every edit silently closed it up.
    return re.compile(
        rf"^###[^\S\n]+{re.escape(rule_id)}[^\S\n]*"
        rf"(?:[·•|][^\S\n]*(?P<name>.*?))?[^\S\n]*$",
        re.MULTILINE)


@dataclass
class _Block:
    """Where one rule's heading and YAML body sit in the file."""

    heading_start: int
    heading_end: int
    name: str
    body_start: int                  # first character inside the fence
    body_end: int                    # the character the closing fence starts at
    end: int                         # end of the whole rule, closing fence included


def _locate(text: str, rule_id: str) -> _Block:
    """
    The one rule with this id, or a refusal naming what was found instead.

    Two matches is a refusal rather than a choice. The loader rejects duplicate rule
    ids outright, so a file with two is already broken — and editing one of them would
    produce a file that still does not load, against a diff that looked fine.
    """
    found = list(_heading_pattern(rule_id).finditer(text))
    if not found:
        raise RuleEditError(
            f"No rule {rule_id} in {RULES_FILE}. A rule is a `### {rule_id} · name` "
            f"heading with a yaml block under it.")
    if len(found) > 1:
        raise RuleEditError(
            f"{rule_id} appears {len(found)} times in {RULES_FILE}. The file does not "
            f"load with a duplicate rule id, so fix that first — editing one of them "
            f"would leave it broken against a diff that looked fine.")

    head = found[0]
    fence = re.compile(r"```ya?ml\s*\n(?P<body>.*?)(?P<close>```)", re.DOTALL)
    block = fence.search(text, head.end())
    if block is None:
        raise RuleEditError(f"{rule_id} has no yaml block under its heading.")
    # A fence found after the *next* rule belongs to that one, not to this.
    nxt = re.compile(r"^#{1,3}\s+", re.MULTILINE).search(text, head.end())
    if nxt is not None and block.start() > nxt.start():
        raise RuleEditError(f"{rule_id} has no yaml block under its heading.")

    return _Block(heading_start=head.start(), heading_end=head.end(),
                  name=(head.group("name") or "").strip(),
                  body_start=block.start("body"), body_end=block.start("close"),
                  end=block.end())


def _order(text: str) -> List[str]:
    """Every rule id in the order the run will apply them. Later wins."""
    return [m.group(1) for m in
            re.finditer(r"^###\s+(\S+)", text, re.MULTILINE)]


# ── Writing a rule's YAML ────────────────────────────────────────────────────

def _render_scalar(value: Any) -> str:
    """
    One value, as YAML, on one line.

    Dumped as the value half of a mapping rather than on its own, because a bare
    top-level scalar comes back with a `...` document-end marker after it — valid YAML
    alone, and a syntax error the moment another key follows it. A collection comes
    back from that as a block, so it is re-dumped in flow style, where the marker does
    not appear.
    """
    import yaml

    dumped = yaml.safe_dump({"v": value}, default_flow_style=False,
                            allow_unicode=True, sort_keys=False)
    rendered = dumped[len("v:"):].strip()
    if "\n" in rendered:
        rendered = yaml.safe_dump(value, default_flow_style=True,
                                  allow_unicode=True, sort_keys=False).strip()
    if "\n" in rendered:
        raise RuleEditError(
            f"{value!r} does not fit on one line, and a rule's values are written one "
            f"per line. Edit the file for this one.")
    return rendered


def _render_rationale(text: str) -> str:
    """
    The folded form the file already uses, wrapped so a diff stays readable.

    Written as `>` rather than a quoted one-liner because every rationale in this file
    is a paragraph and a 200-character line in a diff is a line nobody reads to the end
    of. Wrapping is by width, not by sentence: the text is frequently Chinese, where
    splitting on `.` finds nothing and splitting on width is what a reader expects
    anyway.
    """
    import textwrap

    body = collapse(text)
    lines = textwrap.wrap(body, width=74) or [body]
    return "rationale: >\n" + "\n".join(f"  {line}" for line in lines)


def _render_block(rule: Dict[str, Any]) -> str:
    """
    A whole rule's YAML body — used when adding one, never when editing one.

    Key order is fixed rather than taken from the caller so that two rules written
    months apart read the same way.
    """
    lines = [f"scope: {_render_scalar(rule['scope'])}", "set:"]
    for key, value in (rule.get("set") or {}).items():
        lines.append(f"  {key}: {_render_scalar(value)}")
    lines.append(_render_rationale(rule.get("rationale", "")))
    if rule.get("owner"):
        lines.append(f"owner: {_render_scalar(rule['owner'])}")
    lines.append(f"date: {rule.get('date') or date.today().isoformat()}")
    return "\n".join(lines) + "\n"


def _edit_body(body: str, before: Dict[str, Any], changes: Dict[str, Any]) -> str:
    """
    The rule's YAML with only the changed fields rewritten, byte for byte otherwise.

    The same decision as the scalar editor, and for the same reason: re-rendering the
    whole block from the parsed rule would reflow a rationale nobody touched and put
    six lines in the diff for a change to one number. The diff is the thing being
    approved, so the smallest true diff is the product.
    """
    if "scope" in changes and changes["scope"] != before.get("scope"):
        body = _replace_value(body, "scope", _render_scalar(changes["scope"]))
    if "owner" in changes and changes["owner"] != before.get("owner"):
        body = _replace_value(body, "owner", _render_scalar(changes["owner"]))
    if ("rationale" in changes
            and collapse(changes["rationale"]) != collapse(before.get("rationale"))):
        body = _replace_rationale(body, changes["rationale"])
    if "set" in changes:
        body = _replace_set(body, before.get("set") or {}, changes["set"])
    # Always: the file should say when it last moved, and this is that moment.
    return _replace_value(body, "date", date.today().isoformat())


def _line_of(body: str, key: str, indent: str = "") -> re.Match:
    pattern = re.compile(rf"^{indent}{re.escape(key)}:[^\S\n]*(?P<value>[^\n]*)$",
                         re.MULTILINE)
    match = pattern.search(body)
    if match is None:
        raise RuleEditError(
            f"this rule has no `{key}` line to change, so the edit would have to "
            f"rewrite the block. Edit the file for this one.")
    return match


def _replace_value(body: str, key: str, rendered: str) -> str:
    match = _line_of(body, key)
    return body[:match.start("value")] + rendered + body[match.end("value"):]


def _replace_rationale(body: str, text: str) -> str:
    """
    Swap the folded `rationale: >` block, continuation lines and all.

    The block ends where the indentation does, which is how YAML itself reads it —
    looking for the next known key instead would break on a rule that carries one this
    module has never heard of.
    """
    start = _line_of(body, "rationale")
    end = start.end()
    for line in re.finditer(r"^(?P<line>[^\n]*)$", body[start.end():], re.MULTILINE):
        stripped = line.group("line")
        if stripped.strip() and not stripped.startswith((" ", "\t")):
            break
        end = start.end() + line.end()
    return body[:start.start()] + _render_rationale(text) + body[end:]


def _replace_set(body: str, before: Dict[str, Any], after: Dict[str, Any]) -> str:
    """
    Reconcile the `set:` mapping, keeping every untouched line exactly as it was.

    A key that kept its value keeps its line — including any comment written beside it,
    which for a parameter is often the only note saying why that number and not another.
    """
    head = re.search(r"^set:[^\S\n]*$", body, re.MULTILINE)
    if head is None:
        raise RuleEditError("this rule has no `set:` block to change")
    entries = list(re.finditer(r"^(?P<indent>[ \t]+)(?P<key>[\w.-]+):"
                               r"[^\S\n]*(?P<value>[^\n]*)$",
                               body[head.end():], re.MULTILINE))
    # Stop at the first line that leaves the block.
    kept = []
    cursor = 0
    for entry in entries:
        if body[head.end():][cursor:entry.start()].strip():
            break
        kept.append(entry)
        cursor = entry.end()
    if not kept:
        raise RuleEditError("this rule's `set:` block has no entries to change")

    base = head.end()
    indent = kept[0].group("indent")
    lines: List[str] = []
    for entry in kept:
        key = entry.group("key")
        if key not in after:
            continue                      # dropped by this change
        line = body[base + entry.start():base + entry.end()]
        if after[key] != before.get(key):
            line = (body[base + entry.start():base + entry.start("value")]
                    + _render_scalar(after[key])
                    + body[base + entry.end("value"):base + entry.end()])
        lines.append(line)
    for key, value in after.items():
        if not any(e.group("key") == key for e in kept):
            lines.append(f"{indent}{key}: {_render_scalar(value)}")

    return (body[:base] + "\n" + "\n".join(lines)
            + body[base + kept[-1].end():])


def _current(config_dir, rule_id: str) -> Dict[str, Any]:
    """The rule as the loader reads it — not as this module parses the text."""
    from .parameters import PlanningParameters

    params = PlanningParameters(config_root(config_dir) / RULES_FILE)
    for rule in params.rules:
        if rule.rule_id == rule_id:
            return {"rule_id": rule.rule_id, "name": rule.name, "scope": rule.scope,
                    "set": dict(rule.overrides), "rationale": rule.rationale,
                    "owner": rule.owner, "date": rule.date}
    raise RuleEditError(f"No rule {rule_id} in {RULES_FILE}.")


# ── Validation ───────────────────────────────────────────────────────────────

def _validate(text: str, config_dir) -> None:
    """
    Load the edited file with the loader the pipeline uses, in a scratch copy.

    Every rule invariant already lives there and is enforced on every run: a scope that
    does not parse, a missing `set`, a duplicate rule id, a rule with no rationale.
    Restating any of them here would be a second answer the day one of them changed —
    and the rationale rule in particular is enforced in `Rule.__post_init__`, which is
    exactly where it should be.
    """
    import shutil
    import tempfile

    from .parameters import PlanningParameters

    root = config_root(config_dir)
    with tempfile.TemporaryDirectory(prefix="rules-check-") as tmp:
        scratch = Path(tmp) / "config"
        scratch.mkdir()
        for existing in root.glob("*"):
            if existing.is_file():
                shutil.copy2(existing, scratch / existing.name)
        target = scratch / RULES_FILE
        target.write_text(text, encoding="utf-8")
        try:
            PlanningParameters(target)
        except Exception as exc:
            raise RuleEditError(
                f"the edit is refused by the loader that reads {RULES_FILE}: "
                f"{exc}") from exc


def _checked_fields(changes: Dict[str, Any]) -> Dict[str, Any]:
    unknown = sorted(set(changes) - set(EDITABLE))
    if unknown:
        extra = ("  `rule_id` is not editable: the run manifest records each rule's "
                 "hits against it, so renaming one detaches every count ever recorded "
                 "for it." if "rule_id" in unknown else "")
        raise RuleEditError(
            f"Not editable: {', '.join(unknown)}. A form may write "
            f"{', '.join(EDITABLE)}.{extra}")
    if "set" in changes and not isinstance(changes["set"], dict):
        raise RuleEditError("`set` is a mapping of parameter to value")
    if "set" in changes and not changes["set"]:
        raise RuleEditError(
            "a rule that sets nothing matches SKUs and decides nothing. Remove the "
            "rule instead — that says what happened, where an empty one does not.")
    return changes


# ── Proposing ────────────────────────────────────────────────────────────────

def propose_edit(rule_id: str, changes: Dict[str, Any],
                 config_dir=None) -> RuleProposal:
    """Change some fields of one rule, leaving the rest of the file alone."""
    path, text = _read(config_dir)
    block = _locate(text, rule_id)
    before = _current(config_dir, rule_id)
    changes = _checked_fields(dict(changes or {}))

    after = dict(before)
    after.update({k: v for k, v in changes.items() if k != "name"})
    # Stamped, not asked for: `date` is when the change was made, which the form knows.
    after["date"] = date.today().isoformat()
    name = str(changes.get("name") or before["name"]).strip()

    body = _edit_body(text[block.body_start:block.body_end], before,
                      dict(changes, set=changes.get("set", before["set"])))
    edited = (text[:block.heading_start]
              + f"### {rule_id} · {name}"
              + text[block.heading_end:block.body_start]
              + body
              + text[block.body_end:])

    if edited == text:
        return RuleProposal(action="edit", rule_id=rule_id, diff="", basis=digest(text),
                            before=before, after=before, order=_order(text),
                            unchanged=True)

    _validate(edited, config_dir)
    return RuleProposal(
        action="edit", rule_id=rule_id, diff=unified(text, edited, RULES_FILE),
        basis=digest(text), before=before, after=dict(after, name=name),
        order=_order(edited),
        note=_reach_note(rule_id))


def propose_add(rule: Dict[str, Any], config_dir=None) -> RuleProposal:
    """
    Append a rule after the last one, which is where an exception belongs.

    Appended rather than placed, because position is the one thing about a rule that a
    form cannot ask about usefully: "before or after R-003" is a question about what
    R-003 does, and the answer is in the file the person is already looking at. Last is
    also the only position with a stateable meaning — it wins over everything above it.
    """
    path, text = _read(config_dir)
    rule = dict(rule or {})
    rule_id = str(rule.pop("rule_id", "") or "").strip()
    if not rule_id:
        raise RuleEditError("a new rule needs a rule_id — it is what the run reports "
                            "hits against and what a later reader cites")
    if _heading_pattern(rule_id).search(text):
        raise RuleEditError(
            f"{rule_id} already exists. Edit it, or pick another id — two rules under "
            f"one id do not load, and the manifest could not tell their hits apart.")
    rule = _checked_fields(rule)
    for required in ("scope", "set", "rationale"):
        if not rule.get(required):
            raise RuleEditError(f"a new rule needs `{required}`")

    name = str(rule.get("name") or "").strip()
    if not name:
        raise RuleEditError("a new rule needs a name — the heading is what anyone "
                            "scanning the file reads before the scope")
    body = dict(rule, date=date.today().isoformat())

    last = _last_rule_end(text)
    block = (f"\n### {rule_id} · {name}\n\n```yaml\n" + _render_block(body) + "```\n")
    edited = text[:last] + block + text[last:]

    _validate(edited, config_dir)
    return RuleProposal(
        action="add", rule_id=rule_id, diff=unified(text, edited, RULES_FILE),
        basis=digest(text), before={}, after=dict(body, rule_id=rule_id, name=name),
        order=_order(edited),
        note="Appended last, so it wins over every rule above it on any parameter they "
             "both set. The file is applied top to bottom.")


def propose_remove(rule_id: str, config_dir=None) -> RuleProposal:
    """
    Take a rule out of the file.

    The reason goes to the log rather than the file, because after this there is no
    rule to carry it — which is the one case where the log is the only record, and the
    reason `reason` is required here as much as anywhere.
    """
    path, text = _read(config_dir)
    block = _locate(text, rule_id)
    before = _current(config_dir, rule_id)

    # The rule takes its own trailing blank line with it and leaves the one in front of
    # it alone. Taking both closes the gap between the rules either side, and taking
    # neither opens a widening hole in the file every time one is removed — the second
    # is the mistake this made first, and the diff is where it showed.
    start = block.heading_start
    end = block.end
    if text[end:end + 1] == "\n":
        end += 1                          # the newline ending the closing fence
    if text[end:end + 1] == "\n":
        end += 1                          # the blank line before the next rule

    edited = text[:start] + text[end:]
    _validate(edited, config_dir)
    return RuleProposal(
        action="remove", rule_id=rule_id, diff=unified(text, edited, RULES_FILE),
        basis=digest(text), before=before, after={}, order=_order(edited),
        note="Every SKU this rule was deciding falls back to the rule above it, or to "
             "the defaults where there is none.")


def _last_rule_end(text: str) -> int:
    """Where to append: after the final rule, before whatever prose follows it."""
    ids = _order(text)
    if not ids:
        raise RuleEditError(
            f"{RULES_FILE} has no rules to append after. Add the first one by hand — "
            f"where the rules section begins is a judgement about the file, not about "
            f"the rule.")
    return _locate(text, ids[-1]).end + 1


def _reach_note(rule_id: str) -> str:
    return (f"What {rule_id} reaches is decided against a frame of SKUs, so the counts "
            f"beside it belong to the last run under the rules as they were. This "
            f"change detaches them; the next run measures the new scope.")


# ── Applying ─────────────────────────────────────────────────────────────────

def apply(proposal_for, *, reason: str, by: str, basis: str,
          config_dir=None) -> Dict[str, Any]:
    """
    Redo the proposal, check it against the bytes that were approved, and write it.

    `proposal_for` is a callable that reproduces the proposal, so an apply can never
    write something the proposal did not produce — the two paths are one path called
    twice, and there is no second place where the edit is assembled.
    """
    require_attribution(reason, by, "a change to a rule")

    proposal = proposal_for()
    if proposal.unchanged:
        raise RuleEditError(
            f"{proposal.rule_id} already says this — nothing to apply. A change that "
            f"writes nothing should not appear in the log as though something moved.")
    path, text = _read(config_dir)
    check_basis(basis, proposal.basis, RULES_FILE)

    # Reconstruct from the diff the person approved rather than re-deriving the text a
    # second way: the proposal produced an exact file, and applying anything else would
    # make the diff a suggestion.
    edited = _patched(text, proposal)
    path.write_text(edited, encoding="utf-8")

    at = datetime.now().isoformat(timespec="seconds")
    log_path = record(config_dir, {
        "kind": "rule", "action": proposal.action, "at": at, "by": str(by),
        "reason": collapse(reason), "rule_id": proposal.rule_id, "file": RULES_FILE,
        "from": proposal.before, "to": proposal.after,
        "basis": proposal.basis, "digest": digest(edited),
    })
    body = proposal.to_dict()
    body.update({"applied": True, "by": str(by), "reason": collapse(reason),
                 "at": at, "digest": digest(edited), "recorded_in": str(log_path)})
    return body


def _patched(text: str, proposal: RuleProposal) -> str:
    """
    The file the proposal's diff describes, applied to the text it was made against.

    A unified diff over the exact bytes it was produced from is unambiguous, and the
    basis check above has already established that those are the bytes on disk. This is
    what makes "approve the diff" literal rather than approximate.
    """
    lines = text.splitlines(keepends=True)
    out: List[str] = []
    cursor = 0
    for hunk in _hunks(proposal.diff):
        start, old_lines, new_lines = hunk
        out.extend(lines[cursor:start])
        if lines[start:start + len(old_lines)] != old_lines:
            raise RuleEditError(
                f"{RULES_FILE} does not match the diff that was approved. Nothing was "
                f"written; re-read the rule and approve the diff as it stands now.")
        out.extend(new_lines)
        cursor = start + len(old_lines)
    out.extend(lines[cursor:])
    return "".join(out)


def _hunks(diff: str):
    """`@@ -a,b +c,d @@` hunks as (0-based start, old lines, new lines)."""
    header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    lines = diff.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        match = header.match(lines[i])
        if match is None:
            i += 1
            continue
        start = int(match.group(1)) - 1
        old: List[str] = []
        new: List[str] = []
        i += 1
        while i < len(lines) and not header.match(lines[i]):
            line, body = lines[i][0], lines[i][1:]
            if line == " ":
                old.append(body)
                new.append(body)
            elif line == "-":
                old.append(body)
            elif line == "+":
                new.append(body)
            else:
                break
            i += 1
        yield start, old, new


def history(config_dir=None, limit: int = 50) -> List[Dict[str, Any]]:
    """The rule changes, newest first — the shared log, narrowed to this editor."""
    return _history(config_dir, limit=limit, kind="rule")
