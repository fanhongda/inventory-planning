"""
Macro settings — the scalars, proposed as a diff and written back to the file.

INTERFACE.md §2 set the terms and §7 revised who may operate them: git stays the store,
and the form is a validator with a nicer keyboard. Four properties have to survive the
move from an editor to a form — review, a diff, a rationale and an owner — and all four
do, because the storage does not move. What changes is that editing no longer requires a
text editor and a git client.

Four decisions shape this module, and each of them is the reason a simpler one is wrong:

**The text is edited, not the parsed object.** `json.load` then `json.dump` would
reformat the whole file: `fx_rates.json` carries its own documentation in `_what_this_is`
and `_maintenance` keys, and the blank lines between rate blocks are how it is read. A
change of one number that rewrites sixty lines cannot be reviewed, and the diff is the
thing being approved. So one value is replaced where it sits and every other byte is left
alone.

**Only the settings the engine reads are editable**, listed here one by one rather than
derived from the file's keys. `echelon_level`, `parent_node` and `child_nodes` are in
`node_config.json` and belong to multi-node planning, which is not built — a form
offering them would be a switch that reads as a guarantee and honours nothing, which
INTERFACE.md rules out by name. Derived readings like the FX currency list are not
settings at all.

**Validation is a load, not a second opinion.** The edited text is parsed by the same
loader the pipeline uses before anything is written, so `cycle_stock_basis: pekk` is
refused by `PlanningParameters._validate` — the rule that already exists — rather than by
a copy of it here that would drift. The `choices` below drive the form's dropdown; they
do not decide what is legal.

**Applying is pinned to the bytes that were diffed.** A proposal reports the digest of
the file it read. An apply that does not carry that digest, or carries one that no longer
matches, is refused — otherwise someone approves one diff and a different edit lands,
which is the failure the whole propose-then-approve shape exists to prevent.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

CHANGE_LOG_NAME = "macro_changes.jsonl"

# Syntaxes a setting can live in. Both are edited the same way — find the one line the
# value is on, replace the value, leave the rest of the file alone — and differ only in
# how a value is written and how the file is validated afterwards.
JSON = "json"
YAML_BLOCK = "yaml_block"       # a `key: value` line inside a fenced block in markdown


class MacroError(ValueError):
    """A proposed change that cannot be made, with the reason a person needs."""


@dataclass(frozen=True)
class MacroSetting:
    """One editable scalar: where it lives, what it may be, and what it moves."""

    name: str
    filename: str
    syntax: str
    kind: str                   # text | number | choice
    choices: Tuple[str, ...] = ()
    note: str = ""
    # What a change to this does to the numbers. Shown beside the field, because the
    # cost of these settings is wildly uneven: `location_name` is a label and
    # `cycle_stock_basis` silently restates every should-be figure in the workbook.
    impact: str = ""


SETTINGS: Tuple[MacroSetting, ...] = (
    MacroSetting(
        "location_id", "node_config.json", JSON, "text",
        note="the node these figures are planned for",
        impact="a label on the output — no figure moves"),
    MacroSetting(
        "location_name", "node_config.json", JSON, "text",
        impact="a label on the output — no figure moves"),
    MacroSetting(
        "currency", "node_config.json", JSON, "text",
        note="the node's own currency",
        impact="what the node books in; conversion is the FX table's job"),
    MacroSetting(
        "planning_cycle", "node_config.json", JSON, "choice",
        choices=("monthly", "weekly"),
        impact="the cadence the plan is written for"),
    MacroSetting(
        "reporting_currency", "fx_rates.json", JSON, "text",
        note="every figure is restated into this",
        impact="restates every money figure in the run, and any currency without a "
               "rate into the new one is blanked rather than assumed"),
    MacroSetting(
        "cycle_stock_basis", "planning_parameters.md", YAML_BLOCK, "choice",
        choices=("peak", "average"),
        note="convention — changes every figure",
        impact="peak is D×R, average is D×R/2 — it halves or doubles the cycle stock "
               "in every should-be figure"),
    MacroSetting(
        "safety_stock_exposure", "planning_parameters.md", YAML_BLOCK, "choice",
        choices=("review_plus_lt", "lt_only"),
        note="convention — changes every figure",
        impact="lt_only understates safety stock by √((R+LT)/LT) under periodic "
               "review — 41% for a monthly review at a 30-day lead time"),
    MacroSetting(
        "pipeline_basis", "planning_parameters.md", YAML_BLOCK, "choice",
        choices=("incoterm_aware", "none"),
        note="convention — changes every figure",
        impact="whether goods in transit count towards the should-be balance"),
    MacroSetting(
        "transit_share_of_lt", "planning_parameters.md", YAML_BLOCK, "number",
        note="convention — changes every figure",
        impact="splits the lead time into a supplier leg and a transit leg; only the "
               "transit leg can be on the buyer's books"),
    MacroSetting(
        "days_per_year", "planning_parameters.md", YAML_BLOCK, "number",
        note="convention — changes every figure",
        impact="the denominator behind every DIOH figure"),
    MacroSetting(
        "quantity_rounding", "planning_parameters.md", YAML_BLOCK, "choice",
        choices=("integer", "none"),
        note="convention — changes every figure",
        impact="whether a countable quantity is reported as a whole unit"),
)

BY_NAME: Dict[str, MacroSetting] = {s.name: s for s in SETTINGS}


@dataclass
class MacroProposal:
    """A change that has not been made: what it would do, and to which bytes."""

    name: str
    filename: str
    path: str
    before: Any
    after: Any
    diff: str
    # The file as it was read. An apply quotes it back, and a mismatch means the file
    # moved between the diff being shown and the change being approved.
    basis: str
    impact: str = ""
    # True when the file already says this. Kept rather than raising: a form that
    # resubmits the value on screen is not making a mistake, and "nothing to change" is
    # a clearer answer than an error.
    unchanged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "file": self.filename, "path": self.path,
                "from": self.before, "to": self.after, "diff": self.diff,
                "basis": self.basis, "impact": self.impact,
                "unchanged": self.unchanged}


@dataclass
class MacroChange:
    """A change that has been made, and the line recorded about it."""

    proposal: MacroProposal
    by: str
    reason: str
    at: str
    after_digest: str
    log_path: str

    def to_dict(self) -> Dict[str, Any]:
        body = self.proposal.to_dict()
        body.update({"applied": True, "by": self.by, "reason": self.reason,
                     "at": self.at, "digest": self.after_digest,
                     "recorded_in": self.log_path})
        return body


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _config_root(config_dir=None) -> Path:
    if config_dir is None:
        return Path(__file__).parents[2] / "config"
    return Path(config_dir)


# ── Finding the one line a setting is on ─────────────────────────────────────

def _anchor(setting: MacroSetting) -> re.Pattern:
    """
    The line the value sits on, split so the value alone can be replaced.

    Group 1 is everything up to the value, group 2 the value, group 3 the tail — a
    trailing comma in JSON, a trailing `# comment` in YAML. Keeping the tail rather than
    rebuilding the line is what preserves a comment written beside a setting, and those
    comments are frequently the only explanation of what the setting means.
    """
    if setting.syntax == JSON:
        return re.compile(
            rf'^(?P<head>\s*"{re.escape(setting.name)}"\s*:\s*)'
            rf'(?P<value>"(?:[^"\\]|\\.)*"|-?[0-9][0-9.eE+-]*|true|false|null)'
            rf'(?P<tail>\s*,?\s*)$',
            re.MULTILINE)
    return re.compile(
        rf'^(?P<head>\s*{re.escape(setting.name)}\s*:\s+)'
        rf'(?P<value>[^#\n]*?)'
        rf'(?P<tail>\s*(?:#.*)?)$',
        re.MULTILINE)


def _locate(text: str, setting: MacroSetting) -> re.Match:
    """
    The single line to edit, or a refusal.

    Refusing on two matches rather than taking the first is the point. These files carry
    prose, and a setting name occurring twice means either the file says the same thing
    in two places — where editing one and not the other is worse than editing neither —
    or the anchor has caught something that is not the setting. Neither is a case for
    guessing.
    """
    found = list(_anchor(setting).finditer(text))
    if not found:
        raise MacroError(
            f"{setting.name} is not in {setting.filename} as a single line this can "
            f"edit. It is still editable in the file itself.")
    if len(found) > 1:
        lines = ", ".join(str(text[:m.start()].count("\n") + 1) for m in found)
        raise MacroError(
            f"{setting.name} appears {len(found)} times in {setting.filename} "
            f"(lines {lines}). Which one is authoritative is a judgement this will not "
            f"make — edit the file.")
    return found[0]


# ── Reading and writing a value ──────────────────────────────────────────────

def _read_value(raw: str, setting: MacroSetting) -> Any:
    """The value as it stands, typed the way the loader will see it."""
    raw = raw.strip()
    if setting.syntax == JSON:
        return json.loads(raw)
    import yaml
    return yaml.safe_load(raw)


def _render(value: Any, setting: MacroSetting) -> str:
    """
    The value as text for this syntax.

    A number goes in as the shortest text that reads back as itself — `repr` on a float,
    which gives `0.45` rather than `0.45000000000000001` — because these files are read
    by people and a value that grew fifteen digits on being saved would be the most
    visible thing in the diff and the least meaningful.
    """
    if setting.kind == "number":
        number = _as_number(value, setting)
        return repr(number) if isinstance(number, float) else str(number)
    if setting.syntax == JSON:
        return json.dumps(value, ensure_ascii=False)
    # YAML: a bare scalar unless the text would be read as structure once written.
    text = str(value).strip()
    if text and not any(c in text for c in "#:{}[],&*!|>'\"%@`\n"):
        return text
    import yaml
    return yaml.safe_dump(value, default_flow_style=True,
                          allow_unicode=True).strip().rstrip("\n")


def _as_number(value: Any, setting: MacroSetting):
    if isinstance(value, bool):
        raise MacroError(f"{setting.name} is a number, not a true/false")
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    try:
        return int(text) if re.fullmatch(r"-?\d+", text) else float(text)
    except ValueError:
        raise MacroError(f"{setting.name} must be a number — {value!r} is not") from None


def _coerce(value: Any, setting: MacroSetting) -> Any:
    """
    The value in the type the file should hold, before anything is rendered.

    `choices` is checked here because a dropdown's value arriving as free text is the
    one class of mistake the engine cannot catch for us: `pipeline_basis: incoterm_awre`
    parses, loads, and silently changes which branch every SKU takes.
    """
    if setting.kind == "number":
        return _as_number(value, setting)
    text = str(value).strip()
    if setting.kind == "choice" and text not in setting.choices:
        raise MacroError(
            f"{setting.name} must be one of {', '.join(setting.choices)} — "
            f"{value!r} is not. These are the branches the engine actually takes; "
            f"anything else would parse and then be ignored.")
    if not text:
        raise MacroError(f"{setting.name} cannot be blank")
    return text


# ── Validating by loading ────────────────────────────────────────────────────

def _validate(text: str, setting: MacroSetting, config_dir: Path) -> None:
    """
    Parse the edited file with the loader the pipeline uses.

    Not a second implementation of the rules: `cycle_stock_basis` is checked by
    `PlanningParameters._validate` and `quantity_rounding` by `analytics.rounding`, both
    of which already exist and are what the run will enforce. Anything this module
    checked separately would be a second answer the day one of them changed.
    """
    if setting.syntax == JSON:
        try:
            json.loads(text)
        except ValueError as exc:
            raise MacroError(f"the edit leaves {setting.filename} invalid: {exc}") from exc
        if setting.filename == "fx_rates.json":
            _validate_via(_load_fx, text, setting, config_dir)
        return

    _validate_via(_load_parameters, text, setting, config_dir)


def _load_fx(path: Path):
    from ..fx import FxTable

    return FxTable.load(path.parent)


def _load_parameters(path: Path):
    from ..analytics.rounding import Rounding
    from .parameters import PlanningParameters

    params = PlanningParameters(path)
    # The conventions are checked on load. `quantity_rounding` is checked where it is
    # read instead, which on a real run is an hour of work later — so it is checked here
    # too, by calling the same reader rather than by restating what it accepts.
    Rounding.from_conventions(params.conventions)
    return params


def _validate_via(load, text: str, setting: MacroSetting, config_dir: Path) -> None:
    """
    Run `load` over the edited text in a scratch copy of the config directory.

    A copy rather than the real file: a loader that raises after a partial write would
    leave the pipeline's own configuration broken, and the whole point of proposing a
    change before applying it is that nothing is at stake until it is approved.
    """
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(prefix="macro-check-") as tmp:
        scratch = Path(tmp) / "config"
        scratch.mkdir()
        for existing in config_dir.glob("*"):
            if existing.is_file():
                shutil.copy2(existing, scratch / existing.name)
        target = scratch / setting.filename
        target.write_text(text, encoding="utf-8")
        try:
            load(target)
        except MacroError:
            raise
        except Exception as exc:
            raise MacroError(
                f"the edit is refused by the loader that reads {setting.filename}: "
                f"{exc}") from exc


# ── The two operations ───────────────────────────────────────────────────────

def current(config_dir=None) -> Dict[str, Any]:
    """Every editable setting's value as the file holds it, by name."""
    root = _config_root(config_dir)
    out: Dict[str, Any] = {}
    for setting in SETTINGS:
        path = root / setting.filename
        try:
            text = path.read_text(encoding="utf-8")
            out[setting.name] = _read_value(_locate(text, setting).group("value"), setting)
        except (OSError, MacroError, ValueError):
            continue
    return out


def propose(name: str, value: Any, config_dir=None) -> MacroProposal:
    """
    What changing one setting would do, without doing it.

    Everything that can refuse the change refuses it here — the name, the type, the
    choices, and the loader — so that an approved proposal is one that has already been
    shown to work. An apply that could still fail would make the diff a suggestion
    rather than a promise.
    """
    setting = BY_NAME.get(name)
    if setting is None:
        raise MacroError(
            f"{name!r} is not an editable setting. Editable: "
            f"{', '.join(sorted(BY_NAME))}. Anything else in these files is either "
            f"read by nothing yet or derived from something that is.")

    root = _config_root(config_dir)
    path = root / setting.filename
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MacroError(f"{path} cannot be read: {exc}") from exc

    match = _locate(text, setting)
    before = _read_value(match.group("value"), setting)
    after = _coerce(value, setting)
    rendered = _render(after, setting)
    edited = (text[:match.start("value")] + rendered + text[match.end("value"):])

    if edited == text:
        return MacroProposal(name=setting.name, filename=setting.filename,
                             path=str(path), before=before, after=before, diff="",
                             basis=_digest(text), impact=setting.impact, unchanged=True)

    _validate(edited, setting, root)
    diff = "".join(difflib.unified_diff(
        text.splitlines(keepends=True), edited.splitlines(keepends=True),
        fromfile=f"a/{setting.filename}", tofile=f"b/{setting.filename}", n=3))
    return MacroProposal(name=setting.name, filename=setting.filename, path=str(path),
                         before=before, after=after, diff=diff, basis=_digest(text),
                         impact=setting.impact)


def apply(name: str, value: Any, *, reason: str, by: str, basis: str,
          config_dir=None) -> MacroChange:
    """
    Make the change, and record who made it and why.

    `basis` is the digest the proposal reported, and it is required rather than
    optional. Without it an approval means "change this setting", when what the person
    approved was a specific diff against specific bytes; with it, a file that moved in
    between is a refusal instead of a surprise.

    `reason` and `by` are required for the same reason `Declarations.write_override`
    requires them: a change made by clicking has only these two fields to say who
    asserted it and why, and an unattributed change to a convention that restates every
    figure in the run is exactly what this layer exists to replace.
    """
    if not str(reason or "").strip():
        raise MacroError(
            "a macro change written through an interface must carry a reason — these "
            "settings restate the whole run and the reason is the only account of why")
    if not str(by or "").strip():
        raise MacroError("a macro change written through an interface must name who "
                         "made it")

    proposal = propose(name, value, config_dir=config_dir)
    if proposal.unchanged:
        raise MacroError(
            f"{name} is already {proposal.before!r} — nothing to apply. A change that "
            f"writes nothing should not appear in the log as though something moved.")
    if not basis or basis != proposal.basis:
        raise MacroError(
            f"{proposal.filename} has changed since that diff was produced. The change "
            f"is not applied: re-read the setting and approve the diff against the "
            f"file as it stands now.")

    path = Path(proposal.path)
    edited = path.read_text(encoding="utf-8")
    match = _locate(edited, BY_NAME[name])
    edited = (edited[:match.start("value")] + _render(proposal.after, BY_NAME[name])
              + edited[match.end("value"):])
    path.write_text(edited, encoding="utf-8")

    at = datetime.now().isoformat(timespec="seconds")
    log_path = _record(_config_root(config_dir), {
        "at": at, "by": str(by), "reason": " ".join(str(reason).split()),
        "setting": proposal.name, "file": proposal.filename,
        "from": proposal.before, "to": proposal.after,
        "basis": proposal.basis, "digest": _digest(edited),
    })
    return MacroChange(proposal=proposal, by=str(by),
                       reason=" ".join(str(reason).split()), at=at,
                       after_digest=_digest(edited), log_path=str(log_path))


def _record(root: Path, entry: Dict[str, Any]) -> Path:
    """
    Append the change to `config/macro_changes.jsonl`.

    Beside the files it describes, and in the config directory rather than the output
    directory, because it belongs to the configuration and travels with it: a config
    directory copied to another machine carries its own history of who changed what.
    Append-only, one line per change — the same shape as the batch ledger, for the same
    reason, which is that a history that can be rewritten is not one.

    The diff is deliberately not stored. It is recoverable from the file's own history
    and it would be the longest field in a log meant to be read; what cannot be
    recovered from the file is the reason, and that is what this keeps.
    """
    path = root / CHANGE_LOG_NAME
    root.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def history(config_dir=None, limit: int = 50) -> List[Dict[str, Any]]:
    """The recorded changes, newest first. A malformed line is skipped, not fatal."""
    path = _config_root(config_dir) / CHANGE_LOG_NAME
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            continue
    return list(reversed(entries))[:limit]
