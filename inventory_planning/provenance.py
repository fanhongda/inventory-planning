"""
Run identity.

Every output this pipeline writes is a function of three things — the facts it read,
the parameters it resolved, and the code that ran — and until now none of them was
recorded beside the result. A timestamped CSV says when a recommendation was produced
and nothing about what produced it, so the four questions that all reduce to comparing
two runs could not be asked at all:

  drift             the same SKU across runs
  scenario          the same facts under different rules
  feedback learning what a run recommended, against what later happened
  a policy UI       the effect of a rule change, which is a *diff* of two runs

A `RunManifest` records the three inputs to a run and the outputs that came out of it.
Two fingerprints do the work: `input_fingerprint` over the source files and
`config_fingerprint` over the resolved parameters and rules. They are what let
`compare()` say *why* two runs differ, which is the part that stops a comparison being
misread — a lower recommended buy is a policy result when the facts are pinned and a
data problem when they are not.

The manifest never blocks a run. A missing git binary, an unreadable config file, an
input handed in as a DataFrame with no path behind it: each is recorded as unknown and
the run proceeds. Provenance that can fail the pipeline it documents would be worse
than none.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import dataclass, field as dc_field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

RUNS_DIRNAME = "runs"
INDEX_NAME = "index.jsonl"

# Files whose contents decide what the pipeline computes, as opposed to what it reads.
# `planning_parameters.md` is in here because the rule engine is driven by it.
_CONFIG_GLOBS = ("*.json", "planning_parameters.md")


def sha256_file(path: Path) -> Optional[str]:
    """
    Content digest of one file, or None if it cannot be read.

    Public because the interface has to ask the same question of the rules file that a
    run asked — "are these the bytes that run read?" — and a second implementation of
    that hash would be a second answer the day one of them changed.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _sha256_of(parts) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _git(args: List[str], cwd: Path) -> Optional[str]:
    try:
        out = subprocess.run(["git"] + args, cwd=str(cwd), capture_output=True,
                             text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@dataclass
class InputRecord:
    """One source file, as it was when the run read it."""

    doc_type: str
    name: str
    path: Optional[str] = None
    sha256: Optional[str] = None
    bytes: Optional[int] = None
    modified_at: Optional[str] = None
    rows: Optional[int] = None
    adapter: Optional[str] = None
    # From the contract tests. A run reading a degraded key is perfectly valid; the
    # verdict is recorded so a later store knows these rows cannot be superseded.
    key_verdict: Optional[str] = None
    storable: Optional[bool] = None

    @property
    def identity(self) -> str:
        """What makes this input *this* input — content, or failing that, its name."""
        return self.sha256 or f"<unhashed:{self.name}>"


@dataclass
class OutputRecord:
    name: str
    rows: Optional[int] = None
    bytes: Optional[int] = None


@dataclass
class RuleHitRecord:
    """
    What one rule reached in one run.

    The rule ids alone say which rules were in force; they say nothing about which of
    them did anything, and that is the question anyone editing a rule is actually
    asking. The engine has always worked this out — it prints it — and then dropped it
    on the floor, so the only way to see a rule's reach was to run the pipeline and
    read the console.

    Enough of the rule is copied in for the record to be legible on its own. That
    duplicates the rules file, which is normally the wrong trade; here it is the point,
    because the file moves and the manifest describes a run that has already happened.
    A record saying `R-014 matched 23` against a scope that has since been rewritten
    would be worse than no record.
    """

    rule_id: str
    name: str = ""
    scope: str = ""
    sets: Dict[str, Any] = dc_field(default_factory=dict)
    matched: int = 0
    # Where at least one of its values was still standing at the end of the file.
    # Below `matched` means a later rule took some of it back; zero means the rule ran
    # and decided nothing.
    effective: Optional[int] = None
    sample_skus: List[str] = dc_field(default_factory=list)
    # A rule whose scope names a column the run did not have matches nothing and is not
    # a failure — but it is also not a rule that matched zero SKUs, and a screen that
    # showed both as `0` would send someone to rewrite a scope that is perfectly good.
    unavailable_columns: List[str] = dc_field(default_factory=list)
    # param -> the earlier rule whose value this one replaced
    overrides_earlier: Dict[str, str] = dc_field(default_factory=dict)

    @property
    def skipped(self) -> bool:
        return bool(self.unavailable_columns)


@dataclass
class RunManifest:
    """What one run read, resolved and wrote."""

    run_id: str
    run_at: str
    output_dir: str = ""
    config_dir: str = ""
    code: Dict[str, Any] = dc_field(default_factory=dict)
    config_files: Dict[str, str] = dc_field(default_factory=dict)
    # The rules file the run actually resolved against, which is not necessarily one of
    # the files above: `run_policy_analysis(parameters_file=...)` takes a path anywhere
    # on disk, and running an alternate rule set is exactly what a scenario is.
    policy_files: Dict[str, str] = dc_field(default_factory=dict)
    rule_ids: List[str] = dc_field(default_factory=list)
    rule_hits: List[RuleHitRecord] = dc_field(default_factory=list)
    inputs: List[InputRecord] = dc_field(default_factory=list)
    outputs: List[OutputRecord] = dc_field(default_factory=list)
    notes: List[str] = dc_field(default_factory=list)

    # ── Start ────────────────────────────────────────────────────────────────

    @classmethod
    def begin(cls, config_dir: Path = None, output_dir: Path = None,
              policy_file: Path = None) -> "RunManifest":
        now = datetime.now()
        run = cls(
            # Sortable, and unique even when two runs land in the same second — the
            # output CSVs stamp to the minute and would collide long before this does.
            run_id=f"{now.strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6]}",
            run_at=now.isoformat(timespec="seconds"),
            output_dir=str(output_dir or ""),
            config_dir=str(config_dir or ""),
        )
        run._read_code()
        run._read_config(Path(config_dir) if config_dir else None)
        if policy_file is not None:
            run.set_policy_file(policy_file)
        return run

    def _read_code(self) -> None:
        here = Path(__file__).resolve().parent
        sha = _git(["rev-parse", "HEAD"], here)
        if sha is None:
            self.code = {"git_sha": None, "dirty": None}
            self.notes.append("code version unknown — not a git checkout, or git unavailable")
            return
        status = _git(["status", "--porcelain"], here)
        self.code = {
            "git_sha": sha,
            # A dirty tree means the sha does not describe what actually ran. Recording
            # the sha alone would be a more convincing lie than recording nothing.
            "dirty": bool(status) if status is not None else None,
            "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], here),
        }

    def _read_config(self, config_dir: Optional[Path]) -> None:
        if not config_dir or not config_dir.exists():
            self.notes.append("config directory not found — parameters unfingerprinted")
            return
        for pattern in _CONFIG_GLOBS:
            for path in sorted(config_dir.glob(pattern)):
                digest = sha256_file(path)
                if digest:
                    self.config_files[path.name] = digest

    # ── Recording ────────────────────────────────────────────────────────────

    def record_input(self, path, doc_type: str = "", **extra) -> InputRecord:
        p = Path(path) if path is not None else None
        rec = InputRecord(doc_type=doc_type, name=p.name if p else str(path or "<frame>"))
        if p is not None and p.exists():
            stat = p.stat()
            rec.path = str(p.resolve())
            rec.sha256 = sha256_file(p)
            rec.bytes = stat.st_size
            rec.modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        for k, v in extra.items():
            if hasattr(rec, k) and v is not None:
                setattr(rec, k, v)
        # One file re-read through a second entry point is one input, not two.
        for i, existing in enumerate(self.inputs):
            if existing.path and existing.path == rec.path:
                self.inputs[i] = _merge_inputs(existing, rec)
                return self.inputs[i]
        self.inputs.append(rec)
        return rec

    def record_intake(self, intake_result) -> None:
        """Enrich the inputs with what the contract layer worked out about them."""
        documents = getattr(intake_result, "documents", None) or {}
        # `IntakeResult.documents` is keyed by doc_type; accept a plain sequence too so
        # a caller holding just the documents does not have to wrap them.
        for doc in (documents.values() if hasattr(documents, "values") else documents):
            report = getattr(doc, "test_report", None)
            status = getattr(report, "key_status", None) if report is not None else None
            route = getattr(doc, "route", None)
            adapter = getattr(route, "adapter", None) if route is not None else None
            frame = getattr(doc, "frame", None)
            self.record_input(
                getattr(doc, "source_path", None),
                doc_type=getattr(doc, "doc_type", ""),
                rows=len(frame) if frame is not None else None,
                adapter=getattr(adapter, "slug", None),
                key_verdict=status.verdict if status is not None else None,
                storable=status.storable if status is not None else None,
            )

    def record_output(self, path, rows: int = None) -> None:
        """Idempotent by name: the collection runs again after the policy stage."""
        p = Path(path)
        record = OutputRecord(
            name=p.name,
            rows=rows,
            bytes=p.stat().st_size if p.exists() else None,
        )
        for i, existing in enumerate(self.outputs):
            if existing.name == record.name:
                self.outputs[i] = record
                return
        self.outputs.append(record)

    def record_rules(self, rule_ids) -> None:
        """The rules in force, recorded when the file is parsed."""
        self.rule_ids = [str(r) for r in rule_ids]

    def record_rule_hits(self, hits) -> None:
        """
        What those rules reached, recorded once the run has resolved against them.

        Separate from `record_rules` and deliberately later: the ids are known as soon
        as the file is read, the reach only after a frame exists to match against, and
        a run that dies in between should still record which rules were in force.

        Duck-typed over `policy.parameters.RuleHit` rather than importing it. Nothing
        else in this module knows what a rule is, and provenance that could fail to
        import would be provenance that can fail the run it documents.

        Like every other fingerprint input, this never moves one: the hits are an
        outcome of the run, and `policy_fingerprint` is over the rules file, fixed
        before the first batch was stamped with it.
        """
        records: List[RuleHitRecord] = []
        for hit in hits:
            rule = getattr(hit, "rule", None)
            records.append(RuleHitRecord(
                rule_id=str(getattr(rule, "rule_id", "") or getattr(hit, "rule_id", "")),
                name=str(getattr(rule, "name", "") or ""),
                scope=str(getattr(rule, "scope", "") or ""),
                sets=dict(getattr(rule, "overrides", None) or {}),
                matched=int(getattr(hit, "matched", 0) or 0),
                effective=getattr(hit, "effective", None),
                sample_skus=[str(s) for s in (getattr(hit, "sample_skus", None) or [])],
                unavailable_columns=[
                    str(c) for c in (getattr(hit, "unavailable_columns", None) or [])],
                overrides_earlier={
                    str(k): str(v) for k, v in
                    (getattr(hit, "overrides_earlier", None) or {}).items()},
            ))
        self.rule_hits = records
        # A run that resolved rules without anyone recording the ids still knows them.
        if not self.rule_ids:
            self.rule_ids = [r.rule_id for r in records]

    def set_policy_file(self, path) -> None:
        """
        The rule set this run plans under. Set once, when the run begins.

        The rules are an input, not a by-product: they decide the review period that
        sizes an order and the service level that sizes safety stock, so a run under a
        different rule set is a different run with different recommendations — which is
        what a scenario is. Recording them at the start is what lets `policy_fingerprint`
        hold still for the whole run, the same reason `config_fingerprint` is read here.
        """
        p = Path(path)
        digest = sha256_file(p)
        self.policy_files = {str(p): digest} if digest else {}
        if not digest:
            self.notes.append(f"rules file {p} unreadable — policy unfingerprinted")

    def note_policy_override(self, path) -> None:
        """
        A later stage resolved a *different* rules file than the run began with.

        Recorded as a note and deliberately not folded into the fingerprint: a value
        stamped on batches at intake cannot move afterwards without those batches
        ceasing to be joinable to their manifest. It is worth saying out loud, though —
        it means the plan and the report were produced under different rules.
        """
        p = str(Path(path))
        if p in self.policy_files:
            return
        planned_under = ", ".join(self.policy_files) or "<none>"
        self.notes.append(
            f"rules differ within the run: planned under {planned_under}, "
            f"policy report resolved {p}. The recommendations are the first file's. "
            f"For a scenario, construct the planner with parameters_file instead."
        )

    # ── Fingerprints ─────────────────────────────────────────────────────────

    @property
    def input_fingerprint(self) -> str:
        """The facts. Two runs sharing this read the same bytes."""
        return _sha256_of(sorted(f"{i.doc_type}:{i.identity}" for i in self.inputs))[:16]

    @property
    def config_fingerprint(self) -> str:
        """
        The configuration that transformed the inputs — FX table, incoterm rules, node
        map. Fixed at `begin()` and never moved afterwards, because this is the value
        stamped on every batch at intake time and a batch has to be joinable back to
        the manifest that describes the run that wrote it.

        The rules do not belong here. They decide what the pipeline concludes, not what
        the canonical frames contain, and they are resolved long after the batches are
        written. `policy_fingerprint` carries them instead.
        """
        return _sha256_of(
            f"{name}:{digest}" for name, digest in sorted(self.config_files.items())
        )[:16]

    @property
    def policy_fingerprint(self) -> str:
        """
        The rule set in force, which is what a scenario actually varies.

        Over the digest of the rules file the run plans under, which need not be the one
        in the config directory — a scenario is exactly a run under a rule set from
        somewhere else. Hashing the config directory alone made two such runs
        indistinguishable, and `compare` then called the pair `identical`, whose
        description tells the reader that a real policy result is non-determinism.

        Over the *file*, not the rule ids parsed from it. The ids are read during the
        run, and a fingerprint that moved partway through would leave every batch
        already written stamped with a value the manifest no longer holds. The file
        digest covers strictly more anyway: editing a rule's body changes it, where the
        ids would not.
        """
        return _sha256_of(
            f"{name}:{digest}" for name, digest in sorted(self.policy_files.items())
        )[:16]

    @property
    def unstorable_inputs(self) -> List[InputRecord]:
        return [i for i in self.inputs if i.storable is False]

    # ── Serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_at": self.run_at,
            "input_fingerprint": self.input_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "code": self.code,
            "output_dir": self.output_dir,
            "config_dir": self.config_dir,
            "config_files": self.config_files,
            "policy_files": self.policy_files,
            "rule_ids": self.rule_ids,
            "rule_hits": [asdict(h) for h in self.rule_hits],
            "inputs": [asdict(i) for i in self.inputs],
            "outputs": [asdict(o) for o in self.outputs],
            "notes": self.notes,
        }

    def index_entry(self) -> Dict[str, Any]:
        """The one line that goes in the registry index."""
        return {
            "run_id": self.run_id,
            "run_at": self.run_at,
            "input_fingerprint": self.input_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "policy_fingerprint": self.policy_fingerprint,
            "git_sha": (self.code or {}).get("git_sha"),
            "dirty": (self.code or {}).get("dirty"),
            "inputs": len(self.inputs),
            "outputs": len(self.outputs),
        }

    def _reach(self) -> str:
        """`, 9 standing` — silent when nothing recorded the reach."""
        if not self.rule_hits:
            return ""
        return f", {sum(1 for h in self.rule_hits if h.effective)} standing"

    def summary(self) -> str:
        code = self.code or {}
        sha = (code.get("git_sha") or "unknown")[:8]
        dirty = " +uncommitted" if code.get("dirty") else ""
        lines = [
            f"  Run {self.run_id}",
            f"    facts   {self.input_fingerprint}  ({len(self.inputs)} inputs)",
            f"    config  {self.config_fingerprint}  "
            f"({len(self.config_files)} files)",
            f"    policy  {self.policy_fingerprint}  "
            f"({len(self.rule_ids)} rules{self._reach()})",
            f"    code    {sha}{dirty}",
        ]
        inert = [h for h in self.rule_hits if not h.skipped and not h.effective]
        if inert:
            lines.append(f"    · {len(inert)} rule(s) decided nothing this run — "
                         f"{', '.join(h.rule_id for h in inert[:5])}")
        unstorable = self.unstorable_inputs
        if unstorable:
            lines.append(f"    · {len(unstorable)} input(s) on a partial natural key — "
                         f"plannable, not yet storable")
        return "\n".join(lines)


def _merge_inputs(a: InputRecord, b: InputRecord) -> InputRecord:
    """Later knowledge wins; earlier knowledge is kept where the later has none."""
    merged = InputRecord(**asdict(a))
    for key, value in asdict(b).items():
        if value is not None:
            setattr(merged, key, value)
    return merged


@dataclass
class RunComparison:
    """Why two runs differ — which is what decides how their outputs may be read."""

    a: Dict[str, Any]
    b: Dict[str, Any]

    @property
    def same_inputs(self) -> bool:
        return self.a.get("input_fingerprint") == self.b.get("input_fingerprint")

    @property
    def same_config(self) -> bool:
        """Both halves: what transformed the inputs, and what rules were in force."""
        return (self.a.get("config_fingerprint") == self.b.get("config_fingerprint")
                and self.a.get("policy_fingerprint") == self.b.get("policy_fingerprint"))

    @property
    def same_code(self) -> bool:
        return self.a.get("git_sha") == self.b.get("git_sha")

    @property
    def basis(self) -> str:
        if self.same_inputs and self.same_config and self.same_code:
            return "identical"
        if self.same_inputs and self.same_code and not self.same_config:
            return "scenario"          # facts pinned — a difference is a policy result
        if self.same_config and self.same_code and not self.same_inputs:
            return "new_data"          # policy pinned — a difference is the world moving
        return "mixed"                 # more than one axis moved; attribute nothing

    def describe(self) -> str:
        return {
            "identical": "same facts, same parameters, same code — any difference is "
                         "non-determinism, not a finding",
            "scenario":  "same facts and code, different parameters — differences are "
                         "attributable to the policy change",
            "new_data":  "same parameters and code, different facts — differences are "
                         "the world moving, not a policy result",
            "mixed":     "more than one of facts, parameters and code moved — nothing "
                         "here is attributable to any one of them",
        }[self.basis]


class RunRegistry:
    """Append-only record of the runs written under one output directory."""

    def __init__(self, output_dir):
        self.dir = Path(output_dir) / RUNS_DIRNAME

    @property
    def index_path(self) -> Path:
        return self.dir / INDEX_NAME

    def save(self, manifest: RunManifest) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{manifest.run_id}.json"
        path.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        with open(self.index_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(manifest.index_entry(), ensure_ascii=False) + "\n")
        return path

    def index(self) -> List[Dict[str, Any]]:
        """
        One entry per run, newest state last.

        The file is an append-only log and a run is written to it more than once: the
        planning stage records what it read, and the policy stage — which runs
        afterwards and may resolve an entirely different rule file — records what was
        in force. Later lines for a run supersede earlier ones, in the position the run
        first appeared, so ordering still reflects when each run started.
        """
        if not self.index_path.exists():
            return []
        order: List[str] = []
        latest: Dict[str, Dict[str, Any]] = {}
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = entry.get("run_id")
            if run_id is None:
                continue
            if run_id not in latest:
                order.append(run_id)
            latest[run_id] = entry
        return [latest[run_id] for run_id in order]

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self.dir / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def latest_under_rules(self, digest: Optional[str], *, with_hits: bool = False,
                           limit: int = 50) -> Optional[Dict[str, Any]]:
        """
        The newest saved run that resolved exactly these rule bytes.

        Matched on the digest of the rules file rather than on `policy_fingerprint`,
        which folds in the path the run was given: the same rules read from a scenario
        copy fingerprint differently, and the question here is the content one — did
        this run plan under these rules. Matching on content is also what makes the
        answer safe to line up rule by rule, since identical bytes cannot declare a
        different set of rules.

        `with_hits` skips runs that recorded no reach — runs from before it was kept,
        or that died before resolution. An older run under the same bytes says more
        about those rules than a newer one that recorded nothing.
        """
        if not digest:
            return None
        for entry in reversed(self.index()[-limit:]):
            manifest = self.get(entry.get("run_id", ""))
            if manifest is None:
                continue
            if digest not in (manifest.get("policy_files") or {}).values():
                continue
            if with_hits and not manifest.get("rule_hits"):
                continue
            return manifest
        return None

    def compare(self, run_a: str, run_b: str) -> Optional[RunComparison]:
        by_id = {e["run_id"]: e for e in self.index()}
        a, b = by_id.get(run_a), by_id.get(run_b)
        if a is None or b is None:
            return None
        return RunComparison(a=a, b=b)
