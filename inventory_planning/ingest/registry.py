"""
Adapter store and routing.

Given a file, decide which adapter handles it — or, when nothing matches, draft one
from the profile and the contract so the run can proceed and a human can review the
draft afterwards.

The routing order encodes a trust gradient:

  1. frozen/verified adapter matched by fingerprint   — deterministic, already reviewed
  2. draft adapter matched by fingerprint             — deterministic, not yet reviewed
  3. auto-drafted adapter from lexical detection      — best effort, flagged as draft

Only step 3 involves guessing, and its guesses are recorded in the returned adapter
so they show up in the transform log and the review diff rather than disappearing
into a DataFrame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .adapter import Adapter, ADAPTERS_DIR, Fingerprint, ParsingRules
from .contract import ContractRegistry, DocContract, default_registry
from .expressions import Expression, ExpressionError
from .profiler import Profiler, TableProfile, normalize_header

# Sentinel so a missing contract can be probed for `.discriminator` without a guard.
_NO_CONTRACT = DocContract(doc_type="", description="")


@dataclass
class RouteResult:
    """Outcome of routing one file."""

    adapter: Adapter
    contract: DocContract
    profile: TableProfile
    confidence: float
    reason: str
    is_draft: bool

    @property
    def doc_type(self) -> str:
        return self.contract.doc_type

    def summary(self) -> str:
        badge = "DRAFT" if self.is_draft else self.adapter.status.upper()
        return (
            f"  Routed {self.profile.source_name} -> {self.doc_type} "
            f"via {self.adapter.name} [{badge}] "
            f"(confidence {self.confidence:.0%}: {self.reason})"
        )


class AdapterRegistry:
    """Loads adapters from disk and routes profiles to them."""

    def __init__(
        self,
        adapters_dir: Path = None,
        contracts: ContractRegistry = None,
    ):
        self.adapters_dir = Path(adapters_dir) if adapters_dir else ADAPTERS_DIR
        self.contracts = contracts or default_registry()
        self._adapters: List[Adapter] = []
        self._loaded = False

    # ── Store ────────────────────────────────────────────────────────────────

    def _load_all(self) -> None:
        if self._loaded:
            return
        if self.adapters_dir.exists():
            for path in sorted(self.adapters_dir.rglob("*.yaml")):
                try:
                    self._adapters.append(Adapter.load(path))
                except Exception as exc:
                    print(f"  ⚠ Skipping malformed adapter {path.name}: {exc}")
        self._loaded = True

    @property
    def adapters(self) -> List[Adapter]:
        self._load_all()
        return list(self._adapters)

    def save(self, adapter: Adapter) -> Path:
        """Persist an adapter into the store, versioned by tenant/system/doc_type."""
        folder = self.adapters_dir / f"{adapter.tenant}__{adapter.system}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{adapter.doc_type}.v{adapter.version}.yaml"
        path.write_text(adapter.to_yaml(), encoding="utf-8")
        adapter.source_path = path
        if adapter not in self._adapters:
            self._adapters.append(adapter)
        return path

    def candidates_for(self, doc_type: str) -> List[Adapter]:
        return [a for a in self.adapters if a.doc_type == doc_type]

    # ── Routing ──────────────────────────────────────────────────────────────

    def route(
        self,
        df: pd.DataFrame,
        source_name: str = "<dataframe>",
        doc_type_hint: str = None,
        tenant: str = "default",
    ) -> RouteResult:
        """
        Identify a frame: profile it, try every adapter's fingerprint, fall back to
        drafting. `doc_type_hint` short-circuits document classification when the
        caller already knows what the file is.
        """
        profile = Profiler().profile(df, source_name=source_name)

        best: Optional[Tuple[Adapter, float, str]] = None
        for adapter in self.adapters:
            if doc_type_hint and adapter.doc_type != doc_type_hint:
                continue
            matched, confidence, reason = adapter.fingerprint.match(profile)
            if not matched:
                continue
            # Prefer higher confidence; break ties toward reviewed adapters and
            # then toward the newer version.
            rank = (confidence, adapter.status == "frozen", adapter.status == "verified",
                    adapter.version)
            if best is None or rank > (best[1], best[0].status == "frozen",
                                       best[0].status == "verified", best[0].version):
                best = (adapter, confidence, reason)

        if best is not None:
            adapter, confidence, reason = best
            return RouteResult(
                adapter=adapter,
                contract=self.contracts.get(adapter.doc_type),
                profile=profile,
                confidence=confidence,
                reason=reason,
                is_draft=adapter.status == "draft",
            )

        doc_type, doc_confidence, doc_reason = (
            (doc_type_hint, 1.0, "caller-supplied hint")
            if doc_type_hint
            else self.classify(profile, df)
        )
        contract = self.contracts.get(doc_type)
        adapter = self.draft(profile, contract, tenant=tenant, raw=df)
        return RouteResult(
            adapter=adapter,
            contract=contract,
            profile=profile,
            confidence=doc_confidence,
            reason=f"no adapter matched; drafted from profile ({doc_reason})",
            is_draft=True,
        )

    # ── Document classification ──────────────────────────────────────────────

    def classify(
        self, profile: TableProfile, df: pd.DataFrame = None
    ) -> Tuple[str, float, str]:
        """
        Decide which contract a file satisfies, by scoring how well its headers cover
        each contract's aliases — weighted toward required fields, since those are
        what actually distinguish the documents.

        Shape is decisive before scoring: a wide period layout can only be a
        pre-aggregated demand series, and that is precisely the planner-supplied
        time series case that used to need a separate, explicitly-called loader.
        """
        if profile.shape == "wide_periods":
            return (
                "demand_timeseries",
                0.95,
                f"wide layout with {len(profile.period_columns)} period columns",
            )

        scores: Dict[str, float] = {}
        details: Dict[str, str] = {}

        for doc_type, contract in self.contracts.all().items():
            if doc_type == "demand_timeseries":
                continue   # only reachable via shape, never via aliases
            # Score on the mapping that would actually be drafted, not on a separate
            # exact-match pass. When the two disagree, a file gets classified by one
            # rule and mapped by another — which is how `PO_ORDER_QTY` can be
            # unrecognised at classification time yet mapped fine afterwards, leaving
            # the document routed to entirely the wrong contract.
            hit_fields = set(self._assign_columns(profile, contract))

            required = set(contract.required_fields)
            derivable = set(contract.derivable_fields)
            # A required field counts as covered if it is present OR reconstructable
            covered_required = {
                f for f in required
                if f in hit_fields or (f in derivable and self._derivable_now(contract, f, hit_fields))
            }

            req_score = len(covered_required) / max(len(required), 1)
            breadth = len(hit_fields) / max(len(contract.fields), 1)
            # Required coverage dominates; breadth only breaks ties between documents
            # that share an identifier vocabulary (every doc has `sku`).
            score = 0.75 * req_score + 0.25 * breadth

            # A contract that declares what makes it itself, and finds none of it, is
            # not a weak match — it is the wrong document. Without this a contract
            # requiring only `sku` scores a full 0.75 on any file with an item column
            # and outranks the one that genuinely fits.
            missing_identity = [
                group for group in contract.identifying_any
                if not (set(group) & hit_fields)
            ]
            if missing_identity:
                score = 0.0

            scores[doc_type] = score
            details[doc_type] = (
                f"{len(covered_required)}/{len(required)} required, "
                f"{len(hit_fields)}/{len(contract.fields)} fields"
                + (f"; carries none of {missing_identity[0]}" if missing_identity else "")
            )

        if not scores:
            raise ValueError("No contracts available to classify against")

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_type, best_score = ranked[0]
        reason = details[best_type]

        # Every candidate within a whisker of the best. Header scores bunch up because
        # these documents genuinely share most of their vocabulary — a PO line and a
        # receipt line differ by one date, not by their column list — so the runner-up
        # alone is the wrong shortlist.
        contenders = [t for t, s in ranked if best_score - s < 0.08]
        if len(contenders) > 1:
            # Once the headers are known to be ambiguous, put every document with a
            # content test on the shortlist — not just the ones that happened to score
            # inside the band. The header score is exactly the signal we have already
            # judged unreliable here, so using it to gate the shortlist reintroduces
            # the problem the content test exists to solve.
            contenders = list(dict.fromkeys(
                contenders + [t for t, _ in ranked
                              if (self.contracts.all().get(t) or _NO_CONTRACT).discriminator]
            ))
            decided = self._discriminate(df, profile, contenders) if df is not None else None
            if decided:
                winner, share, runner_up, runner_share = decided
                versus = (f"vs {runner_share:.0%} for {runner_up}" if runner_up
                          else "and no rival document's test even applied")
                return winner, best_score, (
                    f"{details[winner]}; headers could not separate "
                    f"{len(contenders)} candidates, so the content decided — "
                    f"its discriminator holds for {share:.0%} of rows {versus}"
                )
            others = ", ".join(f"{t} ({scores[t]:.0%})" for t in contenders[1:])
            reason += f"  ⚠ close call vs {others}"
        return best_type, best_score, reason

    def _discriminate(
        self, df: pd.DataFrame, profile: TableProfile, candidates: List[str]
    ) -> Optional[Tuple[str, float, float]]:
        """
        Choose between look-alike documents by testing each contract's discriminator
        against the rows.

        A provisional mapping is drafted per candidate first, because the discriminator
        is written in canonical field names while the frame still carries the source's.
        Both must hold clearly and differently for the verdict to count — a marginal
        difference is not evidence, and reporting the tie is more useful than guessing.
        """
        shares: Dict[str, float] = {}
        for doc_type in candidates:
            contract = self.contracts.all().get(doc_type)
            if contract is None or not contract.discriminator:
                continue

            # Empty columns are included here and nowhere else. A `Ship Date` column
            # that exists but is entirely blank is the defining evidence of an open
            # order, yet the normal mapping skips empty columns — so without this the
            # one signal that settles the question is the one thing invisible to it.
            column_map = self._assign_columns(profile, contract, include_empty=True)
            renamed = df.rename(columns={v: k for k, v in column_map.items()})
            # The most decisive content signals are usually on a field the export does
            # not carry directly. `open_qty` is the case in point: a purchase extract
            # showing ordered and received quantities says plainly that its lines are
            # closed, but only once the outstanding balance is worked out. Leaving it
            # underivable here made "nothing is open, so this is history" unusable on
            # exactly the files that need it.
            available = set(column_map)
            # Sources are read as text throughout, deliberately — pandas' inference is
            # where a European decimal or a leading-zero part number gets mangled. But
            # a content test asking `open_qty > 0` then compares str to int and raises,
            # and the test is silently skipped. On a real SAP extract that removed the
            # one signal able to tell an open PO from a purchase history.
            renamed = self._numeric_for_tests(renamed, contract, available)
            renamed, derived = self._derive_for_discriminator(renamed, contract, available)
            available |= derived

            # Every applicable test is evaluated and the strongest result stands.
            # A contract's tests are alternative readings of one claim — for a
            # purchase history, "the line was received" and "nothing is left
            # outstanding" both say it is history — and which of them an export can
            # support varies by ERP. Taking merely the first applicable one judged a
            # 52%-received extract on that alone, when 85% of its lines were closed
            # and said so plainly.
            for source in (contract.discriminators or [contract.discriminator]):
                expr = Expression(source)
                # The referenced column must genuinely exist in the source.
                # Substituting NaN for an absent column makes `is_null(...)` vacuously
                # true, and a sales export would then score 100% as an open PO purely
                # by having no receipt-date column at all — absence of evidence read
                # as evidence.
                if not expr.columns <= available:
                    continue
                try:
                    mask = expr.evaluate(renamed)
                except (ExpressionError, TypeError, ValueError):
                    continue
                if not isinstance(mask, pd.Series):
                    continue
                share = float(mask.fillna(False).astype(bool).mean())
                shares[doc_type] = max(shares.get(doc_type, 0.0), share)

        if not shares:
            return None

        ordered = sorted(shares.items(), key=lambda kv: -kv[1])
        winner, top = ordered[0]

        if top < 0.80:
            return None

        if len(ordered) == 1:
            # One candidate's test is applicable and it holds. That is decisive on its
            # own: the others could not even be evaluated because the source lacks the
            # fields they turn on, which is itself the answer.
            return winner, top, "", 0.0

        runner_up, second = ordered[1]
        # With two live tests, demand a clear margin. A narrow gap is not evidence, and
        # reporting the tie is more useful than committing to a coin flip.
        if top - second < 0.30:
            return None
        return winner, top, runner_up, second

    @staticmethod
    def _numeric_for_tests(frame, contract: DocContract, available: set):
        """
        Coerce the numeric fields a content test may touch, for evaluation only.

        Nothing here is kept: the adapter does the real parsing with the source's own
        locale rules. This exists solely so a comparison in a discriminator sees
        numbers instead of the strings every file is read as.
        """
        numeric = {}
        for name in available:
            spec = contract.field(name)
            if spec is None or not spec.is_numeric or name not in frame.columns:
                continue
            numeric[name] = pd.to_numeric(
                frame[name].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )
        return frame.assign(**numeric) if numeric else frame

    @staticmethod
    def _derive_for_discriminator(frame, contract: DocContract, available: set):
        """
        Compute the derivable fields a content test needs, before evaluating it.

        Only fields whose derivation inputs are all present are attempted, and a
        failure is silent — this is a classification aid, not the transform, and the
        adapter derives properly once the document type is settled.
        """
        wanted = {
            f for source in (contract.discriminators or [contract.discriminator])
            if source
            for f in Expression(source).columns
        } - available
        derived = set()
        for name in wanted:
            spec = contract.field(name)
            if spec is None or not spec.derivable_from:
                continue
            for source in spec.derivable_from:
                expr = Expression(source)
                if not expr.columns <= available:
                    continue
                # A single-column derivation is a fallback assumption, not a
                # measurement: `open_qty` from `order_qty` alone just asserts that
                # nothing was delivered. Fine when transforming a document already
                # known to be an open PO, but as a content test it manufactures the
                # very evidence it is being asked for — every order book would read
                # as fully open.
                if len(expr.columns) < 2:
                    continue
                try:
                    numeric = frame[list(expr.columns)].apply(
                        pd.to_numeric, errors="coerce"
                    )
                    value = expr.evaluate(frame.assign(**numeric))
                except (ExpressionError, TypeError, ValueError, KeyError):
                    continue
                if isinstance(value, pd.Series):
                    frame = frame.assign(**{name: value})
                    derived.add(name)
                    break
        return frame, derived

    @staticmethod
    def _derivable_now(contract: DocContract, field_name: str, hit_fields: set) -> bool:
        """True when at least one derivation path for this field has all its inputs."""
        spec = contract.field(field_name)
        if not spec:
            return False
        from .expressions import Expression

        for source in spec.derivable_from:
            if Expression(source).columns <= hit_fields:
                return True
        return False

    # ── Drafting ─────────────────────────────────────────────────────────────

    def draft(
        self,
        profile: TableProfile,
        contract: DocContract,
        tenant: str = "default",
        raw: pd.DataFrame = None,
    ) -> Adapter:
        """
        Build a best-effort adapter from lexical alias matching plus the contract's
        own derivation rules.

        This is deliberately not an LLM call. It is the deterministic floor: whatever
        can be resolved by matching names and applying contract-declared derivations
        is resolved here, so that when a model is asked to fill the remaining gaps it
        is given a small, specific problem rather than a blank adapter.
        """
        column_map = self._assign_columns(profile, contract)
        mapped = set(column_map)
        derivations: Dict[str, str] = {}
        for field_name in contract.derivable_fields:
            if field_name in mapped:
                continue
            spec = contract.field(field_name)
            from .expressions import Expression

            for source in spec.derivable_from:
                if Expression(source).columns <= mapped:
                    derivations[field_name] = source
                    break

        signature = self._signature_columns(profile, column_map)
        unresolved = [
            f for f in contract.required_fields
            if f not in column_map and f not in derivations
        ]

        notes = [
            "AUTO-DRAFTED from profile — review before freezing.",
            f"Source shape: {profile.shape}; {profile.row_count:,} rows.",
        ]

        source_grain, rollup_to = self._detect_grain(profile, contract, column_map, raw, notes)
        if unresolved:
            notes.append(
                f"UNRESOLVED required field(s): {unresolved}. "
                f"Supply a column_map entry or a derivation for each."
            )
        unmapped_cols = [
            c.name for c in profile.columns
            if not c.is_period_column and c.name not in set(column_map.values())
        ]
        if unmapped_cols:
            notes.append(f"Source columns not used: {unmapped_cols[:12]}")

        # Locale is a property of the exporting system, so it is inferred once from
        # the profile and frozen into the adapter rather than re-guessed each run.
        parsing = ParsingRules(dayfirst=profile.dayfirst)
        if profile.decimal_style == "eu":
            parsing.decimal_sep, parsing.thousands_sep = ",", "."

        return Adapter(
            name=f"{tenant}__{self._slugify(profile.source_name)}__{contract.doc_type}",
            doc_type=contract.doc_type,
            version=1,
            tenant=tenant,
            system=self._slugify(profile.source_name),
            description=f"Auto-drafted adapter for {profile.source_name}",
            fingerprint=Fingerprint(
                header_hash=profile.header_hash,
                header_contains=signature,
                min_confidence=0.6,
            ),
            grain=source_grain,
            rollup_to=rollup_to,
            column_map=column_map,
            derivations=derivations,
            parsing=parsing,
            status="draft",
            notes=notes,
        )

    def _detect_grain(
        self,
        profile: TableProfile,
        contract: DocContract,
        column_map: Dict[str, str],
        raw: Optional[pd.DataFrame],
        notes: List[str],
    ) -> Tuple[str, Optional[str]]:
        """
        Decide whether the source sits at a finer grain than the contract expects,
        and if so propose the rollup rather than merely warning about it.

        Detecting this matters more than it looks. A PO *schedule line* export has two
        or three rows per PO line; summed without a rollup, every quantity is inflated
        by that factor, the inventory position looks healthy, and the pipeline quietly
        stops recommending purchases. Nothing downstream can notice, because the
        numbers remain internally consistent.
        """
        declared = contract.grain
        key_fields = [k for k in contract.natural_key if k in column_map]

        # A natural key that loses its time dimension stops being a key. Where the
        # contract names a timestamp among its key fields and *that* column is absent,
        # any other mapped timestamp stands in for it — two lines for one SKU on
        # different dates are different rows, not a duplicate to be summed away.
        #
        # Without this, a sales history exporting an invoice date but no order number
        # and no ship date keys on `sku` alone: 540 transaction lines "collapse" to 30,
        # the rollup sums the quantities, `demand_date` takes the first value, and the
        # entire demand history becomes a single period. Nothing errors — the forecast
        # simply has one point per SKU to work from.
        if not any(contract.field(k) and contract.field(k).role == "timestamp"
                   for k in key_fields):
            wanted = [k for k in contract.natural_key
                      if contract.field(k) and contract.field(k).role == "timestamp"]
            if wanted:
                stand_in = next(
                    (name for name, spec in contract.fields.items()
                     if spec.role == "timestamp" and name in column_map),
                    None,
                )
                if stand_in:
                    key_fields = key_fields + [stand_in]
                    notes.append(
                        f"Natural key {contract.natural_key} needs a timestamp and "
                        f"{wanted} did not map; using {stand_in} so the grain keeps its "
                        f"time dimension."
                    )

        if raw is None or not key_fields or len(raw) == 0:
            return declared, None

        key_cols = [column_map[k] for k in key_fields]
        try:
            distinct = len(raw.drop_duplicates(subset=key_cols))
        except (TypeError, KeyError):
            return declared, None

        if distinct >= len(raw):
            return declared, None

        # Find which extra mapped identifier explains the multiplicity — that column
        # names the true grain.
        ratio = len(raw) / max(distinct, 1)
        discriminator = None
        for canonical in ("po_line_number", "po_schedule_line", "location_id", "bin", "lot"):
            col = column_map.get(canonical)
            if col and col in raw.columns:
                if len(raw.drop_duplicates(subset=key_cols + [col])) > distinct:
                    discriminator = canonical
                    break

        source_grain = f"{declared}__by_{discriminator}" if discriminator else f"{declared}__fine"
        notes.append(
            f"Source is finer than the contract grain: {len(raw):,} rows collapse to "
            f"{distinct:,} on {'+'.join(key_fields)} ({ratio:.1f}x)"
            + (f", discriminated by {discriminator}" if discriminator else "")
            + f". Proposed rollup_to={declared} — measures are summed, dimensions take "
            f"the first value. VERIFY this is correct before freezing."
        )
        return source_grain, declared

    # ── Column assignment ────────────────────────────────────────────────────

    def _assign_columns(
        self, profile: TableProfile, contract: DocContract, include_empty: bool = False
    ) -> Dict[str, str]:
        """
        Match source columns to canonical fields by scoring name *and* data shape,
        then assigning globally rather than first-match-wins.

        Name alone is not enough, and the failure is not hypothetical: in an SAP
        extract, `Item` is the PO line number while `Material` is the SKU. Both match
        an alias of `sku`, and taking the first one silently plants the wrong
        identifier in every downstream join. Scoring resolves it because the profile
        already knows `Item` holds two distinct small integers while `Material` holds
        high-cardinality codes.
        """
        candidates: List[Tuple[float, str, str]] = []

        for canonical, spec in contract.fields.items():
            aliases = [canonical] + spec.aliases
            # First occurrence wins, exactly as `token_rank` below does. A dict
            # comprehension here keeps the *last*, and the two disagreeing is a real
            # bug rather than a nicety: `po_date` normalizes to the same string as its
            # own first alias `po date`, so the exact header `PO Date` was recorded at
            # rank 1 and lost the +0.30 canonical bonus — while `PO del date`, matching
            # the same alias only as a token subset, kept rank 0 and won it. An exact
            # match was beaten by a fuzzy one, and lead time was then computed from the
            # delivery date. 75 fields across the contracts had the same hole.
            alias_rank: Dict[str, int] = {}
            for i, alias in enumerate(aliases):
                alias_rank.setdefault(normalize_header(alias), i)
            # Token sets collapse word-order and separator variants onto one entry, so
            # `QUANTITY_ORDERED`, `Ordered Quantity` and `quantity_ordered` all reach
            # the same alias without three separate list entries. Enumerating those
            # by hand is the unsustainable path the architecture doc warns about.
            token_rank: Dict[frozenset, int] = {}
            for i, alias in enumerate(aliases):
                key = frozenset(normalize_header(alias).split())
                if key:
                    token_rank.setdefault(key, i)

            for col in profile.columns:
                if col.is_period_column:
                    continue
                if col.non_null == 0 and not include_empty:
                    continue
                rank = alias_rank.get(col.normalized)
                penalty = 0.0
                if rank is None:
                    col_tokens = frozenset(col.normalized.split())
                    rank = token_rank.get(col_tokens)
                    penalty = 0.05     # reordered match is slightly weaker evidence
                if rank is None:
                    # A header that merely *contains* an alias's tokens is usually the
                    # same field with a qualifier bolted on — `PO_ORDER_QTY` is the
                    # order qty, `Customer Req. Date` is the request date. Without this
                    # every such prefix needs its own alias entry, which is the
                    # enumeration treadmill the contracts exist to avoid.
                    rank, penalty = self._subset_match(col_tokens, token_rank)
                if rank is None:
                    continue
                candidates.append(
                    (self._score(spec, col, rank) - penalty, canonical, col.name)
                )

        # Greedy global assignment: strongest evidence claims its column first, and
        # each column and each field is used at most once.
        candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
        column_map: Dict[str, str] = {}
        taken_cols: set = set()
        for _score, canonical, col_name in candidates:
            if canonical in column_map or col_name in taken_cols:
                continue
            column_map[canonical] = col_name
            taken_cols.add(col_name)
        return column_map

    @staticmethod
    def _subset_match(col_tokens: frozenset, token_rank: Dict[frozenset, int]) -> Tuple:
        """
        Match a header whose tokens are a superset of some alias's tokens.

        The penalty grows with each unexplained token, so a close containment
        (`po order qty` ⊇ `order qty`) beats a loose one, and a single-token alias like
        `qty` cannot hijack a header that says much more than that. Aliases of one
        token are excluded entirely — `date` inside `delivery date` is not evidence.
        """
        best_rank, best_penalty = None, None
        for alias_tokens, rank in token_rank.items():
            if len(alias_tokens) < 2 or not alias_tokens < col_tokens:
                continue
            extra = len(col_tokens) - len(alias_tokens)
            penalty = 0.12 + 0.08 * extra
            if penalty > 0.40:
                continue
            if best_penalty is None or penalty < best_penalty:
                best_rank, best_penalty = rank, penalty
        return best_rank, (best_penalty or 0.0)

    @staticmethod
    def _score(spec, col, alias_rank: int) -> float:
        """
        Confidence that `col` supplies `spec`. Lexical position sets the baseline;
        type and cardinality evidence from the profile adjust it.
        """
        # Earlier aliases are the more canonical spellings; decay gently so a strong
        # type signal can still overturn a marginally better name match.
        score = 1.0 - min(alias_rank, 20) * 0.02
        if alias_rank == 0:
            score += 0.30                      # the canonical name itself

        if spec.is_numeric:
            score += 0.30 if col.inferred_type in ("integer", "decimal") else -0.60
        elif spec.is_temporal:
            score += 0.30 if col.inferred_type == "date" else -0.70
        elif spec.type == "string":
            if col.inferred_type in ("string", "boolean"):
                score += 0.15
            elif col.inferred_type == "date":
                score -= 0.40

        if spec.role == "identifier":
            # Identifiers vary across rows; a column with a handful of repeated values
            # is a line number or a status, not a key.
            score += 0.35 * col.distinct_rate
            if col.distinct <= 3 and col.distinct_rate < 0.1:
                score -= 0.30
        elif spec.role == "dimension" and col.distinct_rate > 0.95:
            score -= 0.10                      # a dimension that never repeats is suspect

        score -= 0.25 * col.null_rate
        return score

    @staticmethod
    def _signature_columns(profile: TableProfile, column_map: Dict[str, str]) -> List[str]:
        """
        Pick headers that identify this source. Mapped columns are preferred because
        they are the ones the source is guaranteed to keep carrying.
        """
        mapped_cols = list(column_map.values())[:6]
        if len(mapped_cols) >= 3:
            return mapped_cols
        extra = [
            c.name for c in profile.columns
            if not c.is_period_column and c.name not in mapped_cols
        ]
        return (mapped_cols + extra)[:6]

    @staticmethod
    def _slugify(text: str) -> str:
        stem = Path(str(text)).stem
        slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
        # Strip trailing timestamps so re-exports of the same report share an adapter
        slug = re.sub(r"_?\d{6,}$", "", slug)
        return slug or "source"

    def summary(self) -> str:
        self._load_all()
        if not self._adapters:
            return f"Adapter store empty ({self.adapters_dir})"
        lines = [f"Adapter store — {self.adapters_dir} ({len(self._adapters)} adapters)"]
        for a in sorted(self._adapters, key=lambda x: (x.doc_type, x.tenant, -x.version)):
            lines.append(f"  {a.slug:<44} {a.status:<9} {len(a.column_map)} mapped, "
                         f"{len(a.derivations)} derived")
        return "\n".join(lines)
