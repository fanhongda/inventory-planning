"""
Who made a change, and how that was established.

INTERFACE.md §7's second seam. Every declaration, override, void and restatement already
requires a name — the audit trail predates the login, which is the right order, because
the reverse gives you a system that knows who you are and does not record what you did.
The real store carries 1,188 restatements and 30 voids, every one of them named and
reasoned. **What is missing is only that nothing checks the name.**

So this module is not an identity layer. It is the one place a name is turned into an
actor, so that when a token arrives it is a substitution here rather than an edit in
every endpoint, every CLI and every writer.

Named `attribution` and not `identity` because `store/identity.py` is already the
*material* identity layer — which codes name the same part. Two different questions the
word covers, and having both under it is a trap: I walked into it within ten minutes of
writing this, by overwriting that module's tests with this one's.

## Why the basis is recorded and not only the name

A name established by a form field and a name established by a verified token are
different claims, and a store where they are indistinguishable cannot be asked which
kind it holds. The pipeline already ranks its own figures this way — `measured` outranks
`stated` outranks `defaulted`, and every figure says which it is. An audit field is
exactly the same problem and deserves the same treatment.

## Why absent means self-asserted, and nothing is written today

`self_asserted` is the state of the world: there is no login, so every name in the store
was typed by someone. A marker that appeared on every record would distinguish nothing,
and adding one to `declarations.yaml` — a file people edit by hand — would make it
noisier for no reading. So the basis is serialised **only when it is not
self-asserted**, and a record without it reads as self-asserted.

Two consequences worth stating. Nothing on disk changes today, so there is no migration
and every existing record is correctly classified by the new reader. And the day a token
fills a name, the marker appears and means something, which is the only day it could
have meant anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Someone typed it. Nobody checked it. Every record in the store today.
SELF_ASSERTED = "self_asserted"
# An identity layer established it. Nothing produces this yet — it is the substitution
# this module exists to make cheap.
VERIFIED = "verified"

BASES = (SELF_ASSERTED, VERIFIED)

# The field a basis is stored under, beside whatever field holds the name. Named once
# because it has to be read back by code that did not write it.
BASIS_FIELD = "by_basis"


class Unattributed(ValueError):
    """A change that names nobody."""


@dataclass(frozen=True)
class Actor:
    """A name, and how it came to be believed."""

    name: str
    basis: str = SELF_ASSERTED

    def __post_init__(self):
        if self.basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}, not {self.basis!r}")

    @property
    def verified(self) -> bool:
        return self.basis == VERIFIED

    def __str__(self) -> str:
        return self.name if self.verified else f"{self.name} (unverified)"

    @property
    def basis_recorded(self) -> Optional[str]:
        """The basis as it should be stored: None when there is nothing worth recording."""
        return None if self.basis == SELF_ASSERTED else self.basis

    def record(self, field: str = "by") -> Dict[str, str]:
        """
        The fields to store. The basis is included only when it is not self-asserted.

        `field` because the thing is spelled differently depending on what it attributes
        — `by` on a declaration, `restated_by` on a restatement, `voided_by` on a void.
        The basis keeps one name throughout, so a reader does not need to know which.
        """
        out = {field: self.name}
        if self.basis != SELF_ASSERTED:
            out[BASIS_FIELD] = self.basis
        return out


def resolve_actor(claimed: Any = None, *, verified: Any = None,
                  what: str = "this change") -> Actor:
    """
    The single substitution point. Today a claim; tomorrow a token.

    `verified` is what an identity layer will pass and nothing passes yet. When it
    arrives, a claim that *disagrees* with it is refused rather than quietly resolved
    one way or the other: a form field that says `bob` under a token that says `alice`
    is either a mistake or an attempt, and neither should be recorded as a fact about
    who did something.
    """
    name = str(verified or "").strip()
    if name:
        other = str(claimed or "").strip()
        if other and other != name:
            raise Unattributed(
                f"{what} is signed {other!r} but the session is {name!r}. Nothing was "
                f"recorded: a change attributed to someone who did not make it is worse "
                f"than one attributed to nobody.")
        return Actor(name=name, basis=VERIFIED)

    name = str(claimed or "").strip()
    if not name:
        raise Unattributed(
            f"{what} must name who made it. There is no login behind this yet, so the "
            f"name is taken on trust — which is exactly why it cannot be blank.")
    return Actor(name=name, basis=SELF_ASSERTED)


def actor_of(record: Dict[str, Any], field: str = "by") -> Optional[Actor]:
    """
    Read an actor back off a stored record, absent basis meaning self-asserted.

    Returns None for a record that names nobody, rather than an `Actor("")`: an
    unattributed record is a real state — the parser tolerates a hand-written
    declaration without a name — and turning it into a nameless actor would make it
    look attributed to something.
    """
    name = str((record or {}).get(field) or "").strip()
    if not name:
        return None
    basis = str((record or {}).get(BASIS_FIELD) or SELF_ASSERTED)
    return Actor(name=name, basis=basis if basis in BASES else SELF_ASSERTED)
