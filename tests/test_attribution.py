"""
Who made a change, and how that was established — INTERFACE.md §7's second seam.

Not to be confused with `test_identity.py`, which is the *material* identity layer —
which codes name the same part. This is the actor: which person a change is attributed
to. Two different questions that the word "identity" covers, and the reason this file is
named for the attribution rather than for the module.

This is not an identity layer and these tests are not about authentication. They pin
three things about the seam itself:

**One place decides.** A name becomes an actor in `attribution.resolve_actor` and nowhere
else, so the day a token arrives it is a substitution there rather than an edit in every
endpoint, CLI and writer. The tests reach that one place through each of the writers.

**The library no longer has a hole.** `by` used to default to `""` on `ledger.restate`
and `ledger.void`, so enforcement lived entirely at the four entry points and any caller
that went round one of them wrote an unattributed record in silence. That is the same
shape as an argparse default outranking a tenant: the default is the hole.

**Nothing on disk changed.** `self_asserted` is what every record already is, so it is
not written; a record without a basis reads as self-asserted, and there is no migration.
"""

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.attribution import (
    Actor, BASIS_FIELD, SELF_ASSERTED, Unattributed, VERIFIED, actor_of, resolve_actor,
)


class TestAClaimAndAVerifiedIdentityAreDifferentClaims:

    def test_a_name_off_a_form_is_self_asserted(self):
        actor = resolve_actor("jfanhon")
        assert actor.name == "jfanhon"
        assert actor.basis == SELF_ASSERTED
        assert actor.verified is False
        assert "unverified" in str(actor)

    def test_a_name_from_a_session_is_verified(self):
        """Nothing passes `verified` yet. This is the substitution, tested before it."""
        actor = resolve_actor(None, verified="jfanhon")
        assert actor.basis == VERIFIED
        assert actor.verified is True

    def test_a_claim_that_disagrees_with_the_session_is_refused(self):
        """
        Not resolved one way or the other. A form field saying `bob` under a token
        saying `alice` is either a mistake or an attempt, and a change attributed to
        someone who did not make it is worse than one attributed to nobody.
        """
        with pytest.raises(Unattributed, match="but the session is"):
            resolve_actor("bob", verified="alice")

    def test_a_claim_that_agrees_is_fine_and_still_verified(self):
        assert resolve_actor("alice", verified="alice").basis == VERIFIED

    def test_a_blank_name_is_refused_and_says_why(self):
        for blank in (None, "", "   "):
            with pytest.raises(Unattributed, match="taken on trust"):
                resolve_actor(blank)


class TestNothingOnDiskChangedToday:
    """
    A marker on every record distinguishes nothing, and `self_asserted` is what every
    record in the store already is. So it is not written, and absent reads as it.
    """

    def test_a_self_asserted_actor_records_only_the_name(self):
        assert resolve_actor("jfanhon").record() == {"by": "jfanhon"}
        assert resolve_actor("jfanhon").basis_recorded is None

    def test_a_verified_actor_records_the_basis_too(self):
        recorded = resolve_actor(None, verified="jfanhon").record()
        assert recorded == {"by": "jfanhon", BASIS_FIELD: VERIFIED}

    def test_the_field_the_name_sits_under_varies_and_the_basis_does_not(self):
        """`by` on a declaration, `restated_by` on a restatement, `voided_by` on a void."""
        recorded = resolve_actor(None, verified="x").record("restated_by")
        assert recorded == {"restated_by": "x", BASIS_FIELD: VERIFIED}

    def test_a_record_written_before_the_seam_reads_as_self_asserted(self):
        """Which it was. No migration, and every existing record correctly classified."""
        actor = actor_of({"restated_by": "fanhongda"}, "restated_by")
        assert actor == Actor("fanhongda", SELF_ASSERTED)

    def test_a_record_naming_nobody_reads_as_nobody(self):
        """
        None rather than `Actor("")`. An unattributed record is a real state — the
        parser tolerates a hand-written declaration without a name — and a nameless
        actor would make it look attributed to something.
        """
        assert actor_of({"by": ""}) is None
        assert actor_of({}) is None

    def test_an_unrecognised_basis_is_read_down_not_up(self):
        """A record claiming a basis this code does not know is not thereby trusted."""
        assert actor_of({"by": "x", BASIS_FIELD: "sso-ish"}).basis == SELF_ASSERTED


class TestTheLibraryDefaultWasTheHole:
    """
    `by` defaulted to `""` on both ledger writers, so enforcement lived entirely at the
    four entry points. Nothing went round one of them in practice — but the next caller
    would have, silently, and the store has no way to notice a record that names nobody.
    """

    @pytest.fixture
    def ledger(self, tmp_path):
        from inventory_planning.store.ledger import BatchLedger

        return BatchLedger(tmp_path / "store")

    def test_restating_without_a_name_is_refused_at_the_writer(self, ledger):
        from inventory_planning.store.ledger import LAYER_PREPARED

        with pytest.raises(Unattributed):
            ledger.restate("b1", LAYER_PREPARED, reason="r")

    def test_voiding_without_a_name_is_refused_at_the_writer(self, ledger):
        with pytest.raises(Unattributed):
            ledger.void("b1", reason="r")

    def test_a_refused_write_appends_nothing(self, ledger):
        with pytest.raises(Unattributed):
            ledger.void("b1", reason="r")
        assert not ledger.path.exists() or ledger.path.read_text(encoding="utf-8") == ""

    def test_an_attributed_write_is_unchanged_on_disk(self, ledger):
        """The basis is absent because it is self-asserted — the shape a reader expects."""
        ledger.void("b1", reason="wrong extract", by="jfanhon")
        line = json.loads(ledger.path.read_text(encoding="utf-8").splitlines()[-1])
        assert line["voided_by"] == "jfanhon"
        assert BASIS_FIELD not in line


class TestEveryWriterGoesThroughTheOnePlace:
    """
    Reached through each writer rather than by reading the source: the property is that
    a blank name is refused wherever a change is written, and it is worth asserting at
    the surfaces rather than at the helper they share.
    """

    def test_a_macro_change(self, tmp_path):
        import shutil

        from inventory_planning.policy import macro
        from inventory_planning.policy.edits import EditRefused

        config = tmp_path / "config"
        config.mkdir()
        for path in (Path(__file__).parents[1] / "config").glob("*"):
            if path.is_file():
                shutil.copy2(path, config / path.name)
        proposal = macro.propose("days_per_year", 250, config_dir=config)
        with pytest.raises(EditRefused, match="must name who made it"):
            macro.apply("days_per_year", 250, reason="r", by="  ",
                        basis=proposal.basis, config_dir=config)

    def test_a_declaration(self, tmp_path):
        from inventory_planning.store.declarations import (
            DeclarationError, Declarations, Override, SCOPE_VALUE,
        )

        override = Override(scope=SCOPE_VALUE, field="currency", value="CNY",
                            reason="plant books in CNY", by="")
        with pytest.raises(DeclarationError, match="must name who made it"):
            Declarations.write_override(override, config_dir=tmp_path)

    def test_a_gate_waiver(self, tmp_path):
        from inventory_planning.store.declarations import (
            DeclarationError, Declarations, GateWaiver,
        )

        waiver = GateWaiver(check="sku_agreement", doc_type="inventory",
                            expires=date(2027, 12, 31), reason="disjoint", by="")
        with pytest.raises(DeclarationError, match="must name who made it"):
            Declarations.write_waiver(waiver, config_dir=tmp_path)

    def test_the_refusal_names_what_was_being_changed(self):
        """
        `a change to a rule` and `voiding batch b1` are different sentences on purpose:
        a message that said "must name who made it" identically everywhere would be
        telling the reader less than the code knows.
        """
        with pytest.raises(Unattributed, match="voiding batch b1"):
            resolve_actor("", what="voiding batch b1")
        with pytest.raises(Unattributed, match="a change to a rule"):
            resolve_actor("", what="a change to a rule")
