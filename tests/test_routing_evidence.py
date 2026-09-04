"""
Routing evidence — the ranking as data, and what the confidence number is a measure of.

Two things were readable only as prose. Which contracts came close, which leaked into
the reason string as `⚠ close call vs ...` and was read back by searching for that
substring. And what `confidence` measures, which is three different things — a
fingerprint match, an alias score, and a caller's assertion — reported through one
number with nothing to tell them apart.
"""

import pandas as pd
import pytest

from inventory_planning.ingest.intake import Intake
from inventory_planning.ingest.profiler import Profiler
from inventory_planning.ingest.registry import (
    BASIS_CLASSIFICATION, BASIS_HINT, CLOSE_CALL_MARGIN, AdapterRegistry,
)


@pytest.fixture
def registry():
    return AdapterRegistry()


def _classify(registry, name):
    frame = pd.read_csv(f"sample_data/{name}.csv", dtype=str)
    profile = Profiler().profile(frame, source_name=f"{name}.csv")
    return registry.classify(profile, frame), frame


class TestTheRankingIsReturnedNotFlattened:

    def test_every_contract_scored_is_reported_with_its_reasoning(self, registry):
        verdict, _ = _classify(registry, "inventory")
        assert verdict.doc_type == "inventory"
        assert set(verdict.scores) >= {"inventory", "open_so", "po_history"}
        assert "required" in verdict.details["inventory"]

    def test_the_runner_up_is_named(self, registry):
        verdict, _ = _classify(registry, "inventory")
        doc_type, score = verdict.runner_up
        assert doc_type != "inventory"
        assert score < verdict.score

    def test_margin_separates_a_clear_win_from_a_coin_toss(self, registry):
        """
        The number confidence cannot carry: 85% against a field of 82% and 85% against
        a field of 26% are the same figure on screen and not the same situation.
        """
        clear, _ = _classify(registry, "inventory")
        tossup, _ = _classify(registry, "open_so")
        assert clear.margin > CLOSE_CALL_MARGIN
        assert tossup.margin < CLOSE_CALL_MARGIN

    def test_a_layout_decision_reports_an_empty_ranking_rather_than_a_fake_one(
            self, registry):
        wide = pd.DataFrame({"Material": ["A"], "2024-01": ["5"], "2024-02": ["6"],
                             "2024-03": ["7"], "2024-04": ["8"]})
        verdict = registry.classify(
            Profiler().profile(wide, source_name="ts.xlsx"), wide)
        assert verdict.doc_type == "demand_timeseries"
        assert verdict.scores == {}
        assert verdict.runner_up is None
        assert verdict.margin == 1.0


class TestCloseCallIsAFlagAndAMessageTogether:

    def test_both_are_set_at_the_same_point(self, registry):
        verdict, _ = _classify(registry, "open_so")
        assert verdict.close_call is True
        assert "close call" in verdict.reason

    def test_only_genuinely_close_documents_are_named(self, registry):
        """
        The shortlist handed to the content test deliberately includes every document
        with a discriminator, and naming it here reported `close call vs item_master
        (0%)` — not a close call, read by someone deciding whether to look harder.
        """
        verdict, _ = _classify(registry, "open_so")
        assert verdict.contenders == ["open_so", "open_po"]
        assert "item_master" not in verdict.reason

    def test_a_clear_win_sets_neither(self, registry):
        verdict, _ = _classify(registry, "inventory")
        assert verdict.close_call is False
        assert "close call" not in verdict.reason


class TestWhatTheConfidenceNumberMeasures:

    def test_a_scored_document_says_it_was_classified(self, registry):
        frame = pd.read_csv("sample_data/inventory.csv", dtype=str)
        route = registry.route(frame, source_name="inventory.csv")
        assert route.confidence_basis == BASIS_CLASSIFICATION
        assert route.stated is False
        assert route.runner_up is not None

    def test_a_hint_is_stated_and_carries_no_ranking(self, registry):
        """
        Its 1.0 is the caller's assertion, not an observation. A screen drawing a full
        green bar for it would report a statement as a measurement — the distinction the
        rest of this pipeline spends its time preserving.
        """
        frame = pd.read_csv("sample_data/inventory.csv", dtype=str)
        route = registry.route(frame, source_name="x.csv", doc_type_hint="inventory")
        assert route.confidence == 1.0
        assert route.confidence_basis == BASIS_HINT
        assert route.stated is True
        assert route.scores == {}
        assert route.runner_up is None


class TestUncertaintyNoLongerDependsOnTheWording:

    def test_rewording_the_reason_does_not_switch_the_flag_off(self):
        """
        `route_uncertain` decides whether a person is asked to confirm a document and
        whether supersession may rewrite item numbers. It searched the reason for the
        substring "close call", so an edit to the message would have turned it off with
        nothing failing.
        """
        frame = pd.read_csv("sample_data/open_so.csv", dtype=str)
        doc = Intake(verbose=False).load_frame(frame, source_name="open_so.csv")
        assert doc.route_uncertain is True

        doc.route.reason = "reworded by somebody tidying up the messages"
        assert doc.route_uncertain is True
