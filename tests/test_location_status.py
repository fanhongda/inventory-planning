"""
What a storage-location code means, and what happens when nobody has said.

The declaration is required rather than defaulted, and that is the whole content of
these tests. A default of "everything is sellable" is not a neutral starting point: it
is the defect this file exists to remove, because quarantine stock counted as available
overstates the position, so the run orders *less* than it should and no figure on the
report shows that anything was assumed.

The refusal is therefore load-bearing, and so is its scope. One location code, or none,
needs no declaration — there is nothing to separate — because a single-node planner
made to fill in a file describing a distinction they do not have will fill it in
carelessly, and a carelessly filled file is worse than the question being skipped.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.node import (
    SELLABLE,
    STOCK_STATUSES,
    STOCK_STATUS_FILE,
    StockStatusMap,
    UndeclaredStockLocation,
    load_stock_status,
)

CONFIG_DIR = Path(__file__).parents[1] / "config"


def _write(config_dir: Path, statuses: dict) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / STOCK_STATUS_FILE).write_text(
        json.dumps({"statuses": statuses}), encoding="utf-8")
    return config_dir


class TestWhatShips:

    def test_the_shipped_file_declares_nothing(self):
        """
        Deliberate. A shipped map claiming to know this tenant's codes is the
        `DC-01` placeholder problem with a quantity attached: it would read as
        configured, and the first planner to trust it would be entitled to believe
        their quarantine stock had been accounted for.
        """
        shipped = load_stock_status(CONFIG_DIR)
        assert not shipped.declared
        assert shipped.statuses == {}

    def test_the_shipped_file_documents_every_status(self):
        """The vocabulary is a decision; a status the code knows and the file does not
        is one a planner cannot find out about without reading the source."""
        raw = json.loads((CONFIG_DIR / STOCK_STATUS_FILE).read_text(encoding="utf-8"))
        assert set(raw["_statuses"]) == set(STOCK_STATUSES)


class TestLoading:

    def test_absent_file_is_not_an_error(self, tmp_path):
        """
        The refusal belongs where undeclared codes are about to be summed, not here.
        That is where the quantities are known, so that is where a message can name
        them — and naming them is what makes the refusal answerable.
        """
        loaded = load_stock_status(tmp_path)
        assert not loaded.declared

    def test_declared_codes_are_read(self, tmp_path):
        loaded = load_stock_status(_write(tmp_path, {"01": "sellable",
                                                     "02": "quarantine"}))
        assert loaded.declared
        assert loaded.status_of("01") == SELLABLE
        assert loaded.status_of("02") == "quarantine"

    def test_codes_are_matched_case_and_space_insensitively(self, tmp_path):
        """An export's codes arrive however the report wrote them; the declaration is
        typed by hand. Matching them strictly makes a trailing space look like an
        undeclared location."""
        loaded = load_stock_status(_write(tmp_path, {" fg01 ": "sellable"}))
        assert loaded.status_of("FG01") == SELLABLE
        assert loaded.status_of("fg01 ") == SELLABLE

    def test_an_unknown_status_is_refused(self, tmp_path):
        """
        Refused at load rather than treated as not-sellable. A typo would otherwise
        withhold the stock it was meant to admit, and the run would be quietly short of
        its own inventory.
        """
        with pytest.raises(UndeclaredStockLocation, match="sellabel"):
            load_stock_status(_write(tmp_path, {"01": "sellabel"}))

    def test_a_statuses_list_is_refused(self, tmp_path):
        (tmp_path / STOCK_STATUS_FILE).write_text(
            json.dumps({"statuses": ["01", "02"]}), encoding="utf-8")
        with pytest.raises(UndeclaredStockLocation, match="map of location code"):
            load_stock_status(tmp_path)

    def test_an_undeclared_code_has_no_status(self, tmp_path):
        loaded = load_stock_status(_write(tmp_path, {"01": "sellable"}))
        assert loaded.status_of("02") is None
        assert loaded.undeclared(["01", "02", "05"]) == ["02", "05"]

    def test_a_row_with_no_location_is_sellable(self, tmp_path):
        """
        The single-node case. A snapshot with no location column is the position by
        construction, and reading a blank as undeclared would refuse every run that
        never had the distinction to begin with.
        """
        loaded = load_stock_status(_write(tmp_path, {"01": "sellable"}))
        assert loaded.status_of("") == SELLABLE
        assert loaded.status_of(None) == SELLABLE
        assert loaded.undeclared(["", None]) == []


class TestWildcard:
    """
    `"*": "sellable"` answers the question in one keystroke.

    Offered, not withheld: a bin-level export can carry dozens of codes that genuinely
    are all sellable, and a refusal that demands forty lines gets the forty shortest
    lines that make it stop. It is reported on every run that uses it so a blanket
    answer stays visible as a blanket answer.
    """

    def test_wildcard_covers_undeclared_codes(self, tmp_path):
        loaded = load_stock_status(_write(tmp_path, {"*": "sellable"}))
        assert loaded.status_of("ANYTHING") == SELLABLE
        assert loaded.undeclared(["01", "02", "99"]) == []

    def test_a_named_code_outranks_the_wildcard(self, tmp_path):
        """The common real shape: everything sellable except inspection."""
        loaded = load_stock_status(_write(tmp_path, {"*": "sellable",
                                                     "02": "quarantine"}))
        assert loaded.status_of("01") == SELLABLE
        assert loaded.status_of("02") == "quarantine"

    def test_the_wildcard_is_visible_as_one(self, tmp_path):
        loaded = load_stock_status(_write(tmp_path, {"*": "sellable"}))
        assert loaded.wildcard == SELLABLE
        # And it is not mistaken for a code named "*".
        assert "*" not in loaded.statuses


class TestOnlySellableNets:

    def test_sellable_is_the_only_netting_status(self):
        """
        Every other status is reported. Stated as a test because the split is driven by
        equality with one name: a status added to the vocabulary without a decision
        about netting would silently start being withheld, which is the safe direction
        but not an unexamined one.
        """
        assert SELLABLE in STOCK_STATUSES
        assert [s for s in STOCK_STATUSES if s != SELLABLE] == [
            "quarantine", "blocked", "consignment", "in_transit_staging"]

    def test_every_status_says_what_it_does(self):
        for name, why in STOCK_STATUSES.items():
            assert why and len(why) > 20, name


class TestMapConstruction:

    def test_an_empty_map_is_undeclared(self):
        assert not StockStatusMap(statuses={}, source="x").declared

    def test_a_wildcard_alone_counts_as_declared(self):
        assert StockStatusMap(statuses={}, source="x", wildcard=SELLABLE).declared
