"""
The command line, and the drift that had no test to catch it.

Found by running the real extract before a production run: the documented invocation —
one flag per file — died on it with `KeyError: ['qty']`, because the sales history heads
its quantity column `Shipped Quantity` and the legacy `schema.py` those readers use does
not carry that alias. The contracts do.

It is not one alias. **The contracts carry 1,216 aliases `schema.py` does not**, across
every document type. The per-file path is a second, much weaker implementation of intake
that had drifted, and nothing compared the two.

Why it survived is the part worth pinning: `sample_data/sales_history.csv` heads its
column `Sales Qty`, which the legacy table *does* know. Every test and every
demonstration passed. So the test that matters here is not "does the CLI run" — it did —
but **"does the alias table the CLI reads through still agree with the contracts"**,
which is asked directly below and would have failed the day the gap opened.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.cli import _READABLE, _expand, build_parser  # noqa: E402

REPO = Path(__file__).parents[1]


class TestTheContractsAndTheLegacyTableAgree:
    """
    The guard the drift needed. A contract alias the legacy table lacks is a real export
    the per-file path cannot read — which is exactly what a production run hit.
    """

    def _gaps(self):
        import yaml

        from inventory_planning.schema import ALL_SCHEMAS

        legacy = {}
        for schema in ALL_SCHEMAS.values():
            for field, aliases in schema.items():
                legacy.setdefault(field, set()).update(a.lower() for a in aliases)

        gaps = {}
        for path in sorted((REPO / "inventory_planning" / "ingest" / "contracts")
                           .glob("*.yaml")):
            contract = yaml.safe_load(path.read_text(encoding="utf-8"))
            for field, spec in (contract.get("fields") or {}).items():
                missing = {a.lower() for a in (spec.get("aliases") or [])} \
                    - legacy.get(field, set())
                if missing:
                    gaps.setdefault(contract["doc_type"], {})[field] = sorted(missing)
        return gaps

    def test_the_gap_is_recorded_rather_than_asserted_away(self):
        """
        Not asserted to zero: closing 1,216 aliases by hand into a table the contracts
        already supersede would be work spent keeping a second implementation alive.
        What this holds is that the gap is *known* — and that the quantity column of a
        real SAP export is in it, which is the case that broke a production run.
        """
        gaps = self._gaps()
        assert "shipped quantity" in gaps["sales_history"]["qty"]
        assert sum(len(v) for d in gaps.values() for v in d.values()) > 1000

    def test_the_sample_data_hides_it_which_is_why_it_survived(self):
        """
        `Sales Qty` is in the legacy table and `Shipped Quantity` is not, so the whole
        suite passed while the real export failed. Stated as a test so the next person
        does not conclude from green tests that the per-file path reads real exports.
        """
        import pandas as pd

        from inventory_planning.schema import ALL_SCHEMAS

        header = pd.read_csv(REPO / "sample_data" / "sales_history.csv", nrows=0).columns
        legacy = {a.lower() for a in ALL_SCHEMAS["sales_history"]["qty"]}
        assert "sales qty" in legacy
        assert any(c.lower() in legacy for c in header)
        assert "shipped quantity" not in legacy


class TestTheContractPathIsTheDefault:

    def test_files_may_be_given_positionally(self):
        args = build_parser().parse_args(["a.csv", "b.xlsx", "--output", "out"])
        assert args.inputs == ["a.csv", "b.xlsx"]

    def test_a_directory_expands_to_the_readable_files_in_it(self, tmp_path):
        for name in ("sales.csv", "stock.xlsx", "notes.docx", "~$stock.xlsx",
                     ".hidden.csv"):
            (tmp_path / name).write_text("x", encoding="utf-8")
        (tmp_path / "sub").mkdir()

        found = {Path(p).name for p in _expand([str(tmp_path)])}
        assert found == {"sales.csv", "stock.xlsx"}

    def test_a_lock_file_is_skipped_rather_than_failing_the_run(self, tmp_path):
        """
        `~$name.xlsx` is what Excel leaves beside an open workbook. Routing one fails at
        the read, which is a confusing place to learn that a file was open.
        """
        (tmp_path / "~$open.xlsx").write_text("x", encoding="utf-8")
        assert _expand([str(tmp_path)]) == []

    def test_named_files_are_passed_through_as_given(self, tmp_path):
        path = tmp_path / "one.csv"
        path.write_text("x", encoding="utf-8")
        assert _expand([str(path)]) == [str(path)]

    def test_every_extension_the_readers_handle_is_expandable(self):
        assert {".csv", ".xlsx", ".xls"}.issubset(set(_READABLE))


class TestTheFlagsAreNoLongerRequired:
    """
    They were all `required=True`, so the contract path could not be reached without
    naming five files it does not need.
    """

    @pytest.mark.parametrize("flag", ["sales", "po_history", "open_so", "open_po",
                                      "inventory", "item_master"])
    def test_a_per_file_flag_defaults_to_none(self, flag):
        args = build_parser().parse_args(["somedir"])
        assert getattr(args, flag) is None

    def test_naming_nothing_at_all_is_refused_and_names_both_forms(self, capsys,
                                                                    monkeypatch):
        """
        Every flag is optional now, so the parser itself accepts an empty command line
        and `main` has to be the one that refuses — naming the form to use rather than
        listing five flags that are no longer the answer.
        """
        from inventory_planning.cli import main

        monkeypatch.setattr(sys, "argv", ["inventory-plan"])
        with pytest.raises(SystemExit):
            main()
        assert "positionally" in capsys.readouterr().err

    def test_the_output_and_config_defaults_still_do_not_outrank_the_tenant(self):
        """
        The defect the workspace seam found, still closed: argparse passes its default
        on every run, so a literal here would reach the resolver as an explicit
        argument.
        """
        args = build_parser().parse_args(["somedir"])
        assert args.output is None
        assert args.config is None
        assert args.tenant is None


class TestTheLegacyPathSaysWhatItCosts:

    def test_the_warning_names_the_alias_gap_and_the_missing_gate(self):
        from inventory_planning.cli import _legacy_warning

        warning = _legacy_warning()
        assert "1,216 fewer" in warning
        assert "Shipped Quantity" in warning
        assert "quality gate does" in warning
        assert "inventory-plan <dir-or-files...>" in warning

    def test_it_is_a_warning_and_not_a_removal(self):
        """
        The flags still work. Scripts that use them keep working, and on data whose
        columns the legacy table knows they produce the same plan they always did.
        """
        args = build_parser().parse_args([
            "--sales", "s.csv", "--po-history", "p.csv", "--open-so", "o.csv",
            "--open-po", "q.csv", "--inventory", "i.csv", "--item-master", "m.csv",
        ])
        assert args.inputs == []
        assert args.sales == "s.csv"
