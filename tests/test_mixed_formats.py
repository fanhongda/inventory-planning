"""
One column, two ways of writing the same thing.

The parser assumed a column speaks one convention — a reasonable assumption that is
wrong in a way producing no error at all. Excel breaks it routinely: opened under a
locale that disagrees with the file, it converts the cells it *can* read as local order
— the ones where both components are 12 or less — into real dates, swapping month and
day as it goes, and leaves the rest as text because `10/14/2024` is not a valid
day-first date. Read back out, half the column is `2024-04-10 00:00:00` and half is
`10/14/2024`.

Whichever parser wins, the other half becomes NaT. On the real sales extract that
removed 8,737 of 34,128 rows — 9.7% of shipped quantity across 759 of 1,256 SKUs — and
the rows that survived were exactly those whose day exceeded 12, which is why nothing
downstream looked wrong: the demand that remained was real, there was simply less of it.

Two behaviours are held here. The **detector** reports any column carrying two
incompatible representations, for a human to look at. The **repair** restores the
swapped half, but only where the evidence is conclusive — and where it is not, the
repair declines and the detector's flag is all the reader gets, which is the correct
outcome.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.ingest.adapter import _parse_mixed_dates
from inventory_planning.ingest.profiler import (
    classify_representation,
    detect_representation_mix,
)


def excel_mangled(dates, order: str = "MDY") -> pd.Series:
    """
    Write dates the way Excel leaves them after opening a file in the wrong locale.

    Values whose two leading components are both <= 12 are ambiguous, so Excel converts
    them — and swaps month and day doing it. The rest it cannot parse, so they survive
    as the original text.
    """
    out = []
    for stamp in pd.to_datetime(pd.Series(dates)):
        first, second = ((stamp.month, stamp.day) if order == "MDY"
                         else (stamp.day, stamp.month))
        # Ambiguous exactly when both components could be a month. The month always
        # can, so this reduces to a day at or below the 12th.
        if stamp.day <= 12:
            # Whichever order the file is in, the spreadsheet reads the components the
            # other way round, so month and day come out exchanged.
            out.append(pd.Timestamp(year=stamp.year, month=stamp.day, day=stamp.month)
                       .strftime("%Y-%m-%d %H:%M:%S"))
        else:
            out.append(f"{first}/{second}/{stamp.year}")
    return pd.Series(out)


TRUE_DATES = pd.to_datetime([
    # Deliberately spans both sides of the 12th, in both month and day, so the
    # ambiguous and unambiguous halves are each substantial.
    *(f"2025-{m:02d}-{d:02d}" for m in range(1, 13) for d in (3, 8, 11, 14, 19, 25)),
])


class TestTheDetector:

    def test_a_split_date_column_is_flagged(self):
        mix = detect_representation_mix(excel_mangled(TRUE_DATES))
        assert mix is not None
        assert "iso_date" in mix and "delimited_date" in mix

    def test_a_clean_delimited_column_is_not_flagged(self):
        clean = pd.Series([f"{m}/{d}/2025" for m in range(1, 13) for d in (3, 14, 25)])
        assert detect_representation_mix(clean) is None

    def test_a_clean_iso_column_is_not_flagged(self):
        clean = pd.Series(TRUE_DATES).dt.strftime("%Y-%m-%d")
        assert detect_representation_mix(clean) is None

    def test_a_few_stray_values_are_not_a_convention(self):
        """A footnote row is not a second format; flagging it trains the reader to skip."""
        clean = [f"{m}/{d}/2025" for m in range(1, 13) for d in (3, 14, 25)]
        assert detect_representation_mix(pd.Series(clean + ["2025-01-01"])) is None

    def test_dates_next_to_free_text_are_not_flagged(self):
        """Untidy, not corrupting — and `_count_total_rows` already reports those."""
        values = [f"{m}/14/2025" for m in range(1, 13)] * 3 + ["see remarks"] * 12
        assert detect_representation_mix(pd.Series(values)) is None

    def test_mixed_decimal_conventions_are_flagged(self):
        """The other half of the same failure: 1.200,50 beside 1,200.50."""
        values = [f"1.{n:03d},50" for n in range(100, 130)] + [f"1,{n:03d}.50" for n in range(100, 130)]
        mix = detect_representation_mix(pd.Series(values))
        assert mix is not None
        assert "eu_number" in mix and "us_number" in mix

    def test_it_reads_the_whole_column_not_the_head(self):
        """
        A partial conversion follows whichever values were ambiguous, not row order, so
        a split column is often uniform for its first several hundred rows.
        """
        head = pd.Series([f"{m}/25/2025" for m in range(1, 13)] * 60)   # 720 text rows
        tail = pd.Series(["2025-04-10 00:00:00"] * 60)
        assert detect_representation_mix(pd.concat([head, tail], ignore_index=True)) is not None

    def test_an_empty_column_is_not_flagged(self):
        assert detect_representation_mix(pd.Series([np.nan, None], dtype="object")) is None

    @pytest.mark.parametrize("value,kind", [
        ("2025-04-10 00:00:00", "iso_date"),
        ("2025-04-10", "iso_date"),
        ("10/14/2024", "delimited_date"),
        ("01.07.2026", "delimited_date"),
        ("1.200,50", "eu_number"),
        ("1,200.50", "us_number"),
        ("see remarks", "text"),
        ("", None),
    ])
    def test_representation_classes(self, value, kind):
        assert classify_representation(value) == kind


class TestTheRepair:

    def test_both_halves_survive_and_the_swap_is_undone(self):
        parsed, detail = _parse_mixed_dates(excel_mangled(TRUE_DATES), dayfirst=False)
        assert parsed.notna().all(), "no row may be lost to the format split"
        assert list(parsed) == list(TRUE_DATES), "the swapped half must be restored"
        assert "swapped" in detail

    def test_it_works_when_the_source_is_day_first(self):
        """Symmetric: the spreadsheet swaps month and day whichever way the file reads."""
        parsed, detail = _parse_mixed_dates(excel_mangled(TRUE_DATES, order="DMY"),
                                            dayfirst=True)
        assert list(parsed) == list(TRUE_DATES)
        assert "DMY" in detail

    def test_the_repaired_half_lands_in_the_same_span_as_the_text_half(self):
        """The property that proved the diagnosis on the real file."""
        parsed, _ = _parse_mixed_dates(excel_mangled(TRUE_DATES), dayfirst=False)
        text_half = pd.to_datetime([d for d in TRUE_DATES if d.day > 12])
        assert parsed.min() >= text_half.min() - pd.Timedelta(days=40)
        assert parsed.max() <= text_half.max() + pd.Timedelta(days=40)

    def test_a_single_format_column_is_left_to_the_normal_parser(self):
        clean = pd.Series([f"{m}/14/2025" for m in range(1, 13)])
        assert _parse_mixed_dates(clean, dayfirst=False) is None

    def test_a_mixed_column_with_no_swap_evidence_keeps_both_halves_anyway(self):
        """
        The converted half here contains days past the 12th, so it was never filtered
        to the ambiguous values and no swap happened. Recovering the rows is still the
        fix — it is only the month/day correction that needs evidence.
        """
        series = pd.Series(
            [f"{m}/25/2025" for m in range(1, 13)] * 3
            + ["2025-01-28 00:00:00", "2025-03-30 00:00:00"] * 15
        )
        parsed, detail = _parse_mixed_dates(series, dayfirst=False)
        assert parsed.notna().all()
        assert "swap" not in detail or "not supported" in detail
        assert pd.Timestamp("2025-01-28") in set(parsed)

    def test_too_few_converted_rows_to_call_it_evidence(self):
        """`all days <= 12` over three values is a coincidence, not a fingerprint."""
        series = pd.Series([f"{m}/25/2025" for m in range(1, 13)] * 4
                           + ["2025-04-10 00:00:00"] * 3)
        parsed, detail = _parse_mixed_dates(series, dayfirst=False)
        assert parsed.notna().all()
        assert "swapped" not in detail
        assert pd.Timestamp("2025-04-10") in set(parsed)

    def test_a_swap_that_moves_dates_out_of_span_is_refused(self):
        """
        Self-correcting: where the fingerprint is misleading, the repair loses its own
        comparison and the values stand as they are.
        """
        # Text half sits in April; the converted half is already inside it, so swapping
        # month for day would throw the values across the whole year.
        series = pd.Series(
            [f"4/{d}/2025" for d in (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)] * 4
            + [f"2025-04-{d:02d} 00:00:00" for d in range(1, 12)] * 2
        )
        parsed, detail = _parse_mixed_dates(series, dayfirst=False)
        assert parsed.notna().all()
        assert "not supported by the date span" in detail
        assert parsed.dt.month.eq(4).all()


class TestThroughTheIntake:

    def test_a_mangled_export_loses_no_rows(self, tmp_path):
        from inventory_planning.ingest.intake import Intake

        n = len(TRUE_DATES)
        pd.DataFrame({
            "Part Number": [f"{5000000 + i % 40}" for i in range(n)],
            "Order Number": [f"SO{i:05d}" for i in range(n)],
            "Billto Customer Name": ["ACME"] * n,
            "Invdate Date": excel_mangled(TRUE_DATES),
            "Shipped Quantity": np.arange(1, n + 1),
            "Sales Revenue (USD)": np.arange(1, n + 1) * 10,
        }).to_excel(tmp_path / "sales.xlsx", index=False)

        doc = Intake(verbose=False).load_files([tmp_path / "sales.xlsx"]).get("sales_history")
        dates = pd.to_datetime(doc.frame["demand_date"], errors="coerce")

        assert dates.notna().all(), "the format split must not null out half the demand"
        # Before the repair the survivors were exactly the days past the 12th.
        assert int(dates.dt.day.min()) <= 12
        assert "swapped" in doc.explain()

    def test_the_flag_survives_the_repair(self, tmp_path):
        """
        The repair rescues this run's numbers; it does not fix the file. The export on
        the share drive is still broken and the next one will be broken the same way,
        so a warning that vanishes the moment the symptom is handled is how a source
        defect becomes permanent.
        """
        from inventory_planning.ingest.intake import Intake

        n = len(TRUE_DATES)
        pd.DataFrame({
            "Part Number": [f"{5000000 + i % 40}" for i in range(n)],
            "Order Number": [f"SO{i:05d}" for i in range(n)],
            "Billto Customer Name": ["ACME"] * n,
            "Invdate Date": excel_mangled(TRUE_DATES),
            "Shipped Quantity": np.arange(1, n + 1),
        }).to_excel(tmp_path / "sales.xlsx", index=False)

        result = Intake(verbose=False).load_files([tmp_path / "sales.xlsx"])
        flagged = [note for note in result.notes if "MIXED FORMATS" in note]

        assert len(flagged) == 1, "the source defect must be reported even once repaired"
        assert "Invdate Date" in flagged[0]
        assert "ISO" in flagged[0], "the note has to say what to fix upstream"
        # And the repair still happened.
        dates = pd.to_datetime(result.frame("sales_history")["demand_date"], errors="coerce")
        assert dates.notna().all()

    def test_the_profile_flags_it_for_a_human(self, tmp_path):
        from inventory_planning.ingest.intake import load_file
        from inventory_planning.ingest.profiler import Profiler

        n = len(TRUE_DATES)
        pd.DataFrame({
            "Material": [f"{5000000 + i % 40}" for i in range(n)],
            "Invdate Date": excel_mangled(TRUE_DATES),
            "Shipped Quantity": np.arange(1, n + 1),
        }).to_excel(tmp_path / "sales.xlsx", index=False)

        profile = Profiler().profile(load_file(tmp_path / "sales.xlsx"), "sales.xlsx")
        flagged = [n for n in profile.notes if "MIXED FORMATS" in n]
        assert len(flagged) == 1
        assert "Invdate Date" in flagged[0]
        assert profile.column("Invdate Date").representation_mix is not None
