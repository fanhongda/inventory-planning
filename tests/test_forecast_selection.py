"""
Which forecasting model is used, and how that is decided.

It used to be decided by a decision tree that compared nothing. The demand pattern
picked Croston or the smooth branch; the smooth branch took ETS if it fitted, else
ARIMA, else a moving average. Accuracy played no part — ETS was used whenever it did
not raise — and `forecast_rmse`, which sizes safety stock, was the in-sample residual:
a measure of how closely a model traces the data it was fitted on, not of anything it
predicts.

Make-to-stock items now compete. Every candidate is refitted on a rolling origin, made
to forecast periods it has not seen, and scored against what happened. Make-to-order
items go straight to Croston and are not backtested: nothing is stocked against a
forecast there, the demand is intermittent by construction, and the competition would
spend the time to arrive where it started.

The naive forecast is an entrant, not a footnote. Without it, "best of five" can still
be worse than repeating last month and nothing would say so — and on this extract 35
of 178 backtested items are exactly that.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from inventory_planning.analytics.forecaster import Forecaster

PERIODS = pd.period_range("2024-01", periods=22, freq="M")
RAMP = list(np.arange(1, 23) * 10.0)
LUMPY = [0, 0, 50, 0, 0, 80, 0, 0, 0, 60, 0, 0, 70, 0, 0, 0, 90, 0, 0, 55, 0, 0]
# Nearly empty, then a ramp in the last three months. SARIMAX reads the ramp as a trend
# and runs it out to 148 against a historical maximum of 25 — the shape that makes the
# ceiling do work, taken from the one SKU on the real extract where it fires.
SPARSE_RAMP = [0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 2, 25]


def _run(values, policy=None, sku="A", horizon=6, periods=PERIODS):
    ts = pd.DataFrame({sku: [float(v) for v in values]}, index=periods)
    pol = (pd.DataFrame({"sku": [sku], "stocking_policy": [policy]})
           if policy is not None else None)
    detail = Forecaster(horizon=horizon).forecast_all(ts, policy=pol)
    return detail, detail[detail["is_next_period"]].iloc[0]


class TestThePolicyDecidesWhetherThereIsACompetition:

    def test_make_to_stock_is_backtested(self):
        _, row = _run(RAMP, "MTS")
        assert row["selected_by"].startswith("backtest")

    def test_make_to_order_goes_straight_to_croston(self):
        """No competition, and the cost of one is the reason."""
        _, row = _run(LUMPY, "MTO")
        assert row["model_used"] == "Croston"
        assert row["selected_by"] == "policy:MTO"

    def test_the_policy_outranks_the_pattern(self):
        """
        A clean ramp is exactly what Croston is wrong for, and MTO still takes it: the
        ERP's policy is a statement about how the item is planned, not a guess to be
        overridden by the shape of a series.
        """
        _, row = _run(RAMP, "MTO")
        assert row["model_used"] == "Croston"

    def test_no_policy_leaves_the_old_routing_alone(self):
        """A missing policy is not evidence of one, so nothing about the SKU changes."""
        _, row = _run(RAMP, policy=None)
        assert row["selected_by"] == "pattern"

    def test_a_series_too_short_to_hold_anything_out_is_not_backtested(self):
        short = pd.period_range("2025-01", periods=8, freq="M")
        _, row = _run(RAMP[:8], "MTS", periods=short)
        assert not row["selected_by"].startswith("backtest:")


class TestTheCompetitionPicksOnEvidence:

    def test_a_trend_is_not_forecast_flat(self):
        """
        The case the old router got right by luck and would have got wrong on a lumpy
        classification: a clean ramp needs a model that extrapolates, and the winner
        has to beat repeating last month by a distance.
        """
        _, row = _run(RAMP, "MTS")
        assert row["model_used"] in ("ARIMA", "ETS")
        assert row["vs_naive"] < 0.5

    def test_intermittent_demand_still_lands_on_croston(self):
        """Not by routing this time — by winning."""
        _, row = _run(LUMPY, "MTS")
        assert row["model_used"] == "Croston"
        assert row["selected_by"].startswith("backtest")

    def test_where_no_model_helps_the_naive_one_wins(self):
        """
        Flat demand with noise: there is nothing to model, and the honest answer is to
        repeat the last observation rather than dress it up.
        """
        noisy = 100 + np.random.default_rng(0).normal(0, 5, 22)
        _, row = _run(noisy, "MTS")
        assert row["model_used"] == "Naive"
        assert row["vs_naive"] == 1.0

    def test_vs_naive_is_a_ratio_and_not_a_verdict(self):
        """
        It was a boolean and it was always true — the winner is the lowest-scoring
        model and Naive is an entrant, so "beats naive" could not come out false. The
        ratio is the part that carries information.
        """
        _, ramp = _run(RAMP, "MTS")
        _, flat = _run(100 + np.random.default_rng(0).normal(0, 5, 22), "MTS")
        assert ramp["vs_naive"] < flat["vs_naive"]

    def test_the_error_reported_is_the_backtest_one(self):
        """
        Not the in-sample residual. This number sizes safety stock, and a model's fit
        to the data it was trained on is not its error against data it has not seen.
        """
        _, row = _run(LUMPY, "MTS")
        assert round(row["backtest_rmse"], 3) == row["forecast_rmse"]


class TestAWinnerStaysOnTheGround:
    """
    SARIMAX on one real SKU forecast 4,578 for next month and 63 million a month on
    average. What stops that is the backtest scoring three steps ahead rather than one:
    a model walking away from the data loses before it is ever chosen.

    I wrote a ceiling on the horizon forecast for this as well, then measured it —
    across the whole extract it never once changed which model was chosen. It was
    removed. These tests pin the outcome, which is what matters, rather than the
    mechanism, which turned out not to be needed.
    """

    def test_a_late_ramp_out_of_nothing_is_not_extrapolated(self):
        detail, row = _run(SPARSE_RAMP, "MTS")
        assert detail["forecast_qty"].max() <= max(SPARSE_RAMP) * 3.0
        assert row["model_used"] != "ARIMA"

    def test_and_the_model_that_would_have_run_away_is_a_real_candidate(self):
        """Not a hypothetical: ARIMA does fit this shape, and does run 25 out to 148."""
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in SPARSE_RAMP], index=PERIODS)
        paths = f._candidates(series, 6)
        assert np.max(paths["ARIMA"]) > max(SPARSE_RAMP) * 3.0

    def test_it_does_not_flatten_a_genuine_trend(self):
        """Rejecting runaway growth must not become a refusal to forecast growth."""
        detail, _ = _run(RAMP, "MTS")
        assert detail["forecast_qty"].iloc[0] > RAMP[-1] * 0.9


class TestTheForecastSheet:

    def test_it_reads_left_to_right_as_time(self):
        ts = pd.DataFrame({"A": [float(v) for v in RAMP]}, index=PERIODS)
        f = Forecaster(horizon=6)
        detail = f.forecast_all(ts, policy=pd.DataFrame({"sku": ["A"],
                                                         "stocking_policy": ["MTS"]}))
        sheet = f.history_and_forecast(ts, detail)

        cols = list(sheet.columns)
        hist = [c for c in cols if c.startswith("hist ")]
        fcst = [c for c in cols if c.startswith("fcst ")]
        assert cols[0] == "sku"
        assert hist == sorted(hist) and fcst == sorted(fcst)
        assert cols.index(hist[-1]) < cols.index(fcst[0]), "history must precede forecast"
        assert len(hist) == len(PERIODS) and len(fcst) == 6

    def test_the_history_on_the_row_is_the_history(self):
        ts = pd.DataFrame({"A": [float(v) for v in RAMP]}, index=PERIODS)
        f = Forecaster(horizon=6)
        detail = f.forecast_all(ts, policy=pd.DataFrame({"sku": ["A"],
                                                         "stocking_policy": ["MTS"]}))
        row = f.history_and_forecast(ts, detail).iloc[0]
        assert row[f"hist {PERIODS[-1]}"] == RAMP[-1]
        assert row[f"hist {PERIODS[0]}"] == RAMP[0]

    def test_the_provenance_travels_with_the_numbers(self):
        """A forecast whose model earned nothing reads differently from one that did."""
        ts = pd.DataFrame({"A": [float(v) for v in RAMP]}, index=PERIODS)
        f = Forecaster(horizon=6)
        detail = f.forecast_all(ts, policy=pd.DataFrame({"sku": ["A"],
                                                         "stocking_policy": ["MTS"]}))
        sheet = f.history_and_forecast(ts, detail)
        for col in ("model_used", "selected_by", "vs_naive", "forecast_rmse"):
            assert col in sheet.columns

    def test_an_empty_forecast_is_an_empty_sheet_not_a_crash(self):
        assert Forecaster().history_and_forecast(pd.DataFrame(), pd.DataFrame()).empty


class TestTheModelIsPerSkuNotPerRun:
    """
    Each SKU is scored on its own history and keeps its own winner. Stated as a test
    because the run summary prints a count of models across SKUs, and that reads at a
    glance as one choice for the whole run.
    """

    def test_two_skus_with_different_shapes_get_different_models(self):
        ts = pd.DataFrame({
            "RAMP": [float(v) for v in RAMP],
            "LUMPY": [float(v) for v in LUMPY],
        }, index=PERIODS)
        policy = pd.DataFrame({"sku": ["RAMP", "LUMPY"], "stocking_policy": ["MTS", "MTS"]})
        detail = Forecaster(horizon=6).forecast_all(ts, policy=policy)
        chosen = detail[detail["is_next_period"]].set_index("sku")["model_used"]
        assert chosen["RAMP"] != chosen["LUMPY"]

    def test_the_winner_is_recorded_on_every_row_of_its_own_forecast(self):
        """The record is per SKU per period, so nothing has to be inferred later."""
        ts = pd.DataFrame({"A": [float(v) for v in LUMPY]}, index=PERIODS)
        detail = Forecaster(horizon=6).forecast_all(
            ts, policy=pd.DataFrame({"sku": ["A"], "stocking_policy": ["MTS"]}))
        assert len(detail) == 6
        assert detail["model_used"].nunique() == 1
        assert detail["model_used"].iloc[0] in ("ETS", "ARIMA", "SMA", "Croston", "Naive")


class TestVsNaive:
    """
    What the number means, since it decides how much of a forecast to believe.

    It is the winner's backtest error divided by the naive forecast's, on the same
    metric and the same folds. Below 1 the model earned something; at 1 it did not, and
    that happens when Naive itself won — nothing in the pool beat repeating last month.
    """

    def test_one_means_no_model_beat_repeating_last_month(self):
        noisy = 100 + np.random.default_rng(0).normal(0, 5, 22)
        _, row = _run(noisy, "MTS")
        assert row["model_used"] == "Naive"
        assert row["vs_naive"] == 1.0

    def test_below_one_is_the_share_of_the_naive_error_that_is_left(self):
        _, row = _run(RAMP, "MTS")
        assert 0 <= row["vs_naive"] < 1.0

    def test_it_is_absent_where_the_naive_forecast_has_no_error_to_divide_by(self):
        """A perfectly flat series: naive is exact, so the ratio is undefined, not zero."""
        _, row = _run([100.0] * 22, "MTS")
        assert pd.isna(row["vs_naive"])



class TestAPolicyIsSuggestedWhereverThereIsASeries:
    """
    The ERP's `stocking_policy` is a decision in force, often set years before anyone
    planned on the item. What the last twelve months did is a separate claim, and the
    two disagreeing is the finding — so the suggestion is made for every SKU with a
    series, including those the planner worksheet has never heard of, and it never
    overwrites the ERP's value.
    """

    @staticmethod
    def _classify(series_by_sku):
        from inventory_planning.analytics.demand_classifier import DemandClassifier
        ts = pd.DataFrame(series_by_sku, index=PERIODS[-12:])
        summary = pd.DataFrame({"sku": list(series_by_sku),
                                "demand_mean": 0.0, "demand_std": 0.0})
        cfg = Path(__file__).parents[1] / "config"
        return DemandClassifier(cfg).classify(summary, ts).set_index("sku")

    def test_demand_in_most_months_suggests_mts(self):
        out = self._classify({"A": [10.0] * 12})
        assert out.loc["A", "suggested_stocking_policy"] == "MTS"

    def test_sporadic_demand_suggests_mto(self):
        out = self._classify({"A": [0.0] * 10 + [5.0, 7.0]})
        assert out.loc["A", "suggested_stocking_policy"] == "MTO"

    def test_the_evidence_travels_with_the_verdict(self):
        """A label a planner cannot check is a label they have to take on trust."""
        out = self._classify({"A": [0.0] * 10 + [5.0, 7.0]})
        assert "2/12 months" in out.loc["A", "policy_basis"]

    def test_a_sku_with_no_series_gets_no_suggestion(self):
        """`有时间序列就给出建议` — and where there is none, silence rather than a guess."""
        from inventory_planning.analytics.demand_classifier import DemandClassifier
        ts = pd.DataFrame({"A": [10.0] * 12}, index=PERIODS[-12:])
        summary = pd.DataFrame({"sku": ["A", "NO-SERIES"],
                                "demand_mean": [10.0, 0.0], "demand_std": [0.0, 0.0]})
        out = (DemandClassifier(Path(__file__).parents[1] / "config")
               .classify(summary, ts).set_index("sku"))
        assert out.loc["A", "suggested_stocking_policy"] == "MTS"
        assert pd.isna(out.loc["NO-SERIES", "suggested_stocking_policy"])
