"""
Which forecasting model is used, and how that is decided.

It used to be decided by a decision tree that compared nothing. The demand pattern
picked Croston or the smooth branch; the smooth branch took ETS if it fitted, else
ARIMA, else a moving average. Accuracy played no part — ETS was used whenever it did
not raise — and `forecast_rmse`, which sizes safety stock, was the in-sample residual:
a measure of how closely a model traces the data it was fitted on, not of anything it
predicts.

Items now compete. Every candidate is refitted on a rolling origin, made to forecast
periods it has not seen, and scored against what happened. Whether there is a
competition is decided by how much history there is, not by the ERP's stocking policy:
`MTO` means the item is not held in stock, which is a decision about money and says
nothing about how often it sells. Croston stays in the pool and wins wherever it
deserves to — it simply has to win.

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


class TestTheHistoryDecidesWhetherThereIsACompetition:

    def test_make_to_stock_is_backtested(self):
        _, row = _run(RAMP, "MTS")
        assert row["selected_by"].startswith("backtest")

    def test_make_to_order_is_backtested_too(self):
        """
        It used to go straight to Croston because its demand was "intermittent by
        construction". It is not: `stocking_policy` is a decision not to hold stock,
        and a configured-to-order item can sell every month. The pipeline says as much
        elsewhere, reporting *held as MTO, demand recurs* as a finding.
        """
        _, row = _run(LUMPY, "MTO")
        assert row["selected_by"].startswith("backtest")

    def test_croston_still_wins_on_a_series_it_suits(self):
        """Which is the point: it wins on merit rather than by being routed to."""
        _, row = _run(LUMPY, "MTO")
        assert row["model_used"] == "Croston"

    def test_a_ramp_is_not_forced_onto_croston_by_its_policy(self):
        """
        A clean ramp is exactly what Croston is wrong for, and the policy used to take
        it anyway — answering with one number repeated across the horizon.
        """
        _, row = _run(RAMP, "MTO")
        assert row["model_used"] != "Croston"

    def test_no_policy_is_backtested_on_the_same_terms(self):
        _, row = _run(RAMP, policy=None)
        assert row["selected_by"].startswith("backtest")

    def test_a_series_too_short_to_hold_anything_out_falls_back_to_the_policy(self):
        short = pd.period_range("2025-01", periods=8, freq="M")
        _, row = _run(RAMP[:8], "MTO", periods=short)
        assert row["selected_by"] == "policy:MTO"

    def test_and_a_short_series_with_no_policy_falls_back_to_the_pattern(self):
        short = pd.period_range("2025-01", periods=8, freq="M")
        _, row = _run(RAMP[:8], policy=None, periods=short)
        assert row["selected_by"] == "pattern"


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

    def test_the_model_that_would_run_away_still_would(self):
        """Not a hypothetical: ARIMA does fit this shape, and does run 25 out to 148."""
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in SPARSE_RAMP], index=PERIODS)
        assert np.max(f._fit_arima(series, 6)) > max(SPARSE_RAMP) * 3.0

    def test_it_is_no_longer_offered_on_a_series_this_empty(self):
        """
        It used to be offered and lose. It lost to Croston, and Croston only won
        because its interval estimate started at 1.0 and never converged — so the
        defence against a 63-million-a-month forecast was resting on a bias. Correcting
        the bias handed the series to ARIMA. Not offering it is the defence that does
        not depend on another model being wrong.
        """
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in SPARSE_RAMP], index=PERIODS)
        assert float((series > 0).mean()) < 0.5
        assert "ARIMA" not in f._candidates(series, 6)

    def test_it_is_still_offered_where_there_is_a_series_to_model(self):
        """Every other month is intermittent, and ARIMA recovers the phase on it."""
        f = Forecaster(horizon=6)
        series = pd.Series([0.0, 150.0] * 11, index=PERIODS)
        assert "ARIMA" in f._candidates(series, 6)

    def test_every_fold_is_offered_the_same_models(self):
        """
        Admission is judged on density, and the folds differ by a period each, so a
        series sitting on the boundary let ARIMA into some folds and not others — six
        scored points against everyone else's nine. Pooling across folds exists so that
        no model is ranked on a different sample from its rivals; deciding admission
        inside the loop undid it.
        """
        # 15 non-zero in the first 31 periods, then demand at 32 and 33: fold densities
        # run 0.4839, 0.5000, 0.5152 and cross the boundary mid-backtest.
        values = [0.0] * 36
        for i in range(15):
            values[i * 2] = 100.0
        values[31] = values[32] = 100.0
        series = pd.Series(values, index=pd.period_range("2023-01", periods=36, freq="M"))

        seen = []
        f = Forecaster(horizon=6)
        original = f._candidates
        f._candidates = lambda s, h, allow=None: (seen.append(allow) or original(s, h, allow))
        f._select_by_backtest(series)

        assert len(seen) > 1, "expected more than one fold"
        assert len(set(seen)) == 1, f"admission differed between folds: {seen}"

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


class TestIntermittentDemandIsScoredHonestly:
    """
    The metric that ranks the models has to be defined on the months with no demand,
    because on an intermittent item those are most of the months.

    MAPE is not: it divides by the actual, so it can only score the months that carried
    demand and drops the rest. The drop has a direction. A model forecasting the order
    size every month is never charged for the months it invented, while one forecasting
    near zero is charged in full on the few it missed — so the highest line wins, and an
    item ordering once a quarter is forecast as if it ordered monthly.

    RMSE over every period tilts the other way on the same series: mostly-zero actuals
    are best fitted by predicting zero, so a genuinely recurring lumpy item is forecast
    flat at nothing and its should-be goes with it.
    """

    # Demand in two months of every three: enough non-zero actuals for MAPE to have
    # been chosen under the old rule, and eight zero months for it to have discarded.
    ERRATIC = [80, 0, 120, 90, 0, 110, 70, 0, 95, 105, 0, 85,
               90, 0, 100, 115, 0, 75, 95, 0, 105, 88]

    def test_a_series_with_zero_months_is_not_ranked_on_a_percentage(self):
        _, row = _run(self.ERRATIC)
        assert row["selected_by"] == "backtest:mase"

    def test_a_series_without_zero_months_still_reads_as_a_percentage(self):
        """MAPE is what a planner reads, and it is honest where nothing was dropped."""
        _, row = _run(list(np.linspace(100, 130, 22)))
        assert row["selected_by"] == "backtest:mape"

    def test_the_level_is_not_inflated_towards_the_months_that_had_demand(self):
        _, row = _run(self.ERRATIC)
        true_rate = float(np.mean(self.ERRATIC))
        assert row["forecast_qty"] == pytest.approx(true_rate, rel=0.35)

    def test_a_lumpy_item_is_not_forecast_at_nothing(self):
        """
        Two orders a year, and a backtest window that lands between them.

        Every actual the window holds is zero, so forecasting zero scores perfectly and
        the ranking is decided by which model happened to sit lowest — not by which one
        describes the item. The window has measured nothing, and the run has to say so
        rather than take the answer.
        """
        bursty = [0, 0, 200, 180, 0, 0, 0, 0, 0, 0, 0,
                  0, 0, 190, 210, 0, 0, 0, 0, 0, 0, 0]
        _, row = _run(bursty)
        assert row["forecast_qty"] > 0
        assert "deferred to TSB" in row["selected_by"]


class TestAnItemThatStoppedIsAllowedToDecay:
    """
    Croston updates its interval only when an order arrives, and measures the gaps
    *between* orders — never the gap between the last order and today. So the one run of
    zeros that means "this has stopped" is the one run it cannot read, and an item that
    ordered monthly for six months and nothing in the eighteen since keeps its old rate
    for ever. TSB updates the probability every period, zeros included.
    """

    DORMANT = [100] * 6 + [0] * 16

    def test_croston_cannot_see_the_run_of_zeros_it_ends_on(self):
        f = Forecaster(horizon=6)
        s = pd.Series([float(v) for v in self.DORMANT], index=PERIODS)
        _, croston, _ = f._croston(s)
        _, tsb, _ = f._tsb(s)
        assert croston[0] == pytest.approx(100.0, rel=0.2)   # unchanged by 16 quiet months
        assert tsb[0] < croston[0] / 10

    def test_the_forecast_for_a_dormant_item_falls_away(self):
        _, row = _run(self.DORMANT)
        assert row["forecast_qty"] < 5.0

    def test_demand_returning_brings_the_probability_back(self):
        f = Forecaster(horizon=6)
        quiet = pd.Series([float(v) for v in self.DORMANT], index=PERIODS)
        resumed = pd.Series([float(v) for v in ([100] * 6 + [0] * 12 + [100] * 4)],
                            index=PERIODS)
        assert f._tsb(resumed)[1][0] > f._tsb(quiet)[1][0] * 5


class TestTheLumpTravelsBesideTheRate:
    """
    Every model answers with a per-period rate, which is what safety stock integrates
    and what a planner misreads. 33 a month on an item that orders 100 once a quarter is
    the right rate and a false plan: it claims demand in two months out of three that
    are known to be empty, and a projection built on it draws a ramp where the real
    position is a sawtooth — hiding the month the stock runs out.
    """

    def test_a_quarterly_item_reports_its_order_size_and_interval(self):
        quarterly = [100.0 if i % 3 == 2 else 0.0 for i in range(22)]
        _, row = _run(quarterly)
        assert row["expected_order_size"] == pytest.approx(100.0, rel=0.05)
        assert row["expected_interval"] == pytest.approx(3.0, rel=0.2)

    def test_a_smooth_item_has_an_interval_of_one(self):
        """Where demand arrives every period the two descriptions are the same one."""
        _, row = _run([100.0] * 22)
        assert row["expected_interval"] == pytest.approx(1.0)

    def test_the_quiet_run_since_the_last_order_counts_towards_the_interval(self):
        """A gap still open is evidence about the rhythm, and the only evidence of a stop."""
        f = Forecaster(horizon=6)
        s = pd.Series([0.0, 100.0, 0.0, 100.0] + [0.0] * 18, index=PERIODS)
        assert f._demand_shape(s)["expected_interval"] > 2.0
