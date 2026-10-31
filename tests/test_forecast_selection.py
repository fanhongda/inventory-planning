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

from inventory_planning.analytics.forecaster import Forecaster, _MAX_DENSITY_FOR_CROSTON

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

    def test_an_intermittent_method_is_competitive_on_a_series_it_suits(self):
        """
        Asserted as *competitive* rather than as the winner, because on an intermittent
        item it is not a two-horse race and never was — see
        `TestTheRateModelsAllAnswerTheSameNumber`. Croston, TSB, SES, SMA and plain ETS
        all estimate a rate, and on LUMPY they pool within 14% of each other with
        forecasts between 16.4 and 20.4. Which name comes top is decided by sampling
        noise across nine backtest points; asserting a particular one encodes the noise.

        It was asserted that way because ETS could not fit at the time and so was never
        among them. What the test is really for is that an intermittent-demand method
        competes on merit instead of being routed to, which is what this checks.
        """
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in LUMPY], index=PERIODS)
        offered = f._candidates(series, 6)
        assert "Croston" in offered and "TSB" in offered
        _, row = _run(LUMPY, "MTO")
        assert row["selected_by"].startswith("backtest")
        assert row["forecast_qty"] == pytest.approx(float(np.mean(LUMPY)), rel=0.35)

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

    def test_intermittent_demand_lands_on_a_rate_and_gets_there_by_competing(self):
        """
        Not by routing. The winner is whichever rate model scored best and they are all
        within a couple of percent on this fixture, so what is asserted is the rate and
        the route, not the name — see the sibling test above.
        """
        _, row = _run(LUMPY, "MTS")
        assert row["selected_by"].startswith("backtest")
        assert row["forecast_qty"] == pytest.approx(float(np.mean(LUMPY)), rel=0.35)

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
        f._candidates = lambda s, h, arima=None, croston=None: (
            seen.append((arima, croston)) or original(s, h, arima, croston))
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
        """
        Asserted over the horizon rather than on t+1, because the forecast is no longer
        obliged to be flat. This fixture has demand in two months of every three, in a
        fixed phase, and a seasonal model now reproduces that phase — so t+1 is zero
        when a zero month is next, which is the right answer and not an under-forecast.
        What must not happen is the months with no demand being forecast as if they had
        some, and that shows in the total.
        """
        detail, _ = _run(self.ERRATIC)
        true_total = float(np.mean(self.ERRATIC)) * 6
        assert detail["forecast_qty"].sum() == pytest.approx(true_total, rel=0.35)

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


class TestCrostonIsOnlyOfferedWhereThereAreGaps:
    """
    Croston and TSB split a demand size from the interval between arrivals. Given a
    series with demand in every period there is no interval: every gap is one, `n_hat`
    initialises to 1.0 and its update has 1.0 as a fixed point, so Croston returns
    `z_hat` — exponential smoothing at alpha 0.2 and nothing more.

    It was offered anyway and it won, because a fixed-alpha level forecast is hard to
    beat on noisy trendless demand: 25 of the 65 items on the IN30 extract that sold in
    all 22 months came out `model_used = Croston`. The number was defensible; the label
    said the item was sporadic, which those 25 were not. So the level forecast stands in
    the pool as `SES` and the two intermittent-demand methods are withheld from items
    that have no intermittence.
    """

    # Demand every month, varying in size — the real shape of the mislabelled items.
    NOISY_CONTINUOUS = [46, 49, 25, 112, 32, 27, 38, 19, 44, 37, 43, 38,
                        49, 54, 71, 101, 30, 18, 21, 75, 40, 52]

    def test_a_series_with_no_empty_period_is_not_offered_croston(self):
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in self.NOISY_CONTINUOUS], index=PERIODS)
        offered = f._candidates(series, 6)
        assert "Croston" not in offered
        assert "TSB" not in offered
        assert "SES" in offered, "withdrawing Croston must not remove the level forecast"

    def test_the_winner_is_no_longer_named_after_an_intermittent_method(self):
        _, row = _run(self.NOISY_CONTINUOUS)
        assert row["model_used"] not in ("Croston", "TSB")

    def test_ses_gives_the_same_number_croston_did(self):
        """
        The gate is a correction to the label, not to the forecast. On a dense series
        Croston computes exactly this, so the quantity a planner buys must not move.
        """
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in self.NOISY_CONTINUOUS], index=PERIODS)
        assert f._ses(series, 6)[0] == pytest.approx(f._croston(series, 6)[1][0])

    def test_it_is_still_offered_where_the_demand_is_genuinely_sporadic(self):
        """The gate must not fire on the items the method exists for."""
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in LUMPY], index=PERIODS)
        offered = f._candidates(series, 6)
        assert "Croston" in offered and "TSB" in offered

    def test_an_irregularly_sporadic_item_still_wins_with_an_intermittent_method(self):
        """
        Irregular gaps, which is the case Croston is actually for. A *regularly* spaced
        item is better served by a model that reproduces the phase than by the average
        of it — see `TestARegularRhythmIsForecastAsOne` — so the fixture here is one
        whose rhythm no season can describe.
        """
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in LUMPY], index=PERIODS)
        assert "Croston" in f._candidates(series, 6)
        _, row = _run(LUMPY)
        assert row["forecast_qty"] == pytest.approx(float(np.mean(LUMPY)), rel=0.35)


class TestHowOftenDemandArrivesDecidesThePatternFirst:
    """
    The two Syntetos–Boylan axes are frequency and then size variability, in that order.
    They were tested the other way round: CV first, so anything between the two CV
    thresholds fell through to `intermittent` whatever its frequency. An item that sold
    in all twelve of the last twelve months at CV 0.61 came out *intermittent* — 51 of
    the 65 items on the IN30 extract that sold every single month did.

    The word means demand arrives in some periods and not others, and the label is read
    downstream as evidence of that: it routes a short series to TSB, and the policy
    profile shows it to a planner as `demand_continuity`.
    """

    @staticmethod
    def _pattern(values):
        from inventory_planning.analytics.demand_classifier import DemandClassifier
        ts = pd.DataFrame({"A": [float(v) for v in values]}, index=PERIODS[-12:])
        summary = pd.DataFrame({"sku": ["A"], "demand_mean": 0.0, "demand_std": 0.0})
        cfg = Path(__file__).parents[1] / "config"
        out = DemandClassifier(cfg).classify(summary, ts).set_index("sku")
        return out.loc["A", "demand_pattern"], out.loc["A", "demand_cv"]

    def test_demand_in_every_month_is_never_called_intermittent(self):
        pattern, cv = self._pattern([46, 49, 25, 112, 32, 27, 38, 19, 44, 37, 43, 38])
        assert 0.5 < cv < 1.0, "the CV band that used to decide this on its own"
        assert pattern not in ("intermittent", "lumpy")

    def test_a_continuous_item_whose_size_moves_is_called_variable(self):
        pattern, _ = self._pattern([46, 49, 25, 112, 32, 27, 38, 19, 44, 37, 43, 38])
        assert pattern == "variable"

    def test_a_steady_item_is_still_smooth(self):
        pattern, _ = self._pattern([100, 104, 98, 101, 99, 103, 97, 102, 100, 101, 99, 100])
        assert pattern == "smooth"

    def test_a_continuous_item_with_a_wild_size_is_erratic_not_intermittent(self):
        pattern, cv = self._pattern([5, 400, 10, 8, 350, 6, 12, 300, 9, 7, 280, 11])
        assert cv >= 1.0
        assert pattern == "erratic"

    def test_an_item_with_gaps_and_a_steady_size_is_intermittent(self):
        pattern, _ = self._pattern([50, 0, 55, 0, 48, 0, 52, 0, 51, 0, 49, 53])
        assert pattern == "intermittent"

    def test_an_item_with_gaps_and_a_wild_size_is_lumpy(self):
        pattern, _ = self._pattern([5, 0, 400, 0, 8, 0, 350, 0, 12, 0, 300, 6])
        assert pattern == "lumpy"

    def test_a_continuous_pattern_does_not_route_a_short_series_to_tsb(self):
        """`erratic` and `variable` mean gaps-free, and TSB has nothing to offer them."""
        f = Forecaster(horizon=6)
        short = pd.Series([46.0, 49.0, 25.0, 112.0, 32.0, 27.0, 38.0, 19.0],
                          index=PERIODS[-8:])
        rows = f._forecast_sku("A", short, "variable")
        assert rows[0]["model_used"] != "TSB"


class TestACycleIsMeasuredNotAssumed:
    """
    `_detect_seasonality` was a CV test wearing a seasonality test's name:

        monthly = [vals[i::12].mean() for i in range(12)]
        return np.std(monthly) > 0.2 * np.mean(monthly)

    With 22 months of history each bucket holds one observation or two, so `monthly` is
    very nearly the series and the test reduces to `CV > 0.2`. It fired on **65 of 65**
    items on the IN30 extract that sold every month. Nothing downstream saw the claim,
    because `_fit_ets` then discarded it below 24 observations — so a detector that
    answered yes to everything and a detector that worked were indistinguishable from
    the output. On a history longer than 24 months the answer stops being discarded, and
    a universal yes puts a twelve-parameter seasonal fit on the whole catalogue.

    Seasonality means the series resembles itself a cycle ago. That is autocorrelation
    at that lag, and that is what is measured now.
    """

    @staticmethod
    def _series(values):
        return pd.Series(np.asarray(values, dtype=float),
                         index=pd.period_range("2022-01", periods=len(values), freq="M"))

    @staticmethod
    def _wave(n, period, level=100.0, amp=60.0):
        return level + amp * np.sin(2 * np.pi * np.arange(n) / period)

    @pytest.mark.parametrize("period", [3, 4, 6, 12])
    def test_it_finds_the_cycle_that_is_there(self, period):
        f = Forecaster(horizon=6)
        assert f._seasonal_period(self._series(self._wave(36, period))) == period

    def test_a_short_cycle_is_found_on_a_history_too_short_for_an_annual_one(self):
        """
        The whole defect in one assertion. Eighteen months cannot hold two annual
        cycles, so the old code could see no seasonality here at all — the
        24-observation floor was never a fact about the data, only the consequence of
        hardcoding an annual season. A quarterly rhythm is plainly visible in eighteen.
        """
        f = Forecaster(horizon=6)
        assert f._seasonal_period(self._series(self._wave(18, 3))) == 3

    def test_an_annual_cycle_needs_more_than_the_two_cycles_it_takes_to_fit(self):
        """
        Two cycles is the least that can be *fitted*, not the least that can be
        *recognised*. On exactly 24 months an annual ACF reaches 0.43 against a bar of
        0.52 and is refused; around 26 it clears. The asymmetry is deliberate — a missed
        cycle leaves the forecast where it already was, a fabricated one reshapes a
        purchase recommendation.
        """
        f = Forecaster(horizon=6)
        assert f._seasonal_period(self._series(self._wave(22, 12))) is None
        assert f._seasonal_period(self._series(self._wave(24, 12))) is None
        assert f._seasonal_period(self._series(self._wave(30, 12))) == 12

    def test_the_shortest_qualifying_cycle_wins(self):
        """
        A three-month cycle also correlates at six and twelve, because they are
        multiples of it. Taking the largest autocorrelation would read a quarterly
        rhythm as an annual one and spend four times the parameters saying so.
        """
        f = Forecaster(horizon=6)
        assert f._seasonal_period(self._series(self._wave(48, 3))) == 3

    def test_noise_is_not_a_cycle(self):
        f = Forecaster(horizon=6)
        noise = 100 + 30 * np.random.default_rng(0).standard_normal(36)
        assert f._seasonal_period(self._series(noise)) is None

    def test_a_trend_is_not_a_cycle(self):
        """
        A climbing series correlates with itself at every lag: the raw ramp scored 0.99
        at lag 3 and came out quarterly. Measuring the first difference removes the
        climb — and flattens a ramp to a constant, whose variance is zero.
        """
        f = Forecaster(horizon=6)
        assert f._seasonal_period(self._series(np.arange(1, 37) * 10.0)) is None

    def test_a_flat_series_is_not_a_cycle(self):
        f = Forecaster(horizon=6)
        assert f._seasonal_period(self._series(np.full(36, 100.0))) is None

    def test_the_normal_catalogue_is_not_all_seasonal(self):
        """The gate that fired on everything. Varying is not cycling."""
        f = Forecaster(horizon=6)
        rng = np.random.default_rng(3)
        fired = sum(f._seasonal_period(
            self._series(100 + 40 * rng.standard_normal(30))) is not None
            for _ in range(30))
        assert fired <= 3, f"{fired}/30 plain noisy series called seasonal"


class TestETSIsActuallyInTheCompetition:
    """
    `ExponentialSmoothing.fit()` takes no `disp` argument. It was called with
    `disp=False`, so it raised `TypeError` every time, and `_candidates` catches every
    exception a candidate raises — which is right for a model that cannot fit this
    series and wrong for one that cannot fit anything. ETS was never once in the pool,
    and `_fit_smooth` fell through it to ARIMA on every call.

    Nothing said so. The model mix on the real extract listed SES, Naive, SMA, Croston,
    TSB and ARIMA and no ETS at all, which reads as a model that never wins rather than
    as a model that never ran.
    """

    def test_ets_is_offered(self):
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in RAMP], index=PERIODS)
        assert "ETS" in f._candidates(series, 6)

    def test_a_seasonal_ets_is_offered_beside_the_plain_one(self):
        """
        Beside, not instead. The detector decides what to try and the backtest decides
        what is used, which is the defence that holds when the detector is wrong.
        """
        f = Forecaster(horizon=6)
        wave = 100 + 60 * np.sin(2 * np.pi * np.arange(36) / 4)
        series = pd.Series(wave, index=pd.period_range("2022-01", periods=36, freq="M"))
        offered = f._candidates(series, 6)
        assert "ETS" in offered and "ETS-seasonal4" in offered

    def test_no_seasonal_entrant_where_there_is_no_cycle(self):
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in RAMP], index=PERIODS)
        assert not [n for n in f._candidates(series, 6) if n.startswith("ETS-seasonal")]


class TestARegularRhythmIsForecastAsOne:
    """
    A per-period rate is the right input to safety stock and the wrong description of an
    item that orders on a fixed beat. Where the beat is regular enough to be measured,
    the forecast can carry it instead of averaging it away.
    """

    def test_a_quarterly_item_is_forecast_on_its_beat(self):
        quarterly = [100.0 if i % 3 == 2 else 0.0 for i in range(22)]
        detail, _ = _run(quarterly)
        qty = detail["forecast_qty"].tolist()
        assert not np.allclose(qty, qty[0]), f"flattened to a rate: {qty}"
        # The next order is due two months out, and nothing is due next month.
        assert qty[0] == pytest.approx(0.0, abs=5)
        assert max(qty) == pytest.approx(100.0, rel=0.2)

    def test_the_total_over_the_horizon_still_matches_the_rate(self):
        """The shape may change; the quantity it integrates to may not drift."""
        quarterly = [100.0 if i % 3 == 2 else 0.0 for i in range(22)]
        detail, _ = _run(quarterly)
        assert detail["forecast_qty"].sum() == pytest.approx(
            float(np.mean(quarterly)) * 6, rel=0.3)


class TestTheMonthsBeforeAnItemExistedAreNotEvidence:
    """
    Density decided whether Croston is offered, and it was measured over the whole
    extract. The months before an item's first sale are zeros that mean *not
    applicable*, so reading them as "no demand" reads the length of the extract as
    evidence about the item.

    An item launched half-way through a 24-month history and selling every month since
    has a true density of 1.0, measured 0.50, and was offered Croston — which then won,
    and reported a continuously selling item as intermittent with a flat rate across the
    horizon. The bias grows with the history: a 48-month extract and two years of
    continuous sales still measured 0.50.

    Only the leading run is excluded. Trailing zeros are evidence that an item may have
    stopped, which is the question TSB exists to answer.
    """

    @staticmethod
    def _launched(total, live, level=100.0):
        values = np.concatenate([np.zeros(total - live), np.full(live, level)])
        return pd.Series(values,
                         index=pd.period_range("2023-01", periods=total, freq="M"))

    @pytest.mark.parametrize("total,live", [(24, 12), (36, 16), (48, 24)])
    def test_a_late_launch_is_not_read_as_intermittent_demand(self, total, live):
        f = Forecaster(horizon=6)
        series = self._launched(total, live)
        assert f._density(series) == pytest.approx(1.0)
        assert not f._sparse_enough_for_croston(series)
        assert "Croston" not in f._candidates(series, 6)

    def test_the_item_that_won_on_croston_before(self):
        """24 months, selling the last 12 — Croston was offered and chosen."""
        f = Forecaster(horizon=6)
        rng = np.random.default_rng(5)
        values = np.concatenate([np.zeros(12), np.clip(100 + 30 * rng.standard_normal(12), 1, None)])
        series = pd.Series(values, index=pd.period_range("2023-01", periods=24, freq="M"))
        model, _, _, _ = f._select_by_backtest(series)
        assert model not in ("Croston", "TSB")

    def test_trailing_quiet_still_counts(self):
        """
        An item that sold monthly and then stopped is sparse, and must stay so — that is
        the case TSB is for, and excluding trailing zeros too would hide it.
        """
        f = Forecaster(horizon=6)
        values = np.concatenate([np.full(10, 100.0), np.zeros(14)])
        series = pd.Series(values, index=pd.period_range("2023-01", periods=24, freq="M"))
        assert f._density(series) == pytest.approx(10 / 24)
        assert f._sparse_enough_for_croston(series)

    def test_a_genuinely_sporadic_item_is_unaffected(self):
        """Gaps after the first sale are still gaps."""
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in LUMPY], index=PERIODS)
        assert f._density(series) < _MAX_DENSITY_FOR_CROSTON
        assert f._sparse_enough_for_croston(series)

    def test_density_ignores_only_the_leading_run(self):
        f = Forecaster(horizon=6)
        series = pd.Series([0.0, 0.0, 10.0, 0.0, 10.0, 10.0],
                           index=pd.period_range("2023-01", periods=6, freq="M"))
        # Four live periods from the first sale, three of them with demand.
        assert f._density(series) == pytest.approx(3 / 4)


class TestTheTrendIsMeasuredNotInferredFromAUnitRoot:
    """
    `_detect_trend` asked a unit-root test a question it does not answer:

        clean = series[series > 0]
        return adfuller(clean, autolag="AIC")[1] > 0.05

    `adfuller` with its default `regression="c"` tests for a unit root against a
    stationary mean. A deterministic ramp has no unit root, so the test rejects and the
    rule concluded *no trend*; on a perfectly straight climb the regression is
    degenerate, `adfuller` returns NaN, and `nan > 0.05` is False. A random walk has a
    unit root and no trend, and the rule concluded the opposite. It fired on 60% of the
    catalogue.

    A growing item answered "no trend" gets a level-only ETS and is forecast flat,
    which is the complaint this was found from. Dropping the zero periods first made it
    worse: it compresses the time axis, so an item declining towards nothing has the
    evidence of the decline removed before the test sees it.
    """

    @staticmethod
    def _series(values):
        return pd.Series(np.asarray(values, dtype=float),
                         index=pd.period_range("2024-01", periods=len(values), freq="M"))

    def test_a_clean_upward_ramp_has_a_trend(self):
        """The case that returned NaN and was read as False."""
        f = Forecaster(horizon=6)
        assert f._detect_trend(self._series(np.arange(1, 25) * 10.0))

    def test_a_decline_has_a_trend(self):
        f = Forecaster(horizon=6)
        assert f._detect_trend(self._series(np.arange(24, 0, -1) * 10.0))

    def test_a_flat_series_has_none(self):
        f = Forecaster(horizon=6)
        assert not f._detect_trend(self._series(np.full(24, 100.0)))
        noisy = 100 + 20 * np.random.default_rng(1).standard_normal(24)
        assert not f._detect_trend(self._series(noisy))

    def test_a_slope_too_small_to_matter_is_not_a_trend(self):
        """Significant and material are different bars, and both have to clear."""
        f = Forecaster(horizon=6)
        barely = 100 + np.arange(24) * 0.05        # 1.2 across a level of 100
        assert not f._detect_trend(self._series(barely))

    def test_a_growing_item_is_not_forecast_flat(self):
        """What the defect cost: the whole point of detecting a trend."""
        detail, _ = _run(list(np.arange(1, 23) * 10.0))
        qty = detail["forecast_qty"].tolist()
        assert not np.allclose(qty, qty[0]), f"growing item forecast flat: {qty}"
        assert qty[-1] > qty[0]

    def test_a_trended_ets_is_offered_beside_the_plain_one(self):
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in RAMP], index=PERIODS)
        offered = f._candidates(series, 6)
        assert "ETS" in offered and "ETS-trend" in offered

    def test_no_trended_entrant_without_a_trend(self):
        f = Forecaster(horizon=6)
        noisy = 100 + 20 * np.random.default_rng(1).standard_normal(22)
        assert "ETS-trend" not in f._candidates(self._series(noisy), 6)


class TestTheRateModelsAllAnswerTheSameNumber:
    """
    Five of the seven entrants — Croston, TSB, SES, SMA and plain ETS — estimate a level
    and repeat it across the horizon. On an item with no trend and no cycle that is not
    a shortcut: the conditional mean of a stationary series is the same for t+1 and t+6,
    so a forecast that wiggles is claiming to know which month will be high, and where
    that knowledge is absent the wiggle costs accuracy rather than adding information.

    Two alternatives were measured on the real extract and both lost. Laying Croston's
    rate on the beat it already estimates scored MAE 89.06 against the flat 83.98 across
    359 lumpy items. Recursing the moving average so it feeds its own forecasts back in
    — which does make the horizon move — scored 369.32 against 358.62 over six months.

    The consequence for a reader is that `model_used` on an intermittent item carries
    much less information than its name suggests. This test pins that, so that a future
    change making one of them win by a nose is not mistaken for a finding.
    """

    def test_they_agree_on_an_intermittent_item(self):
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in LUMPY], index=PERIODS)
        offered = f._candidates(series, 6)
        rates = {n: float(v[0]) for n, v in offered.items()
                 if n in ("Croston", "TSB", "SES", "SMA", "ETS")}
        assert len(rates) >= 4, f"expected the rate models in the pool, got {list(rates)}"
        assert max(rates.values()) / min(rates.values()) < 1.5, rates

    def test_each_of_them_repeats_across_the_horizon(self):
        f = Forecaster(horizon=6)
        series = pd.Series([float(v) for v in LUMPY], index=PERIODS)
        for name, values in f._candidates(series, 6).items():
            if name in ("Croston", "TSB", "SES", "SMA", "ETS", "Naive"):
                assert np.allclose(values, values[0]), f"{name} is not a level model"
