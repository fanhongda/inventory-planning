"""
Time series forecaster.

## How the model is chosen

For **make-to-stock** items, by competition: every candidate is refitted on a rolling
origin, made to forecast periods it has not seen, and scored against what actually
happened. Lowest error wins.

That replaces a decision tree which never compared anything. It routed on the demand
pattern and the length of the series, then took whichever model did not raise an
exception — ETS if it fitted, else ARIMA, else a moving average — so accuracy played no
part in the choice, and no model was ever measured against the alternative of not
modelling at all. `forecast_rmse` was in-sample, the residuals of the fit against the
data it was fitted on, which is a measure of how closely a model can trace history
rather than of anything it can predict.

**Every item with enough history is backtested, whatever its policy.** Make-to-order
items used to go straight to Croston on the grounds that their demand is "by
construction intermittent". It is not: `stocking_policy` is an ERP decision not to hold
stock — because the item is configured to order, or expensive, or shipped monthly
against a contract — and says nothing about how often it sells. The pipeline knows
this elsewhere and reports it, in `run_policy_analysis`, as *held as MTO, demand
recurs*. An MTO item that in fact orders every month was being forced onto a method
built for the opposite case, and Croston answers with one number repeated across the
horizon.

Croston remains in the candidate pool and wins wherever it deserves to, which on a
genuinely sporadic item is most of the time. What changed is that it has to win.
Below `_MIN_BACKTEST_POINTS` there is not enough history to rank anything, and the
routing falls back to the policy and then to the demand pattern as before.

**Croston is offered only where there are gaps for it to be about.** On a series with
demand in every period its interval estimate is pinned at 1.0 — see
`_MAX_DENSITY_FOR_CROSTON` — so it computes exponential smoothing at a fixed alpha and
reports itself as an intermittent-demand method while doing it. That won 25 of the 65
items on the IN30 extract that sold in all 22 months. The level forecast is a fair
winner there and now stands in the pool under its own name, `SES`, which produces the
same number; what it no longer does is tell a planner the item is sporadic.

**ETS is in the pool, which it had not been.** `ExponentialSmoothing.fit()` takes no
`disp` argument; it was called with `disp=False` and raised `TypeError` on every series,
and `_candidates` catches whatever a candidate raises — right for a model that cannot
fit *this* series, wrong for one that cannot fit anything. So the model mix on the real
extract listed SES, Naive, SMA, Croston, TSB and ARIMA and no ETS at all, which reads as
a model that never wins rather than as one that never ran, and `_fit_smooth` fell
through ETS to ARIMA on every call. With the argument gone ETS wins 112 of 610 items.

**A cycle is measured, not assumed.** The seasonal period is whatever the series repeats
on — `_seasonal_period` — and no longer hardcoded to twelve months, which is what made
24 observations the price of any seasonality at all. What stood in for detection was
`std(monthly buckets) > 0.2 x mean`, which on a 22-month history is `CV > 0.2` with
extra steps and said yes to 65 of 65 items that sold every month; the answer was then
discarded below 24 observations, so nothing downstream could tell the difference. On a
longer history it is no longer discarded.

A **naive forecast is always in the pool**. Without it, "the best of five models" can
still be worse than repeating last month's number, and nothing in the output would say
so. `vs_naive` is the winner's error over the naive one's: 1.0 means the model earned
nothing, and a model that cannot beat repeating last month simply loses to it.

## How the winner is scored, and why not on a percentage

MAPE divides by the actual, so it exists only on the periods that carried demand. On an
intermittent item that is a minority of them, and discarding the rest is not a smaller
sample of the same question — it has a direction. A model forecasting the order size
every month is never charged for the months it invented; one forecasting near zero is
charged in full on the few it missed. The highest line wins, and a quarterly item comes
out forecast as if it ordered monthly.

RMSE over every period tilts the other way: mostly-zero actuals are best fitted by
predicting zero, so a genuinely recurring lumpy item is forecast flat at nothing.

**MASE** decides wherever the backtest saw a zero — defined on every period, scaled by
the series' own period-to-period movement, so neither tilt survives it. MAPE is kept for
the case it is honest in, no zero actuals anywhere, because a percentage is what a
planner reads. Both are reported either way.

Where the backtest window caught **no demand at all**, it has measured nothing: every
model that says zero ties at perfect and the ranking is an accident. Those items defer
to TSB, whose probability decays with the length of the quiet run — which is the actual
question, whether the item has stopped or is merely between orders.

## The rate, and the lump behind it

Every method answers with a per-period rate. That is what safety stock integrates and
what a planner misreads: 33 a month on an item that orders 100 once a quarter is the
right rate and a false plan, and a projection built on it draws a ramp where the real
position is a sawtooth. So `expected_order_size` and `expected_interval` travel beside
the rate. On a smooth item the interval is 1 and the two descriptions coincide.

Key outputs:
  forecast_next_period  — point forecast for t+1 (used by purchase_recommender)
  forecast_avg_monthly  — mean of 6-month window (for planning horizon view only)
  forecast_rmse         — error used for σDL: the backtest error where the model was
                          chosen by one, otherwise the in-sample residual as before
  expected_order_size   — how much arrives when something arrives
  expected_interval     — how many periods apart, counting the run since the last one
"""

import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


# How many origins the backtest rolls through, and how far ahead it forecasts at each.
# Three steps rather than one because a one-step score cannot see a model walking away
# from the data: SARIMAX on one SKU here forecast 4,578 for next month and 63 million a
# month on average, and a one-step backtest would have called it the winner.
_BACKTEST_FOLDS = 3
# Three steps rather than one is also what keeps a runaway model out. A separate ceiling
# on the horizon forecast was written for that and then measured: across the whole
# extract it never once changed which model was chosen, because a model that walks away
# from the data has already lost three-step scoring. It was removed rather than kept as
# insurance that insures nothing.
_BACKTEST_STEPS = 3
# Below this many observed periods there is nothing to hold out, and a competition
# decided on two points is not a competition.
_MIN_BACKTEST_POINTS = 12
# Share of periods that must carry demand before ARIMA is worth offering. Below it the
# model has zeros to explain rather than a series, and explains them by extrapolating.
# Set at a half rather than at the Syntetos-Boylan intermittence boundary (~0.76) so a
# genuine every-other-month item keeps it: on one of those ARIMA recovers the phase —
# `[0, 150, 0, 150, ...]` — which is a better answer than Croston's flat average of it.
_MIN_DENSITY_FOR_ARIMA = 0.5
# Share of periods that must be *empty* before Croston and TSB are worth offering.
#
# Both methods exist to split a demand size from the interval between arrivals. On a
# series with demand in every period there is no interval to estimate: every gap is one,
# `n_hat` initialises to 1.0, and its update — `0.2 x 1 + 0.8 x n_hat` — has 1.0 as a
# fixed point, so it stays there. Croston's forecast is then `z_hat / 1`, which is
# exponential smoothing of the series at alpha = 0.2 and nothing else. TSB degenerates
# the same way, with `p_hat` pinned at 1.0.
#
# Offered anyway, it wins — and it deserves to, because a fixed-alpha level forecast
# beats ETS and ARIMA on noisy trendless demand. On the IN30 extract it took 25 of the
# 65 items that sold in all 22 months. The forecast was defensible and the label was
# not: `model_used = Croston` tells a planner the item is sporadic, which is the one
# thing those 25 are not. So the level forecast now stands in the pool under its own
# name, `SES`, and the two intermittent-demand methods are offered only where there is
# intermittence for them to be about.
#
# 0.76 is the Syntetos–Boylan boundary, the same one `_MIN_DENSITY_FOR_ARIMA` is set
# below deliberately: ARIMA needs density 0.5 or more, Croston needs less than 0.76, and
# they overlap across the middle, which is where either could honestly win.
_MAX_DENSITY_FOR_CROSTON = 0.76
# Candidate cycle lengths for monthly data, shortest first: quarterly, four-monthly,
# half-yearly, annual. Tried in this order because a genuine three-month cycle also
# correlates at six and twelve — see `_seasonal_period`, which takes the shortest that
# qualifies rather than the strongest.
_SEASONAL_PERIODS = (3, 4, 6, 12)
# How much autocorrelation at the candidate lag before a cycle is worth fitting a season
# per phase for. The white-noise significance band decides alongside it, and the stricter
# of the two applies: the band is the right bar on a short series and too low a one on a
# long history, where an ACF of 0.18 clears it and still does not justify twelve
# parameters.
_MIN_SEASONAL_ACF = 0.3
# The band's multiplier. 2.50 rather than the familiar 1.96 because four candidate
# periods are tested and the bar has to hold for the set, not for one lag: at 1.96, one
# noisy series in nine came out seasonal — 10.6% across 1,200 generated ones — which on
# a 30,000-line catalogue is three thousand items handed a cycle they do not have. This
# is 0.05/4 two-sided, and measured at 3.1%.
#
# What it costs: an annual cycle on exactly 24 months is no longer found (ACF 0.43
# against a bar of 0.52). Two cycles is the least that can be fitted, not the least that
# can be recognised — on a clean cycle the measured thresholds are 13 months for a
# quarterly rhythm, 15 for four-monthly, 17 for half-yearly and 26 for annual, and more
# where the series is noisy. A missed cycle leaves the forecast where it already was; a
# fabricated one reshapes a purchase recommendation.
_SEASONAL_ACF_Z = 2.50
# MAPE needs a non-zero actual to divide by. With fewer than this many across the whole
# backtest the ranking would rest on one or two months, so RMSE decides instead — the
# metric is reported either way rather than left to be assumed.
_MIN_MAPE_POINTS = 4
MTS, MTO = "MTS", "MTO"


class Forecaster:

    def __init__(self, horizon: int = 6):
        self.horizon = horizon

    def forecast_all(self, time_series: pd.DataFrame,
                     classified: pd.DataFrame = None,
                     policy: pd.DataFrame = None,
                     rounding=None, uom: pd.Series = None) -> pd.DataFrame:
        """
        time_series: pivot DataFrame (period index × SKU columns).
        classified: output of DemandClassifier.classify() — provides demand_pattern per SKU.
        rounding: the `quantity_rounding` convention, from `analytics.rounding`. A
                forecast of 33.4 of a countable thing is not a forecast, and the decimal
                travels into the safety stock and the reorder point built on it.
        uom:    unit of measure per SKU, so a weight or a volume keeps its decimals.
                Absent for most SKUs in practice, which is why the convention rounds
                unless a unit says not to rather than the other way round.
        policy: frame carrying `sku` and `stocking_policy` (MTS/MTO) from the ERP.
                Both are backtested wherever there is enough history; the policy only
                decides the fallback for series too short to hold anything out, where
                MTO and an intermittent pattern both route to TSB.

        Returns long-format DataFrame:
          sku, period, forecast_qty, model_used, trend_detected, seasonal_detected,
          is_next_period (True for t+1 only), forecast_rmse, selected_by,
          backtest_mape, backtest_rmse, vs_naive
        """
        pattern_map = {}
        if classified is not None and "demand_pattern" in classified.columns:
            pattern_map = classified.set_index("sku")["demand_pattern"].to_dict()

        policy_map = {}
        if policy is not None and len(policy) and "stocking_policy" in policy.columns:
            policy_map = (policy.dropna(subset=["stocking_policy"])
                          .set_index("sku")["stocking_policy"].to_dict())

        results = []
        for sku in time_series.columns:
            series = time_series[sku].copy()
            if series.sum() == 0 or series.notna().sum() < 4:
                continue
            fc = self._forecast_sku(sku, series, pattern_map.get(sku, "smooth"),
                                    policy_map.get(sku))
            results.extend(fc)

        frame = pd.DataFrame(results) if results else pd.DataFrame()
        if rounding is not None and len(frame):
            # Rounded here rather than where each value is produced, so there is one
            # place a reader can check what happened to every forecast quantity, and so
            # the per-SKU unit can be joined once instead of looked up per period.
            per_row = (frame["sku"].astype(str).map(uom)
                       if uom is not None and len(uom) else None)
            frame["forecast_qty"] = rounding.apply(frame["forecast_qty"], per_row)
        return frame

    def _forecast_sku(self, sku: str, series: pd.Series, pattern: str,
                      policy: str = None) -> list:
        series = series.fillna(0).astype(float)
        scores = {}
        # Reported on every row, whichever path produced the forecast. These were left
        # False on the backtest path, which is nearly every item — so the two columns
        # said "no trend, no cycle" about a catalogue nobody had asked, and
        # `summary()` groups by them.
        trend_detected = self._detect_trend(series)
        seasonal_period = self._seasonal_period(series)
        seasonal_detected = seasonal_period is not None

        if series.notna().sum() >= _MIN_BACKTEST_POINTS:
            model_used, forecast_values, rmse, scores = self._select_by_backtest(series)
            selected_by = scores.pop("_metric")
        elif policy == MTO:
            model_used, forecast_values, rmse = self._tsb(series)
            selected_by = "policy:MTO"
        # Only the patterns that mean *gaps*. `erratic` and `variable` are continuous
        # demand whose size moves about, and an intermittent-demand method has nothing to
        # offer them — see `DemandClassifier._demand_pattern`, where frequency decides
        # which family a pattern belongs to before CV names it.
        elif pattern in ("intermittent", "lumpy"):
            model_used, forecast_values, rmse = self._tsb(series)
            selected_by = "pattern"
        else:
            model_used, forecast_values, rmse = self._fit_smooth(
                series, trend_detected, seasonal_period)
            selected_by = "pattern"

        last_period = series.index[-1]
        future_periods = pd.period_range(start=last_period + 1, periods=self.horizon,
                                         freq=last_period.freq)
        shape = self._demand_shape(series)

        return [
            {
                "sku": sku,
                "period": str(p),
                "forecast_qty": max(0, round(float(v), 1)),
                "model_used": model_used,
                "trend_detected": trend_detected,
                "seasonal_detected": seasonal_detected,
                "is_next_period": (i == 0),
                "forecast_rmse": round(rmse, 3),
                "selected_by": selected_by,
                "backtest_mape": scores.get("mape"),
                "backtest_mase": scores.get("mase"),
                "backtest_rmse": scores.get("rmse"),
                "vs_naive": scores.get("vs_naive"),
                # The rate above is what the period carries on average; these two say
                # what actually arrives and how often, so a lumpy item can be read as
                # lumpy downstream instead of as a flat line.
                "expected_order_size": shape["expected_order_size"],
                "expected_interval": shape["expected_interval"],
            }
            for i, (p, v) in enumerate(zip(future_periods, forecast_values))
        ]

    # ── Choosing by competition ──────────────────────────────────────────────

    @staticmethod
    def _density(series: pd.Series) -> float:
        """
        The share of periods that carried demand, counted from the item's first sale.

        The months before an item existed are zeros that mean *not applicable*, and
        reading them as "no demand" is reading the length of the extract as evidence
        about the item. An item launched half-way through a 24-month history and selling
        every month since has a true density of 1.0 and measures 0.50, which is how it
        came to be offered Croston — and the longer the history, the worse the bias
        gets. On 24 months with the last 12 continuous, Croston was offered and won.

        Only the *leading* run is excluded. Trailing zeros are evidence: an item that
        has gone quiet may have stopped, which is the question TSB exists to answer, so
        they stay in.

        This is the window Croston itself already works on — `_croston` starts its loop
        at the first non-zero observation. The gate deciding whether to offer it now
        measures over the same span the method would use.
        """
        values = series.values.astype(float)
        nonzero = np.flatnonzero(values > 0)
        if len(nonzero) == 0:
            return 0.0
        live = values[int(nonzero[0]):]
        return float((live > 0).mean())

    @classmethod
    def _dense_enough_for_arima(cls, series: pd.Series) -> bool:
        """Whether a series has enough demand in it for ARIMA to be about anything."""
        return bool(cls._density(series) >= _MIN_DENSITY_FOR_ARIMA)

    @classmethod
    def _sparse_enough_for_croston(cls, series: pd.Series) -> bool:
        """
        Whether a series has gaps for an intermittent-demand method to be about.

        See `_MAX_DENSITY_FOR_CROSTON`: above that density both Croston and TSB collapse
        into exponential smoothing at a fixed alpha, which `SES` offers under its own
        name, so all that is left to choose between them is which label the output
        carries.
        """
        return bool(cls._density(series) < _MAX_DENSITY_FOR_CROSTON)

    def _candidates(self, series: pd.Series, horizon: int,
                    allow_arima: bool = None, allow_croston: bool = None) -> dict:
        """
        Every model's forecast for the next `horizon` periods, by name.

        Each entry is built the same way it would be for the real forecast, so the
        backtest measures what the run will actually do rather than a simplified stand-in.
        """
        out = {}
        trend = self._detect_trend(series)
        period = self._seasonal_period(series)

        # ARIMA is offered only where there is a series for it to be about.
        #
        # On a mostly-zero history it has no autocorrelation to find and fits the
        # arrangement of the zeros instead: `[0]*6, 4, [0]*12, 14, 2, 25` is read as a
        # ramp and run out to 148, against a maximum ever observed of 25. That case is
        # in the tests because it happened on a real SKU, where SARIMAX forecast 63
        # million a month. What kept it from being chosen was Croston scoring better —
        # and Croston only scored better because its interval estimate started at 1.0
        # and never converged, so the defence was resting on the bias fixed above.
        #
        # It is also 97% of the cost of a backtest: 5.15 ms a fit against 0.16 for ETS
        # and 0.01 for Croston. At 30,000 SKUs that is ten minutes against under one.
        # Correctness and cost point the same way here, which is rare enough to take.
        #
        # `allow_arima` is decided by the caller so that every fold of one backtest
        # decides it the same way. Judged per fold it moves: the training windows differ
        # by a period each, and a series sitting on the boundary admits ARIMA to some
        # folds and not others, leaving it ranked on six points against everyone else's
        # nine. Pooling across folds exists precisely so that no model is scored on a
        # different sample from its rivals.
        if allow_arima is None:
            allow_arima = self._dense_enough_for_arima(series)
        # Decided by the caller for the same reason `allow_arima` is: density moves by a
        # period between folds, and a series near the boundary would otherwise admit
        # Croston to some folds and not others.
        if allow_croston is None:
            allow_croston = self._sparse_enough_for_croston(series)

        candidates = [
            ("ETS", lambda: self._fit_ets(series, trend, None, horizon)),
        ]
        # Offered beside plain ETS rather than in place of it, so a detected cycle has
        # to earn the parameters it costs. The detector decides what to try; the
        # backtest decides what is used, which is the one defence that holds when the
        # detector is wrong — and the detector it replaced was wrong on every item.
        if period is not None:
            candidates.append(
                (f"ETS-seasonal{period}",
                 lambda: self._fit_ets(series, trend, period, horizon)))
        if allow_arima:
            candidates.append(("ARIMA", lambda: self._fit_arima(series, horizon)))
        if allow_croston:
            candidates += [
                ("Croston", lambda: self._croston(series, horizon)[1]),
                # Croston's estimate of the interval cannot fall while an item is quiet,
                # so it competes against a version that can.
                ("TSB", lambda: self._tsb(series, horizon)[1]),
            ]
        for name, build in candidates + [
            ("SMA", lambda: self._fit_sma(series, horizon)),
            # The level forecast, under its own name. Not a duplicate of `ETS`: that
            # entrant lets statsmodels optimise alpha against the fit, needs twelve
            # observations before it is offered at all, and on noisy demand an optimised
            # alpha chases the noise. This one holds alpha at 0.2 and starts from the
            # mean of the series — which is precisely what Croston reduces to on a dense
            # series, so withdrawing Croston from those items costs no accuracy, only
            # the wrong label.
            ("SES", lambda: self._ses(series, horizon)),
            # The floor every other model has to clear: repeat the last observation.
            ("Naive", lambda: np.full(horizon, float(series.iloc[-1]))),
        ]:
            try:
                values = build()
            except Exception:
                continue
            if values is not None and np.all(np.isfinite(values)):
                out[name] = np.clip(np.asarray(values, dtype=float), 0, None)
        return out

    def _select_by_backtest(self, series: pd.Series):
        """
        Refit every candidate on a rolling origin and keep the one that predicted best.

        Errors are pooled across folds rather than averaged per fold, so a model is not
        rewarded for doing well on the one origin that happened to be easy. MAPE decides
        where there are enough non-zero actuals to divide by, and RMSE where there are
        not — an intermittent month of zero demand has no percentage error, and dropping
        those months quietly would rank models on whichever ones survived.
        """
        n = len(series)
        steps = min(_BACKTEST_STEPS, self.horizon)
        origins = [n - steps - _BACKTEST_FOLDS + 1 + i for i in range(_BACKTEST_FOLDS)]
        origins = [o for o in origins if o >= _MIN_BACKTEST_POINTS - steps and o >= 4]

        # Settled once, on the whole series, and applied to every fold. Which model
        # classes are meaningful for an item is a property of the item, not of how much
        # of it a particular fold was shown.
        allow_arima = self._dense_enough_for_arima(series)
        allow_croston = self._sparse_enough_for_croston(series)

        pooled = {}
        for origin in origins:
            train, actual = series.iloc[:origin], series.iloc[origin:origin + steps].values
            if len(actual) == 0:
                continue
            for name, values in self._candidates(train, len(actual), allow_arima,
                                                 allow_croston).items():
                pooled.setdefault(name, {"err": [], "act": []})
                pooled[name]["err"].extend(values[:len(actual)] - actual)
                pooled[name]["act"].extend(actual)

        if not pooled:
            trend, period = self._detect_trend(series), self._seasonal_period(series)
            model, values, rmse = self._fit_smooth(series, trend, period)
            return model, values, rmse, {"_metric": "in-sample (too short to backtest)"}

        # A window that caught no demand has not measured anything.
        #
        # The backtest evaluates the last few periods of the series. On an item that
        # orders twice a year, those periods routinely fall between two orders — so
        # every actual is zero, forecasting zero scores perfectly, and the ranking is
        # decided by which model happened to sit lowest rather than by which one
        # describes the item. A real SKU ordering ~200 twice a year came out of this
        # forecast at zero for all six months, and its should-be with it.
        #
        # The question the window cannot answer is whether the quiet run means the item
        # has stopped or that it is between orders, and that is precisely what TSB
        # answers: its probability decays with the length of the run, so a long-dormant
        # item still falls to nothing while one inside its normal rhythm does not.
        # Deferring to it is narrower than distrusting the backtest generally — the
        # competition still decides every item whose window saw demand.
        if not any(np.any(np.asarray(a["act"], dtype=float) != 0) for a in pooled.values()):
            model, values, rmse = self._tsb(series)
            return model, values, rmse, {
                "_metric": "no demand in backtest window — deferred to TSB"}

        # The scale MASE divides by: how much the series moves from one period to the
        # next. A property of the item, so it is measured once on the whole series
        # rather than per fold, for the same reason `allow_arima` is.
        mase_scale = float(np.mean(np.abs(np.diff(series.values.astype(float)))))

        scored = {}
        for name, acc in pooled.items():
            err = np.asarray(acc["err"], dtype=float)
            act = np.asarray(acc["act"], dtype=float)
            nz = act != 0
            scored[name] = {
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mape": (float(np.mean(np.abs(err[nz] / act[nz]))) if nz.sum() >= _MIN_MAPE_POINTS
                         else None),
                "mase": (float(np.mean(np.abs(err)) / mase_scale) if mase_scale > 0 else None),
                "n_nonzero": int(nz.sum()),
            }

        # Which metric ranks the models, and why it is not always MAPE.
        #
        # MAPE is defined only where the actual is non-zero, so on an intermittent item
        # it scores the months that had demand and silently discards the months that did
        # not. That is not a smaller sample of the same question — it is a different
        # question, and it has a direction: a model forecasting the order size every
        # month is never charged for the months it invented, while a model forecasting
        # near zero is charged in full on the few months it missed. The highest line
        # wins, which is how an item ordering once a quarter comes out forecast as if it
        # ordered monthly.
        #
        # RMSE over every period has the opposite tilt on the same series: it is
        # minimised by predicting zero when most periods are zero, so a genuinely
        # recurring lumpy item comes out forecast flat at nothing.
        #
        # MASE is defined on every period, zeros included, and is scaled by the series'
        # own period-to-period movement, so neither tilt survives it. MAPE is kept for
        # the case it is honest in — no zero actuals anywhere in the backtest — because
        # a percentage is what a planner reads, and reported either way.
        any_zero_actual = any(0 in np.asarray(a["act"], dtype=float) for a in pooled.values())
        if not any_zero_actual and all(s["mape"] is not None for s in scored.values()):
            metric = "mape"
        elif all(s["mase"] is not None for s in scored.values()):
            metric = "mase"
        else:
            metric = "rmse"
        # The same admission the folds were judged under, so the winner of the ranking
        # is guaranteed to be among the models refitted to produce the forecast.
        final = self._candidates(series, self.horizon, allow_arima, allow_croston)
        ranked = sorted(scored, key=lambda k: scored[k][metric])
        best = next((n for n in ranked if n in final), ranked[0])
        values = final.get(best, np.full(self.horizon, float(series.tail(3).mean())))

        # Reported as a ratio, not a verdict. A boolean here would always have said
        # yes: the winner is the lowest-scoring model and Naive is one of the entrants,
        # so "beats naive" is true by construction and says nothing. The ratio does say
        # something — 1.0 is a model that has added nothing over repeating last month.
        naive = scored.get("Naive", {}).get(metric)
        return best, values, scored[best]["rmse"], {
            "_metric": f"backtest:{metric}",
            "mape": scored[best]["mape"],
            "mase": scored[best]["mase"],
            "rmse": scored[best]["rmse"],
            "vs_naive": (None if not naive else round(scored[best][metric] / naive, 3)),
        }

    # ── Smooth demand: ETS → ARIMA → SMA ─────────────────────────────────────

    def _fit_ets(self, series: pd.Series, trend: bool, period=None, horizon: int = None):
        """
        `period` is the cycle length to fit a season for, or None for level and trend
        only. It used to be the pair (`seasonal: bool`, 12), so the one cycle this could
        ever fit was an annual one and the one history that could hold two of those was
        24 months long.
        """
        n = len(series)
        horizon = self.horizon if horizon is None else horizon
        if n < 12:
            return None
        if period is not None and n < 2 * period:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                series,
                trend="add" if trend else None,
                seasonal="add" if period else None,
                seasonal_periods=period if period else None,
                initialization_method="estimated",
            ).fit(optimized=True)
            return model.forecast(horizon).values

    def _fit_arima(self, series: pd.Series, horizon: int):
        if len(series) < 8:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(series, order=(1, 1, 1), enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
            return model.forecast(horizon).values

    def _fit_sma(self, series: pd.Series, horizon: int):
        window = min(len(series), 3)
        return np.full(horizon, float(series.tail(window).mean()))

    @staticmethod
    def _ses(series: pd.Series, horizon: int):
        """
        Exponential smoothing of the level, at a fixed alpha, seeded from the mean.

        The plainest answer a forecaster can give to a series with no trend and no
        seasonality: a weighted average of history that leans on the recent end. It is
        in the pool because it was already winning there in disguise — Croston on a
        series with demand in every period computes exactly this and reported itself as
        an intermittent-demand method while doing it.

        Alpha is fixed rather than fitted. `ETS` is the entrant that fits it, and the two
        differ on purpose: an optimised alpha is chosen to trace the history it was given
        and on noisy demand that means chasing noise, which the backtest then charges it
        for. Which of the two is the better description of an item is the competition's
        question, not something to settle here.
        """
        values = series.values.astype(float)
        alpha = 0.2
        level = float(values.mean())
        for v in values[1:]:
            level = alpha * v + (1 - alpha) * level
        return np.full(horizon, max(0.0, level))

    def _fit_smooth(self, series: pd.Series, trend: bool, period=None):
        """
        The original decision tree, kept for SKUs with no MTS policy and for series too
        short to hold anything out. First model that fits, in a fixed order — the point
        of the backtest above is that this order is not evidence about accuracy.

        `period` is the detected cycle length or None, as in `_fit_ets`. Nothing here
        compares anything, so a detected cycle is fitted on trust — which is the reason
        `_seasonal_period` has to be a real test rather than a CV threshold.
        """
        n = len(series)
        if period is not None and n < 2 * period:
            period = None

        if n >= 12:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    trend_arg = "add" if trend else None
                    model = ExponentialSmoothing(
                        series,
                        trend=trend_arg,
                        seasonal="add" if period else None,
                        seasonal_periods=period if period else None,
                        initialization_method="estimated",
                    ).fit(optimized=True)
                    fc = model.forecast(self.horizon).values
                    rmse = self._insample_rmse(series, model.fittedvalues)
                    return "ETS", fc, rmse
            except Exception:
                pass

        if n >= 8:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(series, order=(1, 1, 1), enforce_stationarity=False,
                                    enforce_invertibility=False).fit(disp=False)
                    fc = model.forecast(self.horizon).values
                    rmse = self._insample_rmse(series, model.fittedvalues)
                    return "ARIMA", fc, rmse
            except Exception:
                pass

        window = min(n, 3)
        avg = series.tail(window).mean()
        # RMSE for SMA: std dev of residuals over last window
        fitted = series.shift(1).rolling(window, min_periods=1).mean()
        rmse = self._insample_rmse(series, fitted)
        return "SMA", np.full(self.horizon, avg), rmse

    # ── Intermittent / Erratic: Croston's Method ──────────────────────────────

    def _croston(self, series: pd.Series, horizon: int = None):
        """
        Croston's method (MIT CTL §3):
          Separate demand magnitude (ẑ) from inter-arrival interval (n̂).
          Forecast = ẑ / n̂ per period.
        """
        horizon = self.horizon if horizon is None else horizon
        alpha = 0.2  # magnitude smoothing
        beta = 0.2   # interval smoothing

        values = series.values.astype(float)
        n = len(values)

        # Initialise from what was actually observed, not from a guess of 1.0.
        #
        # `n_hat = 1.0` says "demand every period", which is the one thing an
        # intermittent series is not. With beta = 0.2 the estimate crawls towards the
        # truth — 0.2 of the gap per demand epoch — so the starting value survives for
        # as long as there are few epochs, and few epochs is the definition of the
        # items this method is chosen for. An item ordered every fourth month came out
        # 92% too high after three orders, 44% after five, and needed about thirty
        # before the error fell under a percent: ten years of history for a quarterly
        # reorder. The forecast is a rate, so the error goes straight into safety stock
        # and lands on exactly the sparse, make-to-order items that should be holding
        # least.
        #
        # `z_hat` moves from one observation to the mean of them for the same reason —
        # not to remove a bias, since a single draw is unbiased, but because one draw
        # on a series with five of them is a needlessly noisy place to start.
        nonzero = np.flatnonzero(values > 0)
        if len(nonzero) == 0:
            avg = series.mean()
            return "Croston", np.full(horizon, avg), float(series.std())

        first_nonzero = int(nonzero[0])
        z_hat = float(values[nonzero].mean())     # magnitude estimate
        n_hat = (float(np.diff(nonzero).mean()) if len(nonzero) > 1
                 # A single epoch has no interval to observe. What is known is that one
                 # order arrived across the whole span, and that is the estimate — the
                 # alternative is to assert an interval of one period, which is the
                 # error being fixed.
                 else max(1.0, float(n)))
        last_nonzero = first_nonzero

        residuals = []
        for t in range(first_nonzero + 1, n):
            forecast_t = z_hat / n_hat
            if values[t] > 0:
                interval = t - last_nonzero
                z_hat = alpha * values[t] + (1 - alpha) * z_hat
                n_hat = beta * interval + (1 - beta) * n_hat
                last_nonzero = t
                residuals.append(values[t] - forecast_t)

        per_period_forecast = z_hat / n_hat
        fc = np.full(horizon, max(0.0, per_period_forecast))
        rmse = float(np.sqrt(np.mean(np.array(residuals) ** 2))) if residuals else float(series.std())
        return "Croston", fc, rmse

    # ── Intermittent that may have stopped: TSB ───────────────────────────────

    def _tsb(self, series: pd.Series, horizon: int = None):
        """
        Teunter–Syntetos–Babai: Croston's split, with the probability updated every
        period instead of only when an order arrives.

        Croston updates its interval estimate at demand epochs only, and measures the
        gaps *between* orders — never the gap between the last order and today. So an
        item that ordered monthly for six months and has ordered nothing in the
        eighteen since carries an interval of 1.0 forever, and is forecast at its old
        rate indefinitely. Nothing in the method can express "this has stopped",
        because the only evidence that it stopped is the run of zeros it never reads.

        TSB reads them. `p` — the probability a period carries demand — decays by
        `(1-beta)` on every zero period, so the forecast `p·z` falls away on its own as
        an item goes quiet, and recovers when it orders again. The magnitude estimate
        `z` still updates only on demand epochs, because a period with no order carries
        no information about how big the next one will be.
        """
        horizon = self.horizon if horizon is None else horizon
        alpha = 0.2   # magnitude smoothing
        beta = 0.2    # probability smoothing

        values = series.values.astype(float)
        nonzero = np.flatnonzero(values > 0)
        if len(nonzero) == 0:
            avg = float(series.mean())
            return "TSB", np.full(horizon, avg), float(series.std())

        # Initialised from what was observed, for the same reason Croston's is: an
        # estimate that starts at a guess and moves 20% of the way per period survives
        # for as long as the history is short, and short history is the case.
        z_hat = float(values[nonzero].mean())
        p_hat = float(len(nonzero) / len(values))

        residuals = []
        for t in range(int(nonzero[0]) + 1, len(values)):
            forecast_t = p_hat * z_hat
            if values[t] > 0:
                z_hat = alpha * values[t] + (1 - alpha) * z_hat
                p_hat = beta * 1.0 + (1 - beta) * p_hat
                residuals.append(values[t] - forecast_t)
            else:
                p_hat = (1 - beta) * p_hat

        fc = np.full(horizon, max(0.0, p_hat * z_hat))
        rmse = (float(np.sqrt(np.mean(np.array(residuals) ** 2))) if residuals
                else float(series.std()))
        return "TSB", fc, rmse

    @staticmethod
    def _demand_shape(series: pd.Series) -> dict:
        """
        The lump behind the rate: how big an order is, and how often one arrives.

        Every method in the pool answers with a per-period rate, because that is what
        safety stock and the projection integrate. For an item that orders 100 once a
        quarter the rate is 33 a month, and it is the right number to carry into σDL —
        but read as a plan it says demand arrives every month, which is the one thing
        known not to happen. The S&OP sheet shows it to a planner who can see that it
        is wrong, and a projection built on it draws a smooth ramp where the real
        position is a sawtooth, hiding the month the stock actually runs out.

        So the shape travels beside the rate rather than replacing it: `ẑ` is what
        arrives when something arrives, and the interval is how many periods apart. On
        a smooth item the interval is 1 and the two descriptions coincide.
        """
        values = series.values.astype(float)
        nonzero = np.flatnonzero(values > 0)
        if len(nonzero) == 0:
            return {"expected_order_size": 0.0, "expected_interval": None}
        size = float(values[nonzero].mean())
        # Intervals between orders, plus the run of zeros since the last one — a gap
        # still open is evidence about the rhythm too, and it is the only evidence that
        # an item has gone quiet.
        gaps = list(np.diff(nonzero))
        trailing = len(values) - 1 - int(nonzero[-1])
        if trailing > 0:
            gaps.append(trailing)
        interval = float(np.mean(gaps)) if gaps else float(len(values))
        return {"expected_order_size": round(size, 1),
                "expected_interval": round(max(1.0, interval), 2)}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _detect_trend(self, series: pd.Series) -> bool:
        clean = series[series > 0]
        if len(clean) < 4:
            return False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = adfuller(clean, autolag="AIC")
            return result[1] > 0.05
        except Exception:
            return False

    def _seasonal_period(self, series: pd.Series):
        """
        The cycle length the series repeats on, or None where it does not repeat.

        What this replaces was a CV test wearing a seasonality test's name:

            monthly = [vals[i::12].mean() for i in range(12)]
            return np.std(monthly) > 0.2 * np.mean(monthly)

        With 22 months of history each of those twelve buckets holds one observation or
        two, so `monthly` is very nearly the series itself and the test reduces to
        `CV > 0.2`. Every item that varies at all passes it: on the IN30 extract it
        fired on **65 of 65** items that sold every month, and the answer was then
        discarded by the `n >= 24` gate in `_fit_ets`, so nothing downstream ever saw
        what it claimed. A detector that fires on everything is detecting nothing, and
        one whose answer is always thrown away hides that fact. With more than 24 months
        in hand the answer stops being thrown away, and a universal yes would put a
        twelve-parameter seasonal fit on every item in the catalogue.

        Seasonality means the series resembles itself a cycle ago, which is
        autocorrelation at that lag, so that is what this measures. `_SEASONAL_PERIODS`
        are the candidate business cycles for monthly data; a two-month "cycle" is left
        out because alternating noise and a real fortnightly rhythm are the same
        arithmetic, and an every-other-month item is better served by ARIMA recovering
        its phase, which `_MIN_DENSITY_FOR_ARIMA` already exists to allow.

        The bar is the white-noise significance band, widened for the number of lags
        tested (`_SEASONAL_ACF_Z`), and an absolute floor besides. The band alone loosens
        as the series lengthens — on 120 months an ACF of 0.18 clears it — and an effect
        that is distinguishable from noise is not therefore worth twelve parameters. Both
        must hold.

        The **shortest** qualifying period wins, not the strongest. A genuine
        three-month cycle also correlates at six and twelve, because they are multiples
        of it; picking the largest ACF would read a quarterly rhythm as an annual one and
        spend four times the parameters saying so.

        A period is only offered where the history covers it twice, which is what
        estimating a season per phase needs at a minimum — and is where the old `n >= 24`
        came from. It was never a fact about the data, only that 12 was hardcoded.

        Measured on the **first difference**, because a trend correlates with itself at
        every lag. A plain linear ramp scored 0.99 at lag 3 on the raw series and came
        out quarterly: the longer a series climbs, the more any lag looks like a cycle.
        Differencing removes the climb and leaves the rhythm — a period-m signal differs
        with period m — and it flattens a ramp to a constant, whose variance is zero, so
        the one series most likely to be misread cannot be read at all.
        """
        observed = len(series)
        values = np.diff(series.values.astype(float))
        if len(values) < 2:
            return None
        centred = values - values.mean()
        denom = float(np.dot(centred, centred))
        if denom <= 0:
            return None
        bar = max(_MIN_SEASONAL_ACF, _SEASONAL_ACF_Z / np.sqrt(len(values)))
        for m in _SEASONAL_PERIODS:
            # Two full cycles judged on the observations, not on the differences: 24
            # months hold two annual cycles, and differencing costs one point, not one
            # cycle. The lag still has to fit inside the differenced series.
            if observed < 2 * m or len(values) <= m:
                continue
            acf = float(np.dot(centred[:-m], centred[m:]) / denom)
            if acf >= bar:
                return m
        return None

    def _detect_seasonality(self, series: pd.Series) -> bool:
        """Whether there is a cycle at all. The period itself is what callers want."""
        return self._seasonal_period(series) is not None

    @staticmethod
    def _insample_rmse(actual: pd.Series, fitted: pd.Series) -> float:
        residuals = actual.values - fitted.values
        valid = residuals[~np.isnan(residuals)]
        return float(np.sqrt(np.mean(valid ** 2))) if len(valid) > 0 else 0.0

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-SKU summary:
          forecast_next_period  — point forecast for t+1 (use this for purchase qty)
          forecast_avg_monthly  — mean over full horizon (use for planning view)
          forecast_6m_total     — 6-month total
          forecast_rmse         — in-sample RMSE (use for safety stock σDL)
        """
        if forecast_df.empty:
            return pd.DataFrame()

        # Both numbers where a sales review has been applied: `forecast_next_period` is
        # what the plan runs on, `statistical_next_period` is what the model said before
        # anyone argued with it. Carrying only the first would make the review
        # permanently unscoreable — next period could measure the error but never say
        # whether the adjustment caused it or prevented it.
        carry = ["sku", "forecast_qty", "forecast_rmse"]
        for column in ("statistical_qty", "forecast_source"):
            if column in forecast_df.columns:
                carry.append(column)
        next_period = (
            forecast_df[forecast_df["is_next_period"]][carry]
            .rename(columns={"forecast_qty": "forecast_next_period",
                             "statistical_qty": "statistical_next_period"})
        )

        horizon_agg = (
            forecast_df.groupby(["sku", "model_used", "trend_detected", "seasonal_detected"])
            .agg(
                forecast_6m_total=("forecast_qty", "sum"),
                forecast_avg_monthly=("forecast_qty", "mean"),
            )
            .reset_index()
            .round(1)
        )

        return horizon_agg.merge(next_period, on="sku", how="left").round(3)

    # ── The forecast, as a sheet a planner can read ──────────────────────────

    def history_and_forecast(self, time_series: pd.DataFrame,
                             forecast_detail: pd.DataFrame) -> pd.DataFrame:
        """
        One row per SKU: what it did, then what it is expected to do.

        The pipeline's other forecast outputs are long-format and per-period, which is
        the right shape for joining and the wrong one for reading. A planner checking
        whether a forecast is sane reads along the row — twelve months of history and
        the six that follow, in the same units, on one line — and the judgement is
        immediate in a way that no summary statistic makes it.

        The provenance travels with the numbers: which model, how it was chosen, what
        it scored, and how that compares to having no model at all. A forecast whose
        `vs_naive` is 1.0 is worth reading differently from one at 0.4.
        """
        if forecast_detail is None or forecast_detail.empty:
            return pd.DataFrame()

        meta_cols = ["model_used", "selected_by", "backtest_mape", "backtest_mase",
                     "backtest_rmse", "vs_naive", "forecast_rmse",
                     # The lump behind the rate. A reader checking whether a forecast is
                     # sane needs to know that the flat 33 a month is one order of 100 a
                     # quarter, and the row is where that judgement is being made.
                     "expected_order_size", "expected_interval"]
        meta = (forecast_detail[forecast_detail["is_next_period"]]
                .set_index("sku")
                .reindex(columns=[c for c in meta_cols if c in forecast_detail.columns]))

        future = (forecast_detail.pivot_table(index="sku", columns="period",
                                              values="forecast_qty", aggfunc="first")
                  .rename(columns=lambda c: f"fcst {c}"))

        history = time_series.T
        history.index.name = "sku"
        history = history.rename(columns=lambda c: f"hist {c}")

        out = meta.join(history, how="left").join(future, how="left").reset_index()
        # History then forecast, each in date order, so the row reads left to right as
        # time. A pivot sorts them lexically and interleaves the two.
        lead = ["sku"] + [c for c in meta_cols if c in out.columns]
        hist = sorted(c for c in out.columns if c.startswith("hist "))
        fcst = sorted(c for c in out.columns if c.startswith("fcst "))
        return out[lead + hist + fcst]

