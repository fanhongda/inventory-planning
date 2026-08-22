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

**Make-to-order items go straight to Croston** and are not backtested. Nothing is
stocked against a forecast there, the demand is by construction intermittent, and
Croston is the method for it — a competition would spend the time to arrive where it
started. The policy comes from the ERP (`stocking_policy`, MTS/MTO); where a SKU has
none, the old pattern-based routing still applies, because a guess about the policy is
not a reason to change how an item is planned.

A **naive forecast is always in the pool**. Without it, "the best of five models" can
still be worse than repeating last month's number, and nothing in the output would say
so. `vs_naive` is the winner's error over the naive one's: 1.0 means the model earned
nothing, and a model that cannot beat repeating last month simply loses to it.

Key outputs:
  forecast_next_period  — point forecast for t+1 (used by purchase_recommender)
  forecast_avg_monthly  — mean of 6-month window (for planning horizon view only)
  forecast_rmse         — error used for σDL: the backtest error where the model was
                          chosen by one, otherwise the in-sample residual as before
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
                     policy: pd.DataFrame = None) -> pd.DataFrame:
        """
        time_series: pivot DataFrame (period index × SKU columns).
        classified: output of DemandClassifier.classify() — provides demand_pattern per SKU.
        policy: frame carrying `sku` and `stocking_policy` (MTS/MTO) from the ERP.
                MTS is backtested; MTO goes straight to Croston; absent means the old
                pattern-based routing.

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

        return pd.DataFrame(results) if results else pd.DataFrame()

    def _forecast_sku(self, sku: str, series: pd.Series, pattern: str,
                      policy: str = None) -> list:
        series = series.fillna(0).astype(float)
        trend_detected = seasonal_detected = False
        scores = {}

        if policy == MTS and series.notna().sum() >= _MIN_BACKTEST_POINTS:
            model_used, forecast_values, rmse, scores = self._select_by_backtest(series)
            selected_by = scores.pop("_metric")
        elif policy == MTO:
            model_used, forecast_values, rmse = self._croston(series)
            selected_by = "policy:MTO"
        elif pattern in ("intermittent", "erratic", "lumpy"):
            model_used, forecast_values, rmse = self._croston(series)
            selected_by = "pattern"
        else:
            trend_detected = self._detect_trend(series)
            seasonal_detected = self._detect_seasonality(series)
            model_used, forecast_values, rmse = self._fit_smooth(
                series, trend_detected, seasonal_detected)
            selected_by = "pattern"

        last_period = series.index[-1]
        future_periods = pd.period_range(start=last_period + 1, periods=self.horizon,
                                         freq=last_period.freq)

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
                "backtest_rmse": scores.get("rmse"),
                "vs_naive": scores.get("vs_naive"),
            }
            for i, (p, v) in enumerate(zip(future_periods, forecast_values))
        ]

    # ── Choosing by competition ──────────────────────────────────────────────

    def _candidates(self, series: pd.Series, horizon: int) -> dict:
        """
        Every model's forecast for the next `horizon` periods, by name.

        Each entry is built the same way it would be for the real forecast, so the
        backtest measures what the run will actually do rather than a simplified stand-in.
        """
        out = {}
        trend = self._detect_trend(series)
        seasonal = self._detect_seasonality(series)
        for name, build in (
            ("ETS", lambda: self._fit_ets(series, trend, seasonal, horizon)),
            ("ARIMA", lambda: self._fit_arima(series, horizon)),
            ("SMA", lambda: self._fit_sma(series, horizon)),
            ("Croston", lambda: self._croston(series, horizon)[1]),
            # The floor every other model has to clear: repeat the last observation.
            ("Naive", lambda: np.full(horizon, float(series.iloc[-1]))),
        ):
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

        pooled = {}
        for origin in origins:
            train, actual = series.iloc[:origin], series.iloc[origin:origin + steps].values
            if len(actual) == 0:
                continue
            for name, values in self._candidates(train, len(actual)).items():
                pooled.setdefault(name, {"err": [], "act": []})
                pooled[name]["err"].extend(values[:len(actual)] - actual)
                pooled[name]["act"].extend(actual)

        if not pooled:
            trend, seasonal = self._detect_trend(series), self._detect_seasonality(series)
            model, values, rmse = self._fit_smooth(series, trend, seasonal)
            return model, values, rmse, {"_metric": "in-sample (too short to backtest)"}

        scored = {}
        for name, acc in pooled.items():
            err = np.asarray(acc["err"], dtype=float)
            act = np.asarray(acc["act"], dtype=float)
            nz = act != 0
            scored[name] = {
                "rmse": float(np.sqrt(np.mean(err ** 2))),
                "mape": (float(np.mean(np.abs(err[nz] / act[nz]))) if nz.sum() >= _MIN_MAPE_POINTS
                         else None),
                "n_nonzero": int(nz.sum()),
            }

        metric = ("mape" if all(s["mape"] is not None for s in scored.values()) else "rmse")
        final = self._candidates(series, self.horizon)
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
            "rmse": scored[best]["rmse"],
            "vs_naive": (None if not naive else round(scored[best][metric] / naive, 3)),
        }

    # ── Smooth demand: ETS → ARIMA → SMA ─────────────────────────────────────

    def _fit_ets(self, series: pd.Series, trend: bool, seasonal: bool, horizon: int):
        n = len(series)
        if n < 12:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            seasonal_arg = "add" if seasonal and n >= 24 else None
            model = ExponentialSmoothing(
                series,
                trend="add" if trend else None,
                seasonal=seasonal_arg,
                seasonal_periods=12 if seasonal_arg else None,
                initialization_method="estimated",
            ).fit(optimized=True, disp=False)
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

    def _fit_smooth(self, series: pd.Series, trend: bool, seasonal: bool):
        """
        The original decision tree, kept for SKUs with no MTS policy and for series too
        short to hold anything out. First model that fits, in a fixed order — the point
        of the backtest above is that this order is not evidence about accuracy.
        """
        n = len(series)

        if n >= 12:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    trend_arg = "add" if trend else None
                    seasonal_arg = "add" if seasonal and n >= 24 else None
                    model = ExponentialSmoothing(
                        series,
                        trend=trend_arg,
                        seasonal=seasonal_arg,
                        seasonal_periods=12 if seasonal_arg else None,
                        initialization_method="estimated",
                    ).fit(optimized=True, disp=False)
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

        # Initialise on first non-zero observation
        first_nonzero = next((i for i, v in enumerate(values) if v > 0), None)
        if first_nonzero is None:
            avg = series.mean()
            return "Croston", np.full(horizon, avg), float(series.std())

        z_hat = values[first_nonzero]     # magnitude estimate
        n_hat = 1.0                        # interval estimate (periods between transactions)
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

    def _detect_seasonality(self, series: pd.Series) -> bool:
        if len(series) < 12:
            return False
        vals = series.values
        monthly = [vals[i::12].mean() for i in range(12) if len(vals[i::12]) > 0]
        return np.std(monthly) > 0.2 * np.mean(monthly) if np.mean(monthly) > 0 else False

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

        next_period = (
            forecast_df[forecast_df["is_next_period"]]
            [["sku", "forecast_qty", "forecast_rmse"]]
            .rename(columns={"forecast_qty": "forecast_next_period"})
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

        meta_cols = ["model_used", "selected_by", "backtest_mape", "backtest_rmse",
                     "vs_naive", "forecast_rmse"]
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

