"""
Time series forecaster.

Routing logic (per MIT CTL principles):
  smooth      → ETS (Holt-Winters) → ARIMA → SMA fallback
  intermittent/erratic/lumpy → Croston's method
  non-stocking → skipped

Key outputs:
  forecast_next_period  — point forecast for t+1 (used by purchase_recommender)
  forecast_avg_monthly  — mean of 6-month window (for planning horizon view only)
  forecast_rmse         — in-sample RMSE (used by safety_stock for σDL)
"""

import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller


class Forecaster:

    def __init__(self, horizon: int = 6):
        self.horizon = horizon

    def forecast_all(self, time_series: pd.DataFrame,
                     classified: pd.DataFrame = None) -> pd.DataFrame:
        """
        time_series: pivot DataFrame (period index × SKU columns).
        classified: output of DemandClassifier.classify() — provides demand_pattern per SKU.

        Returns long-format DataFrame:
          sku, period, forecast_qty, model_used, trend_detected, seasonal_detected,
          is_next_period (True for t+1 only), forecast_rmse
        """
        pattern_map = {}
        if classified is not None and "demand_pattern" in classified.columns:
            pattern_map = classified.set_index("sku")["demand_pattern"].to_dict()

        results = []
        for sku in time_series.columns:
            series = time_series[sku].copy()
            if series.sum() == 0 or series.notna().sum() < 4:
                continue
            pattern = pattern_map.get(sku, "smooth")
            fc = self._forecast_sku(sku, series, pattern)
            results.extend(fc)

        return pd.DataFrame(results) if results else pd.DataFrame()

    def _forecast_sku(self, sku: str, series: pd.Series, pattern: str) -> list:
        series = series.fillna(0).astype(float)

        if pattern in ("intermittent", "erratic", "lumpy"):
            model_used, forecast_values, rmse = self._croston(series)
            trend_detected = False
            seasonal_detected = False
        else:
            trend_detected = self._detect_trend(series)
            seasonal_detected = self._detect_seasonality(series)
            model_used, forecast_values, rmse = self._fit_smooth(series, trend_detected, seasonal_detected)

        last_period = series.index[-1]
        future_periods = pd.period_range(start=last_period + 1, periods=self.horizon, freq=last_period.freq)

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
            }
            for i, (p, v) in enumerate(zip(future_periods, forecast_values))
        ]

    # ── Smooth demand: ETS → ARIMA → SMA ─────────────────────────────────────

    def _fit_smooth(self, series: pd.Series, trend: bool, seasonal: bool):
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

    def _croston(self, series: pd.Series):
        """
        Croston's method (MIT CTL §3):
          Separate demand magnitude (ẑ) from inter-arrival interval (n̂).
          Forecast = ẑ / n̂ per period.
        """
        alpha = 0.2  # magnitude smoothing
        beta = 0.2   # interval smoothing

        values = series.values.astype(float)
        n = len(values)

        # Initialise on first non-zero observation
        first_nonzero = next((i for i, v in enumerate(values) if v > 0), None)
        if first_nonzero is None:
            avg = series.mean()
            return "Croston", np.full(self.horizon, avg), float(series.std())

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
        fc = np.full(self.horizon, max(0.0, per_period_forecast))
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
