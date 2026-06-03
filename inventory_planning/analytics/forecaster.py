"""
Time series forecaster.
Per-SKU: detects trend/seasonality, selects best model, forecasts 6 months ahead.
Models tried (in order): ETS (Holt-Winters) → ARIMA → Simple Moving Average fallback.
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

    def forecast_all(self, time_series: pd.DataFrame) -> pd.DataFrame:
        """
        time_series: pivot DataFrame (period index × SKU columns).
        Returns long-format DataFrame with columns:
          sku, period, forecast_qty, model_used, trend_detected, seasonal_detected
        """
        results = []
        for sku in time_series.columns:
            series = time_series[sku].copy()
            # Need at least 6 data points for any meaningful forecast
            if series.sum() == 0 or series.notna().sum() < 4:
                continue
            fc = self._forecast_sku(sku, series)
            results.extend(fc)
        return pd.DataFrame(results)

    def _forecast_sku(self, sku: str, series: pd.Series) -> list:
        series = series.fillna(0).astype(float)
        trend_detected = self._detect_trend(series)
        seasonal_detected = self._detect_seasonality(series)
        model_used, forecast_values = self._fit_and_forecast(series, trend_detected, seasonal_detected)

        # Build future period index
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
            }
            for p, v in zip(future_periods, forecast_values)
        ]

    def _detect_trend(self, series: pd.Series) -> bool:
        """ADF test: if non-stationary, likely has trend."""
        clean = series[series > 0]
        if len(clean) < 4:
            return False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = adfuller(clean, autolag="AIC")
            return result[1] > 0.05  # p > 0.05 → non-stationary → trend present
        except Exception:
            return False

    def _detect_seasonality(self, series: pd.Series) -> bool:
        """Simple seasonality check: enough data and non-zero variance in monthly pattern."""
        if len(series) < 12:
            return False
        # Check if std of monthly averages across years is > 20% of mean
        vals = series.values
        if len(vals) < 12:
            return False
        monthly = [vals[i::12].mean() for i in range(12) if len(vals[i::12]) > 0]
        return np.std(monthly) > 0.2 * np.mean(monthly) if np.mean(monthly) > 0 else False

    def _fit_and_forecast(self, series: pd.Series, trend: bool, seasonal: bool):
        n = len(series)

        # --- Try ETS (Holt-Winters) ---
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
                    fc = model.forecast(self.horizon)
                    return "ETS", fc.values
            except Exception:
                pass

        # --- Try ARIMA ---
        if n >= 8:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(series, order=(1, 1, 1), enforce_stationarity=False,
                                    enforce_invertibility=False).fit(disp=False)
                    fc = model.forecast(self.horizon)
                    return "ARIMA", fc.values
            except Exception:
                pass

        # --- Fallback: Simple Moving Average ---
        window = min(n, 3)
        avg = series.tail(window).mean()
        return "SMA", np.full(self.horizon, avg)

    def summary(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """6-month totals and model info per SKU."""
        return (
            forecast_df.groupby(["sku", "model_used", "trend_detected", "seasonal_detected"])
            .agg(forecast_6m_total=("forecast_qty", "sum"),
                 forecast_avg_monthly=("forecast_qty", "mean"))
            .reset_index()
            .round(1)
        )
