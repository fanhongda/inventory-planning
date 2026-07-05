# Inventory Planning Skill — Principles Audit

> Audited against: `principles/sc-principles.md` (MIT CTL MicroMasters SCM Key Concepts)  
> Date: 2026-07-04  
> Scope: analytics/* — demand_classifier, forecaster, safety_stock, inventory_projector, purchase_recommender

---

## Summary

| Severity | Count | Modules affected | Status |
|---|---|---|---|
| CRITICAL | 2 | forecaster, purchase_recommender | ✅ Fixed 2026-07-04 |
| MAJOR | 3 | demand_classifier, safety_stock, (all) | ✅ Fixed 2026-07-04 |
| MODERATE | 3 | inventory_projector, safety_stock, purchase_recommender | ✅ Fixed 2026-07-04 |
| MINOR | 1 | demand_classifier | ✅ Fixed 2026-07-04 |

---

## CRITICAL

### C1 — No Croston's method; ETS/ARIMA applied to intermittent demand

**File:** `analytics/forecaster.py`  
**Principle:** MIT §3 Croston's Method — "Applying traditional exponential smoothing to intermittent demand produces highly unstable estimates."

**What the code does:**  
All SKUs — including those with only 6 out of 12 active periods — are routed to ETS → ARIMA → SMA. There is no detection of intermittent demand and no Croston's method.

**Why it's wrong:**  
Items active 6–9 periods out of 12 have a ~50–75% fill rate. By MIT definition, these are intermittent. ETS and ARIMA treat zero-demand periods as real signal, producing erratic and biased forecasts. MAPE is also undefined when `At = 0`, so the model cannot be self-evaluated on these SKUs.

**Fix:**  
Add intermittent demand detection (e.g., `active_cycles / total_cycles < 0.75` AND `CV > 0.5`) and route those SKUs to Croston's method:
```
ẑₜ = α·xₜ + (1−α)·ẑₜ₋₁   (smooth magnitude on non-zero periods only)
n̂ₜ = β·nₜ + (1−β)·n̂ₜ₋₁   (smooth inter-arrival interval)
F̂ = ẑₜ / n̂ₜ
```

---

### C2 — Purchase quantity uses 6-month average forecast, not next-period point forecast

**File:** `analytics/purchase_recommender.py` line ~25  
**Principle:** MIT §1 Forecasting Truisms — "Shorter horizon forecasts are more accurate than longer ones." Forecast horizon should match the replenishment cycle.

**What the code does:**
```python
next_cycle = forecast_summary[["sku", "forecast_avg_monthly"]].copy()
# forecast_avg_monthly = mean of 6-month forward forecast
```

**Why it's wrong:**  
Using the 6-month average as the "next cycle demand" systematically underestimates in uptrending periods and overestimates in downtrending periods. The purchase decision covers the next review cycle (1 month), so it should use `forecast[t+1]` — the specific point forecast for the next period.

**Example:**  
If forecast months 1–6 = [100, 110, 120, 130, 140, 150], the avg = 125.  
But next month's demand = 100 → buying for 125 means chronic over-purchasing early in trend.

**Fix:**  
In `forecaster.py`, expose `forecast_next_period` (period t+1 value) separately from the 6-month summary. Use that in `purchase_recommender.py` as `next_cycle_demand`.

---

## MAJOR

### M1 — Demand classification uses frequency only; CV is ignored

**File:** `analytics/demand_classifier.py`  
**Principle:** MIT §1 Forecasting Truisms — "CV = σ/μ is the primary measure of demand volatility." MIT §9 ABC — manage with effort proportional to annual value (D·c) and demand type.

**What the code does:**  
Classification is solely by `active_cycles` (number of months with demand > 0):
- ≥ 9/12 → stocking-high
- ≥ 6/12 → stocking-med
- < 6/12 → non-stocking

**Why it's wrong:**  
An item that sells 1 unit every month (active = 12, CV ≈ 0) gets the same stocking-high label as an item that sells 1,000 units in some months and 1 in others (active = 12, CV = 2.0). These require completely different safety stock and forecasting treatment:
- Low-CV items: ETS/ARIMA works well, lower safety stock needed
- High-CV items: need more safety stock, consider Croston-like or SBA methods

**Fix:**  
Add CV as a secondary axis to classify demand pattern within each stocking tier:
```
CV = demand_std_rolling / demand_mean_rolling

If stocking-high AND CV > 1.0 → flag as "erratic-high" (needs Croston or higher SS)
If stocking-med AND CV > 0.5 → flag as "intermittent" (route to Croston)
```

---

### M2 — Safety stock uses demand std dev, not forecast RMSE

**File:** `analytics/safety_stock.py`  
**Principle:** MIT §7 Safety Stock — "SS = k · σDL where σDL = RMSE of forecast error over lead time."

**What the code does:**
```python
d_std = row["demand_std_rolling"]  # std dev of all historical demand periods
variance = lt * (d_std ** 2) + (d_mean ** 2) * (lt_std ** 2)
```

**Why it's wrong:**  
`demand_std_rolling` is the variability of actual demand, not the forecast error. If your forecast is accurate, the actual residual variance (RMSE²) is much smaller than total demand variance. Using raw demand std dev overestimates safety stock — the more accurate the forecast, the more waste this causes.

MIT principle: σDL should be the in-sample RMSE of the forecasting model, scaled to lead time.

**Fix:**  
After `forecaster.py` generates forecasts, compute in-sample residuals for each SKU and pass `forecast_rmse_monthly` to `safety_stock.py`. Use that in place of `demand_std_rolling`:
```
σDL = forecast_rmse_monthly · √(lt_months)   [for independent errors]
SS = k · σDL
```

---

### M3 — Service level metric (CSL vs IFR) is undocumented and may be the wrong choice

**File:** `config/stocking_policy.json`, `analytics/safety_stock.py`  
**Principle:** MIT §7 Service Level Metrics — "Once any one of {CSL, IFR, CSOE, CIS} is set, the other three are implicitly determined. IFR is always higher than CSL for the same k and Q."

**What the code does:**  
`z_score: 1.645` is set for service_level 0.95. This corresponds to CSL = 95% (Cycle Service Level = probability of no stockout per cycle). There is no documentation of which metric was intended.

**Why it matters:**  
- CSL 95% means: "in 95% of replenishment cycles, there will be no stockout"
- IFR 95% means: "95% of total demand is met from on-hand stock"
- For a DC servicing frequent orders, IFR is far more meaningful to operations
- IFR 95% typically requires a lower k than CSL 95% → less safety stock needed
- The correct formula for IFR requires the unit normal loss function G(k), not just NORMSINV

**Fix:**  
Document in `stocking_policy.json` which service metric `service_level` refers to. If switching to IFR, implement:
```
G(k) = Q · (1 − IFR) / σDL   → solve numerically for k
```

---

## MODERATE

### Mo1 — EXCESS inventory threshold is arbitrary (50% over ROP)

**File:** `analytics/inventory_projector.py` line ~35  
**Principle:** MIT §9 Excess Inventory — "DOS > threshold → action." No MIT principle supports a fixed 50% over-ROP trigger.

**What the code does:**
```python
if s > row["should_be_inventory"] * 0.5:  # >50% over target → EXCESS
```

**Why it's wrong:**  
ROP is the trigger to reorder, not the maximum. Having 50% more than ROP is not inherently excess — it depends on the review cycle and lot size. A proper excess trigger is Days of Supply (DOS):
```
DOS = effective_position / demand_mean_daily
```
Flag as EXCESS only when DOS exceeds policy threshold (e.g., 90 days). This should become a policy parameter.

**Fix:**  
Replace the 50% heuristic with a DOS-based threshold, configurable in `policy/policy.md`.

---

### Mo2 — Safety stock uses "shortest LT" supplier, not ordering supplier

**File:** `analytics/safety_stock.py` line ~20  
```python
best_lt = supplier_lt.sort_values("wma_lead_time_days").groupby("sku").first()
```

**Why it's wrong:**  
"Best" is defined as the shortest WMA lead time, but the supplier with the shortest LT may not be the one you're actually ordering from. Safety stock should be calibrated to the LT of the supplier in use, otherwise SS is under-estimated relative to actual exposure.

**Fix:**  
Cross-reference against `config/supplier_incoterm.json` (approved supplier per SKU). If an approved supplier is defined, use their LT — not the shortest.

---

### Mo3 — demand_mean_rolling excludes zero-demand periods

**File:** `analytics/demand_classifier.py` line ~39  
```python
mean_demand = ts[sku][ts[sku] > 0].mean()  # mean of non-zero periods only
```

**Why it matters:**  
This computes conditional mean E[demand | demand > 0], not the unconditional mean E[demand]. For safety stock and ROP, MIT formulas use the unconditional mean (μDL = total expected demand over lead time). Using conditional mean inflates μ for intermittent items, overstating ROP and leading to excess inventory.

**Note:** The conditional mean IS correct for Croston's method (magnitude component), but it's being used as a general-purpose mean for all downstream calculations.

**Fix:**  
Add `demand_mean_all` (unconditional, including zeros) alongside `demand_mean_rolling` (conditional). Use unconditional for ROP/SS; conditional for Croston's magnitude estimate.

---

## Priority Fix Order

| # | Issue | Impact | Effort |
|---|---|---|---|
| 1 | C2: Next-period forecast vs 6-month avg | Direct error in purchase qty | Low — expose forecast[t+1] from forecaster |
| 2 | M1: Add CV to demand classification | Gates correct routing to Croston | Low — add one computed column |
| 3 | C1: Add Croston's for intermittent demand | Forecast quality for ~30–50% of SKUs | Medium — new model branch |
| 4 | Mo1: DOS-based EXCESS threshold | Reduces false excess flags | Low — replace one formula |
| 5 | M3: Clarify CSL vs IFR, fix if needed | SS quantity correctness | Medium — policy decision first |
| 6 | M2: Use forecast RMSE for SS | SS calibration quality | Medium — requires RMSE pipeline |
| 7 | Mo3: Unconditional mean for ROP | ROP accuracy for intermittent items | Low — add one column |
| 8 | Mo2: Use ordering supplier LT | SS calibration correctness | Low — config lookup |
