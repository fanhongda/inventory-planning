# Supply Chain Principles Reference
> Source: MIT CTL MicroMasters SCM Key Concepts — CTL.SC1x, Pages 59–106

---

## 1. Demand Characterization & Forecasting

### Demand Planning Hierarchy

**Core Principle:** Forecasting is one of three sub-processes of Demand Planning, Forecasting & Management; the time frame of the action dictates the time horizon of the forecast.

**When to apply:** When scoping any forecasting effort — always match method and horizon to the decision.

**Key assumptions:** Strategic (years) → capacity/investment; Tactical (weeks–quarters) → inventory/labor/budget; Operational (hours–days) → replenishment/production/transport.

**Common mistakes:** Using a strategic-horizon model for operational replenishment decisions, or vice versa.

**Inventory planning implication:** Operational forecasts directly drive replenishment quantities and safety stock settings.

---

### Forecasting Truisms

**Core Principle:** All point forecasts are wrong; aggregate forecasts are more accurate than disaggregated ones; shorter horizon forecasts are more accurate than longer ones.

**When to apply:** Any time a forecast is produced or evaluated.

**Formula(s):**
```
CV = σ / μ        (Coefficient of Variation — measure of demand volatility)
```
- σ = standard deviation of demand; μ = mean demand; CV is dimensionless.  
- Lower CV → demand is more predictable; CV > 0.5 signals high variability.

**Key assumptions:** Aggregation pooling effect assumes independent demand streams with uncorrelated peaks.

**Common mistakes:**
- Relying solely on a point forecast rather than a range/interval.
- Not tracking forecast errors over time (missing drift/bias signals).
- Disaggregating SKUs unnecessarily, losing the pooling benefit.

**Inventory planning implication:** Higher CV → more safety stock required. Aggregating by SKU family, region, or time bucket reduces required safety stock.

---

### Forecasting Methods Taxonomy

**Core Principle:** Methods are either Subjective (Judgmental / Experimental) or Objective (Causal / Time Series); objective time-series methods dominate for inventory planning.

**When to apply:** Select based on data availability, product lifecycle stage, and decision type.

| Method | Use when |
|---|---|
| Time Series (exponential smoothing, MA) | Long demand history, repeating patterns |
| Causal (leading indicators) | Identified external driver (e.g., housing starts → lumber) |
| Judgmental (Delphi, sales survey) | New product, no history |
| Experimental (test markets) | Very new product, early lifecycle |

---

## 2. Forecast Error Metrics

### Forecast Error Notation

```
At  = Actual value for period t
Ft  = Forecasted value for period t
et  = Forecast error for period t = At − Ft
n   = number of observations
```

---

### Bias Metrics

**Core Principle:** Bias measures persistent tendency to over- or under-predict; non-zero MD or MPE signals a systematic flaw in the model.

**Formula(s):**
```
MD  (Mean Deviation)         = Σ et / n
MPE (Mean Percent Error)     = Σ (et / At) / n
```
- Units: MD in demand units; MPE is dimensionless (%).
- "Good": MD ≈ 0, MPE ≈ 0%.

**Key assumptions:** These metrics cancel positive and negative errors; must be used alongside accuracy metrics.

**Common mistakes:** Using MD/MPE alone — a model with alternating +100 / −100 errors shows MD = 0 but is terrible.

**Inventory planning implication:** Persistent positive bias → chronic over-stock; persistent negative bias → chronic under-stock and service failures.

---

### Accuracy Metrics

**Core Principle:** Accuracy metrics use absolute or squared errors to prevent cancellation; they must be paired with bias metrics for a complete picture.

**Formula(s):**
```
MAD   (Mean Absolute Deviation)    = Σ |et| / n
MSE   (Mean Squared Error)         = Σ et² / n
RMSE  (Root Mean Squared Error)    = √(Σ et² / n)
MAPE  (Mean Absolute Percent Error)= Σ |et / At| / n
```
- MAD in demand units; MSE in units²; RMSE in demand units; MAPE dimensionless (%).
- RMSE penalises large errors more heavily than MAD.
- MAPE is scale-free; use when comparing across SKUs of different volumes.
- "Good": context-dependent, but MAPE < 20–30% is considered reasonable for tactical planning.

**Key assumptions:** MAPE is undefined when At = 0 (cannot use for intermittent demand).

**Common mistakes:**
- Using MSE as a direct input to safety stock formulas (units are squared — use RMSE).
- Ignoring MAPE when comparing across product families with very different mean demand.

**Inventory planning implication:** Safety stock = k × σDL, where σDL ≈ RMSE of forecast over lead time. The better the forecast accuracy (lower RMSE), the less safety stock is required.

---

### Statistical Aggregation

**Core Principle:** Aggregating n independent demand streams reduces relative variability (CV) by √n.

**Formula(s):**
```
For n streams each with mean μ_i and std dev σ_i:

σ_agg = √(σ₁² + σ₂² + ... + σₙ²)
μ_agg = μ₁ + μ₂ + ... + μₙ

For n streams of equal mean μ and std dev σ:
σ_agg = σ√n
μ_agg = nμ
CV_agg = CV / √n
```

**Key assumptions:** Demands are statistically independent across streams.

**Common mistakes:** Applying the √n reduction when demands are positively correlated (e.g., promotion-driven spikes hit all locations simultaneously).

**Inventory planning implication:** Centralising inventory from n locations to 1 reduces required safety stock by up to √n — the "square root law" of pooling.

---

## 3. Time Series Forecasting Models

### Stationary Demand Models (Level Only)

**Core Principle:** Cumulative, Naïve, and M-Period Moving Average models are all special cases trading off stability vs. responsiveness; all assume stationary (no trend, no seasonality) demand.

**Formula(s):**
```
Cumulative Model:        F̂(t,t+1) = Σ xᵢ / t    (M = t; all history)
Naïve Model:             F̂(t,t+1) = xₜ           (M = 1; last point only)
M-Period Moving Average: F̂(t,t+1) = Σ xᵢ / M    (sum over last M periods)
```
- xₜ = actual demand in period t
- F̂(t,t+1) = forecast for t+1 made at time t
- M = number of periods included

**Key assumptions:** Stationary demand (constant mean). Any trend causes severe lag.

**Common mistakes:** Using MA when demand has a trend — the forecast will persistently lag behind.

**Inventory planning implication:** The choice of M controls responsiveness; small M reacts quickly to demand shifts but is noisy.

---

### Simple Exponential Smoothing (Level Only)

**Core Principle:** Newer observations receive exponentially higher weight than older ones; α controls the balance between responsiveness and stability.

**Formula(s):**
```
F̂(t,t+1) = α·xₜ + (1−α)·F̂(t−1,t)

α: smoothing factor, 0 ≤ α ≤ 1 (in practice: 0 ≤ α ≤ 0.3)
```
- As α → 1: forecast is more volatile/naïve
- As α → 0: forecast is more stable/cumulative

**Key assumptions:** Stationary demand; no trend or seasonality.

**Common mistakes:** Setting α too high (> 0.3) making forecast erratic; using SES when trend is present (creates persistent bias).

**Inventory planning implication:** α must be tuned; use MSE tracking (see below) to adaptively stabilise the error estimate.

---

### MSE Estimate via Exponential Smoothing

**Formula(s):**
```
MSEₜ = ω·(xₜ − F̂(t−1,t))² + (1−ω)·MSEₜ₋₁

ω: MSE smoothing factor, 0.01 ≤ ω ≤ 0.1
```

---

### Holt's Method (Level + Linear Trend)

**Core Principle:** Two smoothing equations track the level and trend separately; α smooths level, β smooths trend.

**Formula(s):**
```
Forecast h periods ahead:
  F̂(t, t+τ) = âₜ + τ·b̂ₜ

Level update:
  âₜ = α·xₜ + (1−α)·(âₜ₋₁ + b̂ₜ₋₁)

Trend update:
  b̂ₜ = β·(âₜ − âₜ₋₁) + (1−β)·b̂ₜ₋₁

α: level smoothing factor, 0 ≤ α ≤ 1
β: trend smoothing factor, 0 ≤ β ≤ 1
```

**Key assumptions:** Linear trend; no seasonality.

**Common mistakes:** Using Holt's when trend is non-linear or seasonal — use damped trend or Holt-Winters instead.

---

### Damped Trend Model

**Formula(s):**
```
F̂(t, t+τ) = âₜ + (φ + φ² + ... + φᵗ)·b̂ₜ

âₜ = α·xₜ + (1−α)·(âₜ₋₁ + φ·b̂ₜ₋₁)
b̂ₜ = β·(âₜ − âₜ₋₁) + (1−β)·φ·b̂ₜ₋₁

φ: dampening factor, 0 ≤ φ ≤ 1
```
- As φ → 1: standard Holt's model (trend persists indefinitely)
- As φ → 0: trend disappears; degenerates to SES

---

### Double Exponential Smoothing (Level + Seasonality)

**Core Principle:** Multiplicative seasonality — each period's forecast is the level estimate multiplied by that period's seasonal index; seasonality indices must be re-normalised regularly.

**Formula(s):**
```
Forecast:
  F̂(t, t+τ) = âₜ · F̂(t+τ−P)

Level update:
  âₜ = α·(xₜ / F̂ₜ₋P) + (1−α)·âₜ₋₁

Seasonality update:
  F̂ₜ = γ·(xₜ / âₜ) + (1−γ)·F̂ₜ₋P

P: number of periods within one full season (e.g., 12 for monthly, 4 for quarterly)
Constraint: Σ Fᵢ over all periods within a season = P

Normalisation:
  F̂ₜ_norm = F̂ₜ_raw · P / (Σ F̂ᵢ_raw for all i within one season)
```

**Key assumptions:** Multiplicative seasonal effect; no trend.

**Common mistakes:** Forgetting to normalise seasonality indices after each update — they will drift significantly.

---

### Holt-Winters Model (Level + Trend + Seasonality)

**Formula(s):**
```
Forecast:
  F̂(t, t+τ) = (âₜ + τ·b̂ₜ) · F̂(t+τ−P)

Level:
  âₜ = α·(xₜ / F̂ₜ₋P) + (1−α)·(âₜ₋₁ + b̂ₜ₋₁)

Trend:
  b̂ₜ = β·(âₜ − âₜ₋₁) + (1−β)·b̂ₜ₋₁

Seasonality:
  F̂ₜ = γ·(xₜ / âₜ) + (1−γ)·F̂ₜ₋P
```

**Key assumptions:** Linear trend with multiplicative seasonality; requires ≥ 2 seasons of historical data for initialisation (≥ 4 preferred).

---

### Croston's Method (Intermittent / Sparse Demand)

**Core Principle:** Separate the demand magnitude from the inter-arrival frequency using two independent smoothing equations; the forecast is the ratio of estimated magnitude to estimated interval.

**When to apply:** Spare parts, slow-moving items; periods with many zero-demand observations.

**Formula(s):**
```
Demand process: xₜ = yₜ · zₜ
  yₜ = 1 if transaction in period t, 0 otherwise
  zₜ = transaction size in period t
  nₜ = periods since last transaction

If xₜ = 0 (no transaction):
  ẑₜ = ẑₜ₋₁     (no update)
  n̂ₜ = n̂ₜ₋₁    (no update)

If xₜ > 0 (transaction occurs):
  ẑₜ = α·xₜ + (1−α)·ẑₜ₋₁    (smooth magnitude)
  n̂ₜ = β·nₜ + (1−β)·n̂ₜ₋₁   (smooth interval)

Forecast:
  F̂(t,t+1) = ẑₜ / n̂ₜ
```
- α: smoothing factor for magnitude
- β: smoothing factor for transaction frequency

**Key assumptions:** Demand process is independent across periods; probability of transaction = 1/n̂.

**Common mistakes:** Applying traditional exponential smoothing to intermittent demand — produces highly unstable estimates. MAPE is undefined for zero-demand periods.

---

### Bass Diffusion Model (New Product Forecasting)

**Core Principle:** New product adoption is driven by two forces — Innovation (intrinsic adopters) and Imitation (word-of-mouth) — and follows a predictable S-curve lifecycle.

**Formula(s):**
```
n(t) = [p + q·N(t−1)/m] · [m − N(t−1)]

p = coefficient of innovation (typical range: 0.01–0.03)
q = coefficient of imitation (typical range: 0.3–0.5)
m = total potential adopter population
n(t) = new adopters in period t
N(t−1) = cumulative adopters by period t−1
```

**Key assumptions:** Fixed market potential m; two-parameter model; no repeat purchases.

**Common mistakes:** Using Bass for replenishment forecasting — it models adoption only, not ongoing demand.

---

## 4. Inventory Management — Cost Structure

### Total Cost (TC) Framework

**Core Principle:** The Total Relevant Cost of any replenishment policy is the sum of ordering and holding costs; purchasing and shortage costs are only relevant when they affect the order quantity decision.

**Formula(s):**
```
TC(Q) = Purchase + Order + Holding + Shortage

TRC(Q) = ct·(D/Q) + ce·(Q/2)        [EOQ case: deterministic demand, no stockouts]

where:
  c   = purchase cost ($/unit)
  ct  = ordering cost ($/order)
  h   = holding rate ($/$ inventory/time)
  ce  = excess holding cost = c·h  ($/unit/time)
  cs  = shortage cost ($/unit)
  D   = demand (units/time)
  Q   = order quantity (units/order)
```

**Key assumptions:** A cost is "relevant" only if (a) it varies with the decision and (b) management can control it. Purchase cost is irrelevant to EOQ because it is fixed per unit regardless of order size.

**Common mistakes:** Including purchase cost in TRC when there are no quantity discounts (it cancels out and doesn't affect Q*).

---

## 5. Lot Sizing — Economic Order Quantity (EOQ)

### EOQ Core Model

**Core Principle:** EOQ minimises the sum of ordering and holding costs under deterministic, uniform demand; the optimal order quantity Q* balances the cost of ordering too frequently against the cost of holding too much.

**When to apply:** Deterministic, stable demand with known fixed ordering cost and proportional holding cost.

**Formula(s):**
```
Q* = √(2·ct·D / ce)        [Optimal Order Quantity]

T* = √(2·ct / (ce·D))      [Optimal Time between Replenishments; ensure units make sense]

N* = 1/T* = D/Q*            [Orders per time period]

TRC(Q*) = √(2·ct·ce·D)     [Optimal Total Relevant Cost]

TC(Q*)  = c·D + √(2·ct·ce·D)  [Optimal Total Cost incl. purchase]
```
- ce = c·h ($/unit/time)
- D in units/year if ct is in $/order and you want Q* in units

**Inventory policy:** "Order Q* units every T* time periods" OR "Order Q* when IP = 0" (with zero lead time), or "Order Q* when IP = D·L" (with lead time L > 0).

**Key assumptions:**
1. Demand is uniform and deterministic
2. Lead time is known and constant (does not affect Q*, only reorder point)
3. No quantity discounts
4. Full order received at once

**Common mistakes:**
- Mixing time units (annual demand with monthly holding cost).
- Confusing IP (Inventory Position) with IOH (Inventory on Hand) when lead time > 0.
- Forgetting that lead time L shifts the reorder point but does NOT change Q*.

**Inventory planning implication:** Lead time adds pipeline inventory (API = D·L) and shifts the reorder point but leaves Q* unchanged. A 50% error in Q only raises TRC by ~8% — EOQ is very robust.

---

### EOQ Sensitivity Analysis

**Core Principle:** EOQ is robust — moderate errors in Q, D, or T cause disproportionately small cost increases.

**Formula(s):**
```
TRC(Q) / TRC(Q*) = (1/2)·(Q*/Q + Q/Q*)         [sensitivity to order quantity]

TRC(Q*_F) / TRC(Q*_A) = (1/2)·√(DA/DF) + √(DF/DA)  [sensitivity to demand forecast error;
                                                        DA = actual D, DF = forecasted D]

TRC(T) / TRC(T*) = (1/2)·(T/T* + T*/T)          [sensitivity to time interval]
```

**Key insight:** Power of Two Policy — ordering in intervals of 2^k time periods guarantees TRC within 6% of optimal. When Q is wrong, it is always better to order slightly more than slightly less.

---

### EOQ with Lead Time

**Formula(s):**
```
Reorder Point (IP trigger): s = D·L     [deterministic case]

Average Pipeline Inventory: API = D·L

TC(Q) with lead time = c·D + ct·(D/Q) + ce·(Q/2) + c·D·L  [added pipeline cost]

Q* remains: Q* = √(2·ct·D / ce)    [lead time does NOT change Q*]
```

---

### Economic Production Quantity (EPQ / Finite Replenishment)

**Core Principle:** When inventory arrives at a finite production rate P rather than all at once, average on-hand inventory is lower, so the optimal batch size is larger than EOQ.

**Formula(s):**
```
TRC(Q) = ct·(D/Q) + (Q/2)·(1 − D/P)·ce

EPQ = Q* = √(2·ct·D / (ce·(1 − D/P)))   =   EOQ / √(1 − D/P)
```
- P = production rate (units/time); must have P > D, otherwise demand can never be satisfied
- As P → ∞: EPQ → EOQ

**Key assumptions:** Constant production rate; no stockouts; D < P.

---

### Volume Discounts

#### All-Units Discount

```
c = c₀  if 0 ≤ Q < Q₁
c = c₁  if Q ≥ Q₁   (c₁ < c₀)

Procedure:
1. Compute Q*_c0 and Q*_c1 (EOQ at each price)
2. If Q*_c1 ≥ Q₁ → order Q*_c1
3. Else: compare TRC(Q*_c0) at price c₀  vs  TC(Q₁) at price c₁; pick lower
```

#### Incremental Discount

```
Fixed cost per tier: F₀ = 0;  Fᵢ = Fᵢ₋₁ + (cᵢ₋₁ − cᵢ)·Qᵢ

Q*ᵢ = √(2·D·(ct + Fᵢ) / (h·cᵢ))

Select tier i where Q*ᵢ is in range [Qᵢ₋₁, Qᵢ]; compare TRC across valid tiers.
```

#### One-Time Discount

```
Q*_g = Q* · √(h·c / (h·c_g)) + D·(c − c_g) / (h·c_g)

where c_g = discounted price (one-time offer)

If Q*_g < Q*: check your arithmetic — result cannot be below regular EOQ.
```

---

## 6. Single-Period (Newsvendor) Model

### Critical Ratio and Optimal Order Quantity

**Core Principle:** In a single-period setting, order until the marginal probability of having too much equals the marginal probability of having too little, as expressed by the Critical Ratio.

**When to apply:** One-shot ordering decision; unsold inventory has zero or salvage value; unmet demand is lost (fashion, perishables, event tickets).

**Formula(s):**
```
P[x ≤ Q*] = CR = cs / (cs + ce)

where:
  cs = shortage cost ($/unit)   = p − c + B    [lost margin + penalty]
  ce = excess cost   ($/unit)   = c − g         [purchase cost minus salvage]
  p  = selling price ($/unit)
  c  = purchase cost ($/unit)
  g  = salvage value ($/unit)
  B  = additional shortage penalty ($/unit)

Simplest case (no salvage, no penalty):
  cs = p − c  (lost margin)
  ce = c      (stranded cost)
  CR = (p − c) / p   = gross margin / price
```
- CR ranges 0–1; higher CR → order more → stock out less.

**Key assumptions:** Single period, stochastic demand with known distribution, no replenishment mid-period, no carry-over of inventory.

**Common mistakes:**
- Using cs = p (full price) instead of cs = p − c (lost margin only).
- Forgetting salvage value reduces ce.

**Inventory planning implication:** High CR (luxury goods, high margin) → high service level target. Low CR (perishables with low margin) → accept more stockouts.

---

### Expected Profitability

**Formula(s):**
```
E[Profit(Q)] = p·E[x] − c·Q − p·E[US]

With salvage g and penalty B:
E[Profit(Q)] = p·(E[x] − E[US]) − c·Q + g·(Q − (E[x] − E[US])) − B·E[US]

E[Units Short] — Normal distribution:
  E[US] = σ · G(k)   where k = (Q − μ) / σ

G(k) = Unit Normal Loss Function:
  G(k) = NORMDIST(k,0,1,0) − k·(1 − NORMSDIST(k))   [in spreadsheet]
```
- E[x] = expected demand = μ (for symmetric distribution)
- σ = standard deviation of demand

---

## 7. Probabilistic Inventory Models (Continuous & Periodic Review)

### Safety Stock

**Core Principle:** Safety stock = k · σDL, where σDL is the RMSE of the forecast over the lead time; the safety factor k is set by the chosen service metric.

**Formula(s):**
```
Safety Stock  = k · σDL
Reorder Point: s = μDL + k · σDL

where:
  μDL  = expected demand over lead time L
  σDL  = std dev (RMSE) of demand forecast error over lead time L
  k    = safety factor (set by service level method below)
```

**Key assumptions:** σDL captures forecast uncertainty. Most companies default to using historical demand std dev as a proxy for RMSE (assumes forecast = mean).

**Common mistakes:** Using the std dev of a single period and forgetting to scale to lead time (see time conversion formulas below).

---

### Time Conversion for σ and μ

```
Long period (L months) ↔ Short period (1 month, n short periods within long):

μ_L = n · μ_S
σ_L = √n · σ_S

Converting long to short:
  μ_S = μ_L / n
  σ_S = σ_L / √n
```

---

### Base Stock Policy

**Core Principle:** One-for-one replenishment — order exactly what was consumed, when it was consumed; the base stock level S* covers expected lead time demand plus safety stock.

**Formula(s):**
```
S* = μDL + k · σDL

Level of Service (LOS) = P[x ≤ S*] = CR = cs / (cs + ce)
```

**Inventory policy:** "Order what was demanded, in the quantity demanded, immediately."

---

### Continuous Review Policy (s, Q)

**Core Principle:** Order Q* (via EOQ) whenever Inventory Position drops to or below the reorder point s; s covers expected lead time demand plus safety stock.

**Formula(s):**
```
Reorder Point:  s = μDL + k · σDL

Order Quantity: Q = Q* (from EOQ)

Inventory Policy: "Order Q* units when IP ≤ s"
```

---

### Service Level Metrics and Safety Factor k

#### Cycle Service Level (CSL)

**Core Principle:** CSL = probability of no stockout within a replenishment cycle.

```
CSL = P[x ≤ s] = 1 − P[x > s]

k = NORMSINV(CSL)       [in spreadsheet]
```
- CSL = 95% → k ≈ 1.65; CSL = 99% → k ≈ 2.33
- At high CSL, diminishing returns: each additional % of service requires disproportionately more safety stock.

#### Item Fill Rate (IFR)

**Core Principle:** IFR = fraction of demand met from on-hand inventory (cycle stock); always higher than CSL for the same safety stock.

```
IFR = 1 − E[US] / Q = 1 − σDL · G(k) / Q

Solving for k given target IFR:
  G(k) = Q · (1 − IFR) / σDL

G(k) = NORMDIST(k,0,1,0) − k·(1−NORMSDIST(k))    [spreadsheet]
```

#### Cost per Stockout Event (CSOE / B1 Cost)

**Core Principle:** Penalise each stockout cycle event by B1; minimise total cost.

```
If B1 · D / (ce · Q · √(2π)) > 1:
  k = √(2 · ln(B1 · D / (ce · Q · σDL · √(2π))))

Else: set k as low as management allows.
```

#### Cost per Item Short (CIS)

**Core Principle:** Penalise each unit short by cs; minimise total cost.

```
If ce · Q / (cs · D) ≤ 1:
  k = NORMSINV(1 − ce · Q / (cs · D))     [spreadsheet]

Else: set k as low as management allows.
```

**Key insight — Metric interdependence:** Once any one of {CSL, IFR, CSOE, CIS} is explicitly set, the other three are implicitly determined. IFR is always higher than CSL for the same k and Q.

**Summary table:**

| Metric | k formula |
|---|---|
| CSL (% service) | k = NORMSINV(CSL) |
| IFR (% service) | Solve G(k) = Q(1−IFR)/σDL |
| CSOE ($ cost) | k = √(2·ln(B1·D / (ce·Q·σDL·√(2π)))) |
| CIS ($ cost) | k = NORMSINV(1 − ce·Q/(cs·D)) |

---

### Periodic Review Policy (R, S)

**Core Principle:** Order up to S* every R periods; exposure window extends to L+R, requiring more safety stock than the continuous policy.

**Formula(s):**
```
Order Up To Point: S* = μ(DL+R) + k · σ(DL+R)

Order Quantity at each review = S* − IP  (variable each cycle)
```

**Equivalence to (s, Q):**

| (s, Q) parameter | Maps to (R, S) as |
|---|---|
| s | S |
| Q | D · R |
| L | R + L |

**Cost of R and L:**
```
Average inventory cost = ce · [D·R/2 + k·σ(DL+R) + D·L]

- Increasing L: increases safety stock non-linearly, increases pipeline linearly
- Increasing R: increases safety stock non-linearly, increases cycle stock linearly
```

**Common mistakes:** Forgetting to use σ(DL+R) instead of σDL for periodic review — understates required safety stock.

---

## 8. Multi-Item & Multi-Location Inventory

### Grouping Like Items — Break Points

**Core Principle:** Items with similar annual value (D·c) should share the same replenishment frequency; the break-even value that separates ordering every w₁ weeks from every w₂ weeks is:

**Formula(s):**
```
Dᵢcᵢ ≥ 5408·ct / (h·w₁·w₂)  → order every w₁ weeks
Else if Dᵢcᵢ ≥ 5408·ct / (h·w₂·w₃)  → order every w₂ weeks
... and so on.
```
- w₀ = 1 week (base period)
- Assumes common ct and h across items

---

### Power of Two Policy

**Core Principle:** Restrict replenishment intervals to powers of 2 (1, 2, 4, 8, 16, ... weeks); guarantees TRC within 6% of optimal for any item.

**Formula(s):**
```
T* = √(2·ct / (ce·D))

T_practical = 2^(ROUNDUP(LN(T*/√2) / LN(2)))   [spreadsheet]
```

---

### Exchange Curves — Cycle Stock

**Core Principle:** A single ct/h ratio governs the system-wide trade-off between total annual orders (N) and total annual cycle stock cost (TACS); management can dial ct/h to meet a budget constraint.

**Formula(s):**
```
TACS = Σᵢ √(Dᵢcᵢ / 2) · √(ct/h)   =   √(ct/h) · Σᵢ √(Dᵢcᵢ/2)

N    = Σᵢ Dᵢ/Qᵢ*   =   (1/√(ct/h)) · Σᵢ √(Dᵢcᵢ/2)
```
- Chart N vs TACS across a range of ct/h values to find the Pareto frontier.

---

### Exchange Curves — Safety Stock

**Formula(s):**
```
TSS   = Σᵢ kᵢ · σDLᵢ · cᵢ

TVIS  = Σᵢ cᵢ · σDLᵢ · G(kᵢ)

Process: vary the target service metric; for each value, compute kᵢ for all SKUs,
         then sum safety stock costs and chart TSS vs. aggregate service level.
```

---

### Inventory Pooling (Square Root Law)

**Core Principle:** Consolidating n independent locations into one reduces the required safety stock by √n.

**Formula(s):**
```
σ_pooled = √(σ₁² + σ₂² + ... + σₙ²)

For n equal-variance locations: σ_pooled = σ · √n

Safety stock saving: k · σ · √n  vs.  k · σ · n  (n separate)
Reduction factor: 1/√n   (save (1 − 1/√n) of total safety stock)
```

**Key assumptions:** Demand across locations is independent; uniform demand; EOQ ordering.

**Common mistakes:** Assuming pooling is always beneficial — it increases transit/transport costs and lengthens lead time for remote demand points.

**Inventory planning implication:** Each time you double the number of stocking locations, safety stock increases by ~41%. Centralisation into a hub reduces safety stock but may increase pipeline and service time.

---

## 9. ABC Segmentation & Item Classification

### ABC Inventory Management Principles

**Core Principle:** Manage inventory with effort proportional to its annual value (D·c); A items (high value/fast-moving) require active management while C items (low value) should be managed passively.

| | A Items | B Items | C Items |
|---|---|---|---|
| Records | Extensive, transactional | Moderate | Rule-based, aggregate |
| Demand interaction | Direct input, high integrity | Modified forecast | Simple or none |
| Supply management | Active | By exception | Non |
| Policy review | Monthly or more | Annually | Very infrequent |
| Shortage strategy | Actively manage | Set SL, manage by exception | Set & forget |
| Demand distribution | Consider non-Normal | Normal | N/A |

**Recommended policies:**

| Item | Continuous Review | Periodic Review |
|---|---|---|
| A | (s, S) | (R, s, S) |
| B | (s, Q) | (R, S) |
| C | — | Manual ~(R, S) |

### Fast vs. Slow Moving A Items

```
Fast movers (μDL or μDL+R ≥ 10 units): use Normal or Lognormal distribution

Slow movers (μDL or μDL+R < 10 units): use Poisson distribution
  P[X = x] = e^(−λ)·λˣ / x!    where λ = mean = variance

Discrete loss function (Cachon & Terwiesch):
  L[Xᵢ] = L[Xᵢ₋₁] − (Xᵢ − Xᵢ₋₁)·(1 − F[Xᵢ₋₁])
```

### Excess Inventory Disposal

```
Days of Supply (DOS) = IOH / D

Action trigger: DOS > threshold (e.g., 2 years)
Actions: convert to alternate use, transfer to another location, mark down, auction.
```

---

## 10. Inventory Position (IP) — Master Definition

```
IP = IOH + IOO − BO − CO

where:
  IOH = Inventory on Hand (physical stock)
  IOO = Inventory on Order (in transit / pipeline)
  BO  = Backorders (demand committed but not yet fulfilled)
  CO  = Committed Orders (reserved for specific customers)
```
- All replenishment decisions should use IP, not IOH alone.
- Average Pipeline Inventory = D · L (Little's Law applied to inventory)

---

*Last updated: 2026-07-04 | Source: MIT CTL MicroMasters SCM Key Concepts pp. 59–106*
