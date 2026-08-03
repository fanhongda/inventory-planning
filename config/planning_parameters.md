# Planning Parameters

计划参数的唯一来源。分档边界、口径约定、分段覆盖规则都写在这里,**不写在代码里**。

改这个文件不需要改代码。每次运行会输出规则命中报告,说明哪些 SKU 被哪条规则改了什么。

---

## 口径约定 (Conventions)

这些决定 should-be inventory 怎么算。改动会影响所有输出数字,请谨慎。

```yaml
# cycle stock 口径
#   peak    = D × R        收货后的峰值水位。与"目标上限"管控口径一致。
#   average = D × R / 2    时间均值。与 DIOH 快照对比时用这个。
# 目前用 peak:与现行人工算法一致,便于对账。
cycle_stock_basis: peak

# 在途库存是否计入 should-be。
#   按 incoterm 逐 SKU 判定(config/incoterm_rules.json)。
#   EXW/FOB/CIF → 已发货部分是买方的,计入。
#   DDP/DAP     → 到货前不是买方的,不计入。
# 未发货部分(供应商生产中)一律不计入 —— 不在任何一方的成品库存上。
pipeline_basis: incoterm_aware

# 在途时长占总 LT 的比例。用于把 LT 拆成"供应商段"和"运输段",
# 只有运输段在买方账上(且仅当 incoterm 判定为买方所有)。
# 有实际发货日期时按实际算,没有时用这个比例估算。
transit_share_of_lt: 0.45

# 安全库存暴露期。周期性补货下是 R + LT,不是 LT。
#   review_plus_lt = R + LT   (正确,周期性补货)
#   lt_only        = LT       (仅连续盯库/min-max 适用)
safety_stock_exposure: review_plus_lt

# 一年的工作天数,用于 DIOH 换算
days_per_year: 365
```

---

## 默认参数 (Defaults)

规则未命中的 SKU 用这套值。

```yaml
review_period_days: 30          # 默认月度 review
replenishment_method: periodic  # periodic (R,S) | min_max (s,S) | reorder_point (s,Q)
service_level: 0.95
order_cost_usd: 350             # 每次下单的行政+跟单成本
holding_cost_rate: 0.22         # 年持有成本率(资金+仓储+保险+呆滞)
min_order_qty: 0
order_multiple: 1               # 起订倍数/箱规
```

---

## 分档边界 (Segmentation)

按年消耗金额分 ABC。边界不是绝对的,按业务实际调整。

```yaml
abc_thresholds:
  A: 0.80      # 累计金额前 80%
  B: 0.95      # 80% ~ 95%
  # 其余为 C

# 按需求波动分档(CV = σ/μ)
volatility_thresholds:
  stable: 0.5
  variable: 1.0
  # 超过 1.0 为 erratic
```

---

## 覆盖规则 (Rules)

每条规则:`scope` 选中一批 SKU,`set` 覆盖参数。**必须写 rationale** —— 三个月后你会忘记为什么写这条,而且它让系统能判断规则边界是否该扩展。

规则按顺序应用,后面的覆盖前面的。两条规则改同一个参数会在命中报告里标为冲突。

### R-001 · A 类物料周度 review

```yaml
scope: abc_class == "A"
set:
  review_period_days: 7
rationale: >
  A 类占用资金最多,cycle stock 对 review 周期最敏感。周度 review 把
  cycle stock 从 30 天降到 7 天,是账面库存最直接的杠杆。
owner: FHD
date: 2026-08-02
```

### R-002 · Actuator 类周度 review 且提高服务水平

```yaml
scope: product_family == "actuator"
set:
  review_period_days: 7
  service_level: 0.98
rationale: >
  涉及数据中心业务,客户停机成本远高于我方持有成本。缺货的代价不对称,
  因此单独提高服务水平,不受 ABC 分档约束。
owner: FHD
date: 2026-08-02
```

### R-003 · C 类低值物料走 min-max

```yaml
scope: abc_class == "C" and unit_cost < 20
set:
  replenishment_method: min_max
  review_period_days: 90
  service_level: 0.90
rationale: >
  低值物料的持有成本远低于管理成本,不值得频繁 review。min-max 由
  EOQ 定批量,让系统自己跑,人工只看异常。
owner: FHD
date: 2026-08-02
```

### R-004 · 长交期物料放宽 review 频率

```yaml
scope: lead_time_days > 90
set:
  review_period_days: 30
rationale: >
  LT 远大于 review 周期时,缩短 review 几乎不降安全库存(SS 按 √(R+LT) 走),
  只增加订货次数。这类物料的杠杆在 LT 稳定性上,不在 review 频率上。
  本条会覆盖 R-001,这是有意的。
owner: FHD
date: 2026-08-02
```

---

## 怎么加一条规则

复制上面任意一条,改 `scope` / `set` / `rationale`。

`scope` 可以引用任何 SKU 属性列:`sku`、`abc_class`、`unit_cost`、`lead_time_days`、
`lt_std_days`、`demand_cv`、`demand_mean`、`annual_value`、`supplier`、`incoterm`、
`product_family`、`stocking_class`、`demand_pattern`、`sopc_classification`。

支持 `and` / `or` / `not`、比较运算、`in [...]`、以及 `startswith(col, "X")`。

不确定 scope 选中了哪些 SKU,先跑一次看命中报告——它会列出每条规则命中的 SKU 数和样例。
