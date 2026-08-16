# 企业供应链知识图谱 — 设计研究

> 状态：设计提案，未实现。
> 前置：`inventory_planning/ingest/`（contract + adapter 层）、`TODO.md` 的拓扑缺口、
> `README.md` 的三层架构（Infrastructure / Domain Skill / World Model）。

---

## 0. 结论先行

**这个图谱不是用来替代 pandas 管道的，它补的是管道结构上缺的四样东西：身份、拓扑、时间、出处。**

现在的管道是「一次运行 = 一批文件 → 一组 DataFrame → 一份建议」，运行结束状态就没了。
它算得对，但它不知道：

- 这次的 `SKU 4711` 和上次那个 `0000004711` 是不是同一个东西（`_normalize_material` 只处理了前导零这一种情况）
- 库位 `02` 是隔离区还是可售区（`TODO.md` 的 P0）
- 三个月前那条 lead time 是谁说的、什么时候说的、当时用的哪个 adapter
- 「那个供应商在雨季的交付方差」这种要跨 5 张表 3 个时间点的问题

知识图谱把这四样东西变成**可累积的资产**，让第 N 次导入比第 1 次更有价值。这正是你说的
「之后导入的数据都可以进一步完善这个知识图谱」——但要做到这点，有一个硬约束：
**累积必须是 append，不能是 overwrite**。第 3 节的设计原则基本都是围绕这一条展开的。

第二个结论：**先别上图数据库**。你真正的瓶颈是本体和身份解析，不是查询引擎。
第一版用 DuckDB + Parquet 存两张表就够，等到出现「SQL 写不出来」的查询再换引擎——
而因为原则 P1（图谱是投影），换引擎的成本很低。详见第 6 节。

---

## 1. 你已经有的资产：contract 层就是本体的 80%

这件事值得单独说，因为它决定了不该从零开始设计。

`DocContract` 现在声明的东西，逐条对应知识图谱本体的构件：

| contract 里的字段 | 图谱里对应什么 |
|---|---|
| `grain`, `natural_key` | 实体的**同一性判据** —— 什么算「一个」东西 |
| `role: identifier` | 节点的 key 属性 |
| `role: dimension` | 指向另一个实体的**边**，或枚举属性 |
| `role: measure` | 事实/度量，挂在事件或关系上 |
| `value_domains` | 受控词表 / 分类体系（incoterm、状态码） |
| `assertions` + `severity` | 图谱的完整性约束，且**分级**（error 拦截，warn 标注） |
| `capabilities` | 这份文档能点亮图谱的哪个子图 |
| `derivable_from` | 推断规则（图谱里的 L3 层） |
| `normalize: material_number` | 实体解析规则的第一条 |

而 `Adapter` 那一层——`column_map` / `derivations` / `value_maps` / `Fingerprint` / `TransformStep` 日志
——在语义网的世界里叫 **R2RML / RML mapping**，你已经独立发明了一遍，而且做得更好：
你的 transform log 让每一列都能追溯到产生它的规则，这就是图谱最难补的**出处（provenance）**。

`config/` 里那三个 JSON 更是直接的本体片段：`incoterm_rules.json` 是一组关于所有权转移的
规则（EXW→GIT 归买方），`node_config.json` 里的 `parent_node` / `child_nodes` 是拓扑的占位符，
`supplier_incoterm.json` 是实体属性的默认值表。它们现在散在 JSON 里，各自被一个模块读，
没有统一的取值和冲突处理。图谱要做的就是把它们收编。

**所以图谱的工作量不在"设计一套新东西"，而在三件事：**

1. 把 contract 里隐含的实体关系**显式化**（现在 `supplier` 只是 item_master 的一个字符串列，
   它应该是一个 Party 节点）
2. 补上 contract 覆盖不到的层：拓扑、时间、身份、决策历史
3. 建一个物化器，把 canonical frame 折叠成节点和边

---

## 2. 第一个用例必须是 `TODO.md` 的 P0

**不要为了建图谱而建图谱。**第一版的验收标准应该是关掉 TODO 里那两个洞：

- 库位 `02` 是隔离/冻结库存，被 `consolidate_to_planning_grain` 加进了可用库存，
  导致头寸虚高、系统少订货
- 两个 DC 的数据被求和成一个规划问题，一边的缺货被另一边的过剩掩盖

这两个都是**拓扑问题**，也就是「实体之间的关系」问题，正好是图谱的主场；而且它们
现在就在造成错误的建议。用它当第一个用例有三个好处：

- 有明确的对错判据（隔离库存不该进头寸），不是「感觉更好」
- 需要的实体很少（Location、Item、StockStatus），本体不会一上来就失控
- 做完之后 `consolidate_to_planning_grain` 从「盲目求和」变成「按拓扑求和」，
  管道的其余部分一行不用改

**阶段 0 的范围就到这里为止。**不要顺手把供应商、订单、预测一起建进去。

---

## 3. 设计原则

每条都写了「否则会怎样」，因为原则不带后果就只是口号。

### P1 — 图谱是投影，不是入口

数据流永远是：

```
原始文件 → adapter → canonical frame → [投影] → 图谱
```

**图谱不是数据入口，任何数据都不能只存在于图谱里。**整个图谱必须能从
「原始文件 + adapter 版本 + 本体版本」完全重建，且结果逐位相同（幂等）。

> 否则：图谱会慢慢变成一个没人敢删、没人说得清里面哪些是导入的、哪些是手改的黑箱。
> 这是企业知识图谱项目最常见的死法。

推论：图谱的重建脚本必须是一等公民，而且要经常跑。如果重建一次要三天，说明 P1 已经破了。

### P2 — 事实与推断分层，推断永不写回事实层

```
L0  raw          原始行（Parquet 冷存，只为可重放）
L1  canonical    contract 校验过的规范帧（现有管道的输出）
L2  resolved     实体解析后：稳定 ID、别名、拓扑
L3  derived      推断出的事实：实测 lead time、需求分类、OTD
L4  judgment     建议、决策、偏差归因、planner 的反馈
```

每层可以单独重算，且**只依赖下层**。L3 的实测 lead time 不覆盖 L2 里 item master 的
计划 lead time —— 它们是**两条并存的边**，冲突本身是一个发现。

> 这直接对应你 README 里那条「never agree with what you were shown」：
> 一旦允许推断写回事实层，master lead time 被实测值覆盖之后，
> 「ERP 里的参数已经不像供应商实际交付了」这个最有价值的发现就永久消失了。

### P3 — 双时间轴（bitemporal）

每条事实带两个时间：

- `valid_time` —— 这件事在业务上从什么时候到什么时候为真（供应商 A 从 3 月起供货）
- `observed_at` / `ingest_batch` —— 系统是什么时候、从哪一批数据知道这件事的

库存是快照，`snapshot_date` 是 valid_time 的一个点；文件是上周导的，那是 observed_at。

> 否则：`feedback/drift.py` 和 `snapshot.py` 永远回答不了「我们当时基于什么下的这个建议」。
> 而偏差归因这个模块的全部价值，就建立在能回放当时的认知状态上。没有双时间轴，
> 你只能拿今天的数据去评判三个月前的决策——这在方法论上是错的。

实现上：`valid_from` / `valid_to` / `observed_at` 三列，`valid_to` 用 `9999-12-31` 而不是 NULL
（省掉一半的 NULL 判断），关闭一条边 = 写一条新行把旧行的 `valid_to` 收口，**不是 DELETE**。

### P4 — 身份是一等公民，且合并必须可撤销

合并两个实体是**加一条 `same_as` 边**，不是把一行 UPDATE 掉。边上带证据
（谁/什么规则/多少置信度/什么时候）。删掉这条边，两个实体就分回去了。

> 否则：某天你发现「上海XX贸易」和「上海XX国际贸易」其实是两家公司，
> 而你已经把它们的 200 条 PO 历史合并计算了六个月的 lead time——没有 unmerge 能力，
> 唯一的补救是全量重建（这时候 P1 就救了你的命，前提是你一直守着它）。

### P5 — 边比节点重要：核心规划实体是 lane，不是 SKU

这条是本文最有实操价值的一条。

**lead time 不是 item 的属性。**它是这条供应通路的属性：

```
Lane = (Item × Supplier × ShipFrom × ShipTo × Incoterm × TransportMode)
```

同一个料，从同一家供应商，走海运 FOB 和走空运 DDP，lead time 差 30 天、方差差一个量级。
现在管道把它们混在一起算，得到一个既不描述海运也不描述空运的平均数，
而**方差被人为放大**，安全库存被系统性高估。

这一条同时解掉了 TODO 里的两个小项：

- 「每个 SKU 的首选供应商配置」→ 变成「lane 的选择」，且有历史数据支撑排序
- 「调拨/返工收货和采购收货混在一起算 lead time」→ 它们是不同 lane（ShipFrom 不同），
  天然分开

MOQ、订货倍数、OTD、价格、incoterm 全部挂到 lane 上。Item 上只留真正跟通路无关的属性
（描述、UoM、产品族、生命周期状态）。

### P6 — 事件优于状态，状态是事件的折叠

所有的量变都建成事件（收货、发货、调拨、盘点调整），当前状态是事件序列的折叠结果。
心智模型可以直接借 GS1 EPCIS 的四元组：**what / when / where / why**
（什么料、什么时间、哪个库位、什么业务动作）。

> 这是「后续导入的数据不断完善图谱」在技术上唯一成立的方式：
> 新数据进来是 append 一批事件，不是 overwrite 一份状态。
> 两个来源对同一时刻的库存说法不同？两条都留着，冲突交给 L3 的对账规则去解释——
> 而这个冲突本身往往就是一个 finding（比如某个库位的账实差）。

现实约束：你手上的很多导出是**快照**（inventory）而不是事件流。处理办法是明确标注实体的
「时间性质」：`snapshot` 类实体只做时点存储 + 相邻快照差分推断事件，
**绝不能假装它是事件**（差分出来的「事件」要标 `inferred: true`）。

### P7 — 开放世界 + 显式未知

图谱里查不到 ≠ 不存在。缺失必须能和「已知为零」区分开。

`assertion` 的 `severity` 直接映射：`error` 的违反进**隔离子图**（quarantine），
带上失败原因，仍然可查询、可修复、可重放；`warn` 的违反正常入图但打标记。

> 否则你会陷入两难：严格校验就丢数据（一个 SKU 没有 UoM，整批 500 行被拒），
> 宽松校验就污染图谱。隔离子图让第三条路存在：数据在图里，但不参与计算，且原因可见。

这和 `IntakeResult.unusable` 现在的处理方式是一致的，只是从「一次运行的报告」升级成
「持久的待修复队列」。

### P8 — 本体演进要有版本和迁移

`ontology_version` 和 contract 版本一起 pin 到每个 ingest batch 上。改本体 = 写迁移脚本，
不是手改数据库。因为 P1，最粗暴的迁移永远可用：改本体 → 全量重投影。

### P9 — 图谱不做计算

不要在图里算安全库存、跑预测、做 EOQ。图谱负责**取数和关系**，算法留在
`analytics/` 和 `policy/`。图谱查询的输出是 DataFrame，喂给现有模块。

> 否则你会得到一堆写在 Cypher 里的业务逻辑，没法单测、没法 review、没法复用
> `principles/sc-principles.md` 里那套公式。

---

## 4. 本体设计

### 4.1 实体（节点）

分四组，按建设顺序排列。

**A 组 · 主数据（阶段 0-1，慢变，进事实账本）**

| 实体 | 同一性判据 | 关键属性 | 来源 |
|---|---|---|---|
| `Item` | 归一化料号 | description, base_uom, product_family, item_status, lifecycle | item_master |
| `Party` | 内部 ID + 别名 | 类型(supplier/customer/internal), 名称, 国家 | po_history, sales_history, item_master |
| `Location` | (company_code, plant, storage_loc) | node_type, **stock_status**, 地理位置, 时区 | 需新建主数据 ← **P0 缺口** |
| `PlanningNode` | 内部 ID | echelon_level, 规划周期, 币种 | node_config.json |
| `Lane` | (item, supplier, from, to, incoterm, mode) | 见 P5 | 从 po_history 推断 + 人工确认 |
| `UoM` / `Currency` | 代码 | 换算关系（带生效期，汇率是时变的） | 需新建 |

`Location` 上的 `stock_status` 是 P0 的解药，取值受控：
`sellable | quarantine | blocked | consignment | in_transit_staging | scrap`。
只有 `sellable` 净进头寸，其余在报告中单列。

**B 组 · 事件（阶段 2）**

| 实体 | what/when/where/why |
|---|---|
| `PurchaseOrderLine` | 承诺发生：料、下单日、交货节点、供应商 |
| `GoodsReceipt` | 实收：料、收货日、收货库位、关联 PO 行、数量 |
| `SalesOrderLine` | 承诺owed：料、CRD、promise date、客户 |
| `GoodsIssue` | 实发：料、发货日、发出库位 |
| `InventorySnapshot` | 时点状态（非事件，标 `temporal_kind: snapshot`） |
| `TransferOrder` | 节点间调拨——发出节点的需求 + 接收节点的供应 |

**C 组 · 推断事实（阶段 2-3，L3 层）**

`MeasuredLeadTime`（挂 Lane）、`DemandClass`（挂 Item×Node）、`OTDObservation`、
`ForecastPoint`、`DemandSeriesPoint`。

**D 组 · 判断与决策（阶段 3，L4 层）**

| 实体 | 说明 |
|---|---|
| `PlanningParameter` | **具体化(reified)的参数** —— 见下 |
| `Policy` | 库存策略配置的一个版本 |
| `Recommendation` | 一条建议 + 产生它的输入快照引用 |
| `PlannerDecision` | 人实际做了什么（执行/改量/忽略） |
| `DeviationAttribution` | 建议与实际的差归因到哪 |

**`PlanningParameter` 必须是节点而不是属性**，因为你的 README 里已经写了
「stated parameters are ranked, not merged」。一个 lead time 是：

```
(subject=Lane:L-4711-SUP01-FOB, name=lead_time_days, value=45,
 source_doc=item_master@2026-03-01, authority=stated,
 valid_from=..., observed_at=..., confidence=0.6)
```

同一个 subject 上并存多条不同 `authority` 的记录（`measured` > `stated` > `default`），
取值时按权威度排序取第一条，**冲突全程可见**。这就是 P2 在数据结构上的落地。

### 4.2 关系（边）

```
Item        --sourced_via-->        Lane          (一个料可以有多条通路)
Lane        --supplied_by-->        Party
Lane        --ships_from/to-->      Location
Location    --belongs_to-->         PlanningNode
PlanningNode--parent_of-->          PlanningNode  (梯级，递归)
Location    --replenished_by-->     Location      (内部补货关系)
Item        --substitutes-->        Item          (替代料，有方向、有条件)
Item        --belongs_to_family-->  ProductFamily
Item        --alias_of-->           Item          (客户/供应商料号，**不合并节点**)
Party       --same_as-->            Party         (实体解析结果，可撤销)
GoodsReceipt--fulfills-->           PurchaseOrderLine
Recommendation --based_on-->        IngestBatch   (出处链)
PlannerDecision--responds_to-->     Recommendation
* 未来:  Item --consumes--> Item     (BOM，多层展开)
```

`parent_of` 和 BOM 是仅有的两条会产生**深度递归**的边，也是将来唯一真正需要图引擎的
地方（第 6 节的换引擎触发条件）。

### 4.3 本体也是数据

和 adapter 一样的理由：能生成、能 diff、能 review、能版本回滚。

```yaml
# inventory_planning/graph/ontology/location.yaml
entity: Location
version: 1
identity:
  natural_key: [company_code, plant, storage_location]
  surrogate_prefix: "loc"
temporal_kind: slowly_changing      # event | snapshot | slowly_changing | static
properties:
  stock_status:
    type: enum
    value_domain: stock_status      # 复用 contract 的 value_domains 机制
    required: true
    description: 只有 sellable 净进可用头寸；其余单独报告
  node_type: {type: enum, value_domain: node_type}
relations:
  belongs_to: {target: PlanningNode, cardinality: many_to_one, required: true}
assertions:
  - expr: "stock_status != 'sellable' or belongs_to is not null"
    severity: error
    description: 可售库位必须归属某个规划节点，否则它的库存无人认领
```

投影规则同样是数据：

```yaml
# inventory_planning/graph/projections/inventory.yaml
from_contract: inventory
version: 1
emit:
  - entity: Location
    key_from: [location_id]          # 需要 adapter 补出 company_code/plant
    on_missing: quarantine           # 而不是丢弃，也不是猜
  - fact: InventorySnapshot
    subject: [sku, location_id]
    valid_time: snapshot_date
    measures: [qty_on_hand, qty_in_transit]
```

---

## 5. 实体解析：最容易翻车的地方

三层 key，任何时候都不要混：

1. **source key** —— 源系统里的原样值（`0000004711`），永远保留原样，永不改写
2. **natural key** —— contract 声明的、归一化后的业务键（`4711`）
3. **surrogate ID** —— 图谱内部稳定 ID（`item:01H8X...`），只有它能被引用

### 按实体类型的策略

**Item** —— 现有 `normalize: material_number`（去前导零）只覆盖了最简单的一种。还需要：
- 大小写与分隔符归一（`AB-123` / `ab123`）
- **供应商料号 / 客户料号 → `alias_of` 边，绝不合并节点**。它们不是同一个东西，
  只是同一个东西的不同称呼；合并了就没法回答「这个供应商用什么号称呼它」
- 替代料是 `substitutes` 边（有方向、可能有条件），不是别名

**Party** —— 最难，必须有人工闸门：
- 归一化（去掉 Co./Ltd./有限公司/空格/全半角）后精确匹配 → 自动 `same_as`
- 模糊匹配（编辑距离 / token 重叠）→ 只生成 `candidate_same_as` 边带 score，
  **达阈值也不自动合并**，进人工确认队列
- 确认动作本身记进图谱（谁、什么时候、依据），因为它可能是错的（见 P4）

**Location** —— 复合键。`(company_code, plant, storage_location)` 三者缺一，
ERP 导出里的 `WH01` 就有歧义。P0 阶段的主要工作量其实在这：
**需要人去填一张库位主数据表**，因为这个信息不在任何导出文件里。
这是唯一一个「图谱解决不了、必须人给」的输入，要早说。

### 一条硬规则

> **合并是可逆的边，不是破坏性的 UPDATE。**

---

## 6. Infra 选型

### 现状（2026 年 8 月，值得知道的变化）

嵌入式图数据库这一块在 2025 下半年发生了洗牌：**Kùzu 在 2025 年 10 月被 Apple 收购，
上游仓库已归档**，0.11.3 是最后一个版本，不再有修复和新特性。代码是 MIT 的，
社区分叉了 **LadybugDB**（保留 Cypher 方言、列存、全文/向量索引，目标是做
「graph lakehouse」，直接读写 Arrow/Parquet 并和 DuckDB 存储层互操作，
2026 年 5 月发到 v0.17）。另一条路是 **DuckPGQ** —— DuckDB 的社区扩展，
实现 SQL:2023 的 SQL/PGQ 语法，CWI 主导，仍是研究项目，部分特性不完整。

**结论：现在把身家压在任何一个嵌入式图引擎上都有风险。**这恰恰是 P1（图谱是投影）
最值钱的时候——只要图谱能从原始文件重建，引擎就是可替换部件。

### 分阶段建议

**阶段一（现在做）：DuckDB + Parquet，不装图数据库**

存两层：

```
graph/
  ledger/                    # 事实账本：append-only，出处 + 双时间轴
    facts_2026-08.parquet
  materialized/              # 物化：账本的折叠结果，给管道用
    node_item.parquet
    node_location.parquet
    node_lane.parquet
    edge_sourced_via.parquet
  events/                    # 高频事务事件，不进账本（见下）
    goods_receipt/*.parquet
  quarantine/                # P7 隔离区
```

**账本表结构**（这是整个设计的核心数据结构）：

```sql
CREATE TABLE facts (
  fact_id      VARCHAR,      -- 内容哈希，保证幂等重放
  subject_id   VARCHAR,      -- surrogate ID
  predicate    VARCHAR,      -- 属性名或关系名
  object_id    VARCHAR,      -- 关系的目标（属性则为 NULL）
  object_value VARCHAR,      -- 属性值（关系则为 NULL）
  value_type   VARCHAR,
  -- 双时间轴 (P3)
  valid_from   DATE,
  valid_to     DATE DEFAULT '9999-12-31',
  observed_at  TIMESTAMP,
  -- 出处 (P1)
  ingest_batch VARCHAR,
  source_doc   VARCHAR,      -- LoadedDocument 的路径 + sheet
  adapter_id   VARCHAR,      -- adapter 名 + 版本
  authority    VARCHAR,      -- measured | stated | default | human
  confidence   DOUBLE,
  retracted_by VARCHAR       -- 撤销用，不删行
);
```

**关键的取舍：不是所有东西都进账本。**主数据、身份、拓扑、参数、决策进账本
（量小、变化慢、出处最值钱）；销售历史行、PO 行这种高频事务事件**留在 Parquet 事件表里**，
按 batch 分区，图谱只引用不打散成三元组。否则一次导入 50 万行销售历史会炸出
400 万条 fact 行，而这些行的出处其实是同一个（batch, adapter），逐行存是纯浪费。

为什么 DuckDB 够用：

- 和 pandas 同进程、零运维，`duckdb.sql(...).df()` 直接回到现有管道
- 递归 CTE 足够做梯级展开和（未来的）BOM 展开
- Parquet 原生，和你现有的 `output/history/` 快照机制风格一致
- 双时间轴的 `as_of` 查询就是两个 `BETWEEN`，SQL 表达得很自然

**阶段二（出现这些信号时才做）：换图引擎**

触发条件——满足任意一条再动：
- 需要**变长路径**查询（「这个料的所有上游供应节点，不限层数」）而递归 CTE 写不动了
- BOM 多层展开成为常规需求
- 需要图算法（社区发现、中心性）做供应商风险分析

到时候的选择：**LadybugDB**（Cypher + 直接读你现有的 Parquet，迁移成本最低，
但要接受它是个年轻的社区 fork）或 **DuckPGQ**（不用换存储，只是给 DuckDB 加 SQL/PGQ 语法，
风险最小，但特性不完整）。因为 P1，评估办法很简单：两个都跑一遍重建，比查询表达力。

**阶段三（多人多写才需要）：服务端图库**

Neo4j / Memgraph / ArcadeDB。只有当出现「多个人并发写图谱、需要事务和权限」时才值得，
这是组织问题不是技术问题。

**什么时候用 RDF/OWL**：只有当你需要和外部标准互操作（GS1 EPCIS 2.0 的追溯事件、
客户要求的可持续性溯源），或者真的需要 OWL 推理机时。代价是整个技术栈换一套，
而你 90% 的查询是分析型的，属性图更合适。**但本体设计时可以借标准的概念**
（EPCIS 的 what/when/where/why 四元组、SCOR 的 source/make/deliver/return 流程分类），
这不需要采用它们的技术栈。

### 明确不需要的

Kafka、Airflow、Spark、独立向量库、Neo4j 集群、图谱可视化平台。
你的数据量是「一个企业的规划数据」，几百万行级别，单机 DuckDB 绰绰有余。
上分布式栈的唯一后果是把一个两周的工作变成两个季度的工作。

---

## 7. 模块设计

接在现有 `inventory_planning/` 结构里，和 `ingest/` 平级：

```
inventory_planning/graph/
  __init__.py
  ontology.py          # 本体加载与校验（对标 ingest/contract.py）
  ontology/*.yaml      # 本体声明（对标 ingest/contracts/）
  projections/*.yaml   # canonical frame → 节点/边 的映射（对标 adapters/）
  projector.py         # 执行投影，产出 fact 行 + 事件表，带 transform log
  identity.py          # 实体解析：归一化、别名、candidate/confirmed same_as
  ledger.py            # 账本读写、幂等 upsert、retract、batch 回滚
  temporal.py          # as_of / as_known_at 查询，双时间轴收口逻辑
  store.py             # 存储后端抽象 —— 换引擎的唯一接缝
  queries.py           # 命名查询库（见下）
  validate.py          # 图谱级 assertion + 隔离区管理（对标 contract_tests.py）
  provenance.py        # 出处链回溯、重建脚本
  topology.py          # 拓扑相关的领域逻辑：净头寸、梯级展开
```

### 几个设计要点

**投影是数据不是代码** —— 理由和 adapter 完全一样：能生成、能 diff、能 review、
能版本回滚。而且一旦是数据，就可以让 LLM 起草一份投影，人 review 它的 transform log，
这和你现在对待 adapter 的方式一致。

**`store.py` 是唯一的接缝** —— 上面的模块只依赖 `store` 的抽象接口
（`upsert_facts` / `query_as_of` / `traverse`）。换引擎只改这一个文件。
这是第 6 节「引擎风险可控」的技术前提，不能省。

**接线点在 `IntakeResult`** —— `LoadedDocument` 已经携带 adapter 出处和 profile，
投影器消费的正是它，出处链天然接上：

```python
# orchestrator 里 load_all() 之后
batch = GraphBatch.from_intake(intake_result)   # 一次运行 = 一个 batch
projector.project(batch)                        # 幂等：重跑同一批不产生新行
topology.net_available_position(as_of=today)    # 替换 consolidate_to_planning_grain
```

**批次可回滚** —— 每条 fact 带 `ingest_batch`。发现某次导入的 adapter 有问题，
按 batch 撤销（写 retract，不删行），重新导入。这是运维上最实用的一个能力。

**`queries.py` 是命名查询库，不是让 LLM 自由写查询** —— agent 层调用的是
`lead_time_history(lane_id, window)` 这种带签名、有单测的函数，不是拼 SQL/Cypher 字符串。
自由生成查询在探索期可以，进管道不行：一个写错的 join 会静默返回少一半的行，
而这正是你 `AUDIT.md` 里在防的那类错误。

---

## 8. 落地顺序

| 阶段 | 范围 | 验收标准 |
|---|---|---|
| **0** | Location 拓扑 + Item 身份 + 账本骨架 | 隔离库位不再进可用头寸；两个 DC 分开规划；`TODO.md` 的 P0 可以删掉 |
| **1** | Party 解析 + Lane + PlanningParameter 出处 | 每个参数能回答「谁说的、什么时候、权威度多少」；lane 级 lead time 和 SKU 级的差异可量化 |
| **2** | 事件层：收货/发货/调拨 | 实测 lead time 按 lane 分开；调拨收货不再污染采购 lead time |
| **3** | 决策与偏差回写，接 `feedback/` | 能回放「三个月前基于什么认知给的建议」；偏差归因有持久历史 |
| **4** | 查询接口 + agent 使用 | 命名查询库覆盖常见问题；agent 拿到本体摘要而不是裸 schema |

每个阶段之间**必须能停下**。如果阶段 1 做完发现价值不够，阶段 2 就该重新论证。

阶段 0 的真实工作量分布值得预警：**60% 在整理库位主数据**（要人去搞清楚每个库位代码
是什么意思），30% 在写账本和投影器，10% 在改 `consolidate_to_planning_grain`。
这个比例在企业知识图谱项目里很典型——技术从来不是瓶颈。

---

## 9. 反模式清单

- **一上来设计大本体。**先建 Location + Item 两个实体，跑通，再加。见过太多项目
  花三个月画本体图，一行数据没进。
- **把图谱当计算引擎。**安全库存留在 `analytics/`（P9）。
- **把 LLM 接到写路径。**LLM 可以起草投影 YAML、起草别名映射、解释查询结果，
  但不能直接往图谱写 fact。人 review 的对象是**声明**，不是结果。
- **忘了单位和币种。**两个 UoM 不同的数量相加是静默的错误，不会报错，只会算错。
  UoM 和汇率都要带生效期。
- **快照当事件用。**从两个库存快照差分出的「事件」是推断，必须标记，
  因为它把期间内的所有进出压成了一个净值。
- **供应商合并过度。**同一集团的不同法人实体，交付表现可以完全不同。
  宁可分开、留 `part_of` 边，也不要合并。
- **图谱写成一次性 ETL 脚本。**没有 P1 的可重建性，第一次本体改动就会让你重头再来。

---

## 10. 与现有代码的具体接口

| 现有位置 | 改动 |
|---|---|
| `readers/inventory_reader.consolidate_to_planning_grain` | 改为按 `stock_status == 'sellable'` 且按 PlanningNode 分组；非可售单列报告 |
| `ingest/intake.IntakeResult` | 新增 `to_graph_batch()`，携带 adapter 出处 |
| `policy/parameters.py` | 参数取值改为查 `PlanningParameter` 的 authority 排序，替代现在的读 JSON |
| `config/node_config.json` | 迁移为 PlanningNode 实体；JSON 保留为 bootstrap 种子 |
| `config/supplier_incoterm.json` | 迁移为 Lane 的 incoterm 属性；JSON 降级为默认值兜底 |
| `feedback/snapshot.py` | 快照改为记 `ingest_batch` 引用而不是复制数据 |
| `analytics/safety_stock.py` | lead time 方差改为按 lane 取，而不是按 SKU |

---

## 参考

- [Kuzu 的终局与嵌入式图数据库新格局 (gdotv)](https://gdotv.com/blog/kuzu-legacy-embedded-graph-database-landscape/)
- [From Kuzu to Ladybug (The Data Quarry)](https://thedataquarry.com/blog/from-kuzu-to-ladybug/)
- [LadybugDB](https://ladybugdb.com/)
- [DuckDB Graph Queries 指南](https://duckdb.org/docs/lts/guides/sql_features/graph_queries)
- [DuckPGQ: Bringing SQL/PGQ to DuckDB (VLDB)](https://www.vldb.org/pvldb/vol16/p4034-wolde.pdf)
- [On the Landscape of Graph Databases (2025 survey)](https://arxiv.org/pdf/2505.24758)
- [Ontotext 参与 GS1 EPCIS 2.0 制定](https://www.ontotext.com/company/news/ontotext-contributes-to-the-development-of-gs1-epcis-2-0/)
- [OntoPedigree: 供应链追溯的本体设计模式](https://doi.org/10.3233/SW-150179)
- [A Knowledge Graph Perspective on Supply Chain Resilience](https://arxiv.org/pdf/2305.08506)
- [SCONTO: A Modular Ontology for Supply Chain Representation](https://www.researchgate.net/publication/352573142_SCONTO_A_Modular_Ontology_for_Supply_Chain_Representation)
