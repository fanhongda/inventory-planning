-- Snowflake 覆盖度盘点 — 只读探测
-- 配合 snowflake_coverage.csv 使用：先跑 1/2 找到候选表，填进 sf_* 列，再对每张表跑 3/4/5。
-- 全部只读，不建对象。

-- 1) 有没有 SAP 原表被复制进来（按表名找）
select table_catalog, table_schema, table_name, row_count, bytes, last_altered
from   snowflake.account_usage.tables            -- 无权限则改用 <DB>.information_schema.tables
where  deleted is null
and (
       upper(table_name) rlike '.*(MARA|MARC|MAKT|MARD|MARDH|MBEW|MARM)($|_).*'   -- 主数据/库存
    or upper(table_name) rlike '.*(EKKO|EKPO|EKET|EKBE|EINA|EINE|EORD)($|_).*'    -- 采购
    or upper(table_name) rlike '.*(VBAK|VBAP|VBEP|VBBE|VBRK|VBRP|LIKP|LIPS)($|_).*' -- 销售
)
order by table_schema, table_name;

-- 2) 没有原表时：按字段名找建模层（一列在哪些表里出现过）
select table_schema, table_name, column_name, data_type
from   snowflake.account_usage.columns
where  deleted is null
and    upper(column_name) in (
         'MATNR','WERKS','LGORT','LABST','INSME','SPEME','UMLME',   -- 库存 + 质检/冻结（TODO P0）
         'EINDT','BEDAT','BUDAT','MENGE','NETPR','PEINH','BWART',   -- 采购 + 收货
         'PLIFZ','BSTMI','BSTRF','EISBE','MINBE','DISPO','MMSTA',   -- MRP 主数据
         'NFMAT','AUSDT',                                           -- 后继物料 / 停产日（TODO P1）
         'OMENG','KWMENG','FKIMG','VDATU','WADAT_IST','KUNNR'       -- 销售 / 需求
       )
order by column_name, table_schema, table_name;

-- 3) 粒度核对：这张表真的是一行一个 PO 行吗
select count(*) as rows_total,
       count(distinct <KEY_COLS>) as rows_distinct_key,
       count(*) - count(distinct <KEY_COLS>) as dupes      -- 不为 0 就不是声明的粒度
from   <DB>.<SCHEMA>.<TABLE>;

-- 4) 刷新频率与 as-of 支持：能不能问"上周三的库存是多少"
select max(<LOAD_TS_COL>)                              as latest_load,
       datediff('hour', max(<LOAD_TS_COL>), current_timestamp()) as hours_stale,
       count(distinct to_date(<LOAD_TS_COL>))          as distinct_load_days,
       min(<LOAD_TS_COL>)                              as earliest_load
from   <DB>.<SCHEMA>.<TABLE>;
-- distinct_load_days = 1 → 是当前态覆盖写，没有历史。MARD 类库存表尤其要查这条：
-- 没有快照戳就永久失去历史，之后再补也补不回来。

-- 5) 物料号形态：判断前导零被谁吃掉了
select typeof(to_variant(<MATNR_COL>))                as sf_type,
       min(length(<MATNR_COL>))                       as len_min,
       max(length(<MATNR_COL>))                       as len_max,
       count_if(<MATNR_COL> rlike '^0+')              as leading_zero_rows,
       count_if(<MATNR_COL> <> trim(<MATNR_COL>))     as untrimmed_rows
from   <DB>.<SCHEMA>.<TABLE>;
-- 与 Excel 导出的同一张表对一下：两边前导零处理不一致 → join 静默丢行，
-- 这是 contract 里 normalize: material_number 存在的原因。

-- 6) 与 Excel 报表对账（决定要不要信 Snowflake）
-- 同一天、同一工厂，跑 Snowflake 和现有 t-code 报表，比 行数 / 数量合计 / 金额合计。
-- 对不上先查：过账日 vs 凭证日、冲销行(BWART 102)、删除标记(LOEKZ)、公司码范围。
