"""
Generate the Snowflake coverage worksheet from the live contracts.

Reads inventory_planning/ingest/contracts/*.yaml so the sheet cannot drift from the
canonical field list. The SAP column is a *candidate* to search for, not an assertion
about this tenant — customised t-codes and a customised replication both move things.
"""
import csv, pathlib, sys, yaml

#     python docs/data_layer/gen_coverage.py
#
# Re-run whenever a contract gains or loses a field, so the worksheet cannot drift
# from the canonical field list it is meant to be checked against.

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTRACTS = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "inventory_planning/ingest/contracts"
OUT = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "docs/data_layer/snowflake_coverage.csv"

# source_class decides who owns the field, which decides whether Snowflake is even
# the right question for it.
ERP_FACT, ERP_MASTER, PLANNER, DERIVED = "ERP事实", "ERP主数据", "planner本地", "派生"

# (sap_table_field, source_class, note)
SAP = {
("sales_history","sku"): ("VBRP-MATNR / LIPS-MATNR", ERP_FACT, ""),
("sales_history","demand_date"): ("LIKP-WADAT_IST 或 VBAK-AUDAT", ERP_FACT, "取实际发货日还是订单日会改变整条需求曲线——先定义再取数"),
("sales_history","qty"): ("VBRP-FKIMG / LIPS-LFIMG", ERP_FACT, "按销售单位 VRKME，需经 MARM 换算到基本单位 MEINS"),
("sales_history","order_date"): ("VBAK-AUDAT", ERP_FACT, ""),
("sales_history","ship_date"): ("LIKP-WADAT_IST", ERP_FACT, "实际过账发货日"),
("sales_history","customer_request_date"): ("VBAK-VDATU", ERP_FACT, ""),
("sales_history","promise_date"): ("VBEP-EDATU", ERP_FACT, "计划行表，不在 VBAP"),
("sales_history","amount"): ("VBRP-NETWR", ERP_FACT, ""),
("sales_history","unit_price"): ("派生 NETWR/FKIMG", DERIVED, ""),
("sales_history","currency"): ("VBRK-WAERK", ERP_FACT, ""),
("sales_history","customer"): ("VBRK-KUNAG / VBAK-KUNNR", ERP_FACT, "售达方还是送达方要定死"),
("sales_history","so_number"): ("VBAK-VBELN / VBRP-AUBEL", ERP_FACT, ""),
("sales_history","so_line_number"): ("VBAP-POSNR / VBRP-AUPOS", ERP_FACT, "SAP 报表里显示成 Item——不是物料"),
("sales_history","location_id"): ("VBRP-WERKS + LGORT", ERP_FACT, "缺了会把多个 DC 混成一个"),
("sales_history","fx_rate"): ("VBRK-KURRF", ERP_FACT, ""),
("sales_history","uom_raw"): ("VBRP-VRKME", ERP_FACT, ""),

("open_so","sku"): ("VBAP-MATNR", ERP_FACT, ""),
("open_so","open_qty"): ("VBBE-OMENG", ERP_FACT, "未交量在 VBBE；VBAP-KWMENG 是订单量不是未交量"),
("open_so","order_qty"): ("VBAP-KWMENG", ERP_FACT, ""),
("open_so","shipped_qty"): ("派生 KWMENG-OMENG 或 LIPS 汇总", DERIVED, ""),
("open_so","customer_request_date"): ("VBAK-VDATU / VBEP-EDATU", ERP_FACT, "必需字段——先确认复制里有"),
("open_so","promise_date"): ("VBEP-EDATU(确认行)", ERP_FACT, "计划行表"),
("open_so","ship_date"): ("LIKP-WADAT", ERP_FACT, ""),
("open_so","order_date"): ("VBAK-AUDAT", ERP_FACT, ""),
("open_so","so_number"): ("VBAK-VBELN", ERP_FACT, ""),
("open_so","so_line_number"): ("VBAP-POSNR", ERP_FACT, ""),
("open_so","customer"): ("VBAK-KUNNR", ERP_FACT, ""),
("open_so","open_amount"): ("派生 OMENG*NETPR", DERIVED, ""),
("open_so","unit_price"): ("VBAP-NETPR / KPEIN", ERP_FACT, "除以价格单位 KPEIN"),
("open_so","currency"): ("VBAK-WAERK", ERP_FACT, ""),
("open_so","location_id"): ("VBAP-WERKS", ERP_FACT, ""),
("open_so","fx_rate"): ("VBAK-KURRF", ERP_FACT, ""),

("open_po","sku"): ("EKPO-MATNR", ERP_FACT, ""),
("open_po","open_qty"): ("派生 EKPO-MENGE − EKBE收货量", DERIVED, "没有现成列；多数复制只给 MENGE，未交量要自己算"),
("open_po","order_qty"): ("EKPO-MENGE", ERP_FACT, ""),
("open_po","delivered_qty"): ("EKBE 汇总(BWART 101−102)", DERIVED, ""),
("open_po","committed_delivery"): ("EKET-EINDT", ERP_FACT, "交期在计划行表 EKET，不在 EKPO——只复制 EKPO 会整列丢失"),
("open_po","order_date"): ("EKKO-BEDAT", ERP_FACT, ""),
("open_po","receive_date"): ("EKBE-BUDAT", ERP_FACT, ""),
("open_po","po_number"): ("EKKO-EBELN", ERP_FACT, ""),
("open_po","po_line_number"): ("EKPO-EBELP", ERP_FACT, "报表里显示成 Item"),
("open_po","supplier"): ("EKKO-LIFNR", ERP_FACT, ""),
("open_po","unit_cost"): ("EKPO-NETPR / EKPO-PEINH", ERP_FACT, "不除价格单位 PEINH 会差 10/100/1000 倍"),
("open_po","open_amount"): ("派生", DERIVED, ""),
("open_po","currency"): ("EKKO-WAERS", ERP_FACT, ""),
("open_po","fx_rate"): ("EKKO-WKURS", ERP_FACT, ""),
("open_po","po_status_raw"): ("EKPO-ELIKZ / LOEKZ", ERP_FACT, "交货完成标记与删除标记；不过滤会把已关闭 PO 当在途"),
("open_po","incoterm_raw"): ("EKKO-INCO1 / INCO2", ERP_FACT, ""),
("open_po","transport_mode"): ("无标准字段", DERIVED, "TODO 已明确不推断运输方式"),
("open_po","location_id"): ("EKPO-WERKS + LGORT", ERP_FACT, ""),

("po_history","sku"): ("EKPO-MATNR", ERP_FACT, ""),
("po_history","supplier"): ("EKKO-LIFNR", ERP_FACT, "必需"),
("po_history","po_date"): ("EKKO-BEDAT", ERP_FACT, "必需"),
("po_history","po_qty"): ("EKPO-MENGE", ERP_FACT, "必需"),
("po_history","receive_date"): ("EKBE-BUDAT (BEWTP='E', BWART=101)", ERP_FACT, "必须扣 102 冲销；调拨/返工收货同为 101，要按 EKKO-BSART 区分——对应 TODO 里 lead time 混淆"),
("po_history","lead_time_days"): ("派生 BUDAT−BEDAT", DERIVED, ""),
("po_history","delivered_qty"): ("EKBE-MENGE 汇总", DERIVED, ""),
("po_history","open_qty"): ("派生", DERIVED, ""),
("po_history","po_number"): ("EKKO-EBELN", ERP_FACT, ""),
("po_history","po_line_number"): ("EKPO-EBELP", ERP_FACT, ""),
("po_history","unit_cost"): ("EKPO-NETPR / PEINH", ERP_FACT, "同上，注意价格单位"),
("po_history","po_amount"): ("派生", DERIVED, ""),
("po_history","currency"): ("EKKO-WAERS", ERP_FACT, ""),
("po_history","fx_rate"): ("EKKO-WKURS", ERP_FACT, ""),
("po_history","freight_cost"): ("EKPO 条件 / KONV", ERP_FACT, "运费在条件表，很少被复制"),
("po_history","incoterm_raw"): ("EKKO-INCO1", ERP_FACT, ""),
("po_history","transport_mode"): ("无标准字段", DERIVED, ""),
("po_history","moq"): ("EINE / MARC-BSTMI", ERP_MASTER, "属于主数据，不属于历史"),
("po_history","location_id"): ("EKPO-WERKS", ERP_FACT, ""),

("inventory","sku"): ("MARD-MATNR", ERP_FACT, ""),
("inventory","qty_on_hand"): ("MARD-LABST", ERP_FACT, "仅非限制库存"),
("inventory","location_id"): ("MARD-WERKS + LGORT", ERP_FACT, "库位语义(可售/质检/冻结)必须单独维护——TODO P0"),
("inventory","qty_in_transit"): ("MARD-UMLME / MSEG", ERP_FACT, "库存转储在途"),
("inventory","qty_allocated"): ("VBBE-OMENG 汇总", DERIVED, "SAP 无现成分配量列"),
("inventory","qty_available"): ("派生", DERIVED, ""),
("inventory","qty_on_order"): ("派生 open_po 汇总", DERIVED, ""),
("inventory","unit_cost"): ("MBEW-VERPR 或 STPRS / PEINH", ERP_MASTER, "移动平均还是标准价要定死；同样有价格单位"),
("inventory","inventory_value"): ("MBEW-SALK3", ERP_MASTER, ""),
("inventory","currency"): ("T001-WAERS(公司码货币)", ERP_MASTER, ""),
("inventory","snapshot_date"): ("MARD 无日期字段", ERP_FACT, "MARD 是当前态表。复制时不打快照戳就永久失去历史；期末历史在 MARDH"),
("inventory","line_number"): ("—", DERIVED, ""),
("inventory","fx_rate"): ("—", PLANNER, ""),
("inventory","uom_raw"): ("MARA-MEINS", ERP_MASTER, ""),

("item_master","sku"): ("MARA-MATNR", ERP_MASTER, ""),
("item_master","description"): ("MAKT-MAKTX", ERP_MASTER, "按语言 SPRAS 过滤，否则行翻倍"),
("item_master","supplier"): ("EORD / EINA-LIFNR", ERP_MASTER, "货源清单；今天代码用 PO 历史里最多的供应商顶替"),
("item_master","lead_time_days"): ("MARC-PLIFZ 或 EINE-APLFZ", ERP_MASTER, "工厂级计划交货时间 vs 采购信息记录，两者常不一致"),
("item_master","min_order_qty"): ("MARC-BSTMI / EINE", ERP_MASTER, "最小批量"),
("item_master","order_multiple"): ("MARC-BSTRF", ERP_MASTER, "舍入值"),
("item_master","unit_cost"): ("MBEW-STPRS / VERPR", ERP_MASTER, ""),
("item_master","currency"): ("T001-WAERS", ERP_MASTER, ""),
("item_master","incoterm"): ("EINE-INCO1 / LFM1", ERP_MASTER, ""),
("item_master","item_status"): ("MARA-MSTAE / MARC-MMSTA", ERP_MASTER, "工厂级状态是判停产的那个"),
("item_master","successor_sku"): ("MARC-NFMAT", ERP_MASTER, "后继物料——直接喂 substitution contract"),
("item_master","discontinued_date"): ("MARC-AUSDT", ERP_MASTER, "有效截止日——TODO P1 的 phase-out 日期就缺这个"),
("item_master","planner_code"): ("MARC-DISPO", ERP_MASTER, "MRP 控制者"),
("item_master","product_family"): ("MARA-MATKL / PRDHA", ERP_MASTER, ""),
("item_master","stocking_policy"): ("MARC-DISMM/DISPO 推导", ERP_MASTER, "MRP 类型不等于备货策略，需映射"),
("item_master","uom_raw"): ("MARA-MEINS", ERP_MASTER, ""),
("item_master","fx_rate"): ("—", PLANNER, ""),

("substitution","old_sku"): ("MARC-MATNR", ERP_MASTER, ""),
("substitution","new_sku"): ("MARC-NFMAT", ERP_MASTER, ""),
("substitution","effective_date"): ("MARC-AUSDT", ERP_MASTER, ""),
("substitution","relation_raw"): ("无", PLANNER, "supersede vs phase 的区分 SAP 不表达，必须人工声明"),
("substitution","ratio"): ("无", PLANNER, ""),
("substitution","rationale"): ("无", PLANNER, ""),
}

# Whole contracts that are not an ERP question at all.
WHOLE = {
 "planning_master": (PLANNER, "planner 自己的工作表，不在 SAP 也不该进 Snowflake 事实层——这正是 override 表要装的东西"),
 "demand_timeseries": (DERIVED, "由 sales_history 汇总而来，或 planner 直接给。不要单独去 Snowflake 找"),
}

HEAD = ["doc_type","field","required","type","role","unit","source_class",
        "sap_candidate","sap_note",
        "sf_database","sf_schema","sf_table","sf_column",
        "grain_ok","refresh","as_of_supported","verdict","owner","notes"]

rows = []
for f in sorted(CONTRACTS.glob("*.yaml")):
    c = yaml.safe_load(f.read_text())
    doc = c.get("doc_type")
    for name, spec in (c.get("fields") or {}).items():
        spec = spec or {}
        if doc in WHOLE:
            cls, note = WHOLE[doc]
            sap = "—"
        else:
            sap, cls, note = SAP.get((doc, name), ("?", "", "未归类——手工确认"))
        rows.append([doc, name,
                     "Y" if spec.get("required") else "",
                     spec.get("type",""), spec.get("role",""), spec.get("unit",""),
                     cls, sap, note] + [""]*10)

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(HEAD)
    w.writerows(rows)

req = sum(1 for r in rows if r[2] == "Y")
unmapped = [f"{r[0]}.{r[1]}" for r in rows if r[7] == "?"]
print(f"{len(rows)} fields, {req} required, {len(rows)-req} optional")
print("unmapped:", unmapped or "none")
