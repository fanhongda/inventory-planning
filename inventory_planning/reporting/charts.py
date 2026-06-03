"""
Chart builder — produces matplotlib figures for the inventory planning report.
All figures are returned as base64-encoded PNG strings for HTML embedding.
"""

import base64
import io
import warnings
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Palette ──────────────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#1A56DB",
    "success":   "#057A55",
    "warning":   "#C27803",
    "danger":    "#C81E1E",
    "neutral":   "#6B7280",
    "light":     "#F3F4F6",
    "stocking_high": "#057A55",
    "stocking_med":  "#C27803",
    "non_stocking":  "#6B7280",
    "excess":        "#C81E1E",
    "shortage":      "#9333EA",
    "ok":            "#057A55",
    "slightly_over": "#C27803",
}

STATUS_COLORS = {
    "EXCESS":        PALETTE["danger"],
    "slightly-over": PALETTE["warning"],
    "OK":            PALETTE["success"],
    "SHORTAGE-RISK": "#9333EA",
    "non-stocking":  PALETTE["neutral"],
    "no-data":       "#D1D5DB",
}

ACTION_COLORS = {
    "PURCHASE-REQUEST":  PALETTE["danger"],
    "ORDER-FOR-BACKLOG": "#7C3AED",
    "PUSH-OUT-OPEN-PO":  PALETTE["warning"],
    "HOLD-OK":           PALETTE["success"],
    "HOLD-EXCESS":       PALETTE["warning"],
    "NO-ACTION":         PALETTE["neutral"],
}

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
    "figure.dpi":       130,
})


def _to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


class ChartBuilder:

    def __init__(self):
        self.charts: Dict[str, str] = {}   # name → base64 PNG

    # ── 1. Stocking Classification Donut ─────────────────────────────────────
    def classification_donut(self, classified: pd.DataFrame) -> str:
        counts = classified["stocking_class"].value_counts()
        labels_map = {
            "stocking-high": "High Service\n(≥9/12 months)",
            "stocking-med":  "Med Service\n(6–8/12 months)",
            "non-stocking":  "Non-Stocking\n(<6/12 months)",
        }
        colors_map = {
            "stocking-high": PALETTE["stocking_high"],
            "stocking-med":  PALETTE["stocking_med"],
            "non-stocking":  PALETTE["non_stocking"],
        }
        labels  = [labels_map.get(k, k) for k in counts.index]
        colors  = [colors_map.get(k, "#888") for k in counts.index]
        total   = counts.sum()

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        wedges, texts, autotexts = ax.pie(
            counts, labels=None, colors=colors,
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_color("white")
            at.set_fontweight("bold")
        ax.text(0, 0, f"{total:,}\nSKUs", ha="center", va="center",
                fontsize=12, fontweight="bold", color="#111827")
        ax.legend(wedges, [f"{l}  ({c:,})" for l, c in zip(labels, counts)],
                  loc="lower center", bbox_to_anchor=(0.5, -0.08),
                  fontsize=8, frameon=False, ncol=1)
        ax.set_title("Stocking Classification", fontsize=12, fontweight="bold", pad=10)
        img = _to_b64(fig)
        self.charts["classification_donut"] = img
        return img

    # ── 2. Inventory Status Bar ───────────────────────────────────────────────
    def inventory_status_bar(self, projection: pd.DataFrame) -> str:
        counts = projection["inventory_status"].value_counts()
        order  = ["SHORTAGE-RISK", "EXCESS", "slightly-over", "OK", "non-stocking", "no-data"]
        counts = counts.reindex([o for o in order if o in counts.index]).dropna()

        label_map = {
            "SHORTAGE-RISK": "Shortage Risk",
            "EXCESS":        "Excess",
            "slightly-over": "Slightly Over",
            "OK":            "On Target",
            "non-stocking":  "Non-Stocking",
            "no-data":       "No Data",
        }
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.barh(
            [label_map.get(k, k) for k in counts.index],
            counts.values,
            color=[STATUS_COLORS.get(k, "#888") for k in counts.index],
            height=0.6, edgecolor="white", linewidth=0.8,
        )
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9)
        ax.set_xlabel("Number of SKUs", fontsize=9)
        ax.set_title("Inventory Position Status", fontsize=12, fontweight="bold")
        ax.set_xlim(0, counts.max() * 1.18)
        ax.invert_yaxis()
        img = _to_b64(fig)
        self.charts["inventory_status"] = img
        return img

    # ── 3. Purchase Recommendation Actions ───────────────────────────────────
    def recommendation_summary(self, recommendations: pd.DataFrame) -> str:
        counts = recommendations["recommended_action"].value_counts()
        order  = ["PURCHASE-REQUEST", "ORDER-FOR-BACKLOG", "PUSH-OUT-OPEN-PO",
                  "HOLD-OK", "HOLD-EXCESS", "NO-ACTION"]
        counts = counts.reindex([o for o in order if o in counts.index]).dropna()

        label_map = {
            "PURCHASE-REQUEST":  "Purchase Request",
            "ORDER-FOR-BACKLOG": "Order for Backlog",
            "PUSH-OUT-OPEN-PO":  "Push Out Open PO",
            "HOLD-OK":           "Hold — OK",
            "HOLD-EXCESS":       "Hold — Excess",
            "NO-ACTION":         "No Action",
        }
        fig, ax = plt.subplots(figsize=(7, 3.5))
        bars = ax.barh(
            [label_map.get(k, k) for k in counts.index],
            counts.values,
            color=[ACTION_COLORS.get(k, "#888") for k in counts.index],
            height=0.6, edgecolor="white", linewidth=0.8,
        )
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=9)
        ax.set_xlabel("Number of SKUs", fontsize=9)
        ax.set_title("Purchase Recommendations by Action", fontsize=12, fontweight="bold")
        ax.set_xlim(0, counts.max() * 1.18)
        ax.invert_yaxis()
        img = _to_b64(fig)
        self.charts["recommendations"] = img
        return img

    # ── 4. Top Demand SKUs — Time Series ─────────────────────────────────────
    def top_demand_timeseries(self, time_series: pd.DataFrame, top_n: int = 8) -> str:
        totals  = time_series.sum().sort_values(ascending=False)
        top_skus = totals.head(top_n).index.tolist()
        ts = time_series[top_skus].copy()

        n_cols = 2
        n_rows = (top_n + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, n_rows * 2.4))
        axes = axes.flatten()

        cmap = plt.cm.get_cmap("tab10", top_n)
        periods = [str(p) for p in ts.index]

        for i, sku in enumerate(top_skus):
            ax  = axes[i]
            vals = ts[sku].values
            ax.fill_between(range(len(vals)), vals, alpha=0.15, color=cmap(i))
            ax.plot(range(len(vals)), vals, color=cmap(i), linewidth=1.5, marker="o",
                    markersize=3)
            ax.set_title(f"{sku}", fontsize=8.5, fontweight="bold", pad=3)
            ax.set_xticks(range(0, len(periods), max(1, len(periods) // 6)))
            ax.set_xticklabels(
                [periods[j] for j in range(0, len(periods), max(1, len(periods) // 6))],
                rotation=30, ha="right", fontsize=7,
            )
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.tick_params(axis="y", labelsize=7)

        # Hide any unused subplots
        for j in range(len(top_skus), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Top {top_n} SKUs — Monthly Demand (36 months)", fontsize=12,
                     fontweight="bold", y=1.01)
        fig.tight_layout()
        img = _to_b64(fig)
        self.charts["top_demand_ts"] = img
        return img

    # ── 5. Forecast — Top Purchase-Request SKUs ───────────────────────────────
    def forecast_chart(self, time_series: pd.DataFrame,
                       forecast_detail: pd.DataFrame,
                       recommendations: pd.DataFrame,
                       top_n: int = 6) -> str:
        # Pick top purchase-request SKUs by net requirement
        pr_skus = (
            recommendations[recommendations["recommended_action"] == "PURCHASE-REQUEST"]
            .sort_values("net_requirement", ascending=False)
            .drop_duplicates("sku")
            .head(top_n)["sku"]
            .tolist()
        )
        if not pr_skus:
            pr_skus = recommendations.drop_duplicates("sku").head(top_n)["sku"].tolist()

        n_cols = 2
        n_rows = (len(pr_skus) + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, n_rows * 3))
        axes = np.array(axes).flatten()

        for i, sku in enumerate(pr_skus):
            ax = axes[i]
            # Historical
            if sku in time_series.columns:
                hist = time_series[sku]
                hist_periods = [str(p) for p in hist.index]
                ax.plot(range(len(hist)), hist.values, color=PALETTE["primary"],
                        linewidth=1.5, label="Historical", marker="o", markersize=2.5)
                n_hist = len(hist)
            else:
                n_hist = 0

            # Forecast
            fc = forecast_detail[forecast_detail["sku"] == sku].copy()
            if len(fc):
                fc_vals = fc["forecast_qty"].values
                fc_x    = range(n_hist, n_hist + len(fc_vals))
                ax.plot(list(fc_x), fc_vals, color=PALETTE["danger"],
                        linewidth=1.8, linestyle="--", label="Forecast", marker="s", markersize=3)
                ax.axvline(n_hist - 0.5, color="#999", linewidth=0.8, linestyle=":")
                model = fc["model_used"].iloc[0] if len(fc) else ""
                ax.text(n_hist + 0.2, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1,
                        f"Model: {model}", fontsize=7, color=PALETTE["danger"], va="top")

            ax.set_title(sku, fontsize=9, fontweight="bold")
            ax.legend(fontsize=7, frameon=False)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.tick_params(labelsize=7)

        for j in range(len(pr_skus), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"Top {len(pr_skus)} Purchase-Request SKUs — History + 6-Month Forecast",
                     fontsize=11, fontweight="bold", y=1.01)
        fig.tight_layout()
        img = _to_b64(fig)
        self.charts["forecast"] = img
        return img

    # ── 6. Safety Stock vs On-Hand — Stocking Items ──────────────────────────
    def safety_stock_vs_onhand(self, projection: pd.DataFrame, top_n: int = 20) -> str:
        stocking = projection[
            projection["stocking_class"].isin(["stocking-high", "stocking-med"])
        ].dropna(subset=["safety_stock", "qty_on_hand"]).copy()

        stocking = stocking.sort_values("surplus_deficit").head(top_n)

        fig, ax = plt.subplots(figsize=(9, max(4, len(stocking) * 0.45)))
        y  = range(len(stocking))
        oh = stocking["qty_on_hand"].values
        ss = stocking["safety_stock"].values
        rop = stocking["reorder_point"].values if "reorder_point" in stocking.columns else ss * 1.5

        ax.barh(list(y), oh,  height=0.4, color=PALETTE["primary"], alpha=0.8, label="On Hand")
        ax.barh([yi + 0.42 for yi in y], ss, height=0.4,
                color=PALETTE["danger"], alpha=0.6, label="Safety Stock Target")

        ax.set_yticks(list(y))
        ax.set_yticklabels(stocking["sku"].values, fontsize=8)
        ax.set_xlabel("Quantity", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"On-Hand vs Safety Stock — Bottom {top_n} SKUs (by surplus)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        img = _to_b64(fig)
        self.charts["safety_stock_vs_onhand"] = img
        return img

    # ── 7. Lead Time Distribution ─────────────────────────────────────────────
    def lead_time_distribution(self, supplier_lt: pd.DataFrame) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # Histogram of WMA LT
        ax = axes[0]
        ax.hist(supplier_lt["wma_lead_time_days"].dropna(), bins=20,
                color=PALETTE["primary"], edgecolor="white", alpha=0.85)
        median_lt = supplier_lt["wma_lead_time_days"].median()
        ax.axvline(median_lt, color=PALETTE["danger"], linewidth=1.5,
                   linestyle="--", label=f"Median: {median_lt:.0f}d")
        ax.set_xlabel("WMA Lead Time (days)", fontsize=9)
        ax.set_ylabel("SKU × Supplier Count", fontsize=9)
        ax.set_title("Lead Time Distribution", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, frameon=False)

        # Top 15 longest LT by supplier
        ax2 = axes[1]
        top_lt = supplier_lt.nlargest(15, "wma_lead_time_days")
        colors = [PALETTE["danger"] if lt > 90 else PALETTE["warning"] if lt > 45
                  else PALETTE["success"] for lt in top_lt["wma_lead_time_days"]]
        ax2.barh(range(len(top_lt)), top_lt["wma_lead_time_days"],
                 color=colors, height=0.7, edgecolor="white")
        ax2.set_yticks(range(len(top_lt)))
        ax2.set_yticklabels(top_lt["sku"].values, fontsize=7)
        ax2.set_xlabel("WMA Lead Time (days)", fontsize=9)
        ax2.set_title("Longest Lead Time SKUs", fontsize=11, fontweight="bold")
        ax2.invert_yaxis()

        fig.tight_layout()
        img = _to_b64(fig)
        self.charts["lead_time"] = img
        return img

    # ── 8. Excess & Push-Out Waterfall ───────────────────────────────────────
    def excess_pushout_chart(self, projection: pd.DataFrame, top_n: int = 15) -> str:
        excess = projection[projection["pushout_candidate"] == True].copy()
        if excess.empty:
            return ""
        excess = excess.nlargest(top_n, "surplus_deficit").drop_duplicates("sku")

        fig, ax = plt.subplots(figsize=(9, max(4, len(excess) * 0.55)))
        y = range(len(excess))

        ax.barh(list(y), excess["effective_position"].values,
                height=0.4, color=PALETTE["warning"], alpha=0.75, label="Current Position")
        ax.barh([yi + 0.42 for yi in y], excess["should_be_inventory"].values,
                height=0.4, color=PALETTE["success"], alpha=0.75, label="Should-Be")

        ax.set_yticks(list(y))
        ax.set_yticklabels(excess["sku"].values, fontsize=8)
        ax.set_xlabel("Quantity", fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title(f"Top {len(excess)} Excess / Push-Out Candidates",
                     fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        fig.tight_layout()
        img = _to_b64(fig)
        self.charts["excess_pushout"] = img
        return img

    # ── 9. Demand Variability Scatter (CV vs Mean) ────────────────────────────
    def demand_variability_scatter(self, classified: pd.DataFrame) -> str:
        df = classified.dropna(subset=["demand_mean_rolling", "demand_std_rolling"]).copy()
        df = df[df["demand_mean_rolling"] > 0].copy()
        df["cv"] = df["demand_std_rolling"] / df["demand_mean_rolling"]

        color_map = {
            "stocking-high": PALETTE["stocking_high"],
            "stocking-med":  PALETTE["stocking_med"],
            "non-stocking":  PALETTE["non_stocking"],
        }

        fig, ax = plt.subplots(figsize=(7, 5))
        for cls, grp in df.groupby("stocking_class"):
            ax.scatter(grp["demand_mean_rolling"], grp["cv"],
                       c=color_map.get(cls, "#888"), alpha=0.5, s=18,
                       label=cls.replace("-", " ").title())

        ax.set_xlabel("Avg Monthly Demand (units)", fontsize=9)
        ax.set_ylabel("Coefficient of Variation (σ/μ)", fontsize=9)
        ax.set_title("Demand Variability by SKU", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, frameon=False)
        ax.set_xscale("log")
        fig.tight_layout()
        img = _to_b64(fig)
        self.charts["variability_scatter"] = img
        return img

    def build_all(self, results: dict) -> Dict[str, str]:
        """Generate all charts from a pipeline results dict. Returns {name: b64_png}."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.classification_donut(results["classified_demand"])
            self.inventory_status_bar(results["projection"])
            self.recommendation_summary(results["recommendations"])
            self.top_demand_timeseries(results["time_series"])
            self.forecast_chart(
                results["time_series"],
                results["forecast_detail"],
                results["recommendations"],
            )
            self.safety_stock_vs_onhand(results["projection"])
            self.lead_time_distribution(results["supplier_lt"])
            self.excess_pushout_chart(results["projection"])
            self.demand_variability_scatter(results["classified_demand"])
        return self.charts
