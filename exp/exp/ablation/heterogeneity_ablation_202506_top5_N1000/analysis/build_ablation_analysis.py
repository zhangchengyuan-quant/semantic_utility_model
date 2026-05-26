#!/usr/bin/env python3
"""Build figures, tables, and markdown for heterogeneity ablation analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
FIG_DIR = ANALYSIS / "figures"
TABLE_DIR = ANALYSIS / "tables"
MD_PATH = ROOT / "ablation_analysis.md"

RUNS = [
    ("baseline_sum_N1000", "Baseline SUM", "Baseline"),
    ("macro_sum_institution", "Macro ablation", "Macro"),
    ("meso_noise_channel", "Meso ablation", "Meso"),
    ("micro_default_gamma_sigma", "Micro ablation", "Micro"),
]

COLORS = {
    "Baseline": "#0072B2",
    "Macro": "#E69F00",
    "Meso": "#009E73",
    "Micro": "#D55E00",
    "Real": "#111827",
    "grid": "#D1D5DB",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    summary_rows = []
    index_data: dict[str, pd.DataFrame] = {}
    stock_data: dict[str, pd.DataFrame] = {}
    step_data: dict[str, pd.DataFrame] = {}

    for folder, label, short in RUNS:
        run_dir = ROOT / folder
        meta = json.load(open(run_dir / "run_metadata.json", encoding="utf-8"))
        idx = pd.read_csv(run_dir / "index_series.csv", encoding="utf-8-sig")
        stocks = pd.read_csv(run_dir / "stock_price_changes.csv", encoding="utf-8-sig")
        steps = pd.read_json(run_dir / "step_metrics.json")

        idx_no_init = idx[idx["date"].astype(str) != "init"]

        def rmse(a: str, b: str) -> float:
            return float(np.sqrt(np.mean((idx_no_init[a].to_numpy(float) - idx_no_init[b].to_numpy(float)) ** 2)))

        total_attempts = float(meta.get("total_order_attempts") or meta.get("total_orders") or 0)
        total_trades = float(meta.get("total_trades") or 0)
        summary_rows.append(
            {
                "folder": folder,
                "experiment": label,
                "short": short,
                "daily_records": meta.get("daily_records"),
                "n_retail": meta.get("n_retail"),
                "n_inst": meta.get("n_inst"),
                "total_orders": meta.get("total_orders"),
                "total_order_attempts": meta.get("total_order_attempts"),
                "total_trades": meta.get("total_trades"),
                "trade_per_attempt": total_trades / total_attempts if total_attempts > 0 else np.nan,
                "initial_gini": meta.get("initial_gini"),
                "final_gini": meta.get("final_gini"),
                "delta_gini": (meta.get("final_gini") or 0) - (meta.get("initial_gini") or 0),
                "elapsed_seconds": meta.get("elapsed_seconds"),
                "total_llm_tokens": meta.get("total_llm_tokens"),
                "total_llm_calls": meta.get("total_llm_calls"),
                "semantic_cache_hits": meta.get("semantic_cache_hits", ""),
                "institution_cache_hits": meta.get("institution_cache_hits", ""),
                "eq_rmse": rmse("equal_weight_sim", "equal_weight_real"),
                "float_rmse": rmse("float_weight_sim", "float_weight_real"),
                "eq_final_gap": float(idx["equal_weight_sim"].iloc[-1] - idx["equal_weight_real"].iloc[-1]),
                "float_final_gap": float(idx["float_weight_sim"].iloc[-1] - idx["float_weight_real"].iloc[-1]),
            }
        )
        index_data[short] = idx
        stocks["experiment"] = short
        stocks["return_gap_pp"] = 100.0 * stocks["return_gap"]
        stock_data[short] = stocks
        step_data[short] = steps

    summary = pd.DataFrame(summary_rows)
    baseline = summary[summary["short"] == "Baseline"].iloc[0]
    for col in ["eq_rmse", "float_rmse", "total_orders", "total_trades", "trade_per_attempt"]:
        summary[f"{col}_delta_vs_baseline"] = summary[col] - baseline[col]
    return summary, index_data, stock_data, step_data


def save_tables(summary: pd.DataFrame, stock_data: dict[str, pd.DataFrame]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary_out = summary[
        [
            "experiment",
            "daily_records",
            "total_orders",
            "total_trades",
            "trade_per_attempt",
            "eq_rmse",
            "float_rmse",
            "eq_final_gap",
            "float_final_gap",
            "final_gini",
        ]
    ].copy()
    summary_out["trade_per_attempt"] *= 100.0
    summary_out.to_csv(TABLE_DIR / "table_ablation_main_metrics.csv", index=False, encoding="utf-8-sig")

    delta_out = summary[
        [
            "experiment",
            "eq_rmse_delta_vs_baseline",
            "float_rmse_delta_vs_baseline",
            "total_orders_delta_vs_baseline",
            "total_trades_delta_vs_baseline",
            "trade_per_attempt_delta_vs_baseline",
        ]
    ].copy()
    delta_out["trade_per_attempt_delta_vs_baseline"] *= 100.0
    delta_out.to_csv(TABLE_DIR / "table_ablation_delta_vs_baseline.csv", index=False, encoding="utf-8-sig")

    stocks = pd.concat(stock_data.values(), ignore_index=True)
    pivot = stocks.pivot(index="stock", columns="experiment", values="return_gap_pp")
    pivot = pivot[["Baseline", "Macro", "Meso", "Micro"]]
    pivot.to_csv(TABLE_DIR / "table_stock_return_gap_pp.csv", encoding="utf-8-sig")


def format_dates(dates: pd.Series) -> list[str]:
    labels = []
    for value in dates.astype(str):
        if value == "init":
            labels.append("Init")
        else:
            labels.append(f"{value[4:6]}-{value[6:8]}")
    return labels


def make_index_paths(index_data: dict[str, pd.DataFrame]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base = index_data["Baseline"]
    x = np.arange(len(base))
    labels = format_dates(base["date"])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), constrained_layout=True)
    panels = [
        ("equal_weight", "Equal-weight index", "equal_weight_sim", "equal_weight_real"),
        ("float_weight", "Float-market-cap-weighted index", "float_weight_sim", "float_weight_real"),
    ]
    for ax, (_, title, sim_col, real_col) in zip(axes, panels):
        ax.plot(x, base[real_col], color=COLORS["Real"], lw=2.4, linestyle="--", label="Real market")
        for short, idx in index_data.items():
            ax.plot(x, idx[sim_col], color=COLORS[short], lw=1.9, marker="o", markersize=3.2, label=short)
        ax.set_title(title)
        ax.set_xticks(x[::2])
        ax.set_xticklabels(labels[::2], rotation=45, ha="right")
        ax.set_ylabel("Index value (base=1000)")
        ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.6)
    axes[0].legend(frameon=False, ncol=2, loc="lower left")
    fig.suptitle("Index paths under three-level heterogeneity ablations", y=1.04, fontsize=11)
    fig.savefig(FIG_DIR / "fig_ablation_index_paths.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_ablation_index_paths.pdf", bbox_inches="tight")
    plt.close(fig)


def make_metric_bars(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.4), constrained_layout=True)
    order = ["Baseline", "Macro", "Meso", "Micro"]
    sub = summary.set_index("short").loc[order].reset_index()
    x = np.arange(len(sub))
    colors = [COLORS[s] for s in sub["short"]]

    ax = axes[0, 0]
    width = 0.36
    ax.bar(x - width / 2, sub["eq_rmse"], width, label="Equal-weight", color="#56B4E9")
    ax.bar(x + width / 2, sub["float_rmse"], width, label="Float-weighted", color="#E69F00")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["short"])
    ax.set_ylabel("RMSE")
    ax.set_title("Index tracking error")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.55)

    ax = axes[0, 1]
    ax.bar(x - width / 2, sub["total_orders"], width, label="Orders", color="#0072B2")
    ax.bar(x + width / 2, sub["total_trades"], width, label="Trades", color="#D55E00")
    ax.set_xticks(x)
    ax.set_xticklabels(sub["short"])
    ax.set_ylabel("Count")
    ax.set_title("Market activity")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.55)

    ax = axes[1, 0]
    ax.bar(x, 100.0 * sub["trade_per_attempt"], color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["short"])
    ax.set_ylabel("Trades / order attempts (%)")
    ax.set_title("Matching efficiency")
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.55)

    ax = axes[1, 1]
    ax.bar(x - width / 2, sub["eq_final_gap"], width, label="Equal-weight", color="#56B4E9")
    ax.bar(x + width / 2, sub["float_final_gap"], width, label="Float-weighted", color="#E69F00")
    ax.axhline(0, color="#111827", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["short"])
    ax.set_ylabel("Simulated minus real index")
    ax.set_title("Terminal index gap")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.55)

    fig.suptitle("Aggregate ablation metrics", y=1.03, fontsize=11)
    fig.savefig(FIG_DIR / "fig_ablation_metric_bars.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_ablation_metric_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def make_stock_heatmap(stock_data: dict[str, pd.DataFrame]) -> None:
    stocks = pd.concat(stock_data.values(), ignore_index=True)
    pivot = stocks.pivot(index="stock", columns="experiment", values="return_gap_pp")
    pivot = pivot[["Baseline", "Macro", "Meso", "Micro"]]
    fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
    vmax = max(1.0, float(np.nanmax(np.abs(pivot.to_numpy()))))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Stock-level return gap (real minus simulated, pp)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=8, color="#111827")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Return gap (percentage points)")
    fig.savefig(FIG_DIR / "fig_ablation_stock_return_gap_heatmap.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_ablation_stock_return_gap_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def make_daily_activity(step_data: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    for short, steps in step_data.items():
        x = np.arange(1, len(steps) + 1)
        axes[0].plot(x, steps["submitted_orders"], lw=1.8, marker="o", markersize=3, color=COLORS[short], label=short)
        axes[1].plot(x, steps["trades"], lw=1.8, marker="o", markersize=3, color=COLORS[short], label=short)
    axes[0].set_title("Daily submitted orders")
    axes[1].set_title("Daily matched trades")
    for ax in axes:
        ax.set_xlabel("Trading day")
        ax.grid(True, axis="y", color=COLORS["grid"], alpha=0.6)
    axes[0].set_ylabel("Orders")
    axes[1].set_ylabel("Trades")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Daily market activity across ablations", y=1.04, fontsize=11)
    fig.savefig(FIG_DIR / "fig_ablation_daily_activity.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_ablation_daily_activity.pdf", bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame, float_cols: dict[str, int] | None = None) -> str:
    float_cols = float_cols or {}
    out = df.copy()
    for col, nd in float_cols.items():
        if col in out:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
    for col in out.columns:
        if col not in float_cols:
            out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(v))
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in out.to_numpy(str)]
    return "\n".join([header, sep, *rows])


def write_markdown(summary: pd.DataFrame, stock_data: dict[str, pd.DataFrame]) -> None:
    order = ["Baseline", "Macro", "Meso", "Micro"]
    sub = summary.set_index("short").loc[order].reset_index()
    baseline = sub[sub["short"] == "Baseline"].iloc[0]

    main_table = sub[
        [
            "short",
            "total_orders",
            "total_trades",
            "trade_per_attempt",
            "eq_rmse",
            "float_rmse",
            "eq_final_gap",
            "float_final_gap",
            "final_gini",
        ]
    ].rename(
        columns={
            "short": "Experiment",
            "total_orders": "Orders",
            "total_trades": "Trades",
            "trade_per_attempt": "Trade/Attempt",
            "eq_rmse": "EW RMSE",
            "float_rmse": "FW RMSE",
            "eq_final_gap": "EW Final Gap",
            "float_final_gap": "FW Final Gap",
            "final_gini": "Final Gini",
        }
    )
    main_table["Trade/Attempt"] *= 100.0

    delta = sub[
        [
            "short",
            "eq_rmse_delta_vs_baseline",
            "float_rmse_delta_vs_baseline",
            "total_orders_delta_vs_baseline",
            "total_trades_delta_vs_baseline",
            "trade_per_attempt_delta_vs_baseline",
        ]
    ].rename(
        columns={
            "short": "Experiment",
            "eq_rmse_delta_vs_baseline": "Delta EW RMSE",
            "float_rmse_delta_vs_baseline": "Delta FW RMSE",
            "total_orders_delta_vs_baseline": "Delta Orders",
            "total_trades_delta_vs_baseline": "Delta Trades",
            "trade_per_attempt_delta_vs_baseline": "Delta Trade/Attempt",
        }
    )
    delta["Delta Trade/Attempt"] *= 100.0

    stock_pivot = pd.concat(stock_data.values(), ignore_index=True).pivot(index="stock", columns="experiment", values="return_gap_pp")
    stock_pivot = stock_pivot[order].reset_index().rename(columns={"stock": "Stock"})

    macro = sub[sub["short"] == "Macro"].iloc[0]
    meso = sub[sub["short"] == "Meso"].iloc[0]
    micro = sub[sub["short"] == "Micro"].iloc[0]

    md = f"""# Three-Level Heterogeneity Ablation Analysis

## 实验设置

本节比较标准 SUM N=1000 实验与三类异质性消融实验。四组实验均使用 2025 年 4 月的五只 MSCI A50 中日均成交额最高股票，交易日数为 21 天，散户数量为 1000，并使用相同的初始随机种子与股票池。标准实验来自 scalability 中的 `sum_N1000`，已复制到 `baseline_sum_N1000`；三组消融实验分别对应 macro、meso、micro 三个层级。

三类消融的含义如下。Macro 消融将原本的机构专用架构替换为 SUM 散户同构架构，并把机构本金按 value:momentum:noise = 3:3:4 分成十个 sleeve 后统一结算。Meso 消融保留散户类型标签和个体状态，但让所有 SUM 散户统一使用 noise 语义通道。Micro 消融保留类型和语义通道，但把所有 SUM 散户的 `gamma` 与 `sigma` 固定为分布均值，即 `gamma=0.08`、`sigma=0.0175`。

为了让消融更接近“只移除某一层异质性”的控制变量实验，三组消融复用了标准 SUM N=1000 的共享语义记录；meso 与 micro 还复用了标准机构 CIO 决策记录。因此，三组消融的 LLM token 与 wall-clock cost 不再用于效率比较，分析重点放在重新执行后的 agent 决策、订单、撮合、价格路径、财富分布与逐股收益拟合。

## 总体指标

表 1 给出四组实验的核心结果。标准 SUM 的等权 RMSE 为 {baseline['eq_rmse']:.2f}，流通市值加权 RMSE 为 {baseline['float_rmse']:.2f}。三个消融后，等权 RMSE 均有所上升，其中 macro、meso、micro 分别为 {macro['eq_rmse']:.2f}、{meso['eq_rmse']:.2f}、{micro['eq_rmse']:.2f}；流通市值加权 RMSE 对 meso 消融最敏感，上升到 {meso['float_rmse']:.2f}。

**Table 1. Aggregate ablation metrics**

{md_table(main_table, {'Trade/Attempt': 2, 'EW RMSE': 2, 'FW RMSE': 2, 'EW Final Gap': 2, 'FW Final Gap': 2, 'Final Gini': 4})}

**Table 2. Metric changes relative to Baseline SUM**

{md_table(delta, {'Delta EW RMSE': 2, 'Delta FW RMSE': 2, 'Delta Orders': 0, 'Delta Trades': 0, 'Delta Trade/Attempt': 2})}

![Index paths under three-level heterogeneity ablations](analysis/figures/fig_ablation_index_paths.png)

图 1 显示，标准 SUM 与三个消融在等权指数上都低估了真实市场 4 月初的快速下跌幅度，但后续路径的偏离方式不同。Macro 消融与标准 SUM 的路径最接近，说明机构层架构替换对聚合指数路径的扰动相对温和；它的流通市值加权 RMSE 仅比标准实验高 {macro['float_rmse'] - baseline['float_rmse']:.2f}。这表明在当前五股、月度窗口和单机构设定下，机构层异质性的主要作用不是决定整体指数走势，而是影响局部成交与尾部股票的价格修正。

Meso 消融的路径差异最大，特别是在流通市值加权指数上，期末 gap 从标准实验的 {baseline['float_final_gap']:.2f} 扩大到 {meso['float_final_gap']:.2f}。这说明把 value、momentum、noise 三种散户全部压成 noise 通道，会显著改变大权重股票上的买卖压力与成交效率。换言之，类型到语义通道的中层映射不是装饰性的分类变量，而是决定公共信息如何转化为异质交易方向的关键机制。

Micro 消融在等权指数上的 RMSE 为 {micro['eq_rmse']:.2f}，虽然高于标准实验，但低于 macro 与 meso；然而其流通市值加权期末 gap 达到 {micro['float_final_gap']:.2f}，比标准实验更负。这说明固定 `gamma` 与 `sigma` 会削弱个体响应强弱的分散性，使市场仍能保留一定方向性，但在大权重资产上的价格压力更容易同步化。

![Aggregate ablation metrics](analysis/figures/fig_ablation_metric_bars.png)

## 市场活动与撮合效率

图 2 的市场活动面板显示，meso 消融产生了最多订单，总订单数为 {int(meso['total_orders']):,}，比标准 SUM 多 {int(meso['total_orders'] - baseline['total_orders']):,}；但其成交数只有 {int(meso['total_trades']):,}，比标准实验少 {int(baseline['total_trades'] - meso['total_trades']):,}。这意味着 noise 通道统一化并没有带来更有效的市场流动性，反而使订单意图更拥挤、更同向，导致更多订单无法匹配。

Macro 消融的总订单数与标准实验接近，成交数略高，为 {int(macro['total_trades']):,}。这符合 macro 消融的设计：机构本金被拆成 SUM-style sleeves 后，机构层不再通过 CIO 风控统一形成大单，而是以多个风格账户表达交易需求，因此局部成交机会增加，但总体路径拟合改善有限。

Micro 消融的订单数为 {int(micro['total_orders']):,}，接近标准实验，但成交数下降到 {int(micro['total_trades']):,}。这说明微观参数异质性主要影响订单幅度与价格偏移的离散程度；当 `gamma` 与 `sigma` 被固定后，不同 agent 对同一信号的响应更集中，撮合空间缩小。

![Daily market activity across ablations](analysis/figures/fig_ablation_daily_activity.png)

图 3 进一步展示日度订单与成交变化。Meso 消融在多个交易日都有显著更高的 submitted orders，但 matched trades 没有同步上升，说明中层通道异质性对双边市场中的“可成交差异”非常重要。标准 SUM、macro 和 micro 的日度活动更加接近，其中 macro 的成交更平滑，反映出机构 SUM-sleeves 对订单簿提供了额外但不剧烈的对手方。

## 逐股误差结构

逐股收益差使用 `real return - simulated return` 的百分点表示。负值表示模拟跌幅小于真实市场，即低估真实下跌；正值表示模拟跌幅大于真实市场，即过度下跌。

**Table 3. Stock-level return gap (real minus simulated, percentage points)**

{md_table(stock_pivot, {'Baseline': 2, 'Macro': 2, 'Meso': 2, 'Micro': 2})}

![Stock-level return gap heatmap](analysis/figures/fig_ablation_stock_return_gap_heatmap.png)

图 4 显示所有设置都明显低估了立讯精密 `002475.SZ` 的真实跌幅，这是本实验窗口中最难拟合的单股。标准 SUM 对该股的收益差为 -11.69 个百分点；macro 消融改善到 -8.80 个百分点，而 meso 消融恶化到 -14.31 个百分点。这进一步支持中层语义通道的重要性：当所有散户都使用 noise 通道时，模型更难对个股层面的基本面和趋势信息作出差异化响应。

贵州茅台 `600519.SH` 呈现相反方向的误差：标准 SUM 的收益差为 +8.04 个百分点，三个消融仍为正，micro 达到 +9.52 个百分点。这说明模型对大权重防御型资产的下跌压力偏强，尤其在微观响应参数被固定时更明显。由于流通市值加权指数对大权重股票更敏感，这也是 micro 和 meso 在 float-weighted terminal gap 上变差的重要原因。

## 层级贡献解释

从三组消融的整体结果看，三层异质性对市场仿真的贡献不是同质的。Macro 层主要影响机构资金如何进入订单簿。将机构改成 SUM-style sleeves 后，交易数略有增加，流通市值加权 RMSE 几乎不变，但逐股误差有所再分配。这说明当前设置中，机构层架构更多影响成交结构和局部价格压力，而不是指数级路径的主导因素。

Meso 层贡献最大。统一使用 noise 通道后，订单数量暴增、成交效率下降、float-weighted RMSE 明显恶化。这说明 value、momentum、noise 三种语义解释通道提供了必要的中层行为分工：value 通道吸收基本面与估值修正，momentum 通道吸收趋势和扩散信号，noise 通道吸收情绪与订单簿扰动。移除这种分工后，市场不再有足够的异质对手盘，公共信息被过度压缩为同类情绪响应。

Micro 层主要控制响应幅度的个体离散性。固定 `gamma` 与 `sigma` 后，等权 RMSE 仅小幅恶化，但大权重股票上的终端偏差扩大。这表明微观异质性对方向判断不是最核心的来源，却对订单强度、成交概率和大权重资产的价格弹性很关键。它提供的是“同一语义下不同 agent 反应强弱不同”的横向扩散，而不是新的语义方向。

## 结论

本组消融支持三层异质性设计的必要性。标准 SUM 在四组中取得最低的等权 RMSE 和最低的流通市值加权 RMSE；移除任一层异质性都会降低至少一个市场层面的拟合指标。最关键的层是 meso 语义通道异质性，其次是 micro 响应参数异质性；macro 机构架构异质性在当前单月五股实验中影响较温和，但仍改变成交结构与逐股误差分布。

因此，本文的三层异质性并非简单增加 agent 多样性的装饰项，而是分别对应三种市场生成机制：macro 层控制大资金结构，meso 层控制公共信息的解释分工，micro 层控制同一解释下的响应幅度分散。三者共同作用，才使 N=1000 的 SUM 市场在指数路径、成交活动和逐股误差之间维持相对均衡的仿真表现。
"""
    MD_PATH.write_text(md, encoding="utf-8")


def write_markdown(summary: pd.DataFrame, stock_data: dict[str, pd.DataFrame]) -> None:
    """Write the June 2025 ablation report in Chinese with English chart/table labels."""
    order = ["Baseline", "Macro", "Meso", "Micro"]
    sub = summary.set_index("short").loc[order].reset_index()
    baseline = sub[sub["short"] == "Baseline"].iloc[0]
    macro = sub[sub["short"] == "Macro"].iloc[0]
    meso = sub[sub["short"] == "Meso"].iloc[0]
    micro = sub[sub["short"] == "Micro"].iloc[0]

    main_table = sub[
        [
            "short",
            "total_orders",
            "total_trades",
            "trade_per_attempt",
            "eq_rmse",
            "float_rmse",
            "eq_final_gap",
            "float_final_gap",
            "final_gini",
        ]
    ].rename(
        columns={
            "short": "Experiment",
            "total_orders": "Orders",
            "total_trades": "Trades",
            "trade_per_attempt": "Trade/Attempt",
            "eq_rmse": "EW RMSE",
            "float_rmse": "FW RMSE",
            "eq_final_gap": "EW Final Gap",
            "float_final_gap": "FW Final Gap",
            "final_gini": "Final Gini",
        }
    )
    main_table["Trade/Attempt"] *= 100.0

    delta = sub[
        [
            "short",
            "eq_rmse_delta_vs_baseline",
            "float_rmse_delta_vs_baseline",
            "total_orders_delta_vs_baseline",
            "total_trades_delta_vs_baseline",
            "trade_per_attempt_delta_vs_baseline",
        ]
    ].rename(
        columns={
            "short": "Experiment",
            "eq_rmse_delta_vs_baseline": "Delta EW RMSE",
            "float_rmse_delta_vs_baseline": "Delta FW RMSE",
            "total_orders_delta_vs_baseline": "Delta Orders",
            "total_trades_delta_vs_baseline": "Delta Trades",
            "trade_per_attempt_delta_vs_baseline": "Delta Trade/Attempt",
        }
    )
    delta["Delta Trade/Attempt"] *= 100.0

    stock_pivot = pd.concat(stock_data.values(), ignore_index=True).pivot(index="stock", columns="experiment", values="return_gap_pp")
    stock_pivot = stock_pivot[order].reset_index().rename(columns={"stock": "Stock"})

    stock_names = {
        "300059.SZ": "East Money",
        "300308.SZ": "Zhongji Innolight",
        "300502.SZ": "Eoptolink",
        "600519.SH": "Kweichow Moutai",
        "002594.SZ": "BYD",
    }
    stock_pivot["Stock"] = stock_pivot["Stock"].map(lambda s: f"{s} ({stock_names.get(s, s)})")

    md = f"""# Three-Level Heterogeneity Ablation Analysis, 2025-06

## 实验设置

本报告汇报 2025 年 6 月的三层异质性消融结果，并与标准 SUM N=1000 baseline 对比。股票池为 MSCI A50 中 2025-06 全月日均成交额最高的五只股票：东方财富 `300059.SZ`、中际旭创 `300308.SZ`、新易盛 `300502.SZ`、贵州茅台 `600519.SH`、比亚迪 `002594.SZ`。四组实验均覆盖 2025-06-03 至 2025-06-30 的 20 个交易日，散户数量 N=1000，seed=42，`mock_llm=False`。

三组消融分别对应三个异质性层级。Macro 消融把机构端替换为 SUM 散户同构架构，并将机构本金按 value:momentum:noise = 3:3:4 拆成十个 sleeve 后统一结算。Meso 消融保留散户类型数量和标签，但所有 SUM 散户都使用 noise 通道。Micro 消融保留类型和通道，但把所有 SUM 散户的 `gamma` 与 `sigma` 固定为分布均值，即 `gamma=0.08`、`sigma=0.0175`。

为了保证消融是控制变量实验，三组消融均复用 baseline 的共享语义缓存；meso 和 micro 还复用 baseline 的机构 CIO 决策缓存。因此本报告比较的是重新撮合后的订单、成交、价格路径、财富分布和逐股收益拟合，不把新增 token 作为效率指标。

## 总体结果

**Table 1. Aggregate ablation metrics**

{md_table(main_table, {'Trade/Attempt': 2, 'EW RMSE': 2, 'FW RMSE': 2, 'EW Final Gap': 2, 'FW Final Gap': 2, 'Final Gini': 4})}

**Table 2. Metric changes relative to Baseline SUM**

{md_table(delta, {'Delta EW RMSE': 2, 'Delta FW RMSE': 2, 'Delta Orders': 0, 'Delta Trades': 0, 'Delta Trade/Attempt': 2})}

从指数拟合看，2025-06 的结果呈现明显的多指标权衡，而不是某一组在所有指标上单调最优。Baseline 的 EW RMSE 为 {baseline['eq_rmse']:.2f}，FW RMSE 为 {baseline['float_rmse']:.2f}。Macro 消融的 EW RMSE 降至 {macro['eq_rmse']:.2f}，但 FW RMSE 升至 {macro['float_rmse']:.2f}，说明机构 SUM 化会让等权路径更接近真实指数，却加重大权重资产上的偏差。Meso 消融的 FW RMSE 为 {meso['float_rmse']:.2f}，略低于 baseline，但 EW RMSE 升至 {meso['eq_rmse']:.2f}，且成交结构明显恶化。Micro 消融的 EW RMSE 为 {micro['eq_rmse']:.2f}、FW RMSE 为 {micro['float_rmse']:.2f}，两个口径均相对 baseline 变差，说明个体响应参数异质性对指数拟合有稳定贡献。

![Index paths under three-level heterogeneity ablations](analysis/figures/fig_ablation_index_paths.png)

图 1 对比等权与流通市值加权指数路径。2025 年 6 月真实市场中，中际旭创和新易盛等 AI 光模块股票涨幅较大，而贵州茅台和比亚迪下跌，形成了明显的结构性分化。四组模拟均能产生上涨方向，但都低估了真实等权指数的后半月上行幅度。Meso 消融的模拟路径更平滑、更同质，说明统一 noise 通道后，agent 对强趋势股票的差异化追随不足，同时对大权重股票的压力更集中。

![Aggregate ablation metrics](analysis/figures/fig_ablation_metric_bars.png)

图 2 汇总指数误差、市场活动、撮合效率和期末指数偏差。Macro 消融的订单数与 baseline 接近，成交数为 {int(macro['total_trades']):,}，略低于 baseline 的 {int(baseline['total_trades']):,}，说明机构端拆成 SUM-style sleeves 后并没有显著放大市场活跃度，主要影响局部成交结构。Meso 消融最特殊：订单数增加到 {int(meso['total_orders']):,}，比 baseline 多 {int(meso['total_orders'] - baseline['total_orders']):,}，但成交数下降到 {int(meso['total_trades']):,}。这表明通道同质化带来更多同向订单，而不是更多有效对手盘。Micro 消融的订单数和成交数都接近 baseline，说明微观参数异质性更像是调节响应幅度和成交概率的机制，而不是改变整体交易意愿的主因。

## 日度活动

![Daily market activity across ablations](analysis/figures/fig_ablation_daily_activity.png)

图 3 展示逐交易日订单和成交。Meso 消融在多数交易日都提交更多订单，但 matched trades 明显偏低，尤其在中后段交易日同向拥挤更突出。这一点是三层异质性设计的重要证据：value、momentum、noise 三类语义通道不仅改变“买或卖”的方向，也决定市场中能否形成足够的双边报价。Macro 和 micro 与 baseline 更接近，说明在本月五股设置下，指数层面的主要敏感源来自 meso 层。

## 逐股误差

逐股收益差定义为 `real return - simulated return`，单位为百分点。正值表示模拟涨幅不足或跌幅过大；负值表示模拟涨幅过高或跌幅不足。

**Table 3. Stock-level return gap (real minus simulated, percentage points)**

{md_table(stock_pivot, {'Baseline': 2, 'Macro': 2, 'Meso': 2, 'Micro': 2})}

![Stock-level return gap heatmap](analysis/figures/fig_ablation_stock_return_gap_heatmap.png)

图 4 显示，所有实验都明显低估了中际旭创 `300308.SZ` 和新易盛 `300502.SZ` 的真实上涨，这是 2025-06 股票池中最难拟合的结构性行情。Baseline 对中际旭创的收益差为 {stock_pivot.loc[stock_pivot['Stock'].str.startswith('300308'), 'Baseline'].iloc[0]:.2f} 个百分点，对新易盛为 {stock_pivot.loc[stock_pivot['Stock'].str.startswith('300502'), 'Baseline'].iloc[0]:.2f} 个百分点。Meso 消融进一步扩大了中际旭创误差，说明统一 noise 通道会削弱对趋势性行情的追随能力。

另一方面，贵州茅台 `600519.SH` 和比亚迪 `002594.SZ` 在真实市场下跌，但模拟价格普遍上涨，因此收益差为负。Macro 消融下贵州茅台误差扩大到 {stock_pivot.loc[stock_pivot['Stock'].str.startswith('600519'), 'Macro'].iloc[0]:.2f} 个百分点，说明机构端 SUM 化会加强部分大权重股票上的买入压力。Micro 消融下比亚迪误差为 {stock_pivot.loc[stock_pivot['Stock'].str.startswith('002594'), 'Micro'].iloc[0]:.2f} 个百分点，显示固定响应参数会降低 agent 对负向分化的离散反应。

## 结论

本次 2025-06 消融支持三层异质性设计的必要性，但证据表现为“拟合指标与市场机制的权衡”。Macro 消融降低了等权 RMSE，却显著提高流通市值加权 RMSE，说明机构层改变会重新分配小权重与大权重股票的价格误差。Meso 消融在 FW RMSE 上略有改善，但订单显著增加、成交显著减少，显示语义通道同质化会破坏市场微观结构。Micro 消融在两个指数 RMSE 上均弱于 baseline，说明个体响应参数异质性对稳定拟合有直接贡献。

因此，三层异质性不是单纯增加 agent 多样性，而是分别承担不同市场生成机制：macro 层控制大资金表达方式，meso 层控制公共信息的解释分工，micro 层控制同一解释下的响应强弱分布。三者共同作用，才使 SUM 市场在指数路径、成交活动和逐股误差之间保持相对更好的平衡。
"""
    MD_PATH.write_text(md, encoding="utf-8")


def main() -> None:
    configure_style()
    summary, index_data, stock_data, step_data = load_data()
    save_tables(summary, stock_data)
    make_index_paths(index_data)
    make_metric_bars(summary)
    make_stock_heatmap(stock_data)
    make_daily_activity(step_data)
    write_markdown(summary, stock_data)
    print(f"Wrote {MD_PATH}")
    print(f"Wrote figures to {FIG_DIR}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
