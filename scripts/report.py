"""
report.py — Generate benchmark tables and figures for the paper.

Outputs:
    Tables:
      - Table 1 : Main benchmark results (models × metrics), incl. IC / ICIR / CI
      - Table 2 : Cost sensitivity (models × cost scenarios)
      - Table 3 : Regime analysis (regime × model × metrics)
      - Table 4 : Feature importance (top features per model)
      - Table 5 : Long-only vs Long-short comparison
    Figures:
      - Figure 1 : Walk-forward timeline diagram
      - Figure 2 : Cost sensitivity curves
      - Figure 3 : Cumulative returns (with benchmarks + drawdown subplot)
      - Figure 4 : Heatmap of model × cost ranking
      - Figure 5 : Feature importance bar chart
      - Figure 6 : Regime-specific performance grouped bars

Usage:
    python scripts/report.py --results_dir backtest/ --output_dir reports/
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

warnings.filterwarnings("ignore")

# Consistent style
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colour palette: active models in blues/greens, baselines dashed grey
MODEL_COLOURS = {
    "LinearRegression":    "#1f77b4",
    "Ridge":               "#1f77b4",
    "LogisticRegression":  "#ff7f0e",
    "RandomForest":        "#2ca02c",
    "LightGBM":            "#d62728",
    "MLP":                 "#9467bd",
    "LSTM":                "#8c564b",
    "MomentumBaseline":    "#7f7f7f",
    "MeanReversion":       "#bcbd22",
    "BuyAndHold_SPY":      "#17becf",
    "EqualWeight":         "#e377c2",
}


def _model_style(name: str):
    """Return (colour, linestyle, linewidth) per model."""
    c = MODEL_COLOURS.get(name, "#333333")
    if name in ("BuyAndHold_SPY", "EqualWeight"):
        return c, "--", 2.0
    if name.endswith("Baseline") or name == "MeanReversion":
        return c, ":", 1.2
    return c, "-", 1.4


# ================================================================== #
#  Table 1: Main benchmark results (with IC / ICIR / bootstrap CI)
# ================================================================== #

def generate_main_table(
    all_metrics: dict[str, dict[str, float]],
    output_dir: str,
) -> pd.DataFrame:
    df = pd.DataFrame(all_metrics).T
    df.index.name = "Model"

    # Sort by Sharpe (gross) descending
    sort_col = "Sharpe (gross)" if "Sharpe (gross)" in df.columns else df.columns[0]
    df = df.sort_values(sort_col, ascending=False)

    csv_path = os.path.join(output_dir, "tables", "table1_main_results.csv")
    df.to_csv(csv_path)

    latex_path = os.path.join(output_dir, "tables", "table1_main_results.tex")
    with open(latex_path, "w") as f:
        f.write("% Table 1: Benchmark main results\n")
        f.write(df.to_latex(float_format="%.3f", bold_rows=True))

    print("\n" + "=" * 80)
    print("TABLE 1: Main Benchmark Results (Test Period)")
    print("=" * 80)
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".3f"))  # type: ignore[arg-type]
    print()
    return df


# ================================================================== #
#  Table 2: Cost sensitivity
# ================================================================== #

def generate_cost_sensitivity_table(
    cost_metrics: dict[str, dict[float, dict]],
    output_dir: str,
) -> pd.DataFrame:
    rows = []
    for model_name, cost_dict in cost_metrics.items():
        for cost_bps, metrics in cost_dict.items():
            rows.append({
                "Model": model_name,
                "Cost (bps)": cost_bps,
                "CAGR (net) %": metrics.get("CAGR (net)", 0),
                "Sharpe (net)": metrics.get("Sharpe (net)", 0),
                "Max DD %": metrics.get("Max DD", 0),
            })
    df = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, "tables", "table2_cost_sensitivity.csv")
    df.to_csv(csv_path, index=False)

    pivot = df.pivot_table(index="Model", columns="Cost (bps)", values="Sharpe (net)")
    latex_path = os.path.join(output_dir, "tables", "table2_cost_sensitivity.tex")
    with open(latex_path, "w") as f:
        f.write("% Table 2: Sharpe across cost scenarios\n")
        f.write(pivot.to_latex(float_format="%.3f"))

    print("\n" + "=" * 80)
    print("TABLE 2: Cost Sensitivity — Sharpe (net)")
    print("=" * 80)
    print(tabulate(pivot, headers="keys", tablefmt="github", floatfmt=".3f"))  # type: ignore[arg-type]
    print()
    return df


# ================================================================== #
#  Table 3: Regime analysis
# ================================================================== #

def generate_regime_table(
    regime_results: dict[str, pd.DataFrame],
    output_dir: str,
) -> pd.DataFrame:
    """
    regime_results: {model_name: DataFrame with (Regime, Days, ...)}
    """
    all_dfs = []
    for model_name, rdf in regime_results.items():
        rdf = rdf.copy()
        rdf.insert(0, "Model", model_name)
        all_dfs.append(rdf)
    if not all_dfs:
        return pd.DataFrame()
    df = pd.concat(all_dfs, ignore_index=True)

    csv_path = os.path.join(output_dir, "tables", "table3_regime_analysis.csv")
    df.to_csv(csv_path, index=False)

    latex_path = os.path.join(output_dir, "tables", "table3_regime_analysis.tex")
    with open(latex_path, "w") as f:
        f.write("% Table 3: Per-regime analysis\n")
        f.write(df.to_latex(index=False, float_format="%.3f"))

    print("\n" + "=" * 80)
    print("TABLE 3: Regime Analysis")
    print("=" * 80)
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".3f", showindex=False))  # type: ignore[arg-type]
    print()
    return df


# ================================================================== #
#  Table 4: Feature importance
# ================================================================== #

def generate_feature_importance_table(
    fi_results: dict[str, pd.DataFrame],
    output_dir: str,
    top_n: int = 10,
) -> pd.DataFrame:
    all_dfs = []
    for model_name, fi in fi_results.items():
        top = fi.head(top_n).copy()
        top.insert(0, "Model", model_name)
        all_dfs.append(top)
    if not all_dfs:
        return pd.DataFrame()
    df = pd.concat(all_dfs, ignore_index=True)

    csv_path = os.path.join(output_dir, "tables", "table4_feature_importance.csv")
    df.to_csv(csv_path, index=False)

    latex_path = os.path.join(output_dir, "tables", "table4_feature_importance.tex")
    with open(latex_path, "w") as f:
        f.write("% Table 4: Permutation feature importance (IC drop)\n")
        f.write(df.to_latex(index=False, float_format="%.4f"))

    print("\n" + "=" * 80)
    print("TABLE 4: Feature Importance (top-10 by IC drop)")
    print("=" * 80)
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f", showindex=False))  # type: ignore[arg-type]
    print()
    return df


# ================================================================== #
#  Table 5: Long-only vs Long-short comparison
# ================================================================== #

def generate_longonly_comparison_table(
    ls_metrics: dict[str, dict],
    lo_metrics: dict[str, dict],
    output_dir: str,
) -> pd.DataFrame:
    rows = []
    for model in ls_metrics:
        lsm = ls_metrics[model]
        lom = lo_metrics.get(model, {})
        rows.append({
            "Model": model,
            "LS Sharpe(g)": lsm.get("Sharpe (gross)", 0),
            "LS Sharpe(n)": lsm.get("Sharpe (net)", 0),
            "LS CAGR(g)%": lsm.get("CAGR (gross)", 0),
            "LO Sharpe(g)": lom.get("Sharpe (gross)", 0),
            "LO Sharpe(n)": lom.get("Sharpe (net)", 0),
            "LO CAGR(g)%": lom.get("CAGR (gross)", 0),
        })
    df = pd.DataFrame(rows).set_index("Model")

    csv_path = os.path.join(output_dir, "tables", "table5_longonly_vs_longshort.csv")
    df.to_csv(csv_path)

    latex_path = os.path.join(output_dir, "tables", "table5_longonly_vs_longshort.tex")
    with open(latex_path, "w") as f:
        f.write("% Table 5: Long-only vs Long-short\n")
        f.write(df.to_latex(float_format="%.3f"))

    print("\n" + "=" * 80)
    print("TABLE 5: Long-Only vs Long-Short")
    print("=" * 80)
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".3f"))  # type: ignore[arg-type]
    print()
    return df


# ================================================================== #
#  Figure 1: Walk-forward timeline
# ================================================================== #

def plot_walk_forward_timeline(output_dir: str):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    phases = [
        ("Train\n2005–2016", 0, 12, "#2196F3"),
        ("Val\n2017–2019", 12, 3, "#FF9800"),
        ("Test\n2020–2024", 15, 5, "#4CAF50"),
    ]
    for label, start, width, color in phases:
        rect = mpatches.FancyBboxPatch(
            (start, 0.3), width, 0.4,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor="black", linewidth=1.2, alpha=0.85
        )
        ax.add_patch(rect)
        ax.text(start + width / 2, 0.5, label,
                ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    for x in [11.9, 14.9]:
        ax.axvline(x, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(x, 0.78, "embargo", ha="center", fontsize=7, color="red")
    ax.set_xlim(-0.5, 21)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Year offset from 2005")
    ax.set_yticks([])
    ax.set_title("Walk-Forward Evaluation Protocol", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    path = os.path.join(output_dir, "figures", "fig1_walk_forward_timeline.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Figure 1 saved: {path}")


# ================================================================== #
#  Figure 2: Cost sensitivity curves
# ================================================================== #

def plot_cost_sensitivity_curves(
    cost_metrics: dict[str, dict[float, dict]],
    output_dir: str,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, cost_dict in cost_metrics.items():
        c, ls, lw = _model_style(model_name)
        costs = sorted(cost_dict.keys())
        sharpes = [cost_dict[c_val].get("Sharpe (net)", 0) for c_val in costs]
        ax.plot(costs, sharpes, marker="o", label=model_name,
                linewidth=lw, linestyle=ls, color=c, markersize=4)
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Sharpe Ratio (net)")
    ax.set_title("Cost Sensitivity: Sharpe Ratio vs. Transaction Costs")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    path = os.path.join(output_dir, "figures", "fig2_cost_sensitivity.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Figure 2 saved: {path}")


# ================================================================== #
#  Figure 3: Cumulative returns + drawdown subplot
# ================================================================== #

def plot_cumulative_returns(
    all_bt_results: dict[str, pd.DataFrame],
    output_dir: str,
    cost_bps: float = 15.0,
):
    """Equity curve (top) + drawdown (bottom) with benchmark reference lines."""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    for model_name, bt in all_bt_results.items():
        c, ls, lw = _model_style(model_name)
        dates = pd.to_datetime(bt["date"])
        ax1.plot(dates, bt["cum_gross"], label=model_name,
                 linewidth=lw, linestyle=ls, color=c)

        # Drawdown
        cum = bt["cum_gross"]
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max
        ax2.fill_between(dates, dd, 0, alpha=0.15, color=c)
        ax2.plot(dates, dd, linewidth=0.7, color=c, alpha=0.6)

    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_ylabel("Cumulative Gross Return")
    ax1.set_title("Equity Curves & Drawdowns (Gross)")
    ax1.legend(fontsize=7, loc="upper left", ncol=3)
    ax1.grid(True, alpha=0.3)

    # Shade regime bands
    for (start, end, label, colour, alpha_val) in [
        ("2020-02-01", "2020-06-30", "COVID", "red", 0.08),
        ("2022-01-01", "2022-12-31", "RateHikes", "orange", 0.08),
    ]:
        for ax in (ax1, ax2):
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=alpha_val, color=colour, label=None)

    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "figures", "fig3_cumulative_returns.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Figure 3 saved: {path}")


# ================================================================== #
#  Figure 4: Heatmap of rankings
# ================================================================== #

def plot_ranking_heatmap(
    cost_metrics: dict[str, dict[float, dict]],
    output_dir: str,
):
    models = list(cost_metrics.keys())
    costs = sorted(next(iter(cost_metrics.values())).keys())
    sharpe_matrix = pd.DataFrame(index=models, columns=costs, dtype=float)
    for m in models:
        for c in costs:
            sharpe_matrix.loc[m, c] = cost_metrics[m][c].get("Sharpe (net)", 0)
    rank_matrix = sharpe_matrix.rank(ascending=False, method="min")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(rank_matrix.astype(float), annot=True, fmt=".0f",
                cmap="RdYlGn_r", linewidths=0.5, ax=ax)
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Model")
    ax.set_title("Model Ranking by Sharpe Across Cost Scenarios")
    path = os.path.join(output_dir, "figures", "fig4_ranking_heatmap.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Figure 4 saved: {path}")


# ================================================================== #
#  Figure 5: Feature importance bar chart
# ================================================================== #

def plot_feature_importance(
    fi_results: dict[str, pd.DataFrame],
    output_dir: str,
    top_n: int = 10,
):
    """Grouped horizontal bar chart of feature importance across models."""
    n_models = len(fi_results)
    if n_models == 0:
        return
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, fi) in zip(axes, fi_results.items()):
        top = fi.head(top_n).sort_values("importance")
        c, _, _ = _model_style(model_name)
        ax.barh(top["feature"], top["importance"], xerr=top["std"],
                color=c, alpha=0.8, capsize=3)
        ax.set_xlabel("IC Drop (higher = more important)")
        ax.set_title(model_name, fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Permutation Feature Importance", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = os.path.join(output_dir, "figures", "fig5_feature_importance.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Figure 5 saved: {path}")


# ================================================================== #
#  Figure 6: Regime performance grouped bars
# ================================================================== #

def plot_regime_bars(
    regime_results: dict[str, pd.DataFrame],
    output_dir: str,
):
    """Grouped bar chart: Sharpe(gross) per regime per model."""
    rows = []
    for model_name, rdf in regime_results.items():
        for _, row in rdf.iterrows():
            rows.append({
                "Model": model_name,
                "Regime": row["Regime"],
                "Sharpe (gross)": row["Sharpe (gross)"],
            })
    if not rows:
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    regimes = df["Regime"].unique()
    models = df["Model"].unique()
    x = np.arange(len(regimes))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        c, _, _ = _model_style(model)
        vals = []
        for r in regimes:
            sub = df[(df["Model"] == model) & (df["Regime"] == r)]
            vals.append(sub["Sharpe (gross)"].values[0] if len(sub) > 0 else 0)
        ax.bar(x + i * width, vals, width, label=model, color=c, alpha=0.85)

    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels(regimes, rotation=15)
    ax.set_ylabel("Sharpe (gross)")
    ax.set_title("Per-Regime Performance")
    ax.legend(fontsize=7, ncol=3, loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8)

    path = os.path.join(output_dir, "figures", "fig6_regime_performance.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Figure 6 saved: {path}")


# ================================================================== #
#  Table 6: Diebold-Mariano pairwise p-values
# ================================================================== #

def generate_dm_table(
    dm_pvals: "pd.DataFrame",
    output_dir: str,
    dm_pvals_bh: "pd.DataFrame | None" = None,
) -> "pd.DataFrame":
    csv_path = os.path.join(output_dir, "tables", "table6_dm_test.csv")
    dm_pvals.to_csv(csv_path)

    latex_path = os.path.join(output_dir, "tables", "table6_dm_test.tex")
    with open(latex_path, "w") as f:
        f.write("% Table 6: Diebold-Mariano pairwise p-values (raw)\n")
        f.write(dm_pvals.to_latex(float_format="%.3f"))

    print("\n" + "=" * 80)
    print("TABLE 6: Diebold-Mariano Pairwise p-values (gross returns)")
    print("=" * 80)
    print(tabulate(dm_pvals, headers="keys", tablefmt="github", floatfmt=".3f"))  # type: ignore[arg-type]

    # BH-corrected table
    if dm_pvals_bh is not None:
        csv_bh = os.path.join(output_dir, "tables", "table6b_dm_test_bh.csv")
        dm_pvals_bh.to_csv(csv_bh)
        latex_bh = os.path.join(output_dir, "tables", "table6b_dm_test_bh.tex")
        with open(latex_bh, "w") as f:
            f.write("% Table 6b: Diebold-Mariano pairwise p-values (BH-corrected)\n")
            f.write(dm_pvals_bh.to_latex(float_format="%.3f"))
        print("\nTABLE 6b: BH-Corrected p-values")
        print("-" * 40)
        n = len(dm_pvals_bh)
        n_sig_raw = int((dm_pvals < 0.05).sum().sum()) // 2
        n_sig_bh = int((dm_pvals_bh < 0.05).sum().sum()) // 2
        total = n * (n - 1) // 2
        print(f"  Raw significant: {n_sig_raw}/{total}")
        print(f"  BH-corrected significant: {n_sig_bh}/{total}")

    print()
    return dm_pvals


# ================================================================== #
#  Table 7: Rebalance-frequency sensitivity
# ================================================================== #

def generate_rebalance_sensitivity_table(
    rebal_results: dict[str, dict[int, dict]],
    output_dir: str,
) -> "pd.DataFrame":
    """
    rebal_results: {model: {freq: {metric: val, ...}}}
    """
    rows = []
    for model, freq_dict in rebal_results.items():
        for freq, metrics in freq_dict.items():
            rows.append({
                "Model": model,
                "Rebal Freq (days)": freq,
                "Sharpe (gross)": metrics.get("Sharpe (gross)", 0),
                "Sharpe (net)": metrics.get("Sharpe (net)", 0),
                "Avg Turnover": metrics.get("Avg Turnover", 0),
            })
    df = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, "tables", "table7_rebal_sensitivity.csv")
    df.to_csv(csv_path, index=False)

    pivot = df.pivot_table(index="Model", columns="Rebal Freq (days)",
                           values="Sharpe (gross)")
    latex_path = os.path.join(output_dir, "tables", "table7_rebal_sensitivity.tex")
    with open(latex_path, "w") as f:
        f.write("% Table 7: Rebalance frequency sensitivity\n")
        f.write(pivot.to_latex(float_format="%.3f"))

    print("\n" + "=" * 80)
    print("TABLE 7: Rebalance Frequency Sensitivity — Sharpe (gross)")
    print("=" * 80)
    print(tabulate(pivot, headers="keys", tablefmt="github", floatfmt=".3f"))  # type: ignore[arg-type]
    print()
    return df


# ================================================================== #
#  Table 8: Top-K sensitivity
# ================================================================== #

def generate_topk_sensitivity_table(
    topk_results: dict[str, dict[int, dict]],
    output_dir: str,
) -> "pd.DataFrame":
    rows = []
    for model, k_dict in topk_results.items():
        for k, metrics in k_dict.items():
            rows.append({
                "Model": model,
                "Top-K": k,
                "Sharpe (gross)": metrics.get("Sharpe (gross)", 0),
                "Sharpe (net)": metrics.get("Sharpe (net)", 0),
                "Avg Turnover": metrics.get("Avg Turnover", 0),
            })
    df = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, "tables", "table8_topk_sensitivity.csv")
    df.to_csv(csv_path, index=False)

    pivot = df.pivot_table(index="Model", columns="Top-K",
                           values="Sharpe (gross)")
    latex_path = os.path.join(output_dir, "tables", "table8_topk_sensitivity.tex")
    with open(latex_path, "w") as f:
        f.write("% Table 8: Top-K sensitivity\n")
        f.write(pivot.to_latex(float_format="%.3f"))

    print("\n" + "=" * 80)
    print("TABLE 8: Top-K Sensitivity — Sharpe (gross)")
    print("=" * 80)
    print(tabulate(pivot, headers="keys", tablefmt="github", floatfmt=".3f"))  # type: ignore[arg-type]
    print()
    return df


# ================================================================== #
#  Figure 7: Rebalance-freq sensitivity line plot
# ================================================================== #

def plot_rebalance_sensitivity(
    rebal_results: dict[str, dict[int, dict]],
    output_dir: str,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model, freq_dict in rebal_results.items():
        c, ls, lw = _model_style(model)
        freqs = sorted(freq_dict.keys())
        sharpe_g = [freq_dict[f]["Sharpe (gross)"] for f in freqs]
        sharpe_n = [freq_dict[f]["Sharpe (net)"] for f in freqs]
        ax1.plot(freqs, sharpe_g, marker="o", label=model,
                 color=c, linestyle=ls, linewidth=lw, markersize=4)
        ax2.plot(freqs, sharpe_n, marker="s", label=model,
                 color=c, linestyle=ls, linewidth=lw, markersize=4)

    ax1.set_xlabel("Rebalance Frequency (days)")
    ax1.set_ylabel("Sharpe (gross)")
    ax1.set_title("Gross Sharpe vs Rebalance Freq")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", linewidth=0.8)

    ax2.set_xlabel("Rebalance Frequency (days)")
    ax2.set_ylabel("Sharpe (net)")
    ax2.set_title("Net Sharpe vs Rebalance Freq (15 bps)")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="gray", linewidth=0.8)

    fig.tight_layout()
    path = os.path.join(output_dir, "figures", "fig7_rebal_sensitivity.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Figure 7 saved: {path}")


# ================================================================== #
#  Figure 8: Top-K sensitivity line plot
# ================================================================== #

def plot_topk_sensitivity(
    topk_results: dict[str, dict[int, dict]],
    output_dir: str,
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model, k_dict in topk_results.items():
        c, ls, lw = _model_style(model)
        ks = sorted(k_dict.keys())
        sharpe_g = [k_dict[k]["Sharpe (gross)"] for k in ks]
        sharpe_n = [k_dict[k]["Sharpe (net)"] for k in ks]
        ax1.plot(ks, sharpe_g, marker="o", label=model,
                 color=c, linestyle=ls, linewidth=lw, markersize=4)
        ax2.plot(ks, sharpe_n, marker="s", label=model,
                 color=c, linestyle=ls, linewidth=lw, markersize=4)

    ax1.set_xlabel("Top-K")
    ax1.set_ylabel("Sharpe (gross)")
    ax1.set_title("Gross Sharpe vs Top-K")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="gray", linewidth=0.8)

    ax2.set_xlabel("Top-K")
    ax2.set_ylabel("Sharpe (net)")
    ax2.set_title("Net Sharpe vs Top-K (15 bps)")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color="gray", linewidth=0.8)

    fig.tight_layout()
    path = os.path.join(output_dir, "figures", "fig8_topk_sensitivity.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Figure 8 saved: {path}")


# ================================================================== #
#  Figure 9: DM test p-value heatmap
# ================================================================== #

def plot_dm_heatmap(
    dm_pvals: "pd.DataFrame",
    output_dir: str,
):
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.eye(len(dm_pvals), dtype=bool)
    sns.heatmap(
        dm_pvals.astype(float), annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0, vmax=0.2,
        linewidths=0.5, ax=ax, mask=mask,
    )
    ax.set_title("Diebold-Mariano Test p-values (gross returns)")
    path = os.path.join(output_dir, "figures", "fig9_dm_heatmap.pdf")
    fig.savefig(path)
    plt.close()
    print(f"  Figure 9 saved: {path}")
