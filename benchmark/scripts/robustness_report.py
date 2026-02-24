"""
robustness_report.py — Generate tables and figures for robustness experiments.

New outputs:
  Tables:
    - Table 9  : Adversarial robustness (model × epsilon × metrics)
    - Table 10 : Fuzzing stress-test heatmap (model × scenario × Sharpe)
    - Table 11 : Label poisoning resilience (model × poison_rate × IC)
    - Table 12 : Alpha decay half-life (model → half-life in days)

  Figures:
    - Figure 10 : Adversarial Sharpe degradation curve
    - Figure 11 : Signal flip rate vs epsilon
    - Figure 12 : Fuzzing crash heatmap
    - Figure 13 : Label poisoning IC curve
    - Figure 14 : Alpha decay curve (IC vs horizon)
"""

from __future__ import annotations

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colour palette
MODEL_COLOURS = {
    "LinearRegression":     "#1f77b4",
    "Ridge":                "#1f77b4",
    "LogisticRegression":   "#ff7f0e",
    "RandomForest":         "#2ca02c",
    "LightGBM":            "#d62728",
    "MLP":                 "#9467bd",
    "LSTM":                "#8c564b",
    "Ensemble":            "#e377c2",
    "MomentumBaseline":    "#7f7f7f",
    "MeanReversionBaseline": "#bcbd22",
}


# ================================================================== #
#  Table 9: Adversarial Robustness
# ================================================================== #

def generate_adversarial_table(
    adv_results: dict,
    output_dir: str,
    primary_epsilon: float = 0.10,
) -> pd.DataFrame:
    """
    Generate Table 9: Adversarial robustness comparison.

    Shows for each model at the primary epsilon level:
      - Signal Flip Rate
      - Rank Correlation (clean vs adversarial)
      - Sharpe (clean)
      - Sharpe (adversarial)
      - Sharpe Drop %
      - Max DD (adversarial)
    """
    rows = []
    for model_name, eps_dict in adv_results.items():
        entry = eps_dict.get(primary_epsilon, {})
        if not entry:
            continue
        rows.append({
            "Model": model_name,
            "Signal Flip Rate": entry.get("signal_flip_rate", 0),
            "Rank Correlation": entry.get("rank_correlation", 0),
            "Sharpe (clean)": entry.get("sharpe_clean", 0),
            "Sharpe (adv)": entry.get("sharpe_adversarial", 0),
            "Sharpe Drop %": entry.get("sharpe_drop_%", 0),
            "Max DD (adv) %": entry.get("max_dd_adversarial", 0),
        })

    df = pd.DataFrame(rows).set_index("Model")
    # Sort by Sharpe Drop descending (most vulnerable first)
    df = df.sort_values("Sharpe Drop %", ascending=False)

    # Save
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    csv_path = os.path.join(output_dir, "tables", "table9_adversarial_robustness.csv")
    df.to_csv(csv_path)

    tex_path = os.path.join(output_dir, "tables", "table9_adversarial_robustness.tex")
    with open(tex_path, "w") as f:
        f.write(f"% Table 9: Adversarial Robustness at epsilon={primary_epsilon:.2f}\n")
        f.write(df.to_latex(float_format="%.3f"))

    print(f"  Table 9 saved: {csv_path}")
    return df


def generate_adversarial_multi_epsilon_table(
    adv_results: dict,
    output_dir: str,
    metric: str = "sharpe_adversarial",
) -> pd.DataFrame:
    """
    Pivot table: model × epsilon → metric value.
    """
    rows = []
    for model_name, eps_dict in adv_results.items():
        for eps, entry in eps_dict.items():
            rows.append({
                "Model": model_name,
                "Epsilon": eps,
                "Value": entry.get(metric, np.nan),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="Model", columns="Epsilon", values="Value")
    pivot = pivot.sort_index()

    csv_path = os.path.join(output_dir, "tables", "table9b_adversarial_sharpe_by_epsilon.csv")
    pivot.to_csv(csv_path)
    print(f"  Table 9b saved: {csv_path}")
    return pivot


# ================================================================== #
#  Table 10: Fuzzing Stress-Test Heatmap
# ================================================================== #

def generate_fuzzing_table(
    fuzzing_results: dict,
    output_dir: str,
    metric: str = "Sharpe (gross)",
) -> pd.DataFrame:
    """
    Generate Table 10: Fuzzing stress-test heatmap.
    Rows = models, Columns = scenarios, Values = Sharpe.
    """
    from scripts.robustness import compute_crash_heatmap
    df = compute_crash_heatmap(fuzzing_results, metric=metric)

    if df.empty:
        return df

    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    csv_path = os.path.join(output_dir, "tables", "table10_fuzzing_stress_test.csv")
    df.to_csv(csv_path)

    tex_path = os.path.join(output_dir, "tables", "table10_fuzzing_stress_test.tex")
    with open(tex_path, "w") as f:
        f.write("% Table 10: Fuzzing Stress-Test (Sharpe Ratio)\n")
        f.write(df.to_latex(float_format="%.3f"))

    # Also compute MaxDD version
    from scripts.robustness import compute_crash_heatmap as _hm
    df_dd = _hm(fuzzing_results, metric="Max DD")
    if not df_dd.empty:
        dd_path = os.path.join(output_dir, "tables", "table10b_fuzzing_max_drawdown.csv")
        df_dd.to_csv(dd_path)

    print(f"  Table 10 saved: {csv_path}")
    return df


# ================================================================== #
#  Table 11: Label Poisoning Resilience
# ================================================================== #

def generate_poisoning_table(
    poison_results: dict,
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 11: Label poisoning resilience.
    Rows = models, Columns = poison rates, Values = IC.
    """
    rows = []
    for model_name, rate_dict in poison_results.items():
        row = {"Model": model_name}
        for rate, entry in sorted(rate_dict.items()):
            pct_label = f"{rate*100:.0f}%"
            row[f"IC ({pct_label})"] = entry.get("IC", np.nan)
            row[f"Sharpe ({pct_label})"] = entry.get("Sharpe (gross)", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    csv_path = os.path.join(output_dir, "tables", "table11_label_poisoning.csv")
    df.to_csv(csv_path)

    tex_path = os.path.join(output_dir, "tables", "table11_label_poisoning.tex")
    with open(tex_path, "w") as f:
        f.write("% Table 11: Label Poisoning Resilience\n")
        f.write(df.to_latex(float_format="%.4f"))

    print(f"  Table 11 saved: {csv_path}")
    return df


# ================================================================== #
#  Table 12: Alpha Decay Half-Life
# ================================================================== #

def generate_alpha_decay_table(
    decay_results: dict,
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 12: Alpha decay half-life.
    Shows IC at each horizon + estimated half-life.
    """
    from scripts.robustness import compute_decay_halflife

    rows = []
    for model_name, df_decay in decay_results.items():
        if df_decay.empty:
            continue
        row = {"Model": model_name}
        for _, r in df_decay.iterrows():
            h = int(r["horizon"])
            row[f"IC ({h}d)"] = r["IC"]

        # Compute half-life
        hl = compute_decay_halflife(df_decay["IC"], df_decay["horizon"].tolist())
        row["Half-life (days)"] = hl if hl != np.inf else "∞"
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")

    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    csv_path = os.path.join(output_dir, "tables", "table12_alpha_decay.csv")
    df.to_csv(csv_path)

    tex_path = os.path.join(output_dir, "tables", "table12_alpha_decay.tex")
    with open(tex_path, "w") as f:
        f.write("% Table 12: Alpha Decay Half-Life\n")
        f.write(df.to_latex(float_format="%.4f"))

    print(f"  Table 12 saved: {csv_path}")
    return df


# ================================================================== #
#  Figure 10: Adversarial Sharpe Degradation Curves
# ================================================================== #

def plot_adversarial_sharpe_degradation(
    adv_results: dict,
    output_dir: str,
):
    """
    Plot: X-axis = epsilon, Y-axis = Sharpe ratio.
    Each line = one model.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Sharpe ratio
    ax = axes[0]
    for model_name, eps_dict in adv_results.items():
        epsilons = sorted(eps_dict.keys())
        sharpes = [eps_dict[e].get("sharpe_adversarial", np.nan) for e in epsilons]
        colour = MODEL_COLOURS.get(model_name, "#333333")
        ax.plot(epsilons, sharpes, marker="o", label=model_name, color=colour, linewidth=1.5)

    ax.set_xlabel("Perturbation Budget (ε × σ)")
    ax.set_ylabel("Adversarial Sharpe Ratio")
    ax.set_title("(a) Sharpe Ratio Under Adversarial Attack")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

    # Right panel: Sharpe drop %
    ax2 = axes[1]
    for model_name, eps_dict in adv_results.items():
        epsilons = sorted(eps_dict.keys())
        drops = [eps_dict[e].get("sharpe_drop_%", 0) for e in epsilons]
        colour = MODEL_COLOURS.get(model_name, "#333333")
        ax2.plot(epsilons, drops, marker="s", label=model_name, color=colour, linewidth=1.5)

    ax2.set_xlabel("Perturbation Budget (ε × σ)")
    ax2.set_ylabel("Sharpe Ratio Drop (%)")
    ax2.set_title("(b) Relative Sharpe Degradation")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "fig10_adversarial_sharpe.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Figure 10 saved: {fig_path}")


# ================================================================== #
#  Figure 11: Signal Flip Rate vs Epsilon
# ================================================================== #

def plot_signal_flip_rate(
    adv_results: dict,
    output_dir: str,
):
    """Bar chart: signal flip rate at each epsilon for each model."""
    fig, ax = plt.subplots(figsize=(10, 5))

    models = list(adv_results.keys())
    if not models:
        return

    all_eps = sorted(list(adv_results[models[0]].keys()))
    x = np.arange(len(all_eps))
    width = 0.8 / max(len(models), 1)

    for i, model_name in enumerate(models):
        flips = [adv_results[model_name].get(e, {}).get("signal_flip_rate", 0) for e in all_eps]
        colour = MODEL_COLOURS.get(model_name, "#333333")
        ax.bar(x + i * width, flips, width, label=model_name, color=colour, alpha=0.8)

    ax.set_xlabel("Perturbation Budget (ε × σ)")
    ax.set_ylabel("Signal Flip Rate")
    ax.set_title("Trading Signal Flip Rate Under Adversarial Perturbation")
    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels([f"{e:.2f}" for e in all_eps])
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "fig11_signal_flip_rate.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Figure 11 saved: {fig_path}")


# ================================================================== #
#  Figure 12: Fuzzing Crash Heatmap
# ================================================================== #

def plot_fuzzing_heatmap(
    fuzzing_results: dict,
    output_dir: str,
    metric: str = "Sharpe (gross)",
):
    """Heatmap: model × stress scenario → metric value."""
    from scripts.robustness import compute_crash_heatmap

    df = compute_crash_heatmap(fuzzing_results, metric=metric)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        df.astype(float),
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Model Robustness Under Synthetic Market Stress\n({metric})")
    ax.set_ylabel("Model")
    ax.set_xlabel("Stress Scenario")

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "fig12_fuzzing_heatmap.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Figure 12 saved: {fig_path}")

    # Also generate MaxDD heatmap
    df_dd = compute_crash_heatmap(fuzzing_results, metric="Max DD")
    if not df_dd.empty:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            df_dd.astype(float),
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            linewidths=0.5,
            ax=ax2,
        )
        ax2.set_title("Max Drawdown Under Synthetic Market Stress (%)")
        ax2.set_ylabel("Model")
        ax2.set_xlabel("Stress Scenario")
        fig2.tight_layout()
        fig2_path = os.path.join(output_dir, "figures", "fig12b_fuzzing_maxdd.pdf")
        fig2.savefig(fig2_path)
        plt.close(fig2)


# ================================================================== #
#  Figure 13: Label Poisoning IC Curve
# ================================================================== #

def plot_poisoning_curve(
    poison_results: dict,
    output_dir: str,
):
    """Line plot: X = poison rate, Y = IC or Sharpe for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # IC panel
    ax = axes[0]
    for model_name, rate_dict in poison_results.items():
        rates = sorted(rate_dict.keys())
        ics = [rate_dict[r].get("IC", np.nan) for r in rates]
        colour = MODEL_COLOURS.get(model_name, "#333333")
        ax.plot([r * 100 for r in rates], ics, marker="o", label=model_name,
                color=colour, linewidth=1.5)

    ax.set_xlabel("Label Poisoning Rate (%)")
    ax.set_ylabel("Information Coefficient (IC)")
    ax.set_title("(a) IC Under Label Poisoning")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Sharpe panel
    ax2 = axes[1]
    for model_name, rate_dict in poison_results.items():
        rates = sorted(rate_dict.keys())
        sharpes = [rate_dict[r].get("Sharpe (gross)", np.nan) for r in rates]
        colour = MODEL_COLOURS.get(model_name, "#333333")
        ax2.plot([r * 100 for r in rates], sharpes, marker="s", label=model_name,
                 color=colour, linewidth=1.5)

    ax2.set_xlabel("Label Poisoning Rate (%)")
    ax2.set_ylabel("Sharpe Ratio (gross)")
    ax2.set_title("(b) Sharpe Under Label Poisoning")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "fig13_label_poisoning.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Figure 13 saved: {fig_path}")


# ================================================================== #
#  Figure 14: Alpha Decay Curve
# ================================================================== #

def plot_alpha_decay_curve(
    decay_results: dict,
    output_dir: str,
):
    """
    Plot IC and ICIR vs prediction horizon for each model.
    Overlay exponential decay fit.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # IC panel
    ax = axes[0]
    for model_name, df_decay in decay_results.items():
        if df_decay.empty:
            continue
        colour = MODEL_COLOURS.get(model_name, "#333333")
        ax.plot(df_decay["horizon"], df_decay["IC"], marker="o", label=model_name,
                color=colour, linewidth=1.5)

        # Overlay exponential fit
        valid = df_decay["IC"] > 0
        if valid.sum() >= 3:
            h = df_decay.loc[valid, "horizon"].values
            ic = df_decay.loc[valid, "IC"].values
            try:
                log_ic = np.log(ic.clip(min=1e-8))
                slope, intercept = np.polyfit(h, log_ic, 1)
                h_fit = np.linspace(h.min(), h.max(), 50)
                ic_fit = np.exp(intercept + slope * h_fit)
                ax.plot(h_fit, ic_fit, "--", color=colour, alpha=0.5, linewidth=1)
            except Exception:
                pass

    ax.set_xlabel("Prediction Horizon (days)")
    ax.set_ylabel("Information Coefficient (IC)")
    ax.set_title("(a) Alpha Decay: IC vs Prediction Horizon")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

    # ICIR panel
    ax2 = axes[1]
    for model_name, df_decay in decay_results.items():
        if df_decay.empty:
            continue
        colour = MODEL_COLOURS.get(model_name, "#333333")
        ax2.plot(df_decay["horizon"], df_decay["ICIR"], marker="s", label=model_name,
                 color=colour, linewidth=1.5)

    ax2.set_xlabel("Prediction Horizon (days)")
    ax2.set_ylabel("ICIR")
    ax2.set_title("(b) ICIR vs Prediction Horizon")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "fig14_alpha_decay.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Figure 14 saved: {fig_path}")


# ================================================================== #
#  Composite summary figure
# ================================================================== #

def plot_robustness_summary(
    adv_results: dict,
    fuzzing_results: dict,
    poison_results: dict,
    decay_results: dict,
    output_dir: str,
):
    """
    Generate a 2×2 composite figure summarising all robustness findings.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) Adversarial Sharpe drop at eps=0.10
    ax = axes[0, 0]
    models = []
    drops = []
    for name, eps_dict in adv_results.items():
        entry = eps_dict.get(0.10, {})
        if entry:
            models.append(name)
            drops.append(entry.get("sharpe_drop_%", 0))
    if models:
        colours = [MODEL_COLOURS.get(m, "#333333") for m in models]
        bars = ax.barh(models, drops, color=colours, alpha=0.8)
        ax.set_xlabel("Sharpe Drop (%)")
        ax.set_title("(a) Adversarial Vulnerability (ε = 0.10σ)")
        ax.grid(True, alpha=0.3, axis="x")

    # (b) Fuzzing: Sharpe across scenarios for top 5 models
    ax2 = axes[0, 1]
    if fuzzing_results:
        from scripts.robustness import compute_crash_heatmap
        df_hm = compute_crash_heatmap(fuzzing_results, metric="Sharpe (gross)")
        if not df_hm.empty:
            sns.heatmap(df_hm.astype(float), annot=True, fmt=".2f",
                        cmap="RdYlGn", center=0, linewidths=0.5, ax=ax2,
                        cbar_kws={"shrink": 0.8})
            ax2.set_title("(b) Stress-Test Sharpe Heatmap")

    # (c) Label poisoning IC
    ax3 = axes[1, 0]
    for model_name, rate_dict in poison_results.items():
        rates = sorted(rate_dict.keys())
        ics = [rate_dict[r].get("IC", np.nan) for r in rates]
        colour = MODEL_COLOURS.get(model_name, "#333333")
        ax3.plot([r * 100 for r in rates], ics, marker="o", label=model_name,
                 color=colour, linewidth=1.5)
    ax3.set_xlabel("Poison Rate (%)")
    ax3.set_ylabel("IC")
    ax3.set_title("(c) Label Poisoning Resilience")
    if poison_results:
        ax3.legend(fontsize=7, loc="best")
    ax3.grid(True, alpha=0.3)

    # (d) Alpha decay
    ax4 = axes[1, 1]
    for model_name, df_decay in decay_results.items():
        if df_decay.empty:
            continue
        colour = MODEL_COLOURS.get(model_name, "#333333")
        ax4.plot(df_decay["horizon"], df_decay["IC"], marker="o", label=model_name,
                 color=colour, linewidth=1.5)
    ax4.set_xlabel("Horizon (days)")
    ax4.set_ylabel("IC")
    ax4.set_title("(d) Alpha Decay Curve")
    if decay_results:
        ax4.legend(fontsize=7, loc="best")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="black", linestyle="--", linewidth=0.8)

    fig.suptitle("Algorithmic Robustness Analysis — Summary", fontsize=15, y=1.01)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, "figures", "fig15_robustness_summary.pdf")
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Figure 15 (summary) saved: {fig_path}")
