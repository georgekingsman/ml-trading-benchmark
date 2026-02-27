"""
generate_defense_figures.py — Publication-quality defense figures (Figure 16 & 17).
生成论文级别的对抗训练防御可视化图表 (图 16 & 17)。

Figure 16: Defense Effectiveness Bar Chart
  - Grouped bars comparing SSR & Clean Sharpe: Standard vs Adversarial Training
  图16: 防御效果对比柱状图 (SSR + Clean Sharpe 分组对比)

Figure 17: Robustness Frontier Curve
  - Sharpe under attack vs epsilon, showing the "gap" = robustness gain
  图17: 鲁棒性前沿曲线 (Sharpe vs ε, 展示防御带来的"鲁棒性增益")

Usage:
    cd benchmark/
    python -m scripts.generate_defense_figures
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Publication Style ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Academic colour palette (colour-blind-friendly)
# 学术配色 (色盲友好)
C_STD   = "#c0392b"   # red — Standard (vulnerable)
C_ADV   = "#27ae60"   # green — Adversarial-trained (robust)
C_STD_L = "#e74c3c"   # lighter red (for secondary use)
C_ADV_L = "#2ecc71"   # lighter green

# Per-model line styles
MODEL_STYLE = {
    "LSTM": {"color_std": "#c0392b", "color_adv": "#2980b9",
             "marker_std": "o",      "marker_adv": "s"},
    "MLP":  {"color_std": "#8e44ad", "color_adv": "#27ae60",
             "marker_std": "D",      "marker_adv": "^"},
}


def load_defense_json(path: str = "reports/adversarial_defense_metrics.json") -> dict:
    """Load the adversarial defense JSON."""
    with open(path) as f:
        return json.load(f)


# ================================================================== #
#  Figure 16: Defense Effectiveness — Grouped Bar Chart
# ================================================================== #

def plot_defense_effectiveness(
    data: dict,
    output_dir: str = "reports/figures",
    filename: str = "fig16_defense_effectiveness.pdf",
) -> None:
    """
    Two-panel grouped bar chart:
      (a) Signal Stability Rate at ε=0.10σ
      (b) Clean Sharpe Ratio

    Narrative: Adversarial training makes models "rock-solid" AND can
    even *improve* clean-data performance (MLP case).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = ["LSTM", "MLP"]
    x = np.arange(len(models))
    width = 0.32

    # ── Panel (a): SSR at ε = 0.10σ ──
    ssr_std = [data[f"{m}_Standard"]["0.1"]["ssr"] * 100 for m in models]
    ssr_adv = [data[f"{m}_Adversarial"]["0.1"]["ssr"] * 100 for m in models]

    bars1 = ax1.bar(x - width/2, ssr_std, width,
                    label="Standard Training",
                    color=C_STD, alpha=0.85, edgecolor="white", linewidth=0.8)
    bars2 = ax1.bar(x + width/2, ssr_adv, width,
                    label="Adversarial Training",
                    color=C_ADV, alpha=0.85, edgecolor="white", linewidth=0.8)

    # Value labels — with improvement annotation
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        ax1.text(b1.get_x() + b1.get_width()/2., b1.get_height() + 0.8,
                 f'{b1.get_height():.1f}%', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=C_STD)
        ax1.text(b2.get_x() + b2.get_width()/2., b2.get_height() + 0.8,
                 f'{b2.get_height():.1f}%', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color="#1a7a3a")
        # Draw improvement arrow
        gain = ssr_adv[i] - ssr_std[i]
        mid_x = x[i]
        mid_y = max(b1.get_height(), b2.get_height()) + 5
        ax1.annotate(f'+{gain:.1f}pp',
                     xy=(mid_x, mid_y), fontsize=8, ha='center',
                     color='#2c3e50', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='#f0f0f0',
                               edgecolor='#bbb', alpha=0.9))

    ax1.set_ylabel("Signal Stability Rate (%)")
    ax1.set_title(r"(a) Signal Stability at $\varepsilon = 0.10\sigma$",
                  fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.set_ylim(50, 106)
    ax1.axhline(50, color='grey', linestyle=':', linewidth=0.6, alpha=0.5,
                label='Random baseline')

    # ── Panel (b): Clean Sharpe Ratio ──
    sharpe_std = [data[f"{m}_Standard"]["0.1"]["sharpe_clean"] for m in models]
    sharpe_adv = [data[f"{m}_Adversarial"]["0.1"]["sharpe_clean"] for m in models]

    bars3 = ax2.bar(x - width/2, sharpe_std, width,
                    label="Standard Training",
                    color=C_STD, alpha=0.85, edgecolor="white", linewidth=0.8)
    bars4 = ax2.bar(x + width/2, sharpe_adv, width,
                    label="Adversarial Training",
                    color=C_ADV, alpha=0.85, edgecolor="white", linewidth=0.8)

    for i, (b3, b4) in enumerate(zip(bars3, bars4)):
        ax2.text(b3.get_x() + b3.get_width()/2., b3.get_height() + 0.02,
                 f'{b3.get_height():.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color=C_STD)
        ax2.text(b4.get_x() + b4.get_width()/2., b4.get_height() + 0.02,
                 f'{b4.get_height():.3f}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold', color="#1a7a3a")
        # Highlight the MLP "surprise" — Sharpe doubling
        if models[i] == "MLP":
            ratio = sharpe_adv[i] / sharpe_std[i] if sharpe_std[i] != 0 else 0
            mid_x = x[i]
            mid_y = max(b3.get_height(), b4.get_height()) + 0.08
            ax2.annotate(f'×{ratio:.1f}',
                         xy=(mid_x, mid_y), fontsize=10, ha='center',
                         color='#c0392b', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef9e7',
                                   edgecolor='#f39c12', linewidth=1.5, alpha=0.95))

    ax2.set_ylabel("Sharpe Ratio (gross, clean data)")
    ax2.set_title("(b) Clean-Data Sharpe Ratio", fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.set_ylim(0, max(sharpe_adv) * 1.35)
    ax2.axhline(0, color='black', linewidth=0.5)

    fig.suptitle(
        "Adversarial Training Defense Effectiveness",
        fontsize=14, fontweight='bold', y=1.02,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    # Also save PNG for quick preview
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ Figure 16 (Defense Effectiveness) saved: {path}")


# ================================================================== #
#  Figure 17: Robustness Frontier Curve
# ================================================================== #

def plot_robustness_frontier(
    data: dict,
    output_dir: str = "reports/figures",
    filename: str = "fig17_robustness_frontier.pdf",
) -> None:
    """
    Two-panel figure:
      (a) Adversarial Sharpe vs ε — Standard vs Adversarial-trained
          Shows the "gap" = robustness gain for each model
      (b) SSR vs ε — how signal stability degrades with attack strength

    The gap between Standard (dashed) and Adversarial (solid) lines
    represents the robustness improvement at each perturbation level.
    Standard 和 Adversarial 之间的"开口"即为防御带来的鲁棒性增益。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    models = ["LSTM", "MLP"]

    for m in models:
        style = MODEL_STYLE[m]

        # Extract epsilon levels and metrics
        for training_type in ["Standard", "Adversarial"]:
            key = f"{m}_{training_type}"
            eps_dict = data[key]
            epsilons = sorted([float(e) for e in eps_dict.keys()])
            sharpes_adv = [eps_dict[str(e)]["sharpe_adv"] for e in epsilons]
            ssrs = [eps_dict[str(e)]["ssr"] * 100 for e in epsilons]
            sharpe_clean = eps_dict[str(epsilons[0])]["sharpe_clean"]

            # Prepend ε=0 (clean baseline)
            epsilons_plot = [0.0] + epsilons
            sharpes_plot = [sharpe_clean] + sharpes_adv
            ssrs_plot = [100.0] + ssrs

            if training_type == "Standard":
                ls, lw, alpha = "--", 1.5, 0.7
                color = style["color_std"]
                marker = style["marker_std"]
                label = f"{m} (Standard)"
            else:
                ls, lw, alpha = "-", 2.5, 1.0
                color = style["color_adv"]
                marker = style["marker_adv"]
                label = f"{m} (Adversarial)"

            ax1.plot(epsilons_plot, sharpes_plot, marker=marker,
                     linestyle=ls, linewidth=lw, alpha=alpha,
                     color=color, label=label, markersize=7, zorder=10 if training_type == "Adversarial" else 5)
            ax2.plot(epsilons_plot, ssrs_plot, marker=marker,
                     linestyle=ls, linewidth=lw, alpha=alpha,
                     color=color, label=label, markersize=7, zorder=10 if training_type == "Adversarial" else 5)

        # Shade the "robustness gap" between Standard and Adversarial
        eps_common = sorted([float(e) for e in data[f"{m}_Standard"].keys()])
        eps_plot = [0.0] + eps_common
        sharpe_std_line = [data[f"{m}_Standard"][str(eps_common[0])]["sharpe_clean"]] + \
                          [data[f"{m}_Standard"][str(e)]["sharpe_adv"] for e in eps_common]
        sharpe_adv_line = [data[f"{m}_Adversarial"][str(eps_common[0])]["sharpe_clean"]] + \
                          [data[f"{m}_Adversarial"][str(e)]["sharpe_adv"] for e in eps_common]

        fill_color = style["color_adv"]
        ax1.fill_between(eps_plot, sharpe_std_line, sharpe_adv_line,
                         alpha=0.08, color=fill_color, zorder=1)

    # ── Panel (a) formatting ──
    ax1.axhline(0, color="grey", linestyle="-", linewidth=0.8, alpha=0.4)
    ax1.set_xlabel(r"Perturbation Budget ($\varepsilon \times \sigma_{\mathrm{feature}}$)",
                   fontsize=11)
    ax1.set_ylabel("Sharpe Ratio", fontsize=11)
    ax1.set_title("(a) Sharpe Ratio Under PGD Attack", fontweight='bold')
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=9)

    # Red shading for negative Sharpe zone
    ylim = ax1.get_ylim()
    ax1.fill_between([0, 0.22], ylim[0], 0, color="red", alpha=0.03, zorder=0)
    ax1.set_ylim(ylim)

    # Annotate the robustness gap
    ax1.annotate("Robustness\nGain",
                 xy=(0.12, -2.5), fontsize=9, ha='center',
                 color='#2c3e50', fontstyle='italic',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#eaf2f8',
                           edgecolor='#5dade2', alpha=0.8))

    # ── Panel (b) formatting ──
    ax2.axhline(50, color="grey", linestyle=":", linewidth=0.8, alpha=0.5,
                label="Random baseline (50%)")
    ax2.set_xlabel(r"Perturbation Budget ($\varepsilon \times \sigma_{\mathrm{feature}}$)",
                   fontsize=11)
    ax2.set_ylabel("Signal Stability Rate (%)", fontsize=11)
    ax2.set_title("(b) Signal Stability Under Attack", fontweight='bold')
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=9)
    ax2.set_ylim(55, 102)

    fig.suptitle(
        "Robustness Frontier: Standard vs. Adversarial-Trained Models",
        fontsize=14, fontweight='bold', y=1.02,
    )
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  ✓ Figure 17 (Robustness Frontier) saved: {path}")


# ================================================================== #
#  Main
# ================================================================== #

def main():
    print("=" * 60)
    print("  Generating Publication-Quality Defense Figures")
    print("  生成论文级防御可视化图表")
    print("=" * 60)

    data = load_defense_json("reports/adversarial_defense_metrics.json")
    output_dir = "reports/figures"

    print("\n[1/2] Figure 16: Defense Effectiveness Bar Chart ...")
    plot_defense_effectiveness(data, output_dir)

    print("\n[2/2] Figure 17: Robustness Frontier Curve ...")
    plot_robustness_frontier(data, output_dir)

    print("\n" + "=" * 60)
    print("  All figures generated successfully!")
    print("  所有图表生成完毕！")
    print("=" * 60)


if __name__ == "__main__":
    main()
