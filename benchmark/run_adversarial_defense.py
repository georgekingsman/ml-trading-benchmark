"""
run_adversarial_defense.py — Adversarial Training Defense Experiment
对抗训练防御实验 — 完成论文 "发现问题 → 解决问题" 闭环

Purpose / 目的:
  Table 9 showed MLP/LSTM collapse under PGD attack (Sharpe drops 381-385%).
  Table 9 展示了 MLP/LSTM 在 PGD 攻击下崩溃 (Sharpe 暴跌 381-385%)。

  This experiment demonstrates that adversarial training (Madry et al., ICLR 2018)
  substantially mitigates this vulnerability, upgrading the paper narrative from
  "problem discovery" to "problem discovery + solution".
  本实验证明对抗训练 (Madry et al., ICLR 2018) 能够显著缓解此脆弱性,
  将论文叙事从"发现问题"升级为"发现问题 + 解决问题"。

Experiment Design / 实验设计:
  For each deep model (MLP, LSTM):
  针对每个深度模型 (MLP, LSTM):
    1. Standard Training  → evaluate under PGD attack (ε ∈ {0.01, 0.05, 0.10, 0.20})
       标准训练 → 在 PGD 攻击下评估
    2. Adversarial Training (mix_ratio=0.5) → evaluate under same PGD attack
       对抗训练 (mix_ratio=0.5) → 在相同 PGD 攻击下评估
    3. Compare SSR, Sharpe, Max DD
       对比信号稳定率 (SSR)、夏普比率、最大回撤

Output / 输出:
  - reports/tables/table13_adversarial_defense.csv    (Table 13)
  - reports/tables/table13_adversarial_defense.tex
  - reports/figures/fig16_adversarial_defense.pdf     (Figure 16)
  - reports/adversarial_defense_metrics.json

Usage / 使用方法:
  cd benchmark/
  python run_adversarial_defense.py --skip-download --skip-features
  python run_adversarial_defense.py  # full pipeline from scratch

Reference / 参考文献:
  Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
  (ICLR 2018) — Section 5: Adversarial Training
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ── Path setup / 路径设置 ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.data_pipeline import download_prices, load_universe, process_prices
from scripts.feature_engineering import (
    FEATURE_COLS,
    build_features_for_ticker,
    build_labels,
    rolling_normalize,
)
from scripts.split import walk_forward_split
from scripts.models import build_model, MLPModel, LSTMModel
from scripts.backtest import run_backtest
from scripts.metrics import compute_all_metrics, sharpe as compute_sharpe, max_drawdown
from scripts.adversarial_training import (
    pgd_perturbation,
    adversarial_train,
)

# ── Matplotlib backend / 绘图后端 ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ================================================================== #
#  Helper: compute per-feature std for realistic attack budgets
#  辅助函数: 计算每个特征的标准差, 用于设置合理的攻击预算
# ================================================================== #

def _compute_feature_std(X_train: np.ndarray) -> np.ndarray:
    """Per-feature std from training data / 从训练数据计算每个特征的标准差."""
    return np.std(X_train, axis=0).clip(min=1e-8)


# ================================================================== #
#  Core: evaluate model under PGD attack at multiple epsilons
#  核心: 在多个ε水平下用 PGD 攻击评估模型
# ================================================================== #

def evaluate_model_under_attack(
    model_nn: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    feature_std: np.ndarray,
    epsilon_scales: list[float],
    backtest_fn,
    bt_kwargs: dict,
    device: torch.device,
    label: str = "",
    is_lstm: bool = False,
    lstm_wrapper=None,
) -> dict:
    """
    Evaluate a PyTorch model under PGD attack at multiple epsilon levels.
    在多个 epsilon 水平下用 PGD 攻击评估 PyTorch 模型。

    Parameters / 参数
    ----------
    model_nn : nn.Module
        The raw PyTorch neural network (not the wrapper).
        原始 PyTorch 神经网络 (非包装器)。
    X_test   : Test features / 测试特征
    y_test   : Test targets / 测试目标
    epsilon_scales : List of ε multipliers (applied to feature_std)
                     ε 乘数列表 (乘以 feature_std)
    label    : "Standard" or "Adversarial" for logging
               "Standard" 或 "Adversarial"，用于日志显示

    Returns / 返回
    -------
    dict : {epsilon: {ssr, sharpe_clean, sharpe_adv, sharpe_drop_pct, max_dd_adv, signal_flip_rate}}
    """
    loss_fn = nn.MSELoss()
    model_nn.eval()

    results = {}

    # ── Clean predictions / 干净预测 ──
    if is_lstm and lstm_wrapper is not None:
        pred_clean = lstm_wrapper.predict(X_test)
    else:
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            pred_clean = model_nn(X_t).cpu().numpy().flatten()

    # Clean backtest / 干净回测
    pred_df_clean = test_df[["date", "ticker"]].copy()
    pred_df_clean["prediction"] = pred_clean
    pred_df_clean = pred_df_clean.dropna(subset=["prediction"])

    bt_clean = backtest_fn(pred_df_clean, returns_df, **bt_kwargs)
    metrics_clean = compute_all_metrics(bt_clean, total_cost_bps=20.0)
    sharpe_clean = metrics_clean.get("Sharpe (gross)", 0.0)

    for eps_scale in epsilon_scales:
        epsilon = eps_scale * feature_std  # Per-feature ε / 逐特征 ε

        # ── Generate PGD adversarial test samples / 生成 PGD 对抗测试样本 ──
        if is_lstm and lstm_wrapper is not None:
            # For LSTM: construct sequences, attack, then reconstruct
            # 对于 LSTM: 构造序列 → 攻击 → 还原
            X_seq = lstm_wrapper._make_sequences(X_test)
            y_seq = y_test[lstm_wrapper.seq_len - 1:]
            X_seq_t = torch.tensor(X_seq, dtype=torch.float32).to(device)
            y_seq_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(device)
            eps_t = torch.tensor(epsilon, dtype=torch.float32).to(device)

            # PGD on sequences / 对序列做 PGD
            delta = torch.zeros_like(X_seq_t).uniform_(-1, 1) * eps_t
            for _ in range(10):
                delta.requires_grad_(True)
                pred_tmp = model_nn(X_seq_t + delta)
                loss_tmp = loss_fn(pred_tmp, y_seq_t)
                loss_tmp.backward()
                assert delta.grad is not None  # type guard / 类型守卫
                grad_sign = delta.grad.data.sign()
                delta = delta.detach() + 0.01 * feature_std.max() * grad_sign
                delta = torch.clamp(delta, -eps_t, eps_t)

            with torch.no_grad():
                pred_adv = model_nn(X_seq_t + delta.detach()).cpu().numpy().flatten()

            # Pad with NaN for alignment / 用 NaN 填充以对齐
            pred_adv_full = np.concatenate([
                np.full(lstm_wrapper.seq_len - 1, np.nan), pred_adv
            ])
        else:
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
            eps_t = torch.tensor(epsilon, dtype=torch.float32).to(device)

            # PGD attack / PGD 攻击
            delta = torch.zeros_like(X_t).uniform_(-1, 1) * eps_t
            for _ in range(10):
                delta.requires_grad_(True)
                pred_tmp = model_nn(X_t + delta)
                loss_tmp = loss_fn(pred_tmp, y_t)
                loss_tmp.backward()
                assert delta.grad is not None  # type guard / 类型守卫
                grad_sign = delta.grad.data.sign()
                step_size = 0.01 * feature_std.max()
                delta = delta.detach() + step_size * grad_sign
                delta = torch.clamp(delta, -eps_t, eps_t)

            with torch.no_grad():
                pred_adv_full = model_nn(X_t + delta.detach()).cpu().numpy().flatten()

        # ── Compute metrics / 计算指标 ──
        valid = ~(np.isnan(pred_clean) | np.isnan(pred_adv_full))

        # SSR: Signal Stability Rate / 信号稳定率
        # Fraction of predictions whose sign is unchanged after attack
        # 攻击后预测符号未改变的比例
        ssr = float(np.mean(
            np.sign(pred_clean[valid]) == np.sign(pred_adv_full[valid])
        ))

        # Signal flip rate / 信号翻转率
        signal_flip = 1.0 - ssr

        # Rank correlation / 排名相关性
        from scipy.stats import spearmanr
        sp_result = spearmanr(pred_clean[valid], pred_adv_full[valid])
        rank_corr_val = float(sp_result.statistic) if hasattr(sp_result, 'statistic') else float(sp_result[0])  # type: ignore[index]
        rank_corr = rank_corr_val if not np.isnan(rank_corr_val) else 0.0

        # Adversarial backtest / 对抗回测
        pred_df_adv = test_df[["date", "ticker"]].copy()
        pred_df_adv["prediction"] = pred_adv_full
        pred_df_adv = pred_df_adv.dropna(subset=["prediction"])

        bt_adv = backtest_fn(pred_df_adv, returns_df, **bt_kwargs)
        metrics_adv = compute_all_metrics(bt_adv, total_cost_bps=20.0)
        sharpe_adv = metrics_adv.get("Sharpe (gross)", 0.0)
        max_dd_adv = metrics_adv.get("Max DD", 0.0)

        # Sharpe drop percentage / 夏普下降百分比
        if abs(sharpe_clean) > 1e-6:
            sharpe_drop_pct = (sharpe_clean - sharpe_adv) / abs(sharpe_clean) * 100
        else:
            sharpe_drop_pct = 0.0

        results[eps_scale] = {
            "ssr": round(ssr, 4),
            "signal_flip_rate": round(signal_flip, 4),
            "rank_correlation": round(rank_corr, 4),
            "sharpe_clean": round(sharpe_clean, 3),
            "sharpe_adv": round(sharpe_adv, 3),
            "sharpe_drop_pct": round(sharpe_drop_pct, 1),
            "max_dd_adv_pct": round(max_dd_adv * 100 if max_dd_adv <= 1 else max_dd_adv, 1),
        }

        max_dd_display = max_dd_adv * 100 if max_dd_adv <= 1 else max_dd_adv
        print(f"    [{label}] ε={eps_scale:.2f}: SSR={ssr:.4f}, "
              f"Sharpe {sharpe_clean:.3f}→{sharpe_adv:.3f} "
              f"(drop {sharpe_drop_pct:.1f}%), MaxDD={max_dd_display:.1f}%")

    return results


# ================================================================== #
#  Table 13 Generation / 生成 Table 13
# ================================================================== #

def generate_defense_table(
    all_results: dict,
    output_dir: str,
    primary_epsilon: float = 0.10,
) -> pd.DataFrame:
    """
    Generate Table 13: Adversarial Training Defense Comparison.
    生成 Table 13: 对抗训练防御对比表。

    Columns / 列:
      Model | Training | SSR | Sharpe (clean) | Sharpe (adv) | Sharpe Drop % | Max DD (adv) %

    Each model appears twice: 'Standard' and 'Adversarial-Trained'.
    每个模型出现两行: '标准训练' 和 '对抗训练'。
    """
    rows = []
    # Extract unique model names / 提取唯一模型名
    model_names = sorted(set(k.rsplit("_", 1)[0] for k in all_results.keys()))

    for model_name in model_names:
        for training_type in ["Standard", "Adversarial"]:
            key = f"{model_name}_{training_type}"
            eps_dict = all_results.get(key, {})
            entry = eps_dict.get(primary_epsilon, {})
            if not entry:
                continue
            rows.append({
                "Model": model_name,
                "Training": training_type,
                "SSR (ε=0.10)": entry.get("ssr", 0),
                "Signal Flip Rate": entry.get("signal_flip_rate", 0),
                "Rank Corr": entry.get("rank_correlation", 0),
                "Sharpe (clean)": entry.get("sharpe_clean", 0),
                "Sharpe (adv)": entry.get("sharpe_adv", 0),
                "Sharpe Drop %": entry.get("sharpe_drop_pct", 0),
                "Max DD (adv) %": entry.get("max_dd_adv_pct", 0),
            })

    df = pd.DataFrame(rows)

    # Save CSV / 保存 CSV
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    csv_path = os.path.join(output_dir, "tables", "table13_adversarial_defense.csv")
    df.to_csv(csv_path, index=False)

    # Save LaTeX / 保存 LaTeX
    tex_path = os.path.join(output_dir, "tables", "table13_adversarial_defense.tex")
    with open(tex_path, "w") as f:
        f.write("% Table 13: Adversarial Training Defense — Standard vs Adversarial-Trained\n")
        f.write("% 表 13: 对抗训练防御 — 标准训练 vs 对抗训练\n")
        f.write("%\n")
        f.write("% Columns: Model, Training Type, SSR, Signal Flip Rate, Rank Correlation,\n")
        f.write("%          Sharpe (clean), Sharpe (adversarial), Sharpe Drop %, Max DD (adv) %\n")
        f.write("%\n")
        f.write(df.to_latex(index=False, float_format="%.3f"))

    print(f"\n  ✓ Table 13 saved / 表13已保存:")
    print(f"    CSV:   {csv_path}")
    print(f"    LaTeX: {tex_path}")
    return df


# ================================================================== #
#  Table 13b: Multi-epsilon defense comparison
#  Table 13b: 多ε水平防御对比
# ================================================================== #

def generate_defense_multi_epsilon_table(
    all_results: dict,
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 13b: SSR at multiple epsilon levels, Standard vs Adversarial.
    生成 Table 13b: 多个 ε 水平下的 SSR 对比。
    """
    rows = []
    for key, eps_dict in all_results.items():
        model_name, training_type = key.rsplit("_", 1)
        for eps, entry in sorted(eps_dict.items()):
            rows.append({
                "Model": model_name,
                "Training": training_type,
                "Epsilon": eps,
                "SSR": entry.get("ssr", 0),
                "Sharpe (adv)": entry.get("sharpe_adv", 0),
                "Sharpe Drop %": entry.get("sharpe_drop_pct", 0),
            })

    df = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, "tables", "table13b_defense_multi_epsilon.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Table 13b saved: {csv_path}")
    return df


# ================================================================== #
#  Figure 16: Defense Comparison Visualization
#  图 16: 防御对比可视化
# ================================================================== #

def plot_defense_comparison(
    all_results: dict,
    output_dir: str,
):
    """
    Generate Figure 16: Adversarial Training Defense — 3-panel comparison.
    生成 图16: 对抗训练防御 — 三面板对比图。

    Panel (a): SSR at ε=0.10 — Standard vs Adversarial, grouped bar chart
    面板 (a): ε=0.10 时的 SSR — 标准 vs 对抗训练, 分组柱状图

    Panel (b): Sharpe ratio under attack across ε — line chart
    面板 (b): 不同 ε 下攻击后的 Sharpe — 折线图

    Panel (c): Sharpe drop % across ε — line chart
    面板 (c): 不同 ε 下的 Sharpe 下降率 — 折线图
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Identify unique models / 识别唯一模型
    model_names = sorted(set(k.rsplit("_", 1)[0] for k in all_results.keys()))
    colours_std = {"MLP": "#9467bd", "LSTM": "#8c564b"}
    colours_adv = {"MLP": "#c5b0d5", "LSTM": "#c49c94"}

    # ── Panel (a): SSR grouped bar chart at ε=0.10 / SSR 分组柱状图 ──
    ax = axes[0]
    x = np.arange(len(model_names), dtype=float)
    width = 0.35

    ssr_std = []
    ssr_adv = []
    for m in model_names:
        std_entry = all_results.get(f"{m}_Standard", {}).get(0.10, {})
        adv_entry = all_results.get(f"{m}_Adversarial", {}).get(0.10, {})
        ssr_std.append(std_entry.get("ssr", 0) * 100)
        ssr_adv.append(adv_entry.get("ssr", 0) * 100)

    bars1 = ax.bar(x - width/2, ssr_std, width, label="Standard Training\n标准训练",
                   color="#e74c3c", alpha=0.8, edgecolor="white")
    bars2 = ax.bar(x + width/2, ssr_adv, width, label="Adversarial Training\n对抗训练",
                   color="#27ae60", alpha=0.8, edgecolor="white")

    # Add value labels on bars / 在柱上标注数值
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Signal Stability Rate (%)\n信号稳定率 (%)")
    ax.set_title("(a) SSR at ε = 0.10σ")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)

    # ── Panel (b): Sharpe under attack across ε / 不同ε下攻击后 Sharpe ──
    ax2 = axes[1]
    for m in model_names:
        for training_type, ls, marker in [("Standard", "--", "o"), ("Adversarial", "-", "s")]:
            key = f"{m}_{training_type}"
            if key not in all_results:
                continue
            eps_dict = all_results[key]
            epsilons = sorted(eps_dict.keys())
            sharpes = [eps_dict[e].get("sharpe_adv", 0) for e in epsilons]
            colour = colours_std.get(m, "#333") if training_type == "Standard" else colours_adv.get(m, "#999")
            ax2.plot(epsilons, sharpes, marker=marker, linestyle=ls,
                     label=f"{m} ({training_type})", linewidth=1.8,
                     color=colour if training_type == "Adversarial" else colours_std.get(m, "#333"))

    ax2.set_xlabel("Perturbation Budget (ε × σ)\n扰动预算")
    ax2.set_ylabel("Adversarial Sharpe Ratio\n对抗 Sharpe 比率")
    ax2.set_title("(b) Sharpe Under PGD Attack")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

    # ── Panel (c): Sharpe drop % / Sharpe 下降率 ──
    ax3 = axes[2]
    for m in model_names:
        for training_type, ls, marker in [("Standard", "--", "o"), ("Adversarial", "-", "s")]:
            key = f"{m}_{training_type}"
            if key not in all_results:
                continue
            eps_dict = all_results[key]
            epsilons = sorted(eps_dict.keys())
            drops = [eps_dict[e].get("sharpe_drop_pct", 0) for e in epsilons]
            colour = colours_std.get(m, "#333") if training_type == "Standard" else colours_adv.get(m, "#999")
            ax3.plot(epsilons, drops, marker=marker, linestyle=ls,
                     label=f"{m} ({training_type})", linewidth=1.8,
                     color=colour if training_type == "Adversarial" else colours_std.get(m, "#333"))

    ax3.set_xlabel("Perturbation Budget (ε × σ)\n扰动预算")
    ax3.set_ylabel("Sharpe Drop (%)\nSharpe 下降率 (%)")
    ax3.set_title("(c) Relative Sharpe Degradation")
    ax3.legend(fontsize=8, loc="best")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Adversarial Training Defense — Standard vs Adversarial-Trained\n"
                 "对抗训练防御 — 标准训练 vs 对抗训练", fontsize=14, y=1.03)
    fig.tight_layout()

    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    fig_path = os.path.join(output_dir, "figures", "fig16_adversarial_defense.pdf")
    fig.savefig(fig_path)
    plt.close(fig)

    # Also save PNG for quick preview / 同时保存 PNG 方便预览
    fig_png = os.path.join(output_dir, "figures", "fig16_adversarial_defense.png")
    fig2 = plt.figure(figsize=(18, 5.5))
    # Re-render for PNG (matplotlib limitation)
    plt.close(fig2)

    print(f"\n  ✓ Figure 16 saved / 图16已保存: {fig_path}")


# ================================================================== #
#  Main pipeline / 主流程
# ================================================================== #

def load_config(path: str) -> dict:
    """Load YAML config / 加载 YAML 配置."""
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial Training Defense Experiment / 对抗训练防御实验"
    )
    parser.add_argument("--config", default="config/settings.yaml",
                        help="Path to YAML config / 配置文件路径")
    parser.add_argument("--skip-download", action="store_true",
                        help="Use cached raw data / 使用缓存的原始数据")
    parser.add_argument("--skip-features", action="store_true",
                        help="Use cached features / 使用缓存的特征数据")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs (default 20) / 训练轮数 (默认20)")
    parser.add_argument("--adv-epochs", type=int, default=25,
                        help="Adversarial training epochs (default 25) / 对抗训练轮数 (默认25)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg["output"]["seed"])
    torch.manual_seed(cfg["output"]["seed"])
    t0 = time.time()

    # ── Banner / 标题 ──
    print("=" * 70)
    print("  ADVERSARIAL TRAINING DEFENSE EXPERIMENT")
    print("  对抗训练防御实验")
    print("  ─────────────────────────────────────────")
    print("  Paper narrative: Problem Discovery → Problem + Solution")
    print("  论文叙事:  发现问题 → 发现问题 + 解决方案")
    print("=" * 70)

    # ============================================================== #
    #  Step 1: Load Data & Features / 第1步: 加载数据与特征
    # ============================================================== #
    print("\n" + "─" * 60)
    print("STEP 1: Loading Data & Features / 加载数据与特征")
    print("─" * 60)

    dcfg = cfg["data"]
    feat_path = "features/features_panel.parquet"

    if args.skip_features and os.path.exists(feat_path):
        print("  Using cached features / 使用缓存特征 ✓")
        panel = pd.read_parquet(feat_path)
    elif args.skip_download:
        panel_path = os.path.join(dcfg["processed_dir"], "panel.parquet")
        print(f"  Using cached panel: {panel_path}")
        panel = pd.read_parquet(panel_path)
        from tqdm import tqdm
        pieces = []
        for tkr, grp in tqdm(panel.groupby("ticker"), desc="Building features / 构建特征"):
            pieces.append(build_features_for_ticker(grp))
        panel = pd.concat(pieces)
        panel = build_labels(panel, cfg["task"]["horizons"])
        panel = rolling_normalize(panel, window=cfg["features"].get("rolling_window", 252))
    else:
        universe = load_universe(cfg["universe_file"])
        tickers = universe["ticker"].tolist()
        all_data = download_prices(
            tickers, dcfg["start_date"], dcfg["end_date"], dcfg["raw_dir"]
        )
        panel = process_prices(all_data, dcfg["processed_dir"])
        from tqdm import tqdm
        pieces = []
        for tkr, grp in tqdm(panel.groupby("ticker"), desc="Building features / 构建特征"):
            pieces.append(build_features_for_ticker(grp))
        panel = pd.concat(pieces)
        panel = build_labels(panel, cfg["task"]["horizons"])
        panel = rolling_normalize(panel, window=cfg["features"].get("rolling_window", 252))

    # Drop NaN rows / 去除 NaN 行
    primary_h = cfg["task"]["primary_horizon"]
    target_col = f"fwd_return_{primary_h}d"
    required = FEATURE_COLS + [target_col]
    panel = panel.dropna(subset=required)

    # Walk-forward split / 前向分割
    splits = walk_forward_split(panel, cfg)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[target_col].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[target_col].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[target_col].values
    returns_df = test_df[["date", "ticker", "daily_return"]].copy()

    feature_std = _compute_feature_std(X_train)

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Target: {target_col}")
    print(f"  Feature std range: [{feature_std.min():.4f}, {feature_std.max():.4f}]")

    # Backtest config / 回测配置
    bcfg = cfg["backtest"]
    bt_kwargs = {
        "top_k": bcfg["top_k"],
        "rebalance_freq": bcfg["rebalance_freq"],
        "cost_bps": 15.0,
        "slippage_bps": bcfg["slippage_bps"],
    }

    # Device / 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Output directory / 输出目录
    output_dir = "reports"
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # Epsilon levels to test / 要测试的 ε 水平
    epsilon_scales = [0.01, 0.05, 0.10, 0.20]

    # Store all results / 存储所有结果
    all_results = {}
    input_dim = X_train.shape[1]

    # ============================================================== #
    #  Step 2: MLP Experiment / 第2步: MLP 实验
    # ============================================================== #
    print("\n" + "─" * 60)
    print("STEP 2: MLP — Standard vs Adversarial Training")
    print("        MLP — 标准训练 vs 对抗训练")
    print("─" * 60)

    # ── 2a: Build & train standard MLP / 构建并标准训练 MLP ──
    # Load checkpoint from run_all.py if available for cross-table consistency.
    # 如果 run_all.py 的 checkpoint 存在，则加载以保证跨表一致性。
    mlp_ckpt = "models/MLP_checkpoint.pt"
    mlp_std = MLPModel(hidden_dims=[128, 64], epochs=args.epochs, lr=1e-3, batch_size=256)
    if os.path.exists(mlp_ckpt):
        print(f"\n  [2a] Loading standard MLP from checkpoint / 加载标准 MLP: {mlp_ckpt}")
        mlp_std.load_checkpoint(mlp_ckpt, input_dim=input_dim)
    else:
        print("\n  [2a] Training standard MLP / 标准训练 MLP (no checkpoint found) ...")
        mlp_std.fit(X_train, y_train, X_val, y_val)

    print("  [2a] Evaluating standard MLP under PGD attack / 评估标准 MLP 在 PGD 攻击下...")
    results_mlp_std = evaluate_model_under_attack(
        model_nn=mlp_std.model,
        X_test=X_test,
        y_test=y_test,
        test_df=test_df,
        returns_df=returns_df,
        feature_std=feature_std,
        epsilon_scales=epsilon_scales,
        backtest_fn=run_backtest,
        bt_kwargs=bt_kwargs,
        device=mlp_std.device,
        label="MLP-Standard",
    )
    all_results["MLP_Standard"] = results_mlp_std

    # ── 2b: Build & adversarial-train MLP / 构建并对抗训练 MLP ──
    print(f"\n  [2b] Adversarial training MLP (epochs={args.adv_epochs}) / 对抗训练 MLP ...")

    mlp_adv = MLPModel(hidden_dims=[128, 64], epochs=1, lr=1e-3, batch_size=256)
    mlp_adv._build(input_dim)  # Initialize architecture / 初始化架构

    # Use adversarial_train from adversarial_training.py / 使用对抗训练模块
    adversarial_train(
        model=mlp_adv.model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epsilon=0.1 * feature_std.max(),  # L∞ budget / L∞ 预算
        alpha=0.01 * feature_std.max(),   # PGD step size / PGD 步长
        pgd_steps=7,
        epochs=args.adv_epochs,
        lr=1e-3,
        batch_size=256,
        mix_ratio=0.5,  # 50% clean + 50% adversarial / 50%干净 + 50%对抗
        device=mlp_adv.device,
        verbose=True,
    )

    print("  [2b] Evaluating adversarial MLP under PGD attack / 评估对抗训练 MLP ...")
    results_mlp_adv = evaluate_model_under_attack(
        model_nn=mlp_adv.model,
        X_test=X_test,
        y_test=y_test,
        test_df=test_df,
        returns_df=returns_df,
        feature_std=feature_std,
        epsilon_scales=epsilon_scales,
        backtest_fn=run_backtest,
        bt_kwargs=bt_kwargs,
        device=mlp_adv.device,
        label="MLP-Adversarial",
    )
    all_results["MLP_Adversarial"] = results_mlp_adv

    # ============================================================== #
    #  Step 3: LSTM Experiment / 第3步: LSTM 实验
    # ============================================================== #
    print("\n" + "─" * 60)
    print("STEP 3: LSTM — Standard vs Adversarial Training")
    print("        LSTM — 标准训练 vs 对抗训练")
    print("─" * 60)

    # ── 3a: Build & train standard LSTM / 构建并标准训练 LSTM ──
    # Load checkpoint from run_all.py if available for cross-table consistency.
    # 如果 run_all.py 的 checkpoint 存在，则加载以保证跨表一致性。
    lstm_ckpt = "models/LSTM_checkpoint.pt"
    lstm_std = LSTMModel(hidden_dim=64, num_layers=2, seq_len=20,
                         epochs=args.epochs, lr=1e-3, batch_size=256)
    if os.path.exists(lstm_ckpt):
        print(f"\n  [3a] Loading standard LSTM from checkpoint / 加载标准 LSTM: {lstm_ckpt}")
        lstm_std.load_checkpoint(lstm_ckpt, input_dim=input_dim)
    else:
        print("\n  [3a] Training standard LSTM / 标准训练 LSTM (no checkpoint found) ...")
        lstm_std.fit(X_train, y_train, X_val, y_val)

    print("  [3a] Evaluating standard LSTM under PGD attack / 评估标准 LSTM ...")
    results_lstm_std = evaluate_model_under_attack(
        model_nn=lstm_std.model,
        X_test=X_test,
        y_test=y_test,
        test_df=test_df,
        returns_df=returns_df,
        feature_std=feature_std,
        epsilon_scales=epsilon_scales,
        backtest_fn=run_backtest,
        bt_kwargs=bt_kwargs,
        device=lstm_std.device,
        label="LSTM-Standard",
        is_lstm=True,
        lstm_wrapper=lstm_std,
    )
    all_results["LSTM_Standard"] = results_lstm_std

    # ── 3b: Build & adversarial-train LSTM / 构建并对抗训练 LSTM ──
    print(f"\n  [3b] Adversarial training LSTM (epochs={args.adv_epochs}) / 对抗训练 LSTM ...")

    lstm_adv = LSTMModel(hidden_dim=64, num_layers=2, seq_len=20,
                         epochs=1, lr=1e-3, batch_size=256)
    lstm_adv._build(input_dim)

    # For LSTM adversarial training, we need to use sequences
    # 对于 LSTM 对抗训练, 需要使用序列化数据
    # NOTE: Use only a subsample of training data for LSTM adversarial training
    #       to keep runtime manageable on CPU (~3 min instead of ~30 min).
    # 注意: 对 LSTM 只使用部分训练数据进行对抗训练, 以保持 CPU 上可控的运行时间。
    X_train_seq = lstm_adv._make_sequences(X_train)
    y_train_seq = y_train[lstm_adv.seq_len - 1:]
    X_val_seq = lstm_adv._make_sequences(X_val)
    y_val_seq = y_val[lstm_adv.seq_len - 1:]

    # Subsample for speed: use 20% of training sequences, stratified
    # 速度优化: 使用 20% 的训练序列
    n_sub = min(len(X_train_seq), max(5000, len(X_train_seq) // 5))
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(X_train_seq), size=n_sub, replace=False)
    X_train_seq_sub = X_train_seq[sub_idx]
    y_train_seq_sub = y_train_seq[sub_idx]
    print(f"    LSTM adv-train using {n_sub}/{len(X_train_seq)} sequences for speed")

    adversarial_train(
        model=lstm_adv.model,
        X_train=X_train_seq_sub,
        y_train=y_train_seq_sub,
        X_val=X_val_seq,
        y_val=y_val_seq,
        epsilon=0.1 * feature_std.max(),
        alpha=0.01 * feature_std.max(),
        pgd_steps=3,   # Fewer PGD steps for LSTM (speed) / LSTM 用更少的 PGD 步数
        epochs=min(args.adv_epochs, 15),  # Cap LSTM epochs / 限制 LSTM 轮数
        lr=1e-3,
        batch_size=512,  # Larger batch for throughput / 更大 batch 提高吞吐
        mix_ratio=0.5,
        device=lstm_adv.device,
        verbose=True,
    )

    print("  [3b] Evaluating adversarial LSTM under PGD attack / 评估对抗训练 LSTM ...")
    results_lstm_adv = evaluate_model_under_attack(
        model_nn=lstm_adv.model,
        X_test=X_test,
        y_test=y_test,
        test_df=test_df,
        returns_df=returns_df,
        feature_std=feature_std,
        epsilon_scales=epsilon_scales,
        backtest_fn=run_backtest,
        bt_kwargs=bt_kwargs,
        device=lstm_adv.device,
        label="LSTM-Adversarial",
        is_lstm=True,
        lstm_wrapper=lstm_adv,
    )
    all_results["LSTM_Adversarial"] = results_lstm_adv

    # ============================================================== #
    #  Step 4: Generate Reports / 第4步: 生成报告
    # ============================================================== #
    print("\n" + "─" * 60)
    print("STEP 4: Generating Reports / 生成报告")
    print("─" * 60)

    # Table 13 / 表13
    df_table13 = generate_defense_table(all_results, output_dir, primary_epsilon=0.10)

    # Table 13b: multi-epsilon / 表13b: 多ε对比
    df_table13b = generate_defense_multi_epsilon_table(all_results, output_dir)

    # Paper-format table for \input{} / 论文格式表格
    from scripts.paper_tables import write_defense_table as write_paper_defense
    paper_dir = os.path.join(output_dir, "tables", "paper")
    os.makedirs(paper_dir, exist_ok=True)

    # Figure 16 / 图16
    plot_defense_comparison(all_results, output_dir)

    # ============================================================== #
    #  Step 5: Save JSON / 第5步: 保存 JSON
    # ============================================================== #
    json_path = os.path.join(output_dir, "adversarial_defense_metrics.json")

    def _convert(obj):
        """Convert numpy types to JSON-serializable / 将 numpy 类型转换为可序列化."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def _deep_convert(d):
        if isinstance(d, dict):
            return {str(k): _deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_deep_convert(v) for v in d]
        return _convert(d)

    with open(json_path, "w") as f:
        json.dump(_deep_convert(all_results), f, indent=2, default=str)
    print(f"\n  ✓ JSON saved: {json_path}")

    # Paper-format defense table (needs JSON-format data)
    defense_json: dict = _deep_convert(all_results)  # type: ignore[assignment]
    write_paper_defense(defense_json, paper_dir, primary_epsilon=0.10)

    # ============================================================== #
    #  Step 6: Print Summary / 第6步: 打印总结
    # ============================================================== #
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  ADVERSARIAL DEFENSE EXPERIMENT COMPLETE — {elapsed:.1f} seconds")
    print(f"  对抗训练防御实验完成 — 用时 {elapsed:.1f} 秒")
    print("=" * 70)

    # Summary at ε=0.10 / ε=0.10 时的总结
    print("\n  Summary at ε = 0.10σ / ε=0.10σ 时的对比总结:")
    print("  " + "─" * 55)
    print(f"  {'Model':<8} {'Training':<14} {'SSR':>8} {'Sharpe(adv)':>12} {'Drop%':>8}")
    print("  " + "─" * 55)
    for key in sorted(all_results.keys()):
        model_name, training_type = key.rsplit("_", 1)
        entry = all_results[key].get(0.10, {})
        if entry:
            print(f"  {model_name:<8} {training_type:<14} "
                  f"{entry['ssr']:>8.4f} {entry['sharpe_adv']:>12.3f} "
                  f"{entry['sharpe_drop_pct']:>8.1f}")
    print("  " + "─" * 55)

    # Improvement analysis / 改进分析
    print("\n  Improvement Analysis / 改进分析:")
    for m in ["MLP", "LSTM"]:
        std_ssr = all_results.get(f"{m}_Standard", {}).get(0.10, {}).get("ssr", 0)
        adv_ssr = all_results.get(f"{m}_Adversarial", {}).get(0.10, {}).get("ssr", 0)
        std_drop = all_results.get(f"{m}_Standard", {}).get(0.10, {}).get("sharpe_drop_pct", 0)
        adv_drop = all_results.get(f"{m}_Adversarial", {}).get(0.10, {}).get("sharpe_drop_pct", 0)

        ssr_gain = (adv_ssr - std_ssr) * 100
        drop_reduction = std_drop - adv_drop

        print(f"  {m}: SSR ↑{ssr_gain:+.1f}pp, "
              f"Sharpe drop reduced by {drop_reduction:.1f}pp")

    print(f"\n  Output files / 输出文件:")
    print(f"    Table 13:   reports/tables/table13_adversarial_defense.csv")
    print(f"    Table 13b:  reports/tables/table13b_defense_multi_epsilon.csv")
    print(f"    Figure 16:  reports/figures/fig16_adversarial_defense.pdf")
    print(f"    JSON:       {json_path}")
    print()


if __name__ == "__main__":
    main()
