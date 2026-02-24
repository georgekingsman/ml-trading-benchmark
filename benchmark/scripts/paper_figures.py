"""
paper_figures.py — Publication-quality figures for adversarial robustness paper.
论文级可视化模块 — 生成对抗鲁棒性论文所需的核心图表。

Generates three key figures from existing experiment data:
从已有实验数据生成三张核心图:
  Figure A / 图A: "Collapse Curve" (崩溃曲线)     — Sharpe/SSR degradation vs epsilon
                                                      夏普比率/信号稳定率 随 ε 退化
  Figure B / 图B: "Attack Case Study" (攻击案例)   — Clean vs adversarial price & position
                                                      干净 vs 对抗样本的价格序列与持仓对比
  Figure C / 图C: "Saliency Heatmap" (显著性热力图) — Input gradient attribution (∂L/∂x)
                                                      输入梯度归因分析

Also includes an adversarial training defense module (Section 4: "Closing the Loop").
另含对抗训练防御模块（第4节："闭环"）。

Usage / 使用方法:
    cd benchmark/
    python -m scripts.paper_figures                  # all figures / 生成全部图表
    python -m scripts.paper_figures --only collapse  # just the collapse curve / 仅崩溃曲线
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Publication Style / 论文排版样式 ──────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",          # Serif font, matches LaTeX / 衬线字体，与 LaTeX 一致
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,              # High resolution for print / 印刷级高分辨率
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,        # Remove top/right spines / 去掉上右边框
    "axes.spines.right": False,
})

# Colour palette — distinct for DL vs traditional
# 配色方案 — 深度学习 vs 传统模型用不同色系区分
# Warm colours (red/pink) = vulnerable / 暖色 (红/粉) = 脆弱
# Cool colours (blue/green) = robust   / 冷色 (蓝/绿) = 鲁棒
PALETTE = {
    "LSTM":                 "#d62728",   # red — most vulnerable / 红 — 最脆弱
    "MLP":                  "#e377c2",   # pink — very vulnerable / 粉 — 很脆弱
    "LightGBM":            "#2ca02c",   # green — robust / 绿 — 鲁棒
    "RandomForest":         "#17becf",   # cyan / 青
    "LinearRegression":     "#1f77b4",   # blue — most robust / 蓝 — 最鲁棒
    "Ridge":                "#aec7e8",   # light blue / 淡蓝
    "LogisticRegression":   "#ff7f0e",   # orange / 橙
    "Ensemble":            "#9467bd",   # purple / 紫
    "MomentumBaseline":    "#7f7f7f",   # grey / 灰
    "MeanReversionBaseline": "#bcbd22", # olive / 橄榄
}

# Marker shapes per model / 每个模型的标记形状
MARKER = {
    "LSTM": "D", "MLP": "^", "LightGBM": "s", "RandomForest": "P",
    "LinearRegression": "o", "Ridge": "v", "LogisticRegression": "X",
    "MomentumBaseline": "<", "MeanReversionBaseline": ">",
}


# ================================================================== #
#  Helper: Load cached robustness metrics
#  辅助函数: 加载缓存的鲁棒性实验结果
# ================================================================== #

def load_robustness_json(path: str = "reports/robustness_metrics.json") -> dict:
    """Load the JSON produced by run_robustness.py / 读取鲁棒性实验生成的 JSON。"""
    with open(path) as f:
        return json.load(f)


# ================================================================== #
#  Figure A: The Collapse Curve (Paper core figure)
#  图 A: 崩溃曲线 — 论文核心卖点
# ================================================================== #

def plot_collapse_curve(
    data: dict | None = None,
    json_path: str = "reports/robustness_metrics.json",
    output_dir: str = "reports/figures",
    filename: str = "fig_collapse_curve.pdf",
    highlight_models: list[str] | None = None,
) -> None:
    """
    Plot the "Collapse Curve": how Sharpe and SSR diverge as ε increases.
    绘制 "崩溃曲线": 随着扰动强度 ε 增大，各模型 Sharpe Ratio 与 SSR 的分化。

    Left panel  / 左面板: Adversarial Sharpe Ratio vs ε
    Right panel / 右面板: Signal Stability Rate (SSR) vs ε

    Deep-learning models (LSTM/MLP) drawn with thick lines to highlight collapse;
    传统模型用细线。深度模型 (LSTM/MLP) 用粗线突出其断崖式下跌。

    Parameters / 参数
    ----------
    data : dict, optional
        Pre-loaded adversarial results dict / 预加载的对抗实验字典。
    json_path : str
        Path to robustness_metrics.json / JSON 路径。
    output_dir : str
        Directory for saving output PDF / PDF 输出目录。
    filename : str
        Output filename / 输出文件名。
    highlight_models : list[str], optional
        Models to highlight (default: LSTM, MLP) / 要高亮的模型。
    """
    if data is None:
        data = load_robustness_json(json_path).get("adversarial", {})
    assert data is not None  # guaranteed by guard above / 上方守卫保证非 None

    if highlight_models is None:
        highlight_models = ["LSTM", "MLP"]

    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left: Sharpe Ratio / 左面板: 夏普比率 ──
    for model_name, eps_dict in data.items():
        epsilons = sorted([float(e) for e in eps_dict.keys()])
        sharpes = [eps_dict[str(e) if str(e) in eps_dict else f"{e}"]["sharpe_adversarial"]
                   for e in epsilons]
        # Prepend clean baseline at ε=0 / 在 ε=0 处添加干净基准
        clean_sharpe = eps_dict[str(epsilons[0])]["sharpe_clean"]
        epsilons = [0.0] + epsilons
        sharpes = [clean_sharpe] + sharpes

        # Thick lines for highlighted (deep) models / 深度模型用粗线
        lw = 2.5 if model_name in highlight_models else 1.2
        alpha = 1.0 if model_name in highlight_models else 0.65
        zorder = 10 if model_name in highlight_models else 5
        colour = PALETTE.get(model_name, "#333333")
        marker = MARKER.get(model_name, "o")

        ax1.plot(epsilons, sharpes, marker=marker, label=model_name,
                 color=colour, linewidth=lw, alpha=alpha, markersize=6, zorder=zorder)

    ax1.axhline(0, color="grey", linestyle="--", linewidth=0.8, zorder=1)
    ax1.set_xlabel(r"Perturbation Budget ($\epsilon \times \sigma_{feature}$)")
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("(a) Adversarial Sharpe Ratio")
    ax1.legend(loc="lower left", framealpha=0.9, ncol=2)

    # Red shading in negative Sharpe zone / 负 Sharpe 区域红色底纹
    ax1.fill_between([0.04, 0.55], -5, 0, color="red", alpha=0.04, zorder=0)
    ax1.set_ylim(bottom=min(-5, ax1.get_ylim()[0]))

    # ── Right: SSR (Signal Stability Rate) / 右面板: 信号稳定率 ──
    for model_name, eps_dict in data.items():
        epsilons = sorted([float(e) for e in eps_dict.keys()])
        # SSR = 1 - flip_rate / 信号稳定率 = 1 - 翻转率
        ssrs = [1.0 - eps_dict[str(e) if str(e) in eps_dict else f"{e}"]["signal_flip_rate"]
                for e in epsilons]
        # Prepend ε=0: SSR=1.0 by definition / ε=0 时 SSR=1.0
        epsilons = [0.0] + epsilons
        ssrs = [1.0] + ssrs

        lw = 2.5 if model_name in highlight_models else 1.2
        alpha = 1.0 if model_name in highlight_models else 0.65
        zorder = 10 if model_name in highlight_models else 5
        colour = PALETTE.get(model_name, "#333333")
        marker = MARKER.get(model_name, "o")

        ax2.plot(epsilons, ssrs, marker=marker, label=model_name,
                 color=colour, linewidth=lw, alpha=alpha, markersize=6, zorder=zorder)

    ax2.axhline(0.5, color="grey", linestyle=":", linewidth=0.8, label="Random baseline")
    ax2.set_xlabel(r"Perturbation Budget ($\epsilon \times \sigma_{feature}$)")
    ax2.set_ylabel("Signal Stability Rate (SSR)")
    ax2.set_title("(b) Signal Stability Under Attack")
    ax2.legend(loc="lower left", framealpha=0.9, ncol=2)
    ax2.set_ylim(0.55, 1.02)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    fig.suptitle(
        "Performance Collapse Under Adversarial Perturbation",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Collapse curve saved / 崩溃曲线已保存: {path}")


# ================================================================== #
#  Figure B: Attack Case Study (Single-ETF visualization)
#  图 B: 攻击案例可视化 — 展示"AI 的愚蠢"
# ================================================================== #

def plot_attack_case_study(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    pred_clean: np.ndarray,
    pred_adv: np.ndarray,
    feature_names: list[str] | None = None,
    window: tuple[int, int] = (0, 30),
    output_dir: str = "reports/figures",
    filename: str = "fig_attack_case_study.pdf",
    title_suffix: str = "",
) -> None:
    """
    Attack sample visualization: show "the AI's foolishness".
    攻击样本可视化: 展示"AI 的愚蠢"。

    Upper panel / 上图: Original vs adversarial feature series (nearly overlapping → imperceptible).
                        原始特征序列 vs 攻击后序列 (几乎重合 → 扰动隐蔽)
    Lower panel / 下图: Model position direction flips between clean and adversarial.
                        模型在原始/攻击数据上的持仓方向翻转

    Parameters / 参数
    ----------
    X_clean : (N, F)  clean features / 干净特征
    X_adv   : (N, F)  adversarial features / 对抗特征
    pred_clean : (N,)  clean predictions / 干净预测
    pred_adv   : (N,)  adversarial predictions / 对抗预测
    feature_names : list[str]  feature name list / 特征名列表
    window : (start, end)  time window to display / 展示的时间窗口
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_clean.shape[1])]

    s, e = window
    e = min(e, len(X_clean))
    t = np.arange(s, e)

    # Select top-3 most perturbed features for display
    # 选择扰动最大的 3 个特征来展示
    perturbation = np.abs(X_adv[s:e] - X_clean[s:e]).mean(axis=0)
    top_feat_idx = np.argsort(perturbation)[-3:][::-1]

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[1.2, 1],
                             sharex=True, gridspec_kw={"hspace": 0.08})

    # ── Upper: Feature Time Series / 上图: 特征时间序列 ──
    ax_top = axes[0]
    for rank, idx in enumerate(top_feat_idx):
        fname = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        if rank == 0:
            # Primary feature — full plot / 主特征 — 完整绘制
            ax_top.plot(t, X_clean[s:e, idx], color="#1f77b4", linewidth=1.5,
                        label=f"Clean: {fname}")
            ax_top.plot(t, X_adv[s:e, idx], color="#d62728", linewidth=1.5,
                        linestyle="--", alpha=0.85, label=f"Adversarial: {fname}")
            # Shade the perturbation region / 填充扰动区域
            ax_top.fill_between(t, X_clean[s:e, idx], X_adv[s:e, idx],
                                color="#d62728", alpha=0.12, label="Perturbation δ")
        else:
            # Secondary features — lighter / 次要特征 — 淡色
            ax_top.plot(t, X_clean[s:e, idx], color="#aec7e8", linewidth=0.8, alpha=0.5)
            ax_top.plot(t, X_adv[s:e, idx], color="#ff9896", linewidth=0.8,
                        linestyle="--", alpha=0.5)

    ax_top.set_ylabel("Feature Value (z-scored)")
    ax_top.set_title(
        "Input Features: Clean vs Adversarial" + (f"  ({title_suffix})" if title_suffix else ""),
        fontsize=12,
    )
    ax_top.legend(loc="upper right", fontsize=8)

    # ── Lower: Position / Signal Direction / 下图: 持仓方向 ──
    ax_bot = axes[1]
    pos_clean = np.sign(pred_clean[s:e])
    pos_adv = np.sign(pred_adv[s:e])

    # Identify signal flips / 识别信号翻转点
    flipped = pos_clean != pos_adv

    ax_bot.step(t, pos_clean, where="mid", color="#1f77b4", linewidth=1.8,
                label="Clean Signal")
    ax_bot.step(t, pos_adv, where="mid", color="#d62728", linewidth=1.8,
                linestyle="--", label="Adversarial Signal")

    # Mark flip points with red × / 用红色 × 标记翻转点
    if flipped.any():
        flip_t = t[flipped]
        ax_bot.scatter(flip_t, pos_adv[flipped], color="#d62728", zorder=5,
                       s=60, marker="x", linewidths=2, label="Signal Flip")

    ax_bot.set_ylim(-1.5, 1.5)
    ax_bot.set_yticks([-1, 0, 1])
    ax_bot.set_yticklabels(["Short", "Flat", "Long"])
    ax_bot.set_xlabel("Time Step (days)")
    ax_bot.set_ylabel("Position")
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.axhline(0, color="grey", linewidth=0.5)

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Attack case study saved / 攻击案例图已保存: {path}")


# ================================================================== #
#  Figure C: Saliency Heatmap (Input-Gradient Attribution)
#  图 C: 梯度归因热力图 — 揭示"为什么模型会崩溃"
# ================================================================== #

def compute_input_saliency(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] | None = None,
    aggregate: str = "mean_abs",
) -> np.ndarray:
    """
    Compute input gradient saliency |∂L/∂x| to reveal feature sensitivity.
    计算输入梯度显著性 |∂L/∂x|，揭示模型对各特征的敏感度。

    The absolute gradient magnitude indicates how much each feature
    contributes to loss change — i.e., which features are "attack surfaces".
    梯度绝对值表示每个特征对 loss 变化的贡献 — 即哪些特征是"攻击面"。

    Parameters / 参数
    ----------
    model : PyTorch model (MLPModel or LSTMModel from benchmark)
        Must have .model (nn.Module) and .device attributes.
        需要具有 .model 和 .device 属性的 benchmark 模型。
    X : np.ndarray, shape (N, F) or prepared sequences (N, seq_len, F)
        Input features / 输入特征。
    y : np.ndarray, shape (N,)
        Targets / 目标值。
    aggregate : str
        'mean_abs' — mean of |gradient| across samples / 对所有样本求 |梯度| 均值 (default)
        'raw'      — return the full gradient tensor / 返回完整的梯度张量

    Returns / 返回
    -------
    saliency : np.ndarray
        Shape (F,) for MLP, or (seq_len, F) for LSTM.
    """
    import torch
    import torch.nn as nn

    # Check if model is LSTM / 检查是否为 LSTM
    is_lstm = hasattr(model, "seq_len")

    if is_lstm:
        # LSTM requires sequence construction / LSTM 需要构建序列
        X_input = model._make_sequences(X)
        y_input = y[model.seq_len - 1:]
    else:
        X_input = X
        y_input = y

    # Enable gradient tracking on input / 对输入开启梯度追踪
    X_t = torch.tensor(X_input, dtype=torch.float32, requires_grad=True).to(model.device)
    y_t = torch.tensor(y_input, dtype=torch.float32).unsqueeze(1).to(model.device)

    model.model.eval()
    pred = model.model(X_t)
    loss = nn.MSELoss()(pred, y_t)
    loss.backward()

    # Extract gradient / 提取梯度: MLP=(N,F), LSTM=(N,seq_len,F)
    grad = X_t.grad.data.cpu().numpy()  # type: ignore[union-attr]

    if aggregate == "raw":
        return grad
    elif aggregate == "mean_abs":
        # Average |gradient| across all samples / 对所有样本的 |梯度| 取均值
        return np.mean(np.abs(grad), axis=0)  # (F,) or (seq_len, F)
    else:
        raise ValueError(f"Unknown aggregate method: {aggregate}")


def plot_saliency_heatmap(
    saliency: np.ndarray,
    feature_names: list[str] | None = None,
    seq_len: int | None = None,
    model_name: str = "Model",
    output_dir: str = "reports/figures",
    filename: str = "fig_saliency_heatmap.pdf",
    top_k_features: int | None = None,
) -> None:
    """
    Plot gradient attribution heatmap.
    绘制梯度归因热力图。

    For MLP: horizontal bar chart of |∂L/∂x| per feature.
    对于 MLP: 横向柱状图展示每个特征的 |∂L/∂x|。
    For LSTM: 2D heatmap with X=time step, Y=feature.
    对于 LSTM: 二维热力图，X轴=时间步, Y轴=特征。

    Parameters / 参数
    ----------
    saliency : np.ndarray
        Shape (F,) for MLP or (seq_len, F) for LSTM.
    feature_names : list[str]
    model_name : str — model identifier for title / 模型名 (用于标题)
    top_k_features : int, optional
        Only display top-K most sensitive features / 仅展示 top-K 最敏感的特征。
    """
    os.makedirs(output_dir, exist_ok=True)

    if saliency.ndim == 1:
        # ── MLP: Horizontal bar chart / MLP: 横向柱状图 ──
        n_features = len(saliency)
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(n_features)]

        # Sort descending by importance / 按重要性降序排列
        order = np.argsort(saliency)[::-1]
        if top_k_features:
            order = order[:top_k_features]

        fig, ax = plt.subplots(figsize=(10, max(4, len(order) * 0.35)))
        names = [feature_names[i] for i in order]
        vals = saliency[order]

        # Colour gradient: darker = higher importance / 颜色渐变: 越深=越重要
        colors = plt.cm.YlOrRd(vals / vals.max() * 0.8 + 0.1)  # type: ignore
        ax.barh(range(len(order)), vals, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel(r"Mean $|\nabla_x \mathcal{L}|$")
        ax.set_title(f"Input Gradient Saliency — {model_name}")

    elif saliency.ndim == 2:
        # ── LSTM: 2D Heatmap / LSTM: 二维热力图 ──
        sl, n_features = saliency.shape
        if feature_names is None:
            feature_names = [f"feat_{i}" for i in range(n_features)]

        # Filter to top-K features by total saliency / 按总显著性筛选 top-K 特征
        if top_k_features and top_k_features < n_features:
            total_sal = saliency.sum(axis=0)
            top_idx = np.argsort(total_sal)[-top_k_features:][::-1]
            saliency = saliency[:, top_idx]
            feature_names = [feature_names[i] for i in top_idx]
            n_features = top_k_features

        fig, ax = plt.subplots(figsize=(max(8, sl * 0.5), max(4, n_features * 0.4)))

        # Time labels: t-19, t-18, ..., t-0 (most recent)
        # 时间标签: t-19, t-18, ..., t-0 (最近)
        time_labels = [f"t-{sl - 1 - i}" for i in range(sl)]

        df_hm = pd.DataFrame(saliency.T, index=feature_names, columns=time_labels)

        sns.heatmap(
            df_hm,
            cmap="YlOrRd",
            annot=True if (sl <= 20 and n_features <= 13) else False,
            fmt=".3f" if (sl <= 20 and n_features <= 13) else "",
            linewidths=0.3,
            ax=ax,
            cbar_kws={"label": r"$|\nabla_x \mathcal{L}|$", "shrink": 0.8},
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Feature")
        ax.set_title(f"Input Gradient Saliency — {model_name}\n"
                     r"(brighter $\Rightarrow$ model is more sensitive to perturbation)"
                     "\n"
                     r"(越亮 $\Rightarrow$ 模型对该特征越敏感)")

    else:
        raise ValueError(f"Unexpected saliency shape: {saliency.shape}")

    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Saliency heatmap saved / 显著性热力图已保存: {path}")


def plot_multi_model_saliency(
    models: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_names: list[str] | None = None,
    output_dir: str = "reports/figures",
    filename: str = "fig_saliency_comparison.pdf",
) -> None:
    """
    Multi-model saliency comparison: side-by-side top-K feature sensitivity.
    多模型显著性对比: 并排展示不同模型的 top-K 特征敏感度。

    Answers the question: "Which features fool each model?"
    回答问题: "哪些特征最容易欺骗各个模型?"

    Parameters / 参数
    ----------
    models : dict[str, model]
        {name: trained_model} — only PyTorch models (MLP/LSTM).
        仅支持 PyTorch 模型 (MLP/LSTM)。
    """
    if model_names is None:
        model_names = [n for n in models if n in ("LSTM", "MLP")]

    os.makedirs(output_dir, exist_ok=True)
    n_models = len(model_names)
    if n_models == 0:
        print("  ⚠ No PyTorch models found for saliency comparison")
        return

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        model = models[name]
        sal = compute_input_saliency(model, X, y, feature_names)

        if sal.ndim == 2:
            # LSTM: collapse time dimension / LSTM: 对时间维度求均值
            sal_1d = sal.mean(axis=0)
        else:
            sal_1d = sal

        order = np.argsort(sal_1d)[::-1]
        top_n = min(10, len(order))
        order = order[:top_n]

        names = [feature_names[i] for i in order]
        vals = sal_1d[order]

        colors = [PALETTE.get(name, "#333333")] * top_n
        ax.barh(range(top_n), vals, color=colors, alpha=0.75,
                edgecolor="white", linewidth=0.3)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel(r"Mean $|\nabla_x \mathcal{L}|$")
        ax.set_title(f"{name}")

    fig.suptitle(
        "Input Gradient Attribution — Which Features Fool the Model?\n"
        "输入梯度归因 — 哪些特征最容易欺骗模型?",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(output_dir, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ Multi-model saliency saved / 多模型显著性对比图已保存: {path}")


# ================================================================== #
#  Bonus: Performance Collapse Summary Table (for LaTeX)
#  附加: 性能崩溃汇总表 (LaTeX 格式)
# ================================================================== #

def generate_collapse_table_latex(
    data: dict | None = None,
    json_path: str = "reports/robustness_metrics.json",
    output_dir: str = "reports/tables",
    filename: str = "table_collapse_summary.tex",
) -> pd.DataFrame:
    """
    Generate a concise LaTeX table highlighting deep model collapse.
    生成一张精简的 LaTeX 表，突出深度模型的"崩溃"。

    Columns / 列: Model | Sharpe(clean) | Sharpe(ε=0.05) | Sharpe(ε=0.1) |
                   SSR(ε=0.1) | Collapse%(ε=0.1) | Category
    """
    if data is None:
        data = load_robustness_json(json_path).get("adversarial", {})
    assert data is not None  # guaranteed by guard above / 上方守卫保证非 None

    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for model_name, eps_dict in data.items():
        e01 = eps_dict.get("0.01", {})
        e05 = eps_dict.get("0.05", {})
        e10 = eps_dict.get("0.1", {})

        clean_s = e01.get("sharpe_clean", e05.get("sharpe_clean", 0))
        s05 = e05.get("sharpe_adversarial", np.nan)
        s10 = e10.get("sharpe_adversarial", np.nan)
        flip10 = e10.get("signal_flip_rate", 0)
        drop10 = e10.get("sharpe_drop_%", 0)

        # Categorise model family / 模型分类
        if model_name in ("LSTM", "MLP"):
            cat = "Deep Learning"
        elif model_name in ("MomentumBaseline", "MeanReversionBaseline"):
            cat = "Heuristic"
        else:
            cat = "Traditional ML"

        rows.append({
            "Model": model_name,
            "Category": cat,
            "Sharpe (clean)": round(clean_s, 3),
            r"Sharpe ($\epsilon$=0.05)": round(s05, 3) if not np.isnan(s05) else "—",
            r"Sharpe ($\epsilon$=0.10)": round(s10, 3) if not np.isnan(s10) else "—",
            "SSR (ε=0.10)": f"{(1 - flip10) * 100:.1f}%",
            "Collapse %": f"{drop10:.1f}%",
        })

    df = pd.DataFrame(rows)
    # Sort: Deep Learning first (most dramatic), then Traditional, then Heuristic
    # 排序: 深度学习在前 (效果最戏剧化)
    cat_order = {"Deep Learning": 0, "Traditional ML": 1, "Heuristic": 2}
    df["_sort"] = df["Category"].map(cat_order)
    df = df.sort_values(["_sort", "Collapse %"], ascending=[True, False]).drop(columns="_sort")

    # Save CSV / 保存 CSV
    csv_path = os.path.join(output_dir, filename.replace(".tex", ".csv"))
    df.to_csv(csv_path, index=False)

    # Save LaTeX / 保存 LaTeX
    tex_path = os.path.join(output_dir, filename)
    with open(tex_path, "w") as f:
        f.write("% Auto-generated collapse summary table\n")
        f.write("% 自动生成的崩溃汇总表\n")
        f.write("% Key finding: LSTM and MLP show catastrophic Sharpe collapse\n")
        f.write("% 核心发现: LSTM 和 MLP 展现灾难性的夏普比率崩溃\n")
        f.write(df.set_index("Model").to_latex(
            escape=False,
            column_format="l" + "c" * (len(df.columns) - 1),
        ))
    print(f"  ✓ Collapse table saved / 崩溃汇总表已保存: {tex_path}")
    return df


# ================================================================== #
#  CLI Entry Point / 命令行入口
# ================================================================== #

def main():
    """
    Command-line interface for generating paper figures.
    命令行接口，用于生成论文图表。
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Paper figures for adversarial robustness / 生成论文图表"
    )
    parser.add_argument("--json", default="reports/robustness_metrics.json",
                        help="Path to robustness_metrics.json / JSON 路径")
    parser.add_argument("--output", default="reports/figures",
                        help="Output directory for figures / 图表输出目录")
    parser.add_argument("--only", choices=["collapse", "table", "all"],
                        default="all")
    args = parser.parse_args()

    data = load_robustness_json(args.json)
    adv = data.get("adversarial", {})

    if args.only in ("collapse", "all"):
        print("\n[1/2] Generating Collapse Curve / 生成崩溃曲线 ...")
        plot_collapse_curve(adv, output_dir=args.output)

    if args.only in ("table", "all"):
        print("\n[2/2] Generating Collapse Summary Table / 生成崩溃汇总表 ...")
        generate_collapse_table_latex(adv, output_dir=os.path.join(
            os.path.dirname(args.output), "tables"))

    print("\nDone. Figures ready for paper submission.")
    print("完成。图表已就绪，可用于论文投稿。")


if __name__ == "__main__":
    main()
