"""
paper_tables.py — Generate paper-format tabular .tex files that match the
exact column layout, headers, and formatting of the EAAI manuscript.
论文表格生成器 — 生成与 EAAI 手稿完全一致的列布局、表头和格式的 .tex 表格文件。

Each function produces a standalone tabular environment (no \\begin{table},
caption, or label) so that the paper .tex can simply \\input{} them.
每个函数生成一个独立的 tabular 环境（不包含 \\begin{table}、caption 或 label），
以便论文 .tex 文件直接通过 \\input{} 引入。

Output directory / 输出目录:  reports/tables/paper/

Mapping (paper table → pipeline source) / 映射关系（论文表格 → 流水线数据源）:
  tab:main          ← table1  (9 列, 斜体基准线, $-$ 负号)
  tab:regime        ← table3  (5 列, 宽格式, $+$/$-$ 符号)
  tab:sensitivity   ← table7 + table8 合并 (10 列, | 分隔符)
  tab:longonly      ← table5  (7 列, multicolumn 多空/纯多)
  tab:adversarial   ← table9  (7 列, 双行表头)
  tab:fuzzing       ← table10 (7 列, 数学模式场景表头)
  tab:poisoning     ← table11 (6 列, 仅 IC 指标)
  tab:adv_defense   ← table13 (8 列, 加粗最优, 按模型分组)
"""

from __future__ import annotations

import os
import re
from typing import Any

import numpy as np
import pandas as pd


# ── Helpers / 辅助函数 ─────────────────────────────────────────────── #

def _ensure_dir(path: str) -> None:
    """Create parent directories if they don't exist.
    如果父目录不存在则自动创建。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _fmt_neg(val: float, fmt: str = ".3f", math_wrap: bool = False) -> str:
    """Format a number with $-$ for negatives, matching paper style.
    将负数格式化为 LaTeX 数学负号，匹配论文排版风格。

    math_wrap=False  → e.g.  $-$0.134   (用于大多数表格 / used in most tables)
    math_wrap=True   → e.g.  $-0.134$   (用于对抗/压力测试表 / used in adversarial/fuzzing)
    """
    if pd.isna(val):
        return "---"
    s = f"{abs(val):{fmt}}"
    if val < 0:
        if math_wrap:
            return f"$-{s}$"
        else:
            return f"$-${s}"
    return s


def _fmt_pos_neg(val: float, fmt: str = ".2f") -> str:
    """Format with explicit $+$/$-$ sign (for regime table).
    使用显式 $+$/$-$ 符号格式化（用于市场区间表）。"""
    if pd.isna(val):
        return "---"
    sign = "$+$" if val >= 0 else "$-$"
    return f"{sign}{abs(val):{fmt}}"


def _fmt_plain(val: float, fmt: str = ".3f") -> str:
    """Plain number formatting (with $-$ for negatives).
    普通数字格式化（负数使用 $-$ 前缀）。"""
    if pd.isna(val):
        return "---"
    if val < 0:
        return f"$-${abs(val):{fmt}}"
    return f"{val:{fmt}}"


def _escape_model(name: str) -> str:
    """Escape underscores in model names for LaTeX.
    转义模型名中的下划线以兼容 LaTeX。"""
    return name.replace("_", r"\_")


def _abbreviate_model(name: str) -> str:
    """Shorten model names as in the paper.
    按论文约定缩写模型名称。"""
    abbrev = {
        "MeanReversionBaseline": "MeanReversionBase.",
        "MeanReversion": "MeanReversion",
    }
    return abbrev.get(name, name)


# Default output directory for paper-format tables
# 论文格式表格的默认输出目录
PAPER_DIR = os.path.join("reports", "tables", "paper")


# ================================================================== #
#  Table 1 / 表1: tab:main
#  Main benchmark results / 主要基准测试结果 (9 columns / 9列)
# ================================================================== #

def write_main_table(
    all_metrics: dict[str, dict[str, Any]],
    output_dir: str = PAPER_DIR,
) -> str:
    """
    Generate the main benchmark results table (tab:main).
    生成主要基准测试结果表（tab:main）。

    Paper columns (9) / 论文列 (9列):
      Model | CAGR(g,%) | Sharpe(gross) | Sharpe(net) | Max DD(%)
            | IC | ICIR | Sharpe CI lo | Sharpe CI hi

    Baselines (BuyAndHold_SPY, EqualWeight) are in \\textit{}, separated by \\midrule.
    基准策略 (BuyAndHold_SPY, EqualWeight) 使用 \\textit{} 斜体显示，并用 \\midrule 分隔。
    Missing IC/ICIR shown as '---'. / 缺失的 IC/ICIR 显示为 '---'。
    Negatives use $-$ style. / 负数使用 $-$ 格式。
    """
    # Column keys expected in all_metrics / all_metrics 中预期的列名映射
    col_map = {
        "CAGR (gross)": "cagr",
        "Sharpe (gross)": "sharpe_g",
        "Sharpe (net)": "sharpe_n",
        "Max DD": "maxdd",
        "IC": "ic",
        "ICIR": "icir",
        "Sharpe CI_lo": "ci_lo",
        "Sharpe CI_hi": "ci_hi",
    }

    # Build sorted dataframe / 构建并按 Sharpe 降序排列的 DataFrame
    df = pd.DataFrame(all_metrics).T
    df.index.name = "Model"
    sort_col = "Sharpe (gross)" if "Sharpe (gross)" in df.columns else df.columns[0]
    df = df.sort_values(sort_col, ascending=False)

    baselines = {"BuyAndHold_SPY", "EqualWeight"}  # 基准策略集合
    baseline_rows: list[str] = []  # 基准策略行
    model_rows: list[str] = []     # 模型策略行

    for model, row in df.iterrows():
        # Extract each metric / 提取各项指标
        cagr = row.get("CAGR (gross)", np.nan)
        sg = row.get("Sharpe (gross)", np.nan)
        sn = row.get("Sharpe (net)", np.nan)
        mdd = row.get("Max DD", np.nan)
        ic = row.get("IC", np.nan)
        icir = row.get("ICIR", np.nan)
        ci_lo = row.get("Sharpe CI_lo", np.nan)
        ci_hi = row.get("Sharpe CI_hi", np.nan)

        # Format each cell / 格式化每个单元格
        cells = [
            _fmt_neg(cagr, ".2f"),
            _fmt_neg(sg, ".3f"),
            _fmt_neg(sn, ".2f"),
            _fmt_plain(abs(mdd) if not pd.isna(mdd) else np.nan, ".2f"),
            "---" if pd.isna(ic) else _fmt_neg(ic, ".3f"),
            "---" if pd.isna(icir) else _fmt_neg(icir, ".3f"),
            _fmt_neg(ci_lo, ".2f"),
            _fmt_neg(ci_hi, ".2f"),
        ]

        name_tex = _escape_model(str(model))
        # Pad model name for alignment / 填充模型名以对齐
        name_tex = f"{name_tex:20s}"

        if str(model) in baselines:
            # Wrap every cell in \textit for baselines
            # 基准策略每个单元格用 \textit 包裹为斜体
            line = r"\textit{" + name_tex.strip() + "}"
            line += " & " + " & ".join(
                [r"\textit{" + c + "}" for c in cells]
            )
            baseline_rows.append(line + r" \\")
        else:
            line = name_tex + " & " + " & ".join(cells)
            model_rows.append(line + r" \\")

    # Assemble tabular environment / 组装 tabular 环境
    lines: list[str] = []
    lines.append(r"\begin{tabular}{l r r r r r r r r}")
    lines.append(r"\toprule")
    # Two-row header / 双行表头
    lines.append(
        r"Model & \multicolumn{1}{c}{CAGR} & \multicolumn{1}{c}{Sharpe} "
        r"& \multicolumn{1}{c}{Sharpe} & \multicolumn{1}{c}{Max DD} "
        r"& \multicolumn{1}{c}{IC} & \multicolumn{1}{c}{ICIR} "
        r"& \multicolumn{2}{c}{Sharpe 95\% CI} \\"
    )
    lines.append(
        r"      & \multicolumn{1}{c}{(g, \%)} & \multicolumn{1}{c}{(gross)} "
        r"& \multicolumn{1}{c}{(net)} & \multicolumn{1}{c}{(\%)} "
        r"& & & \multicolumn{1}{c}{lo} & \multicolumn{1}{c}{hi} \\"
    )
    lines.append(r"\midrule")
    # Baseline rows first, then model rows / 先输出基准策略行，再输出模型行
    for r in baseline_rows:
        lines.append(r)
    lines.append(r"\midrule")
    for r in model_rows:
        lines.append(r)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Write to file / 写入文件
    path = os.path.join(output_dir, "tab_main.tex")
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("% Auto-generated paper-format tabular for tab:main\n")
        f.write("% 自动生成的论文格式表格 tab:main\n")
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Paper table tab:main → {path}")
    return path


# ================================================================== #
#  Table 2 / 表2: tab:regime
#  Regime analysis / 市场区间分析 (5 columns / 5列)
# ================================================================== #

def write_regime_table(
    regime_results: dict[str, pd.DataFrame],
    output_dir: str = PAPER_DIR,
) -> str:
    """
    Generate the regime analysis table (tab:regime).
    生成市场区间分析表（tab:regime）。

    Paper: wide format — Model | COVID Crash | Recovery | Rate Hikes | Normalisation
    论文：宽格式 — 模型 | 新冠崩盘 | 复苏期 | 加息期 | 正常化
    Values = Sharpe(gross) with $+$/$-$ signs.
    数值 = 毛夏普比率，带 $+$/$-$ 符号。
    """
    # Regime display order / 市场区间显示顺序
    regime_order = ["COVID Crash", "Recovery", "Rate Hikes", "Normalisation"]
    # Model display order / 模型显示顺序
    model_order = [
        "BuyAndHold_SPY", "EqualWeight",
        "LogisticRegression", "RandomForest", "LightGBM",
        "MLP", "LSTM", "Ensemble", "MomentumBaseline",
    ]

    # Build wide-format data / 构建宽格式数据 {模型: {区间: Sharpe}}
    wide: dict[str, dict[str, float]] = {}
    for model_name, rdf in regime_results.items():
        wide[model_name] = {}
        for _, row in rdf.iterrows():
            regime = row.get("Regime", "")
            sharpe = row.get("Sharpe (gross)", np.nan)
            wide[model_name][regime] = sharpe

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Model & COVID Crash & Recovery & Rate Hikes & Normalisation \\")
    lines.append(r"\midrule")

    for model in model_order:
        if model not in wide:
            continue
        name_tex = _escape_model(model)
        cells = []
        for regime in regime_order:
            val = wide[model].get(regime, np.nan)
            cells.append(_fmt_pos_neg(val))
        lines.append(f"{name_tex:22s} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Write to file / 写入文件
    path = os.path.join(output_dir, "tab_regime.tex")
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("% Auto-generated paper-format tabular for tab:regime\n")
        f.write("% 自动生成的论文格式表格 tab:regime (市场区间分析)\n")
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Paper table tab:regime → {path}")
    return path


# ================================================================== #
#  Table 3 / 表3: tab:sensitivity
#  Rebal + Top-K merged / 再平衡频率 + Top-K 合并 (10 columns / 10列)
# ================================================================== #

def write_sensitivity_table(
    rebal_results: dict[str, dict[int, dict]],
    topk_results: dict[str, dict[int, dict]],
    output_dir: str = PAPER_DIR,
) -> str:
    """
    Generate the sensitivity analysis table (tab:sensitivity).
    生成敏感性分析表（tab:sensitivity）。

    Paper: Model | rebal 1 5 10 20 | topK 3 5 10 15 20
    论文：模型 | 再平衡频率 1 5 10 20 | Top-K 3 5 10 15 20
    Column spec: l r r r r | r r r r r  (with | separator / 带 | 分隔符)
    Values = Sharpe(gross) with $-$ for negatives.
    数值 = 毛夏普比率，负数使用 $-$ 格式。
    """
    rebal_freqs = [1, 5, 10, 20]       # 再平衡频率（天）
    topk_vals = [3, 5, 10, 15, 20]     # Top-K 选股数量

    # Determine models present in both / 确定同时存在于两组结果中的模型（按论文顺序）
    paper_order = [
        "LogisticRegression", "LightGBM", "MLP", "LSTM", "Ensemble",
    ]
    models = [m for m in paper_order if m in rebal_results and m in topk_results]
    # Add any remaining models / 添加其余模型
    for m in rebal_results:
        if m in topk_results and m not in models:
            models.append(m)

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l r r r r | r r r r r}")
    lines.append(r"\toprule")
    # Multicolumn headers / 多列表头
    lines.append(
        r" & \multicolumn{4}{c|}{Rebalance Frequency (days)} "
        r"& \multicolumn{5}{c}{Top-$K$} \\"
    )
    lines.append(
        r"Model & 1 & 5 & 10 & 20 & 3 & 5 & 10 & 15 & 20 \\"
    )
    lines.append(r"\midrule")

    for model in models:
        name_tex = _escape_model(model)
        cells = []
        # Rebalance frequency columns / 再平衡频率列
        for freq in rebal_freqs:
            entry = rebal_results.get(model, {}).get(freq, {})
            val = entry.get("Sharpe (gross)", np.nan)
            cells.append(_fmt_neg(val, ".2f"))
        # Top-K columns / Top-K 列
        for k in topk_vals:
            entry = topk_results.get(model, {}).get(k, {})
            val = entry.get("Sharpe (gross)", np.nan)
            cells.append(_fmt_neg(val, ".2f"))

        lines.append(f"{name_tex:20s} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Write to file / 写入文件
    path = os.path.join(output_dir, "tab_sensitivity.tex")
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("% Auto-generated paper-format tabular for tab:sensitivity\n")
        f.write("% 自动生成的论文格式表格 tab:sensitivity (敏感性分析)\n")
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Paper table tab:sensitivity → {path}")
    return path


# ================================================================== #
#  Table 4 / 表4: tab:longonly
#  Long-Short vs Long-Only / 多空对比纯多头 (7 columns / 7列)
# ================================================================== #

def write_longonly_table(
    ls_metrics: dict[str, dict],
    lo_metrics: dict[str, dict],
    output_dir: str = PAPER_DIR,
) -> str:
    """
    Generate the Long-Short vs Long-Only comparison table (tab:longonly).
    生成多空 vs 纯多头策略对比表（tab:longonly）。

    Paper columns: Model | Sharpe(g) Sharpe(n) CAGR(g,%) | Sharpe(g) Sharpe(n) CAGR(g,%)
    论文列：模型 | 夏普(毛) 夏普(净) CAGR(毛,%) | 夏普(毛) 夏普(净) CAGR(毛,%)
    with multicolumn Long-Short / Long-Only headers.
    使用 multicolumn 显示 Long-Short / Long-Only 组标题。
    BuyAndHold_SPY long-only shows '---'.
    BuyAndHold_SPY 的纯多头列显示 '---'。
    """
    # Model display order / 模型显示顺序
    model_order = [
        "LogisticRegression", "LightGBM", "MLP", "LSTM",
        "Ensemble", "BuyAndHold_SPY",
    ]

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l r r r r r r}")
    lines.append(r"\toprule")
    # Multicolumn group headers / 多列分组表头
    lines.append(
        r" & \multicolumn{3}{c}{Long-Short} "
        r"& \multicolumn{3}{c}{Long-Only} \\"
    )
    lines.append(
        r"Model & Sharpe(g) & Sharpe(n) & CAGR(g,\%) "
        r"& Sharpe(g) & Sharpe(n) & CAGR(g,\%) \\"
    )
    lines.append(r"\midrule")

    for model in model_order:
        if model not in ls_metrics:
            continue
        lsm = ls_metrics[model]
        lom = lo_metrics.get(model, {})
        name_tex = _escape_model(model)

        # Long-Short metrics / 多空策略指标
        ls_sg = _fmt_neg(lsm.get("Sharpe (gross)", np.nan), ".2f")
        ls_sn = _fmt_neg(lsm.get("Sharpe (net)", np.nan), ".2f")
        ls_cagr = _fmt_neg(lsm.get("CAGR (gross)", np.nan), ".2f")

        # BuyAndHold_SPY has no Long-Only data / BuyAndHold_SPY 无纯多头数据
        if model == "BuyAndHold_SPY":
            lo_sg = "---"
            lo_sn = "---"
            lo_cagr = "---"
        else:
            # Long-Only metrics / 纯多头策略指标
            lo_sg = _fmt_neg(lom.get("Sharpe (gross)", np.nan), ".2f")
            lo_sn = _fmt_neg(lom.get("Sharpe (net)", np.nan), ".2f")
            lo_cagr = _fmt_neg(lom.get("CAGR (gross)", np.nan), ".2f")

        cells = [ls_sg, ls_sn, ls_cagr, lo_sg, lo_sn, lo_cagr]
        lines.append(f"{name_tex:20s} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Write to file / 写入文件
    path = os.path.join(output_dir, "tab_longonly.tex")
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("% Auto-generated paper-format tabular for tab:longonly\n")
        f.write("% 自动生成的论文格式表格 tab:longonly (多空 vs 纯多头)\n")
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Paper table tab:longonly → {path}")
    return path


# ================================================================== #
#  Table 5 / 表5: tab:adversarial
#  Adversarial robustness / 对抗鲁棒性 (7 columns / 7列)
# ================================================================== #

def write_adversarial_table(
    adv_results: dict,
    output_dir: str = PAPER_DIR,
    primary_epsilon: float = 0.10,
) -> str:
    """
    Generate the adversarial robustness table (tab:adversarial).
    生成对抗鲁棒性表（tab:adversarial）。

    Paper columns: Model | Signal Flip Rate | Rank Corr. | Sharpe(clean)
                   | Sharpe(adv.) | Sharpe Drop % | Max DD(adv., %)
    论文列：模型 | 信号翻转率 | 排名相关性 | 夏普(干净)
                | 夏普(对抗) | 夏普下降% | 最大回撤(对抗, %)

    Sorted by Sharpe Drop descending. / 按夏普下降幅度降序排列。
    Negatives use $-...$ (math-wrapped) style. / 负数使用 $-...$ 数学包裹格式。
    Model name 'MeanReversionBaseline' → 'MeanReversionBase.'
    模型名 'MeanReversionBaseline' 缩写为 'MeanReversionBase.'
    """
    # Collect row data from results / 从结果中收集行数据
    rows: list[dict] = []
    for model_name, eps_dict in adv_results.items():
        entry = eps_dict.get(primary_epsilon, {})
        if not entry:
            continue
        rows.append({
            "model": model_name,
            "flip": entry.get("signal_flip_rate", 0),          # 信号翻转率
            "rank_corr": entry.get("rank_correlation", 0),      # 排名相关性
            "sharpe_clean": entry.get("sharpe_clean", 0),       # 干净夏普
            "sharpe_adv": entry.get("sharpe_adversarial", 0),   # 对抗夏普
            "sharpe_drop": entry.get("sharpe_drop_%", 0),       # 夏普下降%
            "max_dd": entry.get("max_dd_adversarial", 0),       # 最大回撤(对抗)
        })

    # Sort by Sharpe Drop descending / 按夏普下降幅度降序排列
    rows.sort(key=lambda r: r["sharpe_drop"], reverse=True)

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l r r r r r r}")
    lines.append(r"\toprule")
    # Two-row header / 双行表头
    lines.append(
        r"Model & Signal Flip & Rank & Sharpe & Sharpe & Sharpe & Max DD \\"
    )
    lines.append(
        r"      & Rate & Corr.\ & (clean) & (adv.) & Drop \% & (adv., \%) \\"
    )
    lines.append(r"\midrule")

    for r in rows:
        name = _abbreviate_model(r["model"])
        name_tex = _escape_model(name)

        # Format: small flip rates get 4 decimal places, otherwise 3
        # 格式：较小的翻转率使用4位小数，否则使用3位
        flip_s = f"{r['flip']:.3f}" if r['flip'] >= 0.01 else f"{r['flip']:.4f}"
        rank_s = f"{r['rank_corr']:.3f}"
        sc_s = _fmt_neg(r["sharpe_clean"], ".3f", math_wrap=True)
        sa_s = _fmt_neg(r["sharpe_adv"], ".3f", math_wrap=True)
        drop_s = _fmt_neg(r["sharpe_drop"], ".1f", math_wrap=True)
        dd_s = _fmt_plain(r["max_dd"], ".2f")

        cells = [flip_s, rank_s, sc_s, sa_s, drop_s, dd_s]
        lines.append(f"{name_tex:22s} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Write to file / 写入文件
    path = os.path.join(output_dir, "tab_adversarial.tex")
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("% Auto-generated paper-format tabular for tab:adversarial\n")
        f.write("% 自动生成的论文格式表格 tab:adversarial (对抗鲁棒性)\n")
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Paper table tab:adversarial → {path}")
    return path


# ================================================================== #
#  Table 6 / 表6: tab:fuzzing
#  Synthetic market fuzzing / 合成市场压力测试 (7 columns / 7列)
# ================================================================== #

def write_fuzzing_table(
    fuzzing_results: dict,
    output_dir: str = PAPER_DIR,
    metric: str = "Sharpe (gross)",
) -> str:
    """
    Generate the synthetic market fuzzing table (tab:fuzzing).
    生成合成市场压力测试表（tab:fuzzing）。

    Paper columns: Model | Clean | Flash -10% | Flash -20%
                   | Vol Spike 3× | Vol Spike 5× | Gap & Reversal
    论文列：模型 | 正常 | 闪崩 -10% | 闪崩 -20%
                | 波动率飙升 3× | 波动率飙升 5× | 跳空反转

    Math-mode scenario headers. / 数学模式场景表头。
    Negatives use $-0.134$ style. / 负数使用 $-0.134$ 格式。
    """
    from scripts.robustness import compute_crash_heatmap
    # Compute heatmap DataFrame from raw fuzzing results
    # 从原始压力测试结果计算热力图 DataFrame
    df = compute_crash_heatmap(fuzzing_results, metric=metric)
    if df.empty:
        print("  ⚠ No fuzzing data — skipping tab:fuzzing")
        print("  ⚠ 无压力测试数据 — 跳过 tab:fuzzing")
        return ""

    # Map pipeline scenario names → paper column order
    # 将流水线场景名映射到论文列顺序
    scenario_map = {
        "Clean": "Clean",
        "Flash Crash (−10%)": "Flash Crash (-10%)",
        "Flash Crash (−20%)": "Flash Crash (-20%)",
        "Volatility Spike (3×)": "Volatility Spike (3x)",
        "Volatility Spike (5×)": "Volatility Spike (5x)",
        "Gap & Reversal": "Gap & Reversal",
    }

    # Normalise column names (the pipeline may use different dash/multiply chars)
    # 标准化列名（流水线可能使用不同的破折号/乘号字符）
    df_cols = list(df.columns)

    def _match_col(pattern_fragments: list[str]) -> str | None:
        """Find the column whose name contains all fragments.
        查找名称包含所有关键词片段的列。"""
        for c in df_cols:
            cn = c.lower()
            if all(f.lower() in cn for f in pattern_fragments):
                return c
        return None

    # Define column order by pattern fragments / 通过关键词片段定义列顺序
    col_order_patterns = [
        (["clean"], "Clean"),
        (["flash", "10"], r"Flash"),
        (["flash", "20"], r"Flash"),
        (["vol", "3"], r"Vol Spike"),
        (["vol", "5"], r"Vol Spike"),
        (["gap"], r"Gap \&"),
    ]
    col_keys: list[str | None] = []
    for frags, _ in col_order_patterns:
        c = _match_col(frags)
        if c:
            col_keys.append(c)
        else:
            col_keys.append(None)

    # Model order matching paper / 按论文顺序排列模型
    model_order = [
        "LSTM", "MLP", "LogisticRegression", "LinearRegression", "Ridge",
        "RandomForest", "LightGBM", "MomentumBaseline", "MeanReversionBaseline",
    ]
    models = [m for m in model_order if m in df.index]
    # Append any extra models not in the predefined order
    # 追加预定义顺序之外的额外模型
    for m in df.index:
        if m not in models:
            models.append(m)

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l r r r r r r}")
    lines.append(r"\toprule")
    # Two-row header with math-mode symbols / 带数学模式符号的双行表头
    lines.append(
        r"Model & Clean & Flash & Flash & Vol Spike & Vol Spike & Gap \& \\"
    )
    lines.append(
        r"      &       & $-10\%$ & $-20\%$ & $3\times$ & $5\times$ & Reversal \\"
    )
    lines.append(r"\midrule")

    for model in models:
        name = _abbreviate_model(model)
        name_tex = _escape_model(name)
        cells = []
        for ck in col_keys:
            if ck is None:
                cells.append("---")
                continue
            # Retrieve cell value; default to NaN if model not in index
            # 取单元格值；模型不在索引中时默认为 NaN
            raw = df.loc[model, ck] if model in df.index else np.nan
            cells.append(_fmt_neg(float(raw), ".3f", math_wrap=True))  # type: ignore[arg-type]
        lines.append(f"{name_tex:22s} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Write to file / 写入文件
    path = os.path.join(output_dir, "tab_fuzzing.tex")
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("% Auto-generated paper-format tabular for tab:fuzzing\n")
        f.write("% 自动生成的论文格式表格 tab:fuzzing (合成市场压力测试)\n")
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Paper table tab:fuzzing → {path}")
    return path


# ================================================================== #
#  Table 7 / 表7: tab:poisoning
#  Label poisoning / 标签投毒攻击 (6 columns / 6列, IC only / 仅 IC)
# ================================================================== #

def write_poisoning_table(
    poison_results: dict,
    output_dir: str = PAPER_DIR,
) -> str:
    """
    Generate the label poisoning table (tab:poisoning).
    生成标签投毒攻击表（tab:poisoning）。

    Paper columns: Model | 0% | 2% | 5% | 10% | 20%
    论文列：模型 | 0% | 2% | 5% | 10% | 20%
    Values = IC only (4 decimal places).
    数值 = 仅 IC 指标（4位小数）。
    """
    rates = [0.0, 0.02, 0.05, 0.10, 0.20]  # 投毒比例

    # Model display order / 模型显示顺序
    model_order = [
        "LSTM", "MLP", "LinearRegression", "Ridge",
        "LogisticRegression", "LightGBM", "RandomForest",
    ]
    models = [m for m in model_order if m in poison_results]
    # Append extra models / 追加额外模型
    for m in poison_results:
        if m not in models:
            models.append(m)

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Model & 0\% & 2\% & 5\% & 10\% & 20\% \\")
    lines.append(r"\midrule")

    for model in models:
        name_tex = _escape_model(model)
        rate_dict = poison_results[model]
        cells = []
        for rate in rates:
            entry = rate_dict.get(rate, {})
            ic = entry.get("IC", np.nan)
            cells.append(_fmt_plain(ic, ".4f"))
        lines.append(f"{name_tex:22s} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Write to file / 写入文件
    path = os.path.join(output_dir, "tab_poisoning.tex")
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("% Auto-generated paper-format tabular for tab:poisoning\n")
        f.write("% 自动生成的论文格式表格 tab:poisoning (标签投毒攻击)\n")
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Paper table tab:poisoning → {path}")
    return path


# ================================================================== #
#  Table 8 / 表8: tab:adv_defense
#  Adversarial training defense / 对抗训练防御 (8 columns / 8列)
# ================================================================== #

def write_defense_table(
    defense_data: dict,
    output_dir: str = PAPER_DIR,
    primary_epsilon: float = 0.10,
) -> str:
    """
    Generate the adversarial training defense table (tab:adv_defense).
    生成对抗训练防御表（tab:adv_defense）。

    Paper columns: Model | Training | SSR | Flip Rate | Sharpe(clean)
                   | Sharpe(adv.) | Sharpe Drop % | Max DD(adv., %)
    论文列：模型 | 训练方式 | SSR | 翻转率 | 夏普(干净)
                | 夏普(对抗) | 夏普下降% | 最大回撤(对抗, %)

    Bold best SSR, Flip Rate, Sharpe Drop % per model group.
    每个模型组内最优的 SSR、翻转率、夏普下降% 加粗显示。

    defense_data: the JSON dict keyed by e.g. "LSTM_Standard", "LSTM_Adversarial"
    defense_data: JSON 字典，键如 "LSTM_Standard", "LSTM_Adversarial"
    """
    eps_key = str(primary_epsilon)          # 扰动强度键
    model_groups = ["LSTM", "MLP"]          # 模型分组

    # Collect rows / 收集行数据
    all_rows: list[dict[str, Any]] = []
    for model in model_groups:
        for training in ["Standard", "Adversarial"]:
            key = f"{model}_{training}"
            eps_dict = defense_data.get(key, {})
            entry = eps_dict.get(eps_key, {})
            if not entry:
                continue
            all_rows.append({
                "model": model,
                "training": training,                                       # 训练方式
                "ssr": entry.get("ssr", 0),                                 # 信号稳定率
                "flip": entry.get("signal_flip_rate",
                                  entry.get("flip_rate", 0)),               # 信号翻转率
                "sharpe_clean": entry.get("sharpe_clean", 0),               # 干净夏普
                "sharpe_adv": entry.get("sharpe_adv", 0),                   # 对抗夏普
                "sharpe_drop": entry.get("sharpe_drop_pct",
                                         entry.get("sharpe_drop_%", 0)),    # 夏普下降%
                "max_dd": entry.get("max_dd_adv_pct",
                                    entry.get("max_dd_adversarial", 0)),    # 最大回撤(对抗)
            })

    # Determine best values per model group for bolding
    # 确定每个模型组的最优值（用于加粗显示）
    best: dict[str, dict[str, float]] = {}
    for row in all_rows:
        m = row["model"]
        if m not in best:
            best[m] = {"ssr": -1, "flip": float("inf"), "drop": float("inf")}
        # Best SSR = highest / 最优 SSR = 最高值
        if row["ssr"] > best[m]["ssr"]:
            best[m]["ssr"] = row["ssr"]
        # Best Flip Rate = lowest / 最优翻转率 = 最低值
        if row["flip"] < best[m]["flip"]:
            best[m]["flip"] = row["flip"]
        # Best Sharpe Drop = smallest absolute / 最优夏普下降 = 绝对值最小
        if abs(row["sharpe_drop"]) < abs(best[m]["drop"]):
            best[m]["drop"] = row["sharpe_drop"]

    def _bold_if(val_s: str, is_best: bool) -> str:
        """Wrap value in \\textbf if it's the best in its group.
        如果是组内最优值则用 \\textbf 包裹加粗。"""
        return rf"\textbf{{{val_s}}}" if is_best else val_s

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l l r r r r r r}")
    lines.append(r"\toprule")
    # Two-row header / 双行表头
    lines.append(
        r"Model & Training & SSR & Flip Rate & Sharpe & Sharpe & Sharpe & Max DD \\"
    )
    lines.append(
        r"      &          &     &           & (clean) & (adv.) & Drop \% & (adv., \%) \\"
    )
    lines.append(r"\midrule")

    prev_model = None
    for row in all_rows:
        m = row["model"]
        # Insert \midrule between different model groups
        # 不同模型组之间插入 \midrule 分隔线
        if prev_model is not None and m != prev_model:
            lines.append(r"\midrule")
        prev_model = m

        # Format cells with bold for best values / 格式化单元格，最优值加粗
        ssr_s = _bold_if(f"{row['ssr']:.3f}",
                         abs(row["ssr"] - best[m]["ssr"]) < 1e-6)
        flip_s = _bold_if(f"{row['flip']:.3f}",
                          abs(row["flip"] - best[m]["flip"]) < 1e-6)
        sc_s = _fmt_neg(row["sharpe_clean"], ".3f", math_wrap=True)
        sa_s = _fmt_neg(row["sharpe_adv"], ".3f", math_wrap=True)
        drop_s = _bold_if(
            _fmt_neg(row["sharpe_drop"], ".1f", math_wrap=True),
            abs(row["sharpe_drop"] - best[m]["drop"]) < 1e-3,
        )
        dd_s = _fmt_plain(row["max_dd"], ".1f")

        cells = [row["training"], ssr_s, flip_s, sc_s, sa_s, drop_s, dd_s]
        lines.append(f"{m} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Write to file / 写入文件
    path = os.path.join(output_dir, "tab_adv_defense.tex")
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("% Auto-generated paper-format tabular for tab:adv_defense\n")
        f.write("% 自动生成的论文格式表格 tab:adv_defense (对抗训练防御)\n")
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Paper table tab:adv_defense → {path}")
    return path


# ================================================================== #
#  Convenience / 便捷工具:
#  Generate all paper tables from saved CSVs / JSONs
#  从已保存的 CSV/JSON 重新生成所有论文格式表格
# ================================================================== #

def generate_all_from_saved(
    tables_dir: str = "reports/tables",
    json_dir: str = "reports",
    output_dir: str = PAPER_DIR,
) -> None:
    """
    Re-generate all paper-format tables from the already-saved pipeline
    CSV/JSON outputs (useful for offline regeneration without re-running
    the full backtest pipeline).
    从已保存的流水线 CSV/JSON 输出重新生成所有论文格式表格
    （适用于无需重跑完整回测流水线的离线再生成场景）。
    """
    import json as _json

    print("=" * 60)
    print("  Generating paper-format tables from saved pipeline data")
    print("  从已保存的流水线数据生成论文格式表格")
    print("=" * 60)

    # ── Table 1 / 表1: Main results / 主要结果 ──
    csv = os.path.join(tables_dir, "table1_main_results.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0)
        # Convert DataFrame to nested dict / 将 DataFrame 转换为嵌套字典
        metrics_dict: dict[str, dict[str, Any]] = {
            str(k): {str(kk): vv for kk, vv in v.items()}
            for k, v in df.to_dict(orient="index").items()
        }
        write_main_table(metrics_dict, output_dir)

    # ── Table 2 / 表2: Regime / 市场区间 ──
    csv = os.path.join(tables_dir, "table3_regime_analysis.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        # Group by Model to build per-model DataFrames
        # 按模型分组构建每个模型的 DataFrame
        regime_results: dict[str, pd.DataFrame] = {}
        for model, gdf in df.groupby("Model"):
            regime_results[str(model)] = gdf.reset_index(drop=True)
        write_regime_table(regime_results, output_dir)

    # ── Table 3 / 表3: Sensitivity / 敏感性分析 (merged rebal + topK / 合并再平衡+TopK) ──
    csv_r = os.path.join(tables_dir, "table7_rebal_sensitivity.csv")
    csv_k = os.path.join(tables_dir, "table8_topk_sensitivity.csv")
    if os.path.exists(csv_r) and os.path.exists(csv_k):
        df_r = pd.read_csv(csv_r)
        df_k = pd.read_csv(csv_k)
        # Build rebal dict: {model: {freq: {metric: value}}}
        # 构建再平衡字典：{模型: {频率: {指标: 值}}}
        rebal: dict[str, dict[int, dict]] = {}
        for _, row in df_r.iterrows():
            m = row["Model"]
            f = int(row["Rebal Freq (days)"])
            rebal.setdefault(m, {})[f] = {"Sharpe (gross)": row["Sharpe (gross)"]}
        # Build topK dict: {model: {k: {metric: value}}}
        # 构建 Top-K 字典：{模型: {K值: {指标: 值}}}
        topk: dict[str, dict[int, dict]] = {}
        for _, row in df_k.iterrows():
            m = row["Model"]
            k = int(row["Top-K"])
            topk.setdefault(m, {})[k] = {"Sharpe (gross)": row["Sharpe (gross)"]}
        write_sensitivity_table(rebal, topk, output_dir)

    # ── Table 4 / 表4: Long-only / 纯多头 ──
    csv = os.path.join(tables_dir, "table5_longonly_vs_longshort.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0)
        ls_m: dict[str, dict] = {}     # Long-Short metrics / 多空指标
        lo_m: dict[str, dict] = {}     # Long-Only metrics / 纯多头指标
        for model, row in df.iterrows():
            ls_m[str(model)] = {
                "Sharpe (gross)": row.get("LS Sharpe(g)", np.nan),
                "Sharpe (net)": row.get("LS Sharpe(n)", np.nan),
                "CAGR (gross)": row.get("LS CAGR(g)%", np.nan),
            }
            lo_m[str(model)] = {
                "Sharpe (gross)": row.get("LO Sharpe(g)", np.nan),
                "Sharpe (net)": row.get("LO Sharpe(n)", np.nan),
                "CAGR (gross)": row.get("LO CAGR(g)%", np.nan),
            }
        write_longonly_table(ls_m, lo_m, output_dir)

    # ── Table 5 / 表5: Adversarial robustness / 对抗鲁棒性 ──
    csv = os.path.join(tables_dir, "table9_adversarial_robustness.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0)
        # Reconstruct adversarial results dict / 重建对抗结果字典
        adv_r: dict[str, dict[float, dict]] = {}
        for model, row in df.iterrows():
            adv_r[str(model)] = {
                0.10: {
                    "signal_flip_rate": row.get("Signal Flip Rate", 0),
                    "rank_correlation": row.get("Rank Correlation", 0),
                    "sharpe_clean": row.get("Sharpe (clean)", 0),
                    "sharpe_adversarial": row.get("Sharpe (adv)", 0),
                    "sharpe_drop_%": row.get("Sharpe Drop %", 0),
                    "max_dd_adversarial": row.get("Max DD (adv) %", 0),
                }
            }
        write_adversarial_table(adv_r, output_dir)

    # ── Table 6 / 表6: Fuzzing / 压力测试 ──
    csv = os.path.join(tables_dir, "table10_fuzzing_stress_test.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0)
        # Already have heatmap DF, write directly / 已有热力图 DF，直接写入
        _write_fuzzing_from_df(df, output_dir)

    # ── Table 7 / 表7: Poisoning / 标签投毒 ──
    csv = os.path.join(tables_dir, "table11_label_poisoning.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv, index_col=0)
        # Reconstruct poisoning results dict / 重建投毒结果字典
        poison_r: dict[str, dict[float, dict]] = {}
        for model, row in df.iterrows():
            poison_r[str(model)] = {}
            for rate in [0.0, 0.02, 0.05, 0.10, 0.20]:
                pct = f"{rate*100:.0f}%"
                ic_key = f"IC ({pct})"
                ic_val = row.get(ic_key, np.nan)
                poison_r[str(model)][rate] = {"IC": ic_val}
        write_poisoning_table(poison_r, output_dir)

    # ── Table 8 / 表8: Adversarial defense / 对抗训练防御 ──
    json_path = os.path.join(json_dir, "adversarial_defense_metrics.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            defense_data = _json.load(f)
        write_defense_table(defense_data, output_dir)

    print("\n" + "=" * 60)
    print("  All paper-format tables generated!")
    print("  所有论文格式表格已生成！")
    print("=" * 60)


def _write_fuzzing_from_df(
    df: pd.DataFrame,
    output_dir: str = PAPER_DIR,
) -> str:
    """Write fuzzing table directly from the heatmap DataFrame.
    从热力图 DataFrame 直接写入压力测试表格。

    This is used by generate_all_from_saved() when loading from saved CSV,
    where we already have aggregated Sharpe data and don't need to re-compute.
    用于 generate_all_from_saved() 从已保存 CSV 加载时，
    此时已具有聚合的 Sharpe 数据，无需重新计算。
    """
    # Model display order / 模型显示顺序
    model_order = [
        "LSTM", "MLP", "LogisticRegression", "LinearRegression", "Ridge",
        "RandomForest", "LightGBM", "MomentumBaseline", "MeanReversionBaseline",
    ]

    # Find columns by pattern matching / 通过关键词模式匹配列名
    df_cols = list(df.columns)

    def _find(frags: list[str]) -> str | None:
        """Find column matching all keyword fragments (case-insensitive).
        查找包含所有关键词片段的列名（不区分大小写）。"""
        for c in df_cols:
            if all(f.lower() in c.lower() for f in frags):
                return c
        return None

    # Map scenario patterns to columns / 将场景模式映射到列
    col_keys: list[str | None] = [
        _find(["clean"]),           # Clean / 正常
        _find(["flash", "10"]),     # Flash Crash -10% / 闪崩 -10%
        _find(["flash", "20"]),     # Flash Crash -20% / 闪崩 -20%
        _find(["vol", "3"]),        # Vol Spike 3× / 波动率飙升 3×
        _find(["vol", "5"]),        # Vol Spike 5× / 波动率飙升 5×
        _find(["gap"]),             # Gap & Reversal / 跳空反转
    ]

    # Filter and order models / 过滤并排序模型
    models = [m for m in model_order if m in df.index]
    for m in df.index:
        if m not in models:
            models.append(m)

    lines: list[str] = []
    lines.append(r"\begin{tabular}{l r r r r r r}")
    lines.append(r"\toprule")
    # Two-row header with math-mode symbols / 带数学模式符号的双行表头
    lines.append(
        r"Model & Clean & Flash & Flash & Vol Spike & Vol Spike & Gap \& \\"
    )
    lines.append(
        r"      &       & $-10\%$ & $-20\%$ & $3\times$ & $5\times$ & Reversal \\"
    )
    lines.append(r"\midrule")

    for model in models:
        name = _abbreviate_model(model)
        name_tex = _escape_model(name)
        cells = []
        for ck in col_keys:
            if ck is None:
                cells.append("---")
                continue
            # Retrieve cell value / 取单元格值
            raw = df.loc[model, ck] if model in df.index else np.nan
            cells.append(_fmt_neg(float(raw), ".3f", math_wrap=True))  # type: ignore[arg-type]
        lines.append(f"{name_tex:22s} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Write to file / 写入文件
    path = os.path.join(output_dir, "tab_fuzzing.tex")
    _ensure_dir(path)
    with open(path, "w") as f:
        f.write("% Auto-generated paper-format tabular for tab:fuzzing\n")
        f.write("% 自动生成的论文格式表格 tab:fuzzing (从已保存 CSV 生成)\n")
        f.write("\n".join(lines) + "\n")
    print(f"  ✓ Paper table tab:fuzzing → {path}")
    return path


# ================================================================== #
#  CLI entry point / 命令行入口
#  Usage / 用法:  python -m scripts.paper_tables
# ================================================================== #

if __name__ == "__main__":
    generate_all_from_saved()
