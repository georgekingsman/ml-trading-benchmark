"""
adv_metrics.py — 对抗鲁棒性评估指标 (Adversarial Robustness Metrics)

提供三个核心指标:
    1. Adversarial Sharpe Ratio (ASR): 受攻击数据上策略的年化夏普比率
    2. Signal Stability Rate (SSR):    攻击前后交易方向保持一致的比例
    3. Performance Collapse:           夏普比率的相对下降程度

典型用法:
    from src.adv_metrics import signal_stability_rate, performance_collapse
    ssr = signal_stability_rate(pred_clean, pred_adv)
    collapse = performance_collapse(sharpe_clean, sharpe_adv)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ================================================================== #
#  Core Adversarial Metrics
# ================================================================== #

def adversarial_sharpe_ratio(
    returns_adv: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Adversarial Sharpe Ratio (ASR) — 受攻击数据上的策略年化夏普比率。

    计算公式:
        ASR = E[R_adv - r_f] / σ(R_adv) × √(252)

    Parameters
    ----------
    returns_adv : pd.Series
        在对抗样本预测驱动下的每日策略收益序列。
    rf : float
        年化无风险利率 (默认 0)。
    periods_per_year : int
        年化周期 (日频=252)。

    Returns
    -------
    float
        年化 Sharpe Ratio; 若标准差为 0 则返回 0.0。
    """
    excess = returns_adv - rf / periods_per_year
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def signal_stability_rate(
    pred_clean: np.ndarray,
    pred_adv: np.ndarray,
) -> float:
    """
    Signal Stability Rate (SSR) — 攻击前后交易信号方向一致率。

    计算公式:
        SSR = mean( sign(pred_clean) == sign(pred_adv) )

    值域 [0, 1]:
        1.0 → 所有信号方向未受影响 (完全稳定)
        0.0 → 所有信号方向翻转 (完全不稳定)

    Parameters
    ----------
    pred_clean : np.ndarray, shape [N]
        干净输入上的模型预测值。
    pred_adv : np.ndarray, shape [N]
        对抗输入上的模型预测值。

    Returns
    -------
    float
        方向一致率 (0~1)。
    """
    clean = np.asarray(pred_clean).flatten()
    adv = np.asarray(pred_adv).flatten()

    # 过滤 NaN
    valid_mask = ~(np.isnan(clean) | np.isnan(adv))
    clean = clean[valid_mask]
    adv = adv[valid_mask]

    if len(clean) == 0:
        return 1.0  # 无有效数据时视为稳定

    sign_match = np.sign(clean) == np.sign(adv)
    return float(np.mean(sign_match))


def performance_collapse(
    sharpe_clean: float,
    sharpe_adv: float,
) -> float:
    """
    Performance Collapse — 攻击导致的夏普比率相对下降。

    计算公式:
        Collapse = (Sharpe_clean - Sharpe_adv) / Sharpe_clean

    值域解释:
        0.0  → 无性能下降
        1.0  → 性能完全崩塌 (Sharpe 降为 0)
        >1.0 → Sharpe 变为负值 (比崩塌更糟)
        <0.0 → 攻击反而提升了性能 (罕见)

    Parameters
    ----------
    sharpe_clean : float
        干净数据上的年化 Sharpe Ratio。
    sharpe_adv : float
        对抗数据上的年化 Sharpe Ratio。

    Returns
    -------
    float
        相对下降比例。若 Sharpe_clean ≈ 0 则返回 0.0 避免除零。
    """
    if abs(sharpe_clean) < 1e-8:
        return 0.0
    return float((sharpe_clean - sharpe_adv) / sharpe_clean)


# ================================================================== #
#  辅助指标 (Supplementary Metrics)
# ================================================================== #

def signal_flip_rate(
    pred_clean: np.ndarray,
    pred_adv: np.ndarray,
) -> float:
    """
    Signal Flip Rate — 攻击导致信号反转的比例。

    等价于 1 - SSR, 即: flip_rate = 1 - signal_stability_rate(...)

    Parameters
    ----------
    pred_clean, pred_adv : np.ndarray, shape [N]

    Returns
    -------
    float
        信号翻转率 (0~1)。
    """
    return 1.0 - signal_stability_rate(pred_clean, pred_adv)


def prediction_rmse(
    pred_clean: np.ndarray,
    pred_adv: np.ndarray,
) -> float:
    """
    预测值 RMSE — 衡量攻击造成的绝对预测偏移。

    Parameters
    ----------
    pred_clean, pred_adv : np.ndarray, shape [N]

    Returns
    -------
    float
    """
    clean = np.asarray(pred_clean).flatten()
    adv = np.asarray(pred_adv).flatten()
    valid = ~(np.isnan(clean) | np.isnan(adv))
    if valid.sum() == 0:
        return 0.0
    return float(np.sqrt(np.mean((clean[valid] - adv[valid]) ** 2)))


def rank_correlation(
    pred_clean: np.ndarray,
    pred_adv: np.ndarray,
) -> float:
    """
    Spearman 秩相关 — 衡量攻击后预测排序的稳定性。

    Parameters
    ----------
    pred_clean, pred_adv : np.ndarray, shape [N]

    Returns
    -------
    float
        Spearman ρ ∈ [-1, 1]; 1.0 = 排序完全一致。
    """
    from scipy import stats as sp_stats

    clean = np.asarray(pred_clean).flatten()
    adv = np.asarray(pred_adv).flatten()
    valid = ~(np.isnan(clean) | np.isnan(adv))
    if valid.sum() < 3:
        return 1.0

    result = sp_stats.spearmanr(clean[valid], adv[valid])
    rho = float(result.statistic) if hasattr(result, "statistic") else float(result[0])  # type: ignore[union-attr,arg-type]
    return rho if not np.isnan(rho) else 0.0


def robustness_summary(
    pred_clean: np.ndarray,
    pred_adv: np.ndarray,
    sharpe_clean: float = 0.0,
    sharpe_adv: float = 0.0,
) -> dict[str, float]:
    """
    一次性计算全部鲁棒性指标的便捷函数。

    Returns
    -------
    dict
        包含 SSR, flip_rate, pred_rmse, rank_corr, perf_collapse
    """
    return {
        "SSR": round(signal_stability_rate(pred_clean, pred_adv), 4),
        "flip_rate": round(signal_flip_rate(pred_clean, pred_adv), 4),
        "pred_rmse": round(prediction_rmse(pred_clean, pred_adv), 6),
        "rank_corr": round(rank_correlation(pred_clean, pred_adv), 4),
        "perf_collapse": round(performance_collapse(sharpe_clean, sharpe_adv), 4),
    }
