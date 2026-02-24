"""
adv_evaluation.py — 对抗鲁棒性评估流程 (Adversarial Robustness Evaluation Pipeline)

将 PGDAttacker 集成到测试集评估循环中，针对多组 ε 值生成对抗样本
并记录模型在攻击下的预测表现。

典型用法:
    from src.adv_evaluation import run_adversarial_evaluation
    results = run_adversarial_evaluation(model, X_test, y_test)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.attacks import PGDAttacker
from src.adv_metrics import (
    adversarial_sharpe_ratio,
    signal_stability_rate,
    performance_collapse,
)


def run_adversarial_evaluation(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilon_list: list[float] | None = None,
    alpha: float = 0.01,
    steps: int = 10,
    loss_fn: str = "mse",
    daily_returns_clean: Optional[pd.Series] = None,
    daily_returns_fn: Optional[Any] = None,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    在不同 ε 下运行 PGD 攻击，收集对抗鲁棒性指标。

    Parameters
    ----------
    model : nn.Module
        已训练的 PyTorch 模型 (MLP / LSTM)。
    X_test : np.ndarray
        测试集输入特征, shape [N, F] 或 [N, seq_len, F]。
    y_test : np.ndarray
        测试集目标值, shape [N]。
    epsilon_list : list[float]
        扰动强度列表 (默认 [0.0, 0.01, 0.05, 0.1])。
    alpha : float
        PGD 单步步长。
    steps : int
        PGD 迭代次数。
    loss_fn : str
        损失函数类型 ('mse' / 'bce')。
    daily_returns_clean : pd.Series, optional
        干净数据上的每日策略收益序列 (用于计算 Sharpe 类指标)。
    daily_returns_fn : callable, optional
        给定 predictions → pd.Series(daily_returns) 的函数。
        若提供，将自动为对抗预测生成收益序列。
    rf : float
        无风险利率 (年化)。
    periods_per_year : int
        年化周期数 (日频=252)。

    Returns
    -------
    pd.DataFrame
        每行对应一个 ε 值，包含:
        - epsilon: 扰动强度
        - pred_clean: 干净预测 (ndarray)
        - pred_adv: 对抗预测 (ndarray)
        - SSR: Signal Stability Rate
        - ASR: Adversarial Sharpe Ratio (若提供收益数据)
        - perf_collapse: Performance Collapse (若提供收益数据)
    """
    if epsilon_list is None:
        epsilon_list = [0.0, 0.01, 0.05, 0.1]

    device = next(model.parameters()).device

    # ---- 干净预测 (ε=0 基准) ----
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        pred_clean = model(x_tensor).cpu().numpy().flatten()

    # 干净 Sharpe (若提供收益数据)
    sharpe_clean = None
    if daily_returns_clean is not None:
        sharpe_clean = _compute_sharpe(daily_returns_clean, rf, periods_per_year)

    records: list[dict[str, Any]] = []

    for eps in epsilon_list:
        row: dict[str, Any] = {"epsilon": eps}

        if eps == 0.0:
            # ε=0: 无扰动基准
            pred_adv = pred_clean.copy()
        else:
            # 构建 PGD 攻击器
            attacker = PGDAttacker(
                model=model,
                epsilon=eps,
                alpha=alpha,
                steps=steps,
                loss_fn=loss_fn,
                random_start=True,
            )
            # 生成对抗样本
            X_test_adv = attacker.generate_numpy(X_test, y_test)

            # 模型在对抗样本上的预测
            model.eval()
            with torch.no_grad():
                x_adv_tensor = torch.tensor(
                    X_test_adv, dtype=torch.float32, device=device
                )
                pred_adv = model(x_adv_tensor).cpu().numpy().flatten()

        # ---- 计算指标 ----
        row["pred_clean"] = pred_clean
        row["pred_adv"] = pred_adv
        row["SSR"] = signal_stability_rate(pred_clean, pred_adv)

        # 若可获取对抗收益，计算 Sharpe 相关指标
        if daily_returns_fn is not None:
            daily_ret_adv = daily_returns_fn(pred_adv)
            sharpe_adv = _compute_sharpe(daily_ret_adv, rf, periods_per_year)
            row["ASR"] = sharpe_adv

            if sharpe_clean is not None and sharpe_clean != 0.0:
                row["perf_collapse"] = performance_collapse(sharpe_clean, sharpe_adv)
            else:
                row["perf_collapse"] = 0.0
        elif daily_returns_clean is not None and eps == 0.0:
            row["ASR"] = sharpe_clean
            row["perf_collapse"] = 0.0

        records.append(row)

    return pd.DataFrame(records)


def _compute_sharpe(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """年化 Sharpe Ratio。"""
    excess = returns - rf / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def run_batch_adversarial_evaluation(
    models: dict[str, nn.Module],
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilon_list: list[float] | None = None,
    alpha: float = 0.01,
    steps: int = 10,
    loss_fn: str = "mse",
) -> dict[str, pd.DataFrame]:
    """
    对多个模型批量运行对抗评估。

    Parameters
    ----------
    models : dict[str, nn.Module]
        {模型名: 模型实例} 字典。
    其余参数同 run_adversarial_evaluation。

    Returns
    -------
    dict[str, pd.DataFrame]
        {模型名: 评估结果 DataFrame}
    """
    if epsilon_list is None:
        epsilon_list = [0.0, 0.01, 0.05, 0.1]

    results = {}
    for name, model in models.items():
        print(f"[Adversarial Eval] {name} ...")
        df = run_adversarial_evaluation(
            model=model,
            X_test=X_test,
            y_test=y_test,
            epsilon_list=epsilon_list,
            alpha=alpha,
            steps=steps,
            loss_fn=loss_fn,
        )
        results[name] = df
        # 打印概要
        summary = df[["epsilon", "SSR"]].copy()
        print(summary.to_string(index=False))
        print()

    return results
