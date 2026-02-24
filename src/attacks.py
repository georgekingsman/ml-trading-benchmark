"""
attacks.py — 对抗性攻击模块 (Adversarial Attack Module)

实现 PGD (Projected Gradient Descent) 白盒梯度攻击，用于评估 PyTorch 模型
（MLP / LSTM）在对抗性扰动下的鲁棒性。

Reference:
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
    (ICLR 2018)

典型用法:
    attacker = PGDAttacker(model, epsilon=0.1, alpha=0.01, steps=10)
    X_adv = attacker.generate(X, y)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class PGDAttacker:
    """
    Projected Gradient Descent (PGD) 白盒对抗攻击器。

    通过多步梯度上升最大化模型损失，从而生成在 ε-球内的最强扰动样本。
    仅对输入 x 施加扰动，不修改目标 y。

    Parameters
    ----------
    model : nn.Module
        待攻击的 PyTorch 模型 (需处于 eval 模式)。
    epsilon : float
        最大扰动强度 (L∞ 范数约束)。
    alpha : float
        每步梯度上升的步长。
    steps : int
        PGD 迭代次数。
    loss_fn : str
        损失函数类型，'mse' 或 'bce'，需与训练时一致。
    random_start : bool
        是否在 ε-球内随机初始化扰动 (True = PGD, False = I-FGSM)。
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.1,
        alpha: float = 0.01,
        steps: int = 10,
        loss_fn: str = "mse",
        random_start: bool = True,
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

        # 选择损失函数
        if loss_fn == "mse":
            self._loss_fn = nn.MSELoss()
        elif loss_fn == "bce":
            self._loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss_fn: {loss_fn}. Use 'mse' or 'bce'.")

    @torch.no_grad()
    def _get_device(self) -> torch.device:
        """自动推断模型所在设备。"""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def generate(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
    ) -> torch.Tensor:
        """
        生成 PGD 对抗样本。

        Parameters
        ----------
        x : array-like, shape [batch, features] 或 [batch, seq_len, features]
            干净的输入样本。
        y : array-like, shape [batch] 或 [batch, 1]
            对应的目标值（不被修改）。
        clamp_min, clamp_max : float, optional
            对抗样本的全局数值范围限制。若为 None 则不额外裁剪。

        Returns
        -------
        x_adv : torch.Tensor
            对抗性扰动后的输入，shape 与 x 一致。

        Notes
        -----
        核心迭代公式:
            δ ← δ + α · sign(∇_x L(f(x + δ), y))
            δ ← clip(δ, -ε, ε)               # 投影回 ε-球
            x_adv ← clip(x + δ, x - ε, x + ε) # 确保邻域约束
        """
        device = self._get_device()

        # ---- 转换为 Tensor ----
        if isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            x_tensor = x.clone().detach().to(device).float()

        if isinstance(y, np.ndarray):
            y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
        else:
            y_tensor = y.clone().detach().to(device).float()

        # 确保 y 形状与模型输出一致 -> [batch, 1]
        if y_tensor.dim() == 1:
            y_tensor = y_tensor.unsqueeze(-1)

        # ---- 初始化扰动 δ ----
        delta = torch.zeros_like(x_tensor, device=device)
        if self.random_start:
            # 在 [-ε, +ε] 均匀采样
            delta = delta.uniform_(-self.epsilon, self.epsilon)
            # 投影: 保证 x + δ 在 [x - ε, x + ε] 范围
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)

        self.model.eval()

        # ---- PGD 迭代 ----
        for _ in range(self.steps):
            delta.requires_grad_(True)

            # 前向传播: 只有 delta 有梯度, x_tensor 被 detach
            x_adv = x_tensor.detach() + delta
            pred = self.model(x_adv)

            # 计算损失 (最大化 loss → 对模型是最坏情况)
            loss = self._loss_fn(pred, y_tensor)
            loss.backward()

            # 梯度符号步进
            grad_sign = delta.grad.data.sign()
            delta = delta.detach() + self.alpha * grad_sign

            # 投影 (Projection): 将 δ 裁剪到 [-ε, +ε] 以满足 L∞ 约束
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)

        # ---- 最终对抗样本 ----
        x_adv = (x_tensor + delta).detach()

        # 可选: 全局数值范围裁剪
        if clamp_min is not None or clamp_max is not None:
            x_adv = torch.clamp(
                x_adv,
                min=clamp_min if clamp_min is not None else float("-inf"),
                max=clamp_max if clamp_max is not None else float("inf"),
            )

        return x_adv

    def generate_numpy(
        self,
        x: np.ndarray,
        y: np.ndarray,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
    ) -> np.ndarray:
        """
        便捷方法: 输入 numpy, 输出 numpy。
        适配现有 PyTorch 模型的 DataLoader 格式。

        Parameters
        ----------
        x : np.ndarray, shape [batch, features] 或 [batch, seq_len, features]
        y : np.ndarray, shape [batch]

        Returns
        -------
        x_adv : np.ndarray, 与 x 同 shape
        """
        x_adv_tensor = self.generate(x, y, clamp_min=clamp_min, clamp_max=clamp_max)
        return x_adv_tensor.cpu().numpy()

    def __repr__(self) -> str:
        return (
            f"PGDAttacker(ε={self.epsilon}, α={self.alpha}, "
            f"steps={self.steps}, loss={self._loss_fn.__class__.__name__})"
        )
