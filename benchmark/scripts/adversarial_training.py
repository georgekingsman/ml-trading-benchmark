"""
adversarial_training.py — Adversarial Training Defense for PyTorch Models.
对抗训练防御模块 — 为 PyTorch 模型实现对抗训练。

Implements the "closing the loop" for the paper:
为论文实现"闭环"逻辑：
  1. Discovery  / 发现:  Models are vulnerable to PGD attacks / 模型易受 PGD 攻击
  2. Analysis   / 分析:  Gradient saliency reveals over-sensitive features / 梯度显著性揭示过敏特征
  3. **Defense** / 防御:  Adversarial training injects PGD samples during training
                          对抗训练在训练过程中注入 PGD 对抗样本

Reference / 参考文献:
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
    (ICLR 2018) — Section 5: Adversarial Training

Usage / 使用方法:
    from scripts.adversarial_training import adversarial_train
    model = adversarial_train(model, X_train, y_train, epsilon=0.1, epochs=30)
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def pgd_perturbation(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """
    Generate PGD adversarial perturbation δ (does not modify x in-place).
    生成 PGD 对抗扰动 δ (不会原地修改 x)。

    Returns x_adv = x + δ where ||δ||_∞ ≤ ε.
    返回 x_adv = x + δ，其中 ||δ||_∞ ≤ ε。
    """
    # Randomly initialize δ within [-ε, ε] / 在 [-ε, ε] 内随机初始化 δ
    delta = torch.zeros_like(x).uniform_(-epsilon, epsilon)
    delta.requires_grad_(True)

    for _ in range(steps):
        pred = model(x + delta)
        loss = loss_fn(pred, y)
        loss.backward()
        # Gradient sign ascent / 梯度符号上升
        grad_sign = delta.grad.data.sign()
        delta = delta.detach() + alpha * grad_sign
        # Project back to L∞ ball / 投影回 L∞ 球
        delta = torch.clamp(delta, -epsilon, epsilon)
        delta.requires_grad_(True)

    return (x + delta.detach()).detach()


def adversarial_train(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    pgd_steps: int = 7,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    mix_ratio: float = 0.5,
    loss_fn_cls: type = nn.MSELoss,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Adversarial training: mix clean and PGD adversarial samples per mini-batch.
    对抗训练: 在每个 mini-batch 中混合干净样本与 PGD 对抗样本。

    Core idea (Madry et al.):
    核心思路:
        min_θ  E_{(x,y)} [ max_{||δ||≤ε}  L(f_θ(x + δ), y) ]

    Practical implementation / 实际实现:
        Per batch / 每个 batch:
          1. PGD generates x_adv from current model / 用当前模型对 x 做 PGD 得到 x_adv
          2. loss = mix_ratio × L(f(x_adv), y) + (1 - mix_ratio) × L(f(x), y)
          3. Backprop and update θ / 反向传播更新 θ

    Parameters / 参数
    ----------
    model : nn.Module
        PyTorch model to train (modified **in-place**).
        待训练的 PyTorch 模型 (会被 **原地修改**)。
    X_train : np.ndarray, shape (N, F) or (N, seq_len, F)
    y_train : np.ndarray, shape (N,)
    epsilon : float
        PGD perturbation budget (L∞) / PGD 扰动预算 (L∞)。
    alpha : float
        PGD step size / PGD 每步步长。
    pgd_steps : int
        PGD iterations (7 is sufficient per Madry) / PGD 迭代次数 (7 步足够)。
    mix_ratio : float ∈ [0, 1]
        Adversarial sample weight. 0.5 = half clean half adversarial. 1.0 = pure adversarial.
        对抗样本权重。0.5 = 一半干净一半对抗。1.0 = 纯对抗训练。
    verbose : bool
        Print per-epoch loss / 是否打印每 epoch loss。

    Returns / 返回
    -------
    model : nn.Module
        Adversarially trained model / 对抗训练后的模型。
    """
    # Auto-select device / 自动选择设备
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    model = model.to(device)
    loss_fn = loss_fn_cls()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare data / 准备数据
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Validation data (optional, for early stopping) / 验证集 (可选，用于早停)
    if X_val is not None and y_val is not None:
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    else:
        X_val_t = None

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 10  # Early stopping patience / 早停耐心值

    from tqdm import tqdm

    pbar = tqdm(range(epochs), desc="Adv-Train", disable=not verbose, leave=False)
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            # ── Step 1: Generate adversarial examples / 第1步: 生成对抗样本 ──
            model.eval()  # freeze BN/Dropout for PGD / 冻结 BN/Dropout 再做 PGD
            with torch.enable_grad():
                xb_adv = pgd_perturbation(
                    model, xb, yb,
                    epsilon=epsilon, alpha=alpha,
                    steps=pgd_steps, loss_fn=loss_fn,
                )
            model.train()

            # ── Step 2: Mixed loss / 第2步: 混合损失 ──
            optimizer.zero_grad()
            pred_clean = model(xb)
            pred_adv = model(xb_adv)
            loss_clean = loss_fn(pred_clean, yb)
            loss_adv = loss_fn(pred_adv, yb)
            loss = (1.0 - mix_ratio) * loss_clean + mix_ratio * loss_adv
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation / 验证
        val_info = ""
        if X_val_t is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = loss_fn(val_pred, y_val_t).item()
            val_info = f"  val={val_loss:.6f}"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        pbar.set_postfix_str(f"loss={avg_loss:.6f}{val_info}")

    # Restore best model / 恢复最优模型
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def compare_standard_vs_adversarial(
    model_builder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    pgd_steps_train: int = 7,
    pgd_steps_eval: int = 10,
    epochs: int = 30,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> dict:
    """
    Compare standard training vs adversarial training:
    automatically produce "discover problem → solve problem" evidence.
    对比标准训练 vs 对抗训练: 自动生成 "发现问题 → 解决问题" 的完整证据。

    Parameters / 参数
    ----------
    model_builder : callable
        Zero-arg callable returning a fresh nn.Module.
        无参调用返回一个全新 nn.Module (e.g., lambda: build_mlp(input_dim))。

    Returns / 返回
    -------
    dict with keys / 包含以下键:
        standard_ssr, robust_ssr, ssr_improvement_pct,
        pred_clean_std, pred_adv_std, pred_clean_rob, pred_adv_rob
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    def _eval_model(model_nn, label=""):
        model_nn.eval()
        with torch.no_grad():
            pred_clean = model_nn(X_t).cpu().numpy().flatten()

        # Generate adversarial test samples / 生成对抗测试样本
        x_adv = pgd_perturbation(
            model_nn, X_t, y_t,
            epsilon=epsilon, alpha=alpha,
            steps=pgd_steps_eval, loss_fn=loss_fn,
        )
        with torch.no_grad():
            pred_adv = model_nn(x_adv).cpu().numpy().flatten()

        # SSR: Signal Stability Rate / 信号稳定率
        valid = ~(np.isnan(pred_clean) | np.isnan(pred_adv))
        ssr = float(np.mean(np.sign(pred_clean[valid]) == np.sign(pred_adv[valid])))

        return pred_clean, pred_adv, ssr

    # ── Standard Training / 标准训练 ──
    if verbose:
        print("  [1/2] Standard training / 标准训练 ...")
    model_std = model_builder().to(device)
    _std_train(model_std, X_train, y_train, X_val, y_val, epochs, batch_size, device, verbose)
    pred_c_std, pred_a_std, ssr_std = _eval_model(model_std, "standard")

    # ── Adversarial Training / 对抗训练 ──
    if verbose:
        print("  [2/2] Adversarial training / 对抗训练 ...")
    model_rob = model_builder().to(device)
    adversarial_train(
        model_rob, X_train, y_train, X_val, y_val,
        epsilon=epsilon, alpha=alpha, pgd_steps=pgd_steps_train,
        epochs=epochs, batch_size=batch_size, device=device, verbose=verbose,
    )
    pred_c_rob, pred_a_rob, ssr_rob = _eval_model(model_rob, "adversarial")

    # Compute improvement: what fraction of residual vulnerability was closed
    # 计算改进比例: 消除了多少比例的残余脆弱性
    improvement = ((ssr_rob - ssr_std) / max(1 - ssr_std, 1e-6)) * 100

    result = {
        "standard_ssr": round(ssr_std, 4),
        "robust_ssr": round(ssr_rob, 4),
        "ssr_improvement_pct": round(improvement, 1),
        "pred_clean_std": pred_c_std,
        "pred_adv_std": pred_a_std,
        "pred_clean_rob": pred_c_rob,
        "pred_adv_rob": pred_a_rob,
    }

    if verbose:
        print(f"\n  Standard Training  / 标准训练:    SSR = {ssr_std:.4f}")
        print(f"  Adversarial Training / 对抗训练: SSR = {ssr_rob:.4f}")
        print(f"  Improvement / 改进: {improvement:.1f}% of residual vulnerability closed")

    return result


def _std_train(model, X_train, y_train, X_val, y_val, epochs, batch_size, device, verbose):
    """Helper: standard training loop / 辅助函数: 标准训练循环。"""
    from tqdm import tqdm

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    pbar = tqdm(range(epochs), desc="Std-Train", disable=not verbose, leave=False)
    for _ in pbar:
        ep_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        pbar.set_postfix(loss=f"{ep_loss / len(dl):.6f}")
