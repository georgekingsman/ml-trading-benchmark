"""
robustness.py — Algorithmic Robustness Analysis for Financial ML Models.

Three experimental directions:
  1. Adversarial Perturbation  (FGSM / PGD / Random baseline)
  2. Synthetic Market Fuzzing  (Flash-Crash, V-Recovery, Liquidity Drought)
  3. Concept Drift & Feature Decay  (Label Poisoning + Alpha Decay Half-life)

Reference:
  Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (ICLR 2015)
  Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018)
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")


# ================================================================== #
#  Direction 1 — Adversarial Perturbation
# ================================================================== #

def _compute_feature_std(X_train: np.ndarray) -> np.ndarray:
    """Historical per-feature standard deviation (used to constrain perturbation)."""
    return np.std(X_train, axis=0).clip(min=1e-8)


def fgsm_attack(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon_scale: float = 0.1,
    feature_std: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fast Gradient Sign Method (FGSM) attack for PyTorch models.

    Perturbs input features along the gradient direction to *maximise* MSE loss.
    The perturbation magnitude per feature is bounded by epsilon_scale × σ_feature,
    ensuring that adversarial inputs remain statistically plausible.

    Parameters
    ----------
    model       : A trained PyTorch-based model (MLPModel or LSTMModel)
    X           : Clean test features (N, F)
    y           : True forward returns (N,)
    epsilon_scale : Perturbation budget as fraction of feature std (default 0.1)
    feature_std : Per-feature historical std; if None, computed from X

    Returns
    -------
    X_adv : Adversarially perturbed features (N, F)
    """
    import torch
    import torch.nn as nn

    if feature_std is None:
        feature_std = _compute_feature_std(X)

    epsilon = epsilon_scale * feature_std  # (F,)

    # Handle LSTM sequence construction
    is_lstm = hasattr(model, 'seq_len')
    if is_lstm:
        X_seq = model._make_sequences(X)
        y_seq = y[model.seq_len - 1:]
        X_t = torch.tensor(X_seq, dtype=torch.float32, requires_grad=True).to(model.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(model.device)
    else:
        X_t = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(model.device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(model.device)

    model.model.eval()
    loss_fn = nn.MSELoss()

    pred = model.model(X_t)
    loss = loss_fn(pred, y_t)
    loss.backward()

    assert X_t.grad is not None
    grad_sign = X_t.grad.data.sign().cpu().numpy()

    if is_lstm:
        # For LSTM: perturb all time steps uniformly, then reconstruct flat array
        # Apply epsilon broadcast: (N_seq, seq_len, F) × (F,)
        eps_broadcast = epsilon[np.newaxis, np.newaxis, :]
        X_seq_adv = X_seq + eps_broadcast * grad_sign
        # Reconstruct flat features: take the last time step of each sequence
        # but also need full X_adv — we perturb the original flat X
        X_adv = X.copy()
        # Average gradient sign across sequences that overlap each row
        flat_grad = np.zeros_like(X)
        counts = np.zeros(len(X))
        for i in range(len(X_seq)):
            start = i
            end = i + model.seq_len
            flat_grad[start:end] += grad_sign[i]
            counts[start:end] += 1
        counts = counts.clip(min=1)[:, np.newaxis]
        flat_grad = np.sign(flat_grad / counts)
        X_adv = X + epsilon[np.newaxis, :] * flat_grad
    else:
        X_adv = X + epsilon[np.newaxis, :] * grad_sign

    return X_adv


def pgd_attack(
    model,
    X: np.ndarray,
    y: np.ndarray,
    epsilon_scale: float = 0.1,
    feature_std: np.ndarray | None = None,
    n_steps: int = 10,
    step_scale: float = 0.025,
) -> np.ndarray:
    """
    Projected Gradient Descent (PGD) attack — iterative version of FGSM.

    Stronger multi-step attack that iteratively perturbs then projects back
    within the ε-ball.

    Parameters
    ----------
    model        : PyTorch model (MLP / LSTM)
    X, y         : Clean features and targets
    epsilon_scale: Budget as fraction of feature std
    n_steps      : Number of PGD steps
    step_scale   : Step size per iteration as fraction of feature std

    Returns
    -------
    X_adv : Adversarially perturbed features (N, F)
    """
    import torch
    import torch.nn as nn

    if feature_std is None:
        feature_std = _compute_feature_std(X)

    epsilon = epsilon_scale * feature_std  # (F,)
    alpha = step_scale * feature_std       # step size per iter

    is_lstm = hasattr(model, 'seq_len')
    model.model.eval()
    loss_fn = nn.MSELoss()

    if is_lstm:
        X_seq = model._make_sequences(X)
        y_seq = y[model.seq_len - 1:]
        X_orig = torch.tensor(X_seq, dtype=torch.float32).to(model.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1).to(model.device)
        eps_t = torch.tensor(epsilon, dtype=torch.float32).to(model.device)
        alpha_t = torch.tensor(alpha, dtype=torch.float32).to(model.device)

        # Random init within ε-ball
        delta = torch.zeros_like(X_orig).uniform_(-1, 1).to(model.device)
        delta = delta * eps_t

        for _ in range(n_steps):
            delta.requires_grad_(True)
            X_adv_t = X_orig + delta
            pred = model.model(X_adv_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            assert delta.grad is not None
            grad_sign = delta.grad.data.sign()
            delta = delta.detach() + alpha_t * grad_sign
            # Project back to ε-ball
            delta = torch.clamp(delta, -eps_t, eps_t)

        X_seq_adv = (X_orig + delta.detach()).cpu().numpy()

        # Reconstruct flat X_adv
        X_adv = X.copy()
        perturbation = np.zeros_like(X)
        counts = np.zeros(len(X))
        for i in range(len(X_seq)):
            pert_i = X_seq_adv[i] - X_seq[i]
            start = i
            end = i + model.seq_len
            perturbation[start:end] += pert_i
            counts[start:end] += 1
        counts = counts.clip(min=1)[:, np.newaxis]
        perturbation = perturbation / counts
        # Re-clip to epsilon
        perturbation = np.clip(perturbation, -epsilon, epsilon)
        X_adv = X + perturbation
    else:
        X_orig = torch.tensor(X, dtype=torch.float32).to(model.device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(model.device)
        eps_t = torch.tensor(epsilon, dtype=torch.float32).to(model.device)
        alpha_t = torch.tensor(alpha, dtype=torch.float32).to(model.device)

        delta = torch.zeros_like(X_orig).uniform_(-1, 1).to(model.device)
        delta = delta * eps_t

        for _ in range(n_steps):
            delta.requires_grad_(True)
            X_adv_t = X_orig + delta
            pred = model.model(X_adv_t)
            loss = loss_fn(pred, y_t)
            loss.backward()
            assert delta.grad is not None
            grad_sign = delta.grad.data.sign()
            delta = delta.detach() + alpha_t * grad_sign
            delta = torch.clamp(delta, -eps_t, eps_t)

        X_adv = (X_orig + delta.detach()).cpu().numpy()

    return X_adv


def random_perturbation(
    X: np.ndarray,
    epsilon_scale: float = 0.1,
    feature_std: np.ndarray | None = None,
    seed: int = 42,
) -> np.ndarray:
    """
    Random baseline perturbation (model-agnostic).

    Adds uniform noise bounded by ε × σ per feature.
    Used for non-differentiable models (sklearn, LightGBM).

    Parameters
    ----------
    X             : Clean features
    epsilon_scale : Budget as fraction of feature std
    feature_std   : Per-feature historical std
    seed          : Random seed

    Returns
    -------
    X_adv : Perturbed features
    """
    if feature_std is None:
        feature_std = _compute_feature_std(X)
    epsilon = epsilon_scale * feature_std
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-1, 1, size=X.shape) * epsilon[np.newaxis, :]
    return X + noise


def adversarial_sharpe_ratio(
    model,
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    clean_predictions: np.ndarray,
) -> dict[str, float]:
    """
    Compute Adversarial Sharpe Ratio (ASR) — the Sharpe ratio of predictions
    made on adversarial inputs vs. clean inputs.

    Also computes:
      - Signal Flip Rate: fraction of signals that change sign (long → short or vice versa)
      - Rank Correlation: Spearman ρ between clean and adversarial rankings
      - Prediction RMSE: RMS difference between clean and adversarial predictions

    Returns
    -------
    dict with keys: signal_flip_rate, rank_correlation, prediction_rmse
    """
    adv_predictions = model.predict(X_adv)

    # Handle NaN from LSTM padding
    valid = ~(np.isnan(clean_predictions) | np.isnan(adv_predictions))
    clean_v = clean_predictions[valid]
    adv_v = adv_predictions[valid]

    if len(clean_v) < 10:
        return {
            "signal_flip_rate": 0.0,
            "rank_correlation": 1.0,
            "prediction_rmse": 0.0,
        }

    # Signal flip rate
    clean_sign = np.sign(clean_v)
    adv_sign = np.sign(adv_v)
    flip_rate = float(np.mean(clean_sign != adv_sign))

    # Rank correlation
    rho = stats.spearmanr(clean_v, adv_v)
    rank_corr = float(rho.statistic) if hasattr(rho, 'statistic') else float(rho[0])  # type: ignore[union-attr]
    if np.isnan(rank_corr):
        rank_corr = 0.0

    # RMSE
    rmse = float(np.sqrt(np.mean((clean_v - adv_v) ** 2)))

    return {
        "signal_flip_rate": round(flip_rate, 4),
        "rank_correlation": round(rank_corr, 4),
        "prediction_rmse": round(rmse, 6),
    }


def run_adversarial_experiment(
    trained_models: dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    feature_cols: list[str],
    epsilon_scales: list[float] | None = None,
    backtest_fn=None,
    backtest_kwargs: dict | None = None,
    metrics_fn=None,
) -> dict:
    """
    Run full adversarial perturbation experiment across all models and epsilon levels.

    Returns nested dict:
      results[model_name][epsilon] = {
          "signal_flip_rate": ...,
          "rank_correlation": ...,
          "prediction_rmse": ...,
          "sharpe_clean": ...,
          "sharpe_adversarial": ...,
          "sharpe_drop_%": ...,
          "max_dd_clean": ...,
          "max_dd_adversarial": ...,
      }
    """
    if epsilon_scales is None:
        epsilon_scales = [0.01, 0.05, 0.10, 0.20, 0.50]

    feature_std = _compute_feature_std(X_train)
    is_torch_model = lambda name: name in ("MLP", "LSTM")
    bkw = backtest_kwargs or {}

    results = {}

    from tqdm import tqdm
    model_pbar = tqdm(trained_models.items(), desc="Adversarial models", leave=True)
    for name, model in model_pbar:
        model_pbar.set_description(f"Adversarial: {name}")
        results[name] = {}

        # Clean predictions + backtest
        clean_preds = model.predict(X_test)

        for eps in tqdm(epsilon_scales, desc=f"  ε levels ({name})", leave=False):
            # Generate adversarial examples
            if is_torch_model(name):
                try:
                    X_adv_fgsm = fgsm_attack(model, X_test, y_test, eps, feature_std)
                    X_adv_pgd = pgd_attack(model, X_test, y_test, eps, feature_std,
                                           n_steps=10, step_scale=eps * 0.25)
                    # Use PGD (stronger) as the primary adversarial input
                    X_adv = X_adv_pgd
                except Exception as e:
                    print(f"      Warning: Gradient attack failed for {name}: {e}")
                    X_adv = random_perturbation(X_test, eps, feature_std)
            else:
                X_adv = random_perturbation(X_test, eps, feature_std)

            # Signal quality metrics
            sq = adversarial_sharpe_ratio(model, X_test, X_adv, clean_preds)

            # Adversarial predictions for backtest
            adv_preds = model.predict(X_adv)

            # Build prediction DataFrame for backtest
            if backtest_fn is not None and metrics_fn is not None:
                # Clean backtest
                pred_clean = test_df[["date", "ticker"]].copy()
                pred_clean["prediction"] = clean_preds
                pred_clean = pred_clean.dropna(subset=["prediction"])

                bt_clean = backtest_fn(pred_clean, returns_df, **bkw)
                m_clean = metrics_fn(bt_clean)

                # Adversarial backtest
                pred_adv = test_df[["date", "ticker"]].copy()
                pred_adv["prediction"] = adv_preds
                pred_adv = pred_adv.dropna(subset=["prediction"])

                bt_adv = backtest_fn(pred_adv, returns_df, **bkw)
                m_adv = metrics_fn(bt_adv)

                sharpe_clean = m_clean.get("Sharpe (gross)", 0)
                sharpe_adv = m_adv.get("Sharpe (gross)", 0)
                dd_clean = m_clean.get("Max DD", 0)
                dd_adv = m_adv.get("Max DD", 0)

                sharpe_drop = 0.0
                if abs(sharpe_clean) > 1e-6:
                    sharpe_drop = (sharpe_clean - sharpe_adv) / abs(sharpe_clean) * 100

                sq["sharpe_clean"] = round(sharpe_clean, 3)
                sq["sharpe_adversarial"] = round(sharpe_adv, 3)
                sq["sharpe_drop_%"] = round(sharpe_drop, 1)
                sq["max_dd_clean"] = round(dd_clean, 2)
                sq["max_dd_adversarial"] = round(dd_adv, 2)

            results[name][eps] = sq

    return results


# ================================================================== #
#  Direction 2 — Synthetic Market Fuzzing
# ================================================================== #

class MarketFuzzer:
    """
    Synthetic market stress scenario generator.

    Injects controlled anomalies into historical market data to test model
    robustness under extreme conditions not present in the training set.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def inject_flash_crash(
        self,
        returns_df: pd.DataFrame,
        crash_magnitude: float = -0.10,
        recovery_days: int = 2,
        n_events: int = 5,
    ) -> pd.DataFrame:
        """
        Inject synthetic flash-crash events: sudden drop followed by partial recovery.

        Parameters
        ----------
        returns_df      : Original returns DataFrame (date, ticker, daily_return)
        crash_magnitude : Single-day return during crash (default -10%)
        recovery_days   : Days after crash with recovery momentum
        n_events        : Number of flash-crash events to inject

        Returns
        -------
        Fuzzed returns DataFrame
        """
        df = returns_df.copy()
        dates = sorted(df["date"].unique())
        n_dates = len(dates)

        # Select random dates (at least 10 days from start and end)
        eligible = dates[10: n_dates - recovery_days - 5]
        crash_dates = self.rng.choice(eligible, size=min(n_events, len(eligible)),
                                      replace=False)

        for crash_date in crash_dates:
            # Crash day: all assets drop
            mask = df["date"] == crash_date
            df.loc[mask, "daily_return"] = crash_magnitude

            # Recovery: partial bounce-back over next N days
            crash_idx = list(dates).index(crash_date)
            recovery_ret = abs(crash_magnitude) * 0.5 / recovery_days
            for d in range(1, recovery_days + 1):
                if crash_idx + d < n_dates:
                    rec_date = dates[crash_idx + d]
                    mask_rec = df["date"] == rec_date
                    df.loc[mask_rec, "daily_return"] += recovery_ret  # type: ignore[assignment]

        return df

    def inject_volatility_spike(
        self,
        returns_df: pd.DataFrame,
        vol_multiplier: float = 3.0,
        spike_days: int = 10,
        n_events: int = 3,
    ) -> pd.DataFrame:
        """
        Inject periods of extreme volatility: amplify returns by a multiplier
        while preserving the sign (direction).
        """
        df = returns_df.copy()
        dates = sorted(df["date"].unique())
        n_dates = len(dates)

        eligible = dates[20: n_dates - spike_days - 5]
        start_dates = self.rng.choice(eligible, size=min(n_events, len(eligible)),
                                      replace=False)

        for start_date in start_dates:
            start_idx = list(dates).index(start_date)
            for d in range(spike_days):
                if start_idx + d < n_dates:
                    spike_date = dates[start_idx + d]
                    mask = df["date"] == spike_date
                    df.loc[mask, "daily_return"] *= vol_multiplier  # type: ignore[assignment]

        return df

    def inject_liquidity_drought(
        self,
        panel_df: pd.DataFrame,
        volume_fraction: float = 0.1,
        spread_multiplier: float = 10.0,
        drought_days: int = 5,
        n_events: int = 3,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simulate liquidity drought: shrink volume and inflate bid-ask spread.

        Returns both the modified panel and a cost-multiplier series.

        Parameters
        ----------
        panel_df          : Full panel with volume column
        volume_fraction   : Volume reduced to this fraction (default 10%)
        spread_multiplier : Effective bid-ask spread multiplied by this factor
        drought_days      : Duration of each drought event
        n_events          : Number of events

        Returns
        -------
        (modified_panel, cost_multiplier_df) where cost_multiplier_df has
        columns [date, cost_multiplier] indicating when costs are inflated.
        """
        df = panel_df.copy()
        dates = sorted(df["date"].unique())
        n_dates = len(dates)

        eligible = dates[20: n_dates - drought_days - 5]
        start_dates = self.rng.choice(eligible, size=min(n_events, len(eligible)),
                                      replace=False)

        cost_mult = pd.DataFrame({"date": dates, "cost_multiplier": 1.0})

        for start_date in start_dates:
            start_idx = list(dates).index(start_date)
            for d in range(drought_days):
                if start_idx + d < n_dates:
                    dd = dates[start_idx + d]
                    mask = df["date"] == dd
                    if "volume" in df.columns:
                        df.loc[mask, "volume"] *= volume_fraction  # type: ignore[assignment]
                    cost_mult.loc[cost_mult["date"] == dd, "cost_multiplier"] = spread_multiplier

        return df, cost_mult

    def inject_gap_and_reversal(
        self,
        returns_df: pd.DataFrame,
        gap_magnitude: float = -0.08,
        reversal_magnitude: float = 0.06,
        n_events: int = 5,
    ) -> pd.DataFrame:
        """
        Inject gap-down followed by next-day reversal (bear-trap pattern).
        Tests whether models chase momentum into traps.
        """
        df = returns_df.copy()
        dates = sorted(df["date"].unique())
        n_dates = len(dates)

        eligible = dates[10: n_dates - 5]
        event_dates = self.rng.choice(eligible, size=min(n_events, len(eligible)),
                                      replace=False)

        for event_date in event_dates:
            idx = list(dates).index(event_date)
            # Gap down
            mask = df["date"] == event_date
            df.loc[mask, "daily_return"] = gap_magnitude
            # Next-day reversal
            if idx + 1 < n_dates:
                next_date = dates[idx + 1]
                mask_next = df["date"] == next_date
                df.loc[mask_next, "daily_return"] = reversal_magnitude

        return df


def run_fuzzing_experiment(
    all_predictions: dict[str, pd.DataFrame],
    returns_df: pd.DataFrame,
    backtest_fn,
    metrics_fn,
    backtest_kwargs: dict | None = None,
    seed: int = 42,
) -> dict:
    """
    Run all fuzzing scenarios and compare model performance under stress.

    Returns nested dict:
      results[scenario][model_name] = {metrics_dict}
    """
    bkw = backtest_kwargs or {}
    fuzzer = MarketFuzzer(seed=seed)

    scenarios = {
        "Clean": returns_df,
        "Flash Crash (−10%)": fuzzer.inject_flash_crash(
            returns_df, crash_magnitude=-0.10, recovery_days=2, n_events=5
        ),
        "Flash Crash (−20%)": fuzzer.inject_flash_crash(
            returns_df, crash_magnitude=-0.20, recovery_days=3, n_events=3
        ),
        "Volatility Spike (3×)": fuzzer.inject_volatility_spike(
            returns_df, vol_multiplier=3.0, spike_days=10, n_events=3
        ),
        "Volatility Spike (5×)": fuzzer.inject_volatility_spike(
            returns_df, vol_multiplier=5.0, spike_days=10, n_events=3
        ),
        "Gap & Reversal": fuzzer.inject_gap_and_reversal(
            returns_df, gap_magnitude=-0.08, reversal_magnitude=0.06, n_events=5
        ),
    }

    results = {}

    from tqdm import tqdm
    for scenario_name, fuzzed_returns in tqdm(scenarios.items(), desc="Fuzzing scenarios", leave=True):
        results[scenario_name] = {}

        for model_name, pred_df in tqdm(all_predictions.items(), desc=f"  {scenario_name}", leave=False):
            try:
                bt = backtest_fn(pred_df, fuzzed_returns, **bkw)
                m = metrics_fn(bt)
                results[scenario_name][model_name] = m
            except Exception as e:
                print(f"      Warning: {model_name} failed under {scenario_name}: {e}")
                results[scenario_name][model_name] = {}

    return results


def compute_crash_heatmap(fuzzing_results: dict, metric: str = "Sharpe (gross)") -> pd.DataFrame:
    """
    Build a model × scenario heatmap for a given metric.

    Returns DataFrame where rows=models, columns=scenarios.
    """
    scenarios = list(fuzzing_results.keys())
    if not scenarios:
        return pd.DataFrame()

    models = list(fuzzing_results[scenarios[0]].keys())
    rows = []
    for model in models:
        row = {"Model": model}
        for scenario in scenarios:
            val = fuzzing_results.get(scenario, {}).get(model, {}).get(metric, np.nan)
            row[scenario] = val
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Model")
    return df


# ================================================================== #
#  Direction 3 — Concept Drift & Feature Decay
# ================================================================== #

def run_label_poisoning_experiment(
    model_configs: list[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    poison_rates: list[float] | None = None,
    backtest_fn=None,
    metrics_fn=None,
    backtest_kwargs: dict | None = None,
    seed: int = 42,
) -> dict:
    """
    Label poisoning experiment: inject random label flips into training data
    and measure model self-healing capacity.

    Parameters
    ----------
    poison_rates : Fraction of training labels to corrupt (default [0, 0.02, 0.05, 0.10, 0.20])

    Returns
    -------
    results[model_name][poison_rate] = {metrics_dict + IC}
    """
    from scripts.models import build_model
    from scripts.metrics import compute_ic_icir

    if poison_rates is None:
        poison_rates = [0.0, 0.02, 0.05, 0.10, 0.20]

    rng = np.random.RandomState(seed)
    bkw = backtest_kwargs or {}

    results = {}

    from tqdm import tqdm
    for mcfg in tqdm(model_configs, desc="Poisoning models", leave=True):
        name = mcfg["name"]
        params = mcfg.get("params", {})
        results[name] = {}

        for rate in tqdm(poison_rates, desc=f"  {name} rates", leave=False):
            # Poison labels
            y_poisoned = y_train.copy()
            if rate > 0:
                n_poison = int(len(y_train) * rate)
                poison_idx = rng.choice(len(y_train), size=n_poison, replace=False)
                # Flip sign (convert long to short and vice versa)
                y_poisoned[poison_idx] = -y_poisoned[poison_idx]

            # Train model on poisoned data
            model = build_model(name, params)
            model.fit(X_train, y_poisoned, X_val, y_val)

            # Predict on clean test data
            preds = model.predict(X_test)

            entry = {}

            # IC measurement
            pred_df = test_df[["date", "ticker"]].copy()
            pred_df["prediction"] = preds
            pred_df = pred_df.dropna(subset=["prediction"])
            target_col = "fwd_return_5d"
            if target_col in test_df.columns:
                mean_ic, icir = compute_ic_icir(pred_df, test_df, target_col)
                entry["IC"] = round(mean_ic, 4)
                entry["ICIR"] = round(icir, 4)

            # Backtest
            if backtest_fn is not None and metrics_fn is not None:
                try:
                    bt = backtest_fn(pred_df, returns_df, **bkw)
                    m = metrics_fn(bt)
                    entry.update(m)
                except Exception:
                    pass

            results[name][rate] = entry

    return results


def run_alpha_decay_experiment(
    panel: pd.DataFrame,
    model_configs: list[dict],
    feature_cols: list[str],
    horizons: list[int] | None = None,
    split_cfg: dict | None = None,
    seed: int = 42,
) -> dict:
    """
    Alpha Decay Half-life experiment.

    Train models to predict forward returns at multiple horizons (1d → 20d)
    and measure how IC decays with prediction horizon.

    Parameters
    ----------
    panel        : Feature panel with fwd_return_{h}d columns
    model_configs: List of model config dicts
    horizons     : List of horizons to test (default [1, 2, 3, 5, 7, 10, 15, 20])

    Returns
    -------
    results[model_name] = pd.DataFrame with columns [horizon, IC, ICIR]
    """
    from scripts.models import build_model
    from scripts.metrics import compute_ic_icir
    from scripts.feature_engineering import build_labels

    if horizons is None:
        horizons = [1, 2, 3, 5, 7, 10, 15, 20]

    results = {}

    # Ensure all horizon labels exist
    existing_horizons = []
    for h in horizons:
        col = f"fwd_return_{h}d"
        if col in panel.columns:
            existing_horizons.append(h)
        else:
            # Build label for this horizon
            try:
                panel[col] = (
                    panel.groupby("ticker")["close"]
                    .transform(lambda x: x.pct_change(h).shift(-h))
                )
                existing_horizons.append(h)
            except Exception:
                print(f"      Warning: Cannot build label for horizon {h}d")

    if not existing_horizons:
        return results

    # Use walk-forward split dates
    if split_cfg:
        train_mask = (panel["date"] >= split_cfg.get("train_start", "2005-01-01")) & \
                     (panel["date"] <= split_cfg.get("train_end", "2016-12-31"))
        val_mask = (panel["date"] >= split_cfg.get("val_start", "2017-01-01")) & \
                   (panel["date"] <= split_cfg.get("val_end", "2019-12-31"))
        test_mask = (panel["date"] >= split_cfg.get("test_start", "2020-01-01")) & \
                    (panel["date"] <= split_cfg.get("test_end", "2024-12-31"))
    else:
        dates = sorted(panel["date"].unique())
        n = len(dates)
        cutoff1 = dates[int(n * 0.6)]
        cutoff2 = dates[int(n * 0.8)]
        train_mask = panel["date"] <= cutoff1
        val_mask = (panel["date"] > cutoff1) & (panel["date"] <= cutoff2)
        test_mask = panel["date"] > cutoff2

    from tqdm import tqdm
    eligible_configs = [m for m in model_configs if "Baseline" not in m["name"]]
    for mcfg in tqdm(eligible_configs, desc="Alpha decay models", leave=True):
        name = mcfg["name"]
        params = mcfg.get("params", {})

        rows = []

        for h in tqdm(existing_horizons, desc=f"  {name} horizons", leave=False):
            target_col = f"fwd_return_{h}d"
            sub = panel.dropna(subset=feature_cols + [target_col])

            # Filter by date ranges (train_mask is on panel, not sub)
            train_dates = panel.loc[train_mask, "date"].unique()  # type: ignore[call-overload]
            val_dates = panel.loc[val_mask, "date"].unique()  # type: ignore[call-overload]
            test_dates = panel.loc[test_mask, "date"].unique()  # type: ignore[call-overload]
            train = sub.loc[sub["date"].isin(train_dates)]
            val = sub.loc[sub["date"].isin(val_dates)]
            test = sub.loc[sub["date"].isin(test_dates)]

            if len(train) < 100 or len(test) < 100:
                continue

            X_tr = np.asarray(train[feature_cols].values)
            y_tr = np.asarray(train[target_col].values)
            X_va = np.asarray(val[feature_cols].values) if len(val) > 0 else None
            y_va = np.asarray(val[target_col].values) if len(val) > 0 else None
            X_te = np.asarray(test[feature_cols].values)
            y_te = np.asarray(test[target_col].values)

            try:
                model = build_model(name, params)
                model.fit(X_tr, y_tr, X_va, y_va)
                preds = model.predict(X_te)

                pred_df = test[["date", "ticker"]].copy()
                pred_df["prediction"] = preds
                pred_df = pred_df.dropna(subset=["prediction"])

                mean_ic, icir = compute_ic_icir(pred_df, test, target_col)

                rows.append({
                    "horizon": h,
                    "IC": round(mean_ic, 4),
                    "ICIR": round(icir, 4),
                })
            except Exception as e:
                print(f"      Warning: {name} @ {h}d failed: {e}")
                rows.append({"horizon": h, "IC": 0.0, "ICIR": 0.0})

        if rows:
            results[name] = pd.DataFrame(rows)

    return results


def compute_decay_halflife(ic_series: pd.Series, horizons: list[int]) -> float:
    """
    Estimate the half-life of alpha decay by fitting IC = IC_0 × exp(-λt).

    Returns half-life in days; np.inf if no decay detected.
    """
    valid = ic_series > 0
    if valid.sum() < 3:
        return np.inf

    h = np.array(horizons)[np.asarray(valid.values)]
    ic = np.asarray(ic_series[valid].values)

    # Log-linear fit: log(IC) = log(IC_0) - λ*h
    try:
        log_ic = np.log(np.clip(ic, a_min=1e-8, a_max=None))
        slope, intercept = np.polyfit(h, log_ic, 1)
        if slope >= 0:
            return np.inf  # no decay
        halflife = np.log(2) / abs(slope)
        return round(float(halflife), 2)
    except Exception:
        return np.inf
