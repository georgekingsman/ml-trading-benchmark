"""
Quick smoke test for robustness.py — validates core functions with small data.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

# Load real data
from scripts.feature_engineering import FEATURE_COLS
from scripts.split import walk_forward_split
from scripts.models import build_model
from scripts.backtest import run_backtest
from scripts.metrics import compute_all_metrics
import yaml

cfg = yaml.safe_load(open("config/settings.yaml"))

# Quick load
print("Loading cached features...")
panel = pd.read_parquet("features/features_panel.parquet")
splits = walk_forward_split(panel, cfg)
train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]

primary_h = cfg["task"]["primary_horizon"]
target_col = f"fwd_return_{primary_h}d"

X_train = train_df[FEATURE_COLS].values
y_train = train_df[target_col].values
X_val = val_df[FEATURE_COLS].values
y_val = val_df[target_col].values
X_test = test_df[FEATURE_COLS].values
y_test = test_df[target_col].values
returns_df = test_df[["date", "ticker", "daily_return"]].copy()

print(f"Data loaded: train={X_train.shape}, test={X_test.shape}")

# === Test 1: Train a quick model (Ridge) ===
print("\n--- Test 1: Training Ridge model ---")
model = build_model("Ridge", {})
model.fit(X_train, y_train, X_val, y_val)
clean_preds = model.predict(X_test)
print(f"  Clean predictions: mean={clean_preds.mean():.6f}, std={clean_preds.std():.6f}")

# === Test 2: Random perturbation ===
print("\n--- Test 2: Random Perturbation ---")
from scripts.robustness import random_perturbation, adversarial_sharpe_ratio, _compute_feature_std
feature_std = _compute_feature_std(X_train)
X_adv = random_perturbation(X_test, epsilon_scale=0.10, feature_std=feature_std)
print(f"  Perturbation L∞ per feature: max={np.max(np.abs(X_adv - X_test)):.6f}")
print(f"  Expected max: ~{0.10 * np.max(feature_std):.6f}")

adv_preds = model.predict(X_adv)
print(f"  Adversarial predictions: mean={adv_preds.mean():.6f}, std={adv_preds.std():.6f}")

sq = adversarial_sharpe_ratio(model, X_test, X_adv, clean_preds)
print(f"  Signal quality: {sq}")

# === Test 3: MLP + FGSM ===
print("\n--- Test 3: MLP + FGSM Attack ---")
mlp = build_model("MLP", {"hidden_dims": [64, 32], "epochs": 5, "lr": 1e-3, "batch_size": 512})
mlp.fit(X_train, y_train, X_val, y_val)
mlp_clean = mlp.predict(X_test)

from scripts.robustness import fgsm_attack, pgd_attack
X_fgsm = fgsm_attack(mlp, X_test, y_test, epsilon_scale=0.10, feature_std=feature_std)
mlp_fgsm = mlp.predict(X_fgsm)
sq_fgsm = adversarial_sharpe_ratio(mlp, X_test, X_fgsm, mlp_clean)
print(f"  FGSM signal quality: {sq_fgsm}")

X_pgd = pgd_attack(mlp, X_test, y_test, epsilon_scale=0.10, feature_std=feature_std, n_steps=5)
mlp_pgd = mlp.predict(X_pgd)
sq_pgd = adversarial_sharpe_ratio(mlp, X_test, X_pgd, mlp_clean)
print(f"  PGD signal quality:  {sq_pgd}")

# === Test 4: Market Fuzzing ===
print("\n--- Test 4: Market Fuzzing ---")
from scripts.robustness import MarketFuzzer
fuzzer = MarketFuzzer(seed=42)

fuzzed = fuzzer.inject_flash_crash(returns_df, crash_magnitude=-0.10, recovery_days=2, n_events=3)
print(f"  Flash crash: {(fuzzed['daily_return'] != returns_df['daily_return']).sum()} modified returns")

fuzzed_vol = fuzzer.inject_volatility_spike(returns_df, vol_multiplier=3.0, spike_days=5, n_events=2)
print(f"  Vol spike: {(fuzzed_vol['daily_return'] != returns_df['daily_return']).sum()} modified returns")

# Quick backtest under stress
pred_df = test_df[["date", "ticker"]].copy()
pred_df["prediction"] = clean_preds
pred_df = pred_df.dropna(subset=["prediction"])

bt_kwargs = {"top_k": 5, "rebalance_freq": 5, "cost_bps": 15.0, "slippage_bps": 5.0}
bt_clean = run_backtest(pred_df, returns_df, **bt_kwargs)
bt_fuzz = run_backtest(pred_df, fuzzed, **bt_kwargs)

m_clean = compute_all_metrics(bt_clean, total_cost_bps=20.0)
m_fuzz = compute_all_metrics(bt_fuzz, total_cost_bps=20.0)
print(f"  Clean Sharpe: {m_clean.get('Sharpe (gross)', 0):.3f}")
print(f"  Fuzzed Sharpe: {m_fuzz.get('Sharpe (gross)', 0):.3f}")

# === Test 5: Label Poisoning ===
print("\n--- Test 5: Label Poisoning (quick, Ridge only) ---")
from scripts.robustness import run_label_poisoning_experiment
poison_results = run_label_poisoning_experiment(
    model_configs=[{"name": "Ridge", "params": {}}],
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test,
    test_df=test_df, returns_df=returns_df,
    poison_rates=[0.0, 0.10, 0.20],
    backtest_fn=run_backtest,
    metrics_fn=lambda bt: compute_all_metrics(bt, total_cost_bps=20.0),
    backtest_kwargs=bt_kwargs,
    seed=42,
)
for rate, entry in poison_results.get("Ridge", {}).items():
    print(f"  Poison rate {rate*100:.0f}%: IC={entry.get('IC', 'N/A')}, Sharpe={entry.get('Sharpe (gross)', 'N/A')}")

# === Test 6: Report generation ===
print("\n--- Test 6: Report Generation ---")
os.makedirs("reports/tables", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)

# Build a small adversarial result dict
adv_results_small = {}
for name, mdl in [("Ridge", model)]:
    adv_results_small[name] = {}
    for eps in [0.05, 0.10, 0.20]:
        X_adv_eps = random_perturbation(X_test, eps, feature_std)
        sq = adversarial_sharpe_ratio(mdl, X_test, X_adv_eps, clean_preds)
        # Add backtest metrics
        pred_adv = test_df[["date", "ticker"]].copy()
        pred_adv["prediction"] = mdl.predict(X_adv_eps)
        pred_adv = pred_adv.dropna(subset=["prediction"])
        bt_adv = run_backtest(pred_adv, returns_df, **bt_kwargs)
        m_adv = compute_all_metrics(bt_adv, total_cost_bps=20.0)
        sq["sharpe_clean"] = m_clean.get("Sharpe (gross)", 0)
        sq["sharpe_adversarial"] = m_adv.get("Sharpe (gross)", 0)
        sq["sharpe_drop_%"] = 0
        adv_results_small[name][eps] = sq

from scripts.robustness_report import (
    generate_adversarial_table, plot_adversarial_sharpe_degradation,
    plot_signal_flip_rate, generate_fuzzing_table, plot_fuzzing_heatmap,
)

generate_adversarial_table(adv_results_small, "reports", primary_epsilon=0.10)
plot_adversarial_sharpe_degradation(adv_results_small, "reports")
plot_signal_flip_rate(adv_results_small, "reports")
print("  Adversarial tables/figures generated OK")

# Fuzzing reports
fuzz_results = {
    "Clean": {"Ridge": m_clean},
    "Flash Crash (−10%)": {"Ridge": m_fuzz},
}
generate_fuzzing_table(fuzz_results, "reports")
plot_fuzzing_heatmap(fuzz_results, "reports")
print("  Fuzzing tables/figures generated OK")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED!")
print("=" * 60)
