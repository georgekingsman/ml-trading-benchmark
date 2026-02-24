"""
run_robustness.py — Robustness analysis pipeline for ML trading models.

Three experimental directions:
  Direction 1: Adversarial Perturbation   (FGSM / PGD / Random)
  Direction 2: Synthetic Market Fuzzing   (Flash-Crash / Volatility-Spike / Gap-Reversal)
  Direction 3: Concept Drift Analysis     (Label Poisoning + Alpha Decay Half-Life)

Usage:
  cd benchmark/
  python run_robustness.py                        # full robustness suite
  python run_robustness.py --skip-download --skip-features  # use cached data
  python run_robustness.py --only adversarial     # run only direction 1
  python run_robustness.py --only fuzzing         # run only direction 2
  python run_robustness.py --only drift           # run only direction 3

Prerequisites:
  Run the main benchmark first (python run_all.py) to build features and train models.
  OR use --skip-download --skip-features to use cached data and train fresh models.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from scripts.data_pipeline import download_prices, load_universe, process_prices
from scripts.feature_engineering import (
    FEATURE_COLS,
    build_features_for_ticker,
    build_labels,
    rolling_normalize,
)
from scripts.split import walk_forward_split
from scripts.models import build_model, MODEL_REGISTRY
from scripts.backtest import run_backtest
from scripts.metrics import compute_all_metrics

# Robustness modules
from scripts.robustness import (
    run_adversarial_experiment,
    run_fuzzing_experiment,
    run_label_poisoning_experiment,
    run_alpha_decay_experiment,
)
from scripts.robustness_report import (
    generate_adversarial_table,
    generate_adversarial_multi_epsilon_table,
    generate_fuzzing_table,
    generate_poisoning_table,
    generate_alpha_decay_table,
    plot_adversarial_sharpe_degradation,
    plot_signal_flip_rate,
    plot_fuzzing_heatmap,
    plot_poisoning_curve,
    plot_alpha_decay_curve,
    plot_robustness_summary,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Robustness Analysis — Adversarial, Fuzzing, Concept Drift"
    )
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--only", choices=["adversarial", "fuzzing", "drift", "all"],
                        default="all", help="Run only one direction")
    args = parser.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg["output"]["seed"])
    t0 = time.time()

    print("=" * 70)
    print("  ROBUSTNESS ANALYSIS PIPELINE")
    print("  Adversarial Perturbation · Synthetic Fuzzing · Concept Drift")
    print("=" * 70)

    # ============================================================== #
    #  Data Loading (same as run_all.py)
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP R1: Loading Data & Features")
    print("=" * 60)

    dcfg = cfg["data"]
    panel_path = os.path.join(dcfg["processed_dir"], "panel.parquet")
    feat_path = "features/features_panel.parquet"

    if args.skip_features and os.path.exists(feat_path):
        print("  Using cached features")
        panel = pd.read_parquet(feat_path)
    elif args.skip_download and os.path.exists(panel_path):
        print("  Using cached panel, rebuilding features...")
        panel = pd.read_parquet(panel_path)
        from tqdm import tqdm
        pieces = []
        for tkr, grp in tqdm(panel.groupby("ticker"), desc="Building features"):
            pieces.append(build_features_for_ticker(grp))
        panel = pd.concat(pieces)
        panel = build_labels(panel, cfg["task"]["horizons"])
        panel = rolling_normalize(panel, window=cfg["features"].get("rolling_window", 252))
        primary_h = cfg["task"]["primary_horizon"]
        required = FEATURE_COLS + [f"fwd_return_{primary_h}d"]
        panel = panel.dropna(subset=required)
    else:
        # Full data pipeline
        universe = load_universe(cfg["universe_file"])
        tickers = universe["ticker"].tolist()
        all_data = download_prices(
            tickers, dcfg["start_date"], dcfg["end_date"], dcfg["raw_dir"]
        )
        panel = process_prices(all_data, dcfg["processed_dir"])
        from tqdm import tqdm
        pieces = []
        for tkr, grp in tqdm(panel.groupby("ticker"), desc="Building features"):
            pieces.append(build_features_for_ticker(grp))
        panel = pd.concat(pieces)
        panel = build_labels(panel, cfg["task"]["horizons"])
        panel = rolling_normalize(panel, window=cfg["features"].get("rolling_window", 252))
        primary_h = cfg["task"]["primary_horizon"]
        required = FEATURE_COLS + [f"fwd_return_{primary_h}d"]
        panel = panel.dropna(subset=required)

    # Split
    splits = walk_forward_split(panel, cfg)
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    primary_h = cfg["task"]["primary_horizon"]
    target_col = f"fwd_return_{primary_h}d"

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[target_col].values
    X_val = val_df[FEATURE_COLS].values
    y_val = val_df[target_col].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df[target_col].values

    returns_df = test_df[["date", "ticker", "daily_return"]].copy()

    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"  Target: {target_col}")

    # ============================================================== #
    #  Train Models
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP R2: Training Models")
    print("=" * 60)

    model_configs = [m for m in cfg["models"] if m.get("enabled", True)]

    # Reduce deep-model epochs for robustness pipeline (we care about
    # relative performance differences, not absolute accuracy, so fewer
    # epochs are sufficient and dramatically speed up the many re-trainings).
    ROBUSTNESS_EPOCH_OVERRIDE = {"MLP": 10, "LSTM": 8}
    for mcfg in model_configs:
        name = mcfg["name"]
        if name in ROBUSTNESS_EPOCH_OVERRIDE:
            mcfg.setdefault("params", {})["epochs"] = ROBUSTNESS_EPOCH_OVERRIDE[name]
            print(f"  [Robustness] {name} epochs overridden → {ROBUSTNESS_EPOCH_OVERRIDE[name]}")

    trained_models = {}
    all_predictions = {}

    from tqdm import tqdm
    for mcfg in tqdm(model_configs, desc="Training models"):
        name = mcfg["name"]
        params = mcfg.get("params", {})
        print(f"  Training: {name} (epochs={params.get('epochs', 'N/A')})")

        model = build_model(name, params)
        model.fit(X_train, y_train, X_val, y_val)
        trained_models[name] = model

        preds = model.predict(X_test)
        pred_df = test_df[["date", "ticker"]].copy()
        pred_df["prediction"] = preds
        pred_df = pred_df.dropna(subset=["prediction"])
        all_predictions[name] = pred_df
        print(f"    ✓ {name}: {len(pred_df)} predictions")

    # Backtest common kwargs
    bcfg = cfg["backtest"]
    bt_kwargs = {
        "top_k": bcfg["top_k"],
        "rebalance_freq": bcfg["rebalance_freq"],
        "cost_bps": 15.0,
        "slippage_bps": bcfg["slippage_bps"],
    }

    # Output directory
    output_dir = "reports"
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # Store all results for JSON export
    all_robustness_results = {}

    # ============================================================== #
    #  Direction 1: Adversarial Perturbation
    # ============================================================== #
    adv_results = {}
    if args.only in ("adversarial", "all"):
        print("\n" + "=" * 60)
        print("DIRECTION 1: Adversarial Perturbation Analysis")
        print("  FGSM / PGD for deep models, Random noise for traditional ML")
        print("  ε ∈ {0.01, 0.05, 0.10, 0.20, 0.50} × σ_feature")
        print("=" * 60)

        adv_results = run_adversarial_experiment(
            trained_models=trained_models,
            X_train=X_train,
            X_test=X_test,
            y_test=y_test,
            test_df=test_df,
            returns_df=returns_df,
            feature_cols=FEATURE_COLS,
            epsilon_scales=[0.01, 0.05, 0.10, 0.20, 0.50],
            backtest_fn=run_backtest,
            backtest_kwargs=bt_kwargs,
            metrics_fn=lambda bt: compute_all_metrics(bt, total_cost_bps=20.0),
        )

        # Generate reports
        print("\n  Generating adversarial reports...")
        generate_adversarial_table(adv_results, output_dir, primary_epsilon=0.10)
        generate_adversarial_multi_epsilon_table(adv_results, output_dir)
        plot_adversarial_sharpe_degradation(adv_results, output_dir)
        plot_signal_flip_rate(adv_results, output_dir)

        all_robustness_results["adversarial"] = {
            model: {str(eps): metrics for eps, metrics in eps_dict.items()}
            for model, eps_dict in adv_results.items()
        }

        print("\n  Direction 1 COMPLETE")

    # ============================================================== #
    #  Direction 2: Synthetic Market Fuzzing
    # ============================================================== #
    fuzzing_results = {}
    if args.only in ("fuzzing", "all"):
        print("\n" + "=" * 60)
        print("DIRECTION 2: Synthetic Market Fuzzing")
        print("  Flash Crash · Volatility Spike · Gap & Reversal")
        print("=" * 60)

        fuzzing_results = run_fuzzing_experiment(
            all_predictions=all_predictions,
            returns_df=returns_df,
            backtest_fn=run_backtest,
            metrics_fn=lambda bt: compute_all_metrics(bt, total_cost_bps=20.0),
            backtest_kwargs=bt_kwargs,
            seed=42,
        )

        # Generate reports
        print("\n  Generating fuzzing reports...")
        generate_fuzzing_table(fuzzing_results, output_dir)
        plot_fuzzing_heatmap(fuzzing_results, output_dir)

        all_robustness_results["fuzzing"] = {
            scenario: {model: metrics for model, metrics in m_dict.items()}
            for scenario, m_dict in fuzzing_results.items()
        }

        print("\n  Direction 2 COMPLETE")

    # ============================================================== #
    #  Direction 3: Concept Drift & Feature Decay
    # ============================================================== #
    poison_results = {}
    decay_results = {}
    if args.only in ("drift", "all"):
        print("\n" + "=" * 60)
        print("DIRECTION 3: Concept Drift & Feature Decay Analysis")
        print("  3a: Label Poisoning (0%, 2%, 5%, 10%, 20%)")
        print("  3b: Alpha Decay Half-Life (horizons 1d → 20d)")
        print("=" * 60)

        # 3a: Label Poisoning
        print("\n  --- 3a: Label Poisoning ---")
        # Use a subset of models for efficiency
        poison_model_configs = [m for m in model_configs
                                if m["name"] not in ("MomentumBaseline", "MeanReversionBaseline")]

        poison_results = run_label_poisoning_experiment(
            model_configs=poison_model_configs,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            test_df=test_df,
            returns_df=returns_df,
            poison_rates=[0.0, 0.02, 0.05, 0.10, 0.20],
            backtest_fn=run_backtest,
            metrics_fn=lambda bt: compute_all_metrics(bt, total_cost_bps=20.0),
            backtest_kwargs=bt_kwargs,
            seed=42,
        )

        generate_poisoning_table(poison_results, output_dir)
        plot_poisoning_curve(poison_results, output_dir)

        all_robustness_results["label_poisoning"] = {
            model: {str(rate): metrics for rate, metrics in rate_dict.items()}
            for model, rate_dict in poison_results.items()
        }

        # 3b: Alpha Decay
        print("\n  --- 3b: Alpha Decay Half-Life ---")
        # Build extra horizon labels if needed
        extra_horizons = [1, 2, 3, 5, 7, 10, 15, 20]
        for h in extra_horizons:
            col = f"fwd_return_{h}d"
            if col not in panel.columns:
                print(f"    Building label for horizon {h}d...")
                panel[col] = (
                    panel.groupby("ticker")["close"]
                    .transform(lambda x: x.pct_change(h).shift(-h))
                )

        decay_model_configs = [m for m in model_configs
                               if m["name"] not in ("MomentumBaseline", "MeanReversionBaseline",
                                                    "LSTM")]  # LSTM too slow for 8 horizons

        decay_results = run_alpha_decay_experiment(
            panel=panel,
            model_configs=decay_model_configs,
            feature_cols=FEATURE_COLS,
            horizons=extra_horizons,
            split_cfg=cfg["split"],
            seed=42,
        )

        generate_alpha_decay_table(decay_results, output_dir)
        plot_alpha_decay_curve(decay_results, output_dir)

        all_robustness_results["alpha_decay"] = {
            model: df.to_dict(orient="records")
            for model, df in decay_results.items()
        }

        print("\n  Direction 3 COMPLETE")

    # ============================================================== #
    #  Summary Composite Figure
    # ============================================================== #
    if args.only == "all":
        print("\n" + "=" * 60)
        print("Generating Composite Summary Figure")
        print("=" * 60)

        plot_robustness_summary(
            adv_results=adv_results,
            fuzzing_results=fuzzing_results,
            poison_results=poison_results,
            decay_results=decay_results,
            output_dir=output_dir,
        )

    # ============================================================== #
    #  Save JSON
    # ============================================================== #
    json_path = os.path.join(output_dir, "robustness_metrics.json")

    def _convert(obj):
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
            return {k: _deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [_deep_convert(v) for v in d]
        return _convert(d)

    with open(json_path, "w") as f:
        json.dump(_deep_convert(all_robustness_results), f, indent=2, default=str)

    # ============================================================== #
    #  Final Summary
    # ============================================================== #
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  ROBUSTNESS ANALYSIS COMPLETE — {elapsed:.1f} seconds")
    print("=" * 70)

    if adv_results:
        # Show most/least vulnerable models at eps=0.10
        drops = {}
        for name, eps_dict in adv_results.items():
            entry = eps_dict.get(0.10, {})
            if entry:
                drops[name] = entry.get("sharpe_drop_%", 0)
        if drops:
            most_vuln = max(drops, key=lambda k: drops[k])
            least_vuln = min(drops, key=lambda k: drops[k])
            print(f"\n  [Adversarial] Most vulnerable:  {most_vuln} "
                  f"(Sharpe drop {drops[most_vuln]:.1f}%)")
            print(f"  [Adversarial] Most robust:      {least_vuln} "
                  f"(Sharpe drop {drops[least_vuln]:.1f}%)")

    if fuzzing_results and "Clean" in fuzzing_results:
        clean = fuzzing_results["Clean"]
        worst_scenario = None
        worst_avg_drop = 0
        for scenario, m_dict in fuzzing_results.items():
            if scenario == "Clean":
                continue
            drops_s = []
            for model, metrics in m_dict.items():
                clean_s = clean.get(model, {}).get("Sharpe (gross)", 0)
                stress_s = metrics.get("Sharpe (gross)", 0)
                if abs(clean_s) > 0.01:
                    drops_s.append((clean_s - stress_s) / abs(clean_s) * 100)
            if drops_s:
                avg_drop = np.mean(drops_s)
                if avg_drop > worst_avg_drop:
                    worst_avg_drop = avg_drop
                    worst_scenario = scenario
        if worst_scenario:
            print(f"\n  [Fuzzing] Most damaging scenario: {worst_scenario} "
                  f"(avg Sharpe drop {worst_avg_drop:.1f}%)")

    if decay_results:
        from scripts.robustness import compute_decay_halflife
        for name, df_d in decay_results.items():
            if not df_d.empty:
                hl = compute_decay_halflife(df_d["IC"], df_d["horizon"].tolist())
                hl_str = f"{hl:.1f} days" if hl != np.inf else "∞"
                print(f"  [Alpha Decay] {name}: half-life = {hl_str}")

    print(f"\n  Output directory: {output_dir}/")
    print(f"  Tables:  reports/tables/table9-12_*.csv")
    print(f"  Figures: reports/figures/fig10-15_*.pdf")
    print(f"  JSON:    {json_path}")
    print()


if __name__ == "__main__":
    main()
