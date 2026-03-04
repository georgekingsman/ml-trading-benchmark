"""
run_all.py — One-click benchmark pipeline  (v2 — enhanced).

  python run_all.py                         # full pipeline
  python run_all.py --skip-download         # skip data download (use cached)
  python run_all.py --skip-features         # skip feature engineering (use cached)
  python run_all.py --skip-download --skip-features  # skip both
  python run_all.py --config config/settings.yaml

Pipeline:
  1. Download & process data
  2. Feature engineering
  3. Train/val/test split
  4. Train models & generate predictions
  5. Backtest  (long-short + long-only + passive benchmarks)
  6. IC / ICIR / Bootstrap CI
  7. Regime analysis
  8. Feature importance
  9. Compute metrics & generate reports  (5 tables + 6 figures)
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
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from scripts.data_pipeline import download_prices, load_universe, process_prices
from scripts.feature_engineering import (
    FEATURE_COLS,
    build_features_for_ticker,
    build_labels,
    rolling_normalize,
)
from scripts.split import walk_forward_split, LEAKAGE_CHECKLIST
from scripts.models import build_model, MODEL_REGISTRY
from scripts.backtest import (
    run_backtest,
    run_cost_scenarios,
    run_backtest_long_only,
    run_buy_and_hold,
    run_equal_weight,
)
from scripts.metrics import (
    compute_all_metrics,
    information_coefficient,
    compute_ic_icir,
    bootstrap_sharpe_ci,
    classify_regimes,
    compute_regime_metrics,
    permutation_importance,
    sharpe,
    pairwise_dm_matrix,
    apply_fdr_correction,
)
from scripts.report import (
    generate_main_table,
    generate_cost_sensitivity_table,
    generate_regime_table,
    generate_feature_importance_table,
    generate_longonly_comparison_table,
    generate_dm_table,
    generate_rebalance_sensitivity_table,
    generate_topk_sensitivity_table,
    plot_walk_forward_timeline,
    plot_cost_sensitivity_curves,
    plot_cumulative_returns,
    plot_ranking_heatmap,
    plot_feature_importance,
    plot_regime_bars,
    plot_rebalance_sensitivity,
    plot_topk_sensitivity,
    plot_dm_heatmap,
)
from scripts.paper_tables import (
    write_main_table,
    write_regime_table,
    write_sensitivity_table,
    write_longonly_table,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ML Trading Benchmark — Full Pipeline")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--skip-download", action="store_true", help="Use cached data")
    parser.add_argument("--skip-features", action="store_true", help="Use cached features")
    args = parser.parse_args()

    cfg = load_config(args.config)
    np.random.seed(cfg["output"]["seed"])

    t0 = time.time()

    # ============================================================== #
    #  Step 1: Data
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 1: Data Download & Processing")
    print("=" * 60)

    dcfg = cfg["data"]
    panel_path = os.path.join(dcfg["processed_dir"], "panel.parquet")

    if args.skip_download and os.path.exists(panel_path):
        print("  Using cached panel data")
        panel = pd.read_parquet(panel_path)
    else:
        universe = load_universe(cfg["universe_file"])
        tickers = universe["ticker"].tolist()
        all_data = download_prices(
            tickers, dcfg["start_date"], dcfg["end_date"], dcfg["raw_dir"]
        )
        panel = process_prices(all_data, dcfg["processed_dir"])

    # ============================================================== #
    #  Step 2: Feature Engineering
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)

    feat_path = "features/features_panel.parquet"
    if args.skip_features and os.path.exists(feat_path):
        print("  Using cached features")
        panel = pd.read_parquet(feat_path)
    else:
        pieces = []
        for tkr, grp in tqdm(panel.groupby("ticker"), desc="Building features"):
            pieces.append(build_features_for_ticker(grp))
        panel = pd.concat(pieces)

        horizons = cfg["task"]["horizons"]
        panel = build_labels(panel, horizons)

        norm_method = cfg["features"].get("normalize", "rolling")
        if norm_method == "rolling":
            win = cfg["features"].get("rolling_window", 252)
            print(f"Rolling z-score normalisation (window={win})...")
            panel = rolling_normalize(panel, window=win)

        primary_h = cfg["task"]["primary_horizon"]
        required = FEATURE_COLS + [f"fwd_return_{primary_h}d"]
        panel = panel.dropna(subset=required)

        os.makedirs("features", exist_ok=True)
        panel.to_parquet(feat_path, index=False)
        print(f"  Features saved ({panel.shape})")

    # ============================================================== #
    #  Step 3: Split
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 3: Walk-Forward Split")
    print("=" * 60)

    splits = walk_forward_split(panel, cfg)
    print(LEAKAGE_CHECKLIST)

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

    # ============================================================== #
    #  Step 4: Train Models & Predict
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 4: Training Models")
    print("=" * 60)

    model_configs = [m for m in cfg["models"] if m.get("enabled", True)]
    all_predictions = {}   # model_name → pred_df
    trained_models = {}    # model_name → model object (for feature importance)

    os.makedirs("models", exist_ok=True)

    for mcfg in model_configs:
        name = mcfg["name"]
        params = mcfg.get("params", {})
        print(f"\n  Training: {name}")

        model = build_model(name, params)
        model.fit(X_train, y_train, X_val, y_val)
        trained_models[name] = model

        # Save checkpoint for PyTorch models (ensures cross-script consistency)
        ckpt_path = f"models/{name}_checkpoint.pt"
        try:
            model.save_checkpoint(ckpt_path)
            print(f"    ✓ Checkpoint saved: {ckpt_path}")
        except NotImplementedError:
            pass  # Traditional ML models don't need checkpointing

        preds = model.predict(X_test)
        pred_df = test_df[["date", "ticker"]].copy()
        pred_df["prediction"] = preds
        pred_df = pred_df.dropna(subset=["prediction"])
        all_predictions[name] = pred_df
        print(f"    {name}: {len(pred_df)} predictions")

    # ============================================================== #
    #  Step 5: Backtest (long-short + long-only + benchmarks)
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 5: Backtesting")
    print("=" * 60)

    bcfg = cfg["backtest"]
    cost_scenarios = bcfg["cost_scenarios_bps"]
    returns_df = test_df[["date", "ticker", "daily_return"]].copy()
    primary_cost = 15.0

    all_bt_results = {}      # model → bt_df (long-short @ primary cost)
    all_metrics_main = {}    # model → metrics dict
    all_cost_metrics = {}    # model → {cost_bps → metrics}
    all_bt_lo = {}           # model → bt_df (long-only @ primary cost)
    all_metrics_lo = {}      # model → metrics dict (long-only)

    os.makedirs("backtest", exist_ok=True)

    for name, pred_df in all_predictions.items():
        print(f"\n  Backtesting: {name}")

        # --- Long-Short (multi-cost) ---
        cost_results = run_cost_scenarios(
            pred_df, returns_df,
            cost_scenarios_bps=cost_scenarios,
            top_k=bcfg["top_k"],
            rebalance_freq=bcfg["rebalance_freq"],
            slippage_bps=bcfg["slippage_bps"],
        )
        bt_main = cost_results.get(primary_cost, list(cost_results.values())[0])
        all_bt_results[name] = bt_main

        cost_m = {}
        for cost_bps, bt_df in cost_results.items():
            m = compute_all_metrics(bt_df, total_cost_bps=cost_bps + bcfg["slippage_bps"])
            cost_m[cost_bps] = m
        all_cost_metrics[name] = cost_m
        all_metrics_main[name] = cost_m[primary_cost]

        bt_main.to_parquet(f"backtest/{name}_backtest.parquet", index=False)

        # --- Long-Only ---
        bt_lo = run_backtest_long_only(
            pred_df, returns_df,
            top_k=bcfg["top_k"],
            rebalance_freq=bcfg["rebalance_freq"],
            cost_bps=primary_cost,
            slippage_bps=bcfg["slippage_bps"],
        )
        all_bt_lo[name] = bt_lo
        all_metrics_lo[name] = compute_all_metrics(
            bt_lo, total_cost_bps=primary_cost + bcfg["slippage_bps"]
        )

        print(f"    LS Sharpe(g)={all_metrics_main[name]['Sharpe (gross)']:.3f}  "
              f"LO Sharpe(g)={all_metrics_lo[name]['Sharpe (gross)']:.3f}")

    # --- Passive benchmarks ---
    print("\n  Computing passive benchmarks...")
    bt_bh = run_buy_and_hold(returns_df, ticker="SPY")
    bt_ew = run_equal_weight(returns_df)
    all_bt_results["BuyAndHold_SPY"] = bt_bh
    all_bt_results["EqualWeight"] = bt_ew
    all_metrics_main["BuyAndHold_SPY"] = compute_all_metrics(bt_bh, total_cost_bps=0)
    all_metrics_main["EqualWeight"] = compute_all_metrics(bt_ew, total_cost_bps=0)
    # Also give BH/EW same cost-scenario entries (cost=0 for all)
    for bench_name, bench_bt in [("BuyAndHold_SPY", bt_bh), ("EqualWeight", bt_ew)]:
        all_cost_metrics[bench_name] = {
            c: compute_all_metrics(bench_bt, total_cost_bps=0) for c in cost_scenarios
        }
    print(f"    SPY B&H Sharpe={all_metrics_main['BuyAndHold_SPY']['Sharpe (gross)']:.3f}")
    print(f"    EW 1/N  Sharpe={all_metrics_main['EqualWeight']['Sharpe (gross)']:.3f}")

    # ============================================================== #
    #  Step 6: IC / ICIR / Bootstrap CI
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 6: Signal Quality & Statistical Significance")
    print("=" * 60)

    for name, pred_df in all_predictions.items():
        # IC / ICIR
        mean_ic, icir = compute_ic_icir(pred_df, test_df, target_col)
        all_metrics_main[name]["IC"] = round(mean_ic, 4)
        all_metrics_main[name]["ICIR"] = round(icir, 4)

        # Bootstrap CI on Sharpe(gross)
        bt = all_bt_results[name]
        point, lo, hi = bootstrap_sharpe_ci(bt["gross_return"], n_bootstrap=1000)
        all_metrics_main[name]["Sharpe CI_lo"] = round(lo, 3)
        all_metrics_main[name]["Sharpe CI_hi"] = round(hi, 3)

        print(f"  {name:25s}  IC={mean_ic:.4f}  ICIR={icir:.3f}  "
              f"Sharpe 95% CI=[{lo:.3f}, {hi:.3f}]")

    # Benchmarks don't have IC — fill NaN
    for bench in ("BuyAndHold_SPY", "EqualWeight"):
        all_metrics_main[bench]["IC"] = float("nan")
        all_metrics_main[bench]["ICIR"] = float("nan")
        bt = all_bt_results[bench]
        point, lo, hi = bootstrap_sharpe_ci(bt["gross_return"], n_bootstrap=1000)
        all_metrics_main[bench]["Sharpe CI_lo"] = round(lo, 3)
        all_metrics_main[bench]["Sharpe CI_hi"] = round(hi, 3)

    # ============================================================== #
    #  Step 7: Regime Analysis
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 7: Regime Analysis")
    print("=" * 60)

    # Get SPY returns for vol regime
    spy_rets = returns_df[returns_df["ticker"] == "SPY"].set_index("date")["daily_return"]
    regime_df = classify_regimes(test_df, spy_returns=spy_rets)

    regime_results = {}  # model → DataFrame
    for name, bt in all_bt_results.items():
        rdf = compute_regime_metrics(bt, regime_df)
        regime_results[name] = rdf
        print(f"  {name}: {len(rdf)} regimes computed")

    # ============================================================== #
    #  Step 8: Feature Importance
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 8: Feature Importance (permutation-based)")
    print("=" * 60)

    fi_models = ["LightGBM", "RandomForest", "LinearRegression"]
    fi_results = {}
    for name in fi_models:
        if name not in trained_models:
            continue
        print(f"  Computing importance for {name}...")
        fi = permutation_importance(
            trained_models[name], X_test, y_test, FEATURE_COLS,
            n_repeats=5, seed=42,
        )
        fi_results[name] = fi
        top3 = fi.head(3)["feature"].tolist()
        print(f"    Top-3: {top3}")

    # ============================================================== #
    #  Step 9: Ensemble (rank-average of all ML models)
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 9: Ensemble Model")
    print("=" * 60)

    # Build ensemble prediction via cross-sectional rank averaging
    ml_models = [n for n in all_predictions if n not in
                 ("MomentumBaseline", "MeanReversionBaseline")]
    ensemble_parts = []
    for name in ml_models:
        pdf = all_predictions[name].copy()
        # Cross-sectional rank per day
        pdf["rank"] = pdf.groupby("date")["prediction"].rank(pct=True)
        pdf = pdf.rename(columns={"rank": f"rank_{name}"})
        ensemble_parts.append(pdf[["date", "ticker", f"rank_{name}"]])

    if len(ensemble_parts) >= 2:
        from functools import reduce
        ens_df = reduce(lambda a, b: a.merge(b, on=["date", "ticker"], how="inner"),
                        ensemble_parts)
        rank_cols = [c for c in ens_df.columns if c.startswith("rank_")]
        ens_df["prediction"] = ens_df[rank_cols].mean(axis=1)
        ens_pred = ens_df[["date", "ticker", "prediction"]].copy()
        all_predictions["Ensemble"] = ens_pred

        # Backtest ensemble (long-short)
        bt_ens = run_backtest(
            ens_pred, returns_df,
            top_k=bcfg["top_k"], rebalance_freq=bcfg["rebalance_freq"],
            cost_bps=primary_cost, slippage_bps=bcfg["slippage_bps"],
        )
        all_bt_results["Ensemble"] = bt_ens
        all_metrics_main["Ensemble"] = compute_all_metrics(
            bt_ens, total_cost_bps=primary_cost + bcfg["slippage_bps"]
        )

        # Ensemble IC/ICIR/CI
        mean_ic, icir = compute_ic_icir(ens_pred, test_df, target_col)
        all_metrics_main["Ensemble"]["IC"] = round(mean_ic, 4)
        all_metrics_main["Ensemble"]["ICIR"] = round(icir, 4)
        point, lo, hi = bootstrap_sharpe_ci(bt_ens["gross_return"], n_bootstrap=1000)
        all_metrics_main["Ensemble"]["Sharpe CI_lo"] = round(lo, 3)
        all_metrics_main["Ensemble"]["Sharpe CI_hi"] = round(hi, 3)

        # Ensemble cost scenarios
        ens_cost = {}
        for cbps in cost_scenarios:
            bt_c = run_backtest(
                ens_pred, returns_df,
                top_k=bcfg["top_k"], rebalance_freq=bcfg["rebalance_freq"],
                cost_bps=cbps, slippage_bps=bcfg["slippage_bps"],
            )
            ens_cost[cbps] = compute_all_metrics(bt_c, total_cost_bps=cbps + bcfg["slippage_bps"])
        all_cost_metrics["Ensemble"] = ens_cost

        # Ensemble long-only
        from scripts.backtest import run_backtest_long_only as _run_lo
        bt_ens_lo = _run_lo(
            ens_pred, returns_df,
            top_k=bcfg["top_k"], rebalance_freq=bcfg["rebalance_freq"],
            cost_bps=primary_cost, slippage_bps=bcfg["slippage_bps"],
        )
        all_metrics_lo["Ensemble"] = compute_all_metrics(
            bt_ens_lo, total_cost_bps=primary_cost + bcfg["slippage_bps"]
        )

        # Ensemble regime
        ens_regime = compute_regime_metrics(bt_ens, regime_df)
        regime_results["Ensemble"] = ens_regime

        print(f"  Ensemble: Sharpe(g)={all_metrics_main['Ensemble']['Sharpe (gross)']:.3f}  "
              f"IC={mean_ic:.4f}  ICIR={icir:.3f}")
    else:
        print("  Skipped — fewer than 2 ML models available")

    # ============================================================== #
    #  Step 10: Diebold-Mariano Test
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 10: Diebold-Mariano Pairwise Test")
    print("=" * 60)

    dm_pvals = pairwise_dm_matrix(all_bt_results, return_col="gross_return")
    sig_pairs = (dm_pvals < 0.05).sum().sum() // 2
    total_pairs = len(dm_pvals) * (len(dm_pvals) - 1) // 2
    print(f"  {sig_pairs}/{total_pairs} pairs significant at 5% (raw)")

    # Benjamini-Hochberg FDR correction
    dm_pvals_bh, n_raw, n_bh = apply_fdr_correction(dm_pvals, alpha=0.05, method="fdr_bh")
    print(f"  {n_bh}/{total_pairs} pairs significant at 5% (BH-corrected)")

    # ============================================================== #
    #  Step 11: Rebalance Frequency Sensitivity
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 11: Rebalance Frequency Sensitivity")
    print("=" * 60)

    rebal_freqs = [1, 5, 10, 20]
    # Run for a subset of representative models to save time
    rebal_models = ["LogisticRegression", "LightGBM", "MLP", "LSTM", "MomentumBaseline"]
    if "Ensemble" in all_predictions:
        rebal_models.append("Ensemble")
    rebal_results = {}

    for name in rebal_models:
        if name not in all_predictions:
            continue
        pred_df = all_predictions[name]
        freq_metrics = {}
        for freq in rebal_freqs:
            bt = run_backtest(
                pred_df, returns_df,
                top_k=bcfg["top_k"], rebalance_freq=freq,
                cost_bps=primary_cost, slippage_bps=bcfg["slippage_bps"],
            )
            freq_metrics[freq] = compute_all_metrics(
                bt, total_cost_bps=primary_cost + bcfg["slippage_bps"]
            )
        rebal_results[name] = freq_metrics
        sharpes = [freq_metrics[f]["Sharpe (gross)"] for f in rebal_freqs]
        print(f"  {name:25s}  freq={rebal_freqs}  Sharpe(g)={[f'{s:.3f}' for s in sharpes]}")

    # ============================================================== #
    #  Step 12: Top-K Sensitivity
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 12: Top-K Sensitivity")
    print("=" * 60)

    topk_values = [3, 5, 10, 15, 20]
    topk_models = ["LogisticRegression", "LightGBM", "MLP", "LSTM"]
    if "Ensemble" in all_predictions:
        topk_models.append("Ensemble")
    topk_results = {}

    for name in topk_models:
        if name not in all_predictions:
            continue
        pred_df = all_predictions[name]
        k_metrics = {}
        for k in topk_values:
            bt = run_backtest(
                pred_df, returns_df,
                top_k=k, rebalance_freq=bcfg["rebalance_freq"],
                cost_bps=primary_cost, slippage_bps=bcfg["slippage_bps"],
            )
            k_metrics[k] = compute_all_metrics(
                bt, total_cost_bps=primary_cost + bcfg["slippage_bps"]
            )
        topk_results[name] = k_metrics
        sharpes = [k_metrics[k]["Sharpe (gross)"] for k in topk_values]
        print(f"  {name:25s}  K={topk_values}  Sharpe(g)={[f'{s:.3f}' for s in sharpes]}")

    # ============================================================== #
    #  Step 13: Reports
    # ============================================================== #
    print("\n" + "=" * 60)
    print("STEP 13: Generating Reports")
    print("=" * 60)

    output_dir = "reports"
    os.makedirs(os.path.join(output_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # Table 1: Main results (now includes IC, ICIR, bootstrap CI)
    generate_main_table(all_metrics_main, output_dir)

    # Table 2: Cost sensitivity
    generate_cost_sensitivity_table(all_cost_metrics, output_dir)

    # Table 3: Regime analysis
    generate_regime_table(regime_results, output_dir)

    # Table 4: Feature importance
    if fi_results:
        generate_feature_importance_table(fi_results, output_dir)

    # Table 5: Long-only vs Long-short
    generate_longonly_comparison_table(all_metrics_main, all_metrics_lo, output_dir)

    # Table 6: DM test (raw + BH-corrected)
    generate_dm_table(dm_pvals, output_dir, dm_pvals_bh=dm_pvals_bh)

    # Table 7: Rebalance sensitivity
    generate_rebalance_sensitivity_table(rebal_results, output_dir)

    # Table 8: Top-K sensitivity
    generate_topk_sensitivity_table(topk_results, output_dir)

    # ── Paper-format tables (for \input{} in manuscript) ──
    paper_dir = os.path.join(output_dir, "tables", "paper")
    os.makedirs(paper_dir, exist_ok=True)
    print("\n  Generating paper-format tables...")
    write_main_table(all_metrics_main, paper_dir)
    write_regime_table(regime_results, paper_dir)
    write_sensitivity_table(rebal_results, topk_results, paper_dir)
    write_longonly_table(all_metrics_main, all_metrics_lo, paper_dir)

    # Figures 1-6
    plot_walk_forward_timeline(output_dir)
    plot_cost_sensitivity_curves(all_cost_metrics, output_dir)
    plot_cumulative_returns(all_bt_results, output_dir, cost_bps=primary_cost)
    plot_ranking_heatmap(all_cost_metrics, output_dir)
    if fi_results:
        plot_feature_importance(fi_results, output_dir)
    plot_regime_bars(regime_results, output_dir)

    # Figures 7-9
    plot_rebalance_sensitivity(rebal_results, output_dir)
    plot_topk_sensitivity(topk_results, output_dir)
    plot_dm_heatmap(dm_pvals, output_dir)

    # Save all metrics as JSON
    json_path = os.path.join(output_dir, "all_metrics.json")
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    serializable = {}
    for model, metrics in all_metrics_main.items():
        serializable[model] = {k: _convert(v) for k, v in metrics.items()}

    # Also save cost-scenario data
    cost_ser = {}
    for model, cost_dict in all_cost_metrics.items():
        cost_ser[model] = {}
        for cost, metrics in cost_dict.items():
            cost_ser[model][str(cost)] = {k: _convert(v) for k, v in metrics.items()}

    with open(json_path, "w") as f:
        json.dump({"main": serializable, "cost_scenarios": cost_ser}, f, indent=2)

    # ============================================================== #
    #  Summary
    # ============================================================== #
    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"BENCHMARK COMPLETE — {elapsed:.1f} seconds")
    print("=" * 60)
    print(f"  Models tested: {len(all_predictions)} (incl. Ensemble) + 2 passive benchmarks")
    print(f"  Assets:        {len(test_df['ticker'].unique())}")
    print(f"  Test period:   {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"  Tables:        reports/tables/ (8 tables)")
    print(f"  Figures:       reports/figures/ (9 figures)")
    print(f"  Metrics JSON:  {json_path}")
    print()


if __name__ == "__main__":
    main()
