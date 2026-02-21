# Reproducibility Guide

This document provides step-by-step instructions to reproduce all results reported in the paper.

## Prerequisites

- Python ≥ 3.10 (see [ENVIRONMENT.md](ENVIRONMENT.md) for platform-specific instructions)
- Internet connection (for initial data download only)
- ~155 MB disk space

## Step-by-Step Reproduction

### 1. Clone and set up

```bash
git clone https://github.com/georgekingsman/ml-trading-benchmark.git
cd ml-trading-benchmark
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python run_all.py
```

This single command executes 13 steps:

| Step | Description | Approximate Time |
|------|-------------|-----------------|
| 1 | Download & cache 50 ETF OHLCV data (Stooq → yfinance fallback) | ~60s |
| 2 | Engineer 13 technical features + rolling z-score normalisation | ~30s |
| 3 | Walk-forward split (train/val/test) with 5-day embargo | <1s |
| 4 | Train 9 models (Linear, Ridge, Logistic, RF, LightGBM, MLP, LSTM, Momentum, MeanReversion) | ~120s |
| 5 | Backtest all models (long-short + long-only + passive benchmarks) at 5 cost scenarios | ~30s |
| 6 | Compute IC/ICIR + bootstrap Sharpe CI (B=1000) | ~15s |
| 7 | Regime analysis (COVID, Recovery, Rate Hikes, Normalisation) | ~5s |
| 8 | Permutation feature importance | ~20s |
| 9 | Ensemble construction (rank-average) | ~10s |
| 10 | Diebold-Mariano pairwise tests + BH-FDR correction | ~5s |
| 11 | Rebalance frequency sensitivity (1/5/10/20 days) | ~30s |
| 12 | Top-K sensitivity (3/5/10/15/20) | ~30s |
| 13 | Generate all tables (9) + figures (9) | ~10s |

**Total: ~6 minutes on Apple M-series.**

### 3. Verify output

After completion, check the `reports/` directory:

```bash
ls reports/tables/    # 9 CSV files + 9 LaTeX files
ls reports/figures/   # 9 PDF figures
cat reports/all_metrics.json  # Complete numerical results
```

### 4. Subsequent runs (skip data download)

```bash
python run_all.py --skip-download                  # use cached OHLCV data
python run_all.py --skip-download --skip-features   # use cached features too
```

## Expected Output Structure

```
reports/
├── tables/
│   ├── table1_main_results.csv          # Table 1: Main benchmark results
│   ├── table2_cost_sensitivity.csv      # Table 2: Cost sensitivity (Sharpe vs bps)
│   ├── table3_regime_analysis.csv       # Table 3: Per-regime Sharpe
│   ├── table4_feature_importance.csv    # Table 4: Permutation importance
│   ├── table5_longonly_vs_longshort.csv # Table 5: Long-only vs long-short
│   ├── table6_dm_test.csv              # Table 6: DM p-values (raw)
│   ├── table6b_dm_test_bh.csv          # Table 6b: DM p-values (BH-corrected)
│   ├── table7_rebal_sensitivity.csv    # Table 7: Rebalance frequency sensitivity
│   ├── table8_topk_sensitivity.csv     # Table 8: Top-K sensitivity
│   └── *.tex                           # LaTeX versions of all tables
├── figures/
│   ├── fig1_walk_forward_timeline.pdf
│   ├── fig2_cost_sensitivity.pdf
│   ├── fig3_equity_curves.pdf
│   ├── fig4_ranking_heatmap.pdf
│   ├── fig5_feature_importance.pdf
│   ├── fig6_regime_performance.pdf
│   ├── fig7_rebalance_sensitivity.pdf
│   ├── fig8_topk_sensitivity.pdf
│   └── fig9_dm_heatmap.pdf
└── all_metrics.json
```

## Configuration

All pipeline parameters are controlled via `config/settings.yaml`:

```yaml
data:
  start_date: "2005-01-01"
  end_date: "2024-12-31"

split:
  train_end: "2016-12-31"
  val_end: "2019-12-31"
  embargo_days: 5

backtest:
  top_k: 10
  rebalance_freq: 5
  cost_scenarios_bps: [0, 5, 10, 15, 25]
  slippage_bps: 5
```

To modify the benchmark (e.g., different universe, split dates, or cost scenarios), edit this file and re-run.

## Numerical Reproducibility

- **Same platform + same Python version**: Results should match within floating-point precision
- **Cross-platform**: Due to CPU architecture differences (x86 vs ARM), expect minor numerical deviations (<0.01 in Sharpe ratios). All qualitative conclusions (CI coverage, significance tests, regime patterns) are robust
- **Random seed**: Fixed at 42 via `config/settings.yaml`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Data download hangs | Re-run with `--skip-download` if partial data cached |
| yfinance rate limit | Wait 5 minutes, then re-run; Stooq data is cached |
| CUDA/MPS errors | Set `CUDA_VISIBLE_DEVICES=""` to force CPU |
| Memory errors (LSTM) | Reduce `batch_size` in settings.yaml |
