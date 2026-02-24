# ML Trading Benchmark Toolkit

**A reproducible benchmark protocol for evaluating ML models in quantitative trading.**

This toolkit accompanies the survey paper *"Machine Learning for Quantitative Trading: Models, Data, Evaluation, and Practical Frontiers"* and provides a unified, end-to-end pipeline from data acquisition to performance reporting.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full benchmark (data → features → train → backtest → report)
python run_all.py

# Or use Make:
make install
make benchmark
```

## What's Inside

```
benchmark/
├── config/
│   ├── settings.yaml          # All parameters (data, models, costs, splits)
│   └── etf_universe.csv       # 50 US ETFs across 5 asset classes
├── scripts/
│   ├── data_pipeline.py       # Download & process OHLCV from yfinance
│   ├── feature_engineering.py # 13 technical features + forward-return labels
│   ├── split.py               # Walk-forward & rolling window (with embargo)
│   ├── models.py              # 9 baselines: LR, Ridge, LogReg, RF, LGBM, MLP, LSTM, Momentum, MeanRev
│   ├── backtest.py            # Top-K long-short strategy + transaction cost model
│   ├── metrics.py             # CAGR, Sharpe, MaxDD, Calmar, Turnover, IC, ICIR, HitRate
│   └── report.py              # Tables (CSV+LaTeX) & figures (PDF) generation
├── run_all.py                 # One-click full pipeline
├── Makefile                   # Convenience targets
└── requirements.txt
```

## Benchmark Design (5-Component Protocol)

| Component | Description |
|-----------|-------------|
| **Task definition** | Return forecasting (1/5/20-day forward returns) → top-K long-short signal |
| **Split protocol** | Walk-forward: Train 2005–2016, Val 2017–2019, Test 2020–2024 + embargo |
| **Cost model** | `cost = (fee + slippage) × |Δposition|`; scenarios: 0, 5, 10, 15, 25 bps |
| **Metrics** | CAGR, Sharpe, MaxDD, Calmar, Turnover, Hit Rate, IC, ICIR |
| **Leakage checklist** | Rolling normalization, embargo gap, no future features, val-only tuning |

## Universe: 50 US ETFs

Covering **equity sectors** (SPY, QQQ, XLK, XLF, XLE, …), **fixed income** (TLT, AGG, HYG, …), **commodities** (GLD, USO, DBA, …), and **currencies** (UUP, FXE, FXY, …) — a true multi-asset benchmark.

## Models

| Category | Models |
|----------|--------|
| Traditional ML | Linear Regression, Ridge, Logistic Regression, Random Forest, LightGBM |
| Deep Learning | MLP (tabular), LSTM (sequential) |
| Strategy baselines | Momentum (past 20d return), Mean Reversion (contrarian) |

## Outputs

After running, you get:

- `reports/tables/table1_main_results.csv` — Main benchmark (models × metrics)
- `reports/tables/table2_cost_sensitivity.csv` — Performance across cost scenarios
- `reports/figures/fig1_walk_forward_timeline.pdf` — Split protocol diagram
- `reports/figures/fig2_cost_sensitivity.pdf` — Sharpe vs. cost curves
- `reports/figures/fig3_cumulative_returns.pdf` — Cumulative PnL
- `reports/figures/fig4_ranking_heatmap.pdf` — Model ranking stability
- `reports/all_metrics.json` — Machine-readable full results

## Reproducing Results

```bash
# Use cached data (skip re-download)
python run_all.py --skip-download

# Use cached features too
python run_all.py --skip-download --skip-features

# Custom config
python run_all.py --config config/my_custom.yaml
```

## Configuration

All parameters are in `config/settings.yaml`. Key sections:

- **data**: source, date range, frequency
- **task**: regression/classification, horizons
- **split**: walk-forward dates, embargo days
- **models**: enable/disable, hyperparameters
- **backtest**: strategy, top-K, cost scenarios
- **metrics**: which metrics to compute

## License

Released for academic research. See the accompanying paper for citation.
