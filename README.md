# ML Trading Benchmark

A reproducible benchmark toolkit for evaluating machine-learning models in cross-sectional quantitative trading. Companion code for the ESWA survey paper *"Machine Learning for Quantitative Trading: Models, Data, Evaluation, and Practical Frontiers"*.

## Overview

| Item | Detail |
|------|--------|
| **Universe** | 50 US-listed ETFs (equity, fixed income, commodity, currency) |
| **Period** | Jan 2005 – Dec 2024 (daily OHLCV from Stooq) |
| **Features** | 13 technical indicators (returns, volatility, momentum, RSI, MA ratios, volume) |
| **Split** | Walk-forward: Train 2005-2016, Val 2017-2019, Test 2020-2024, 5-day embargo |
| **Models** | LinearRegression, Ridge, LogisticRegression, RandomForest, LightGBM, MLP, LSTM, Momentum, MeanReversion, Ensemble |
| **Strategy** | Top-K long-short (K=10, rebalance every 5 days) + long-only variant |
| **Cost model** | fee + slippage × ΔWeight, tested at 0/5/10/15/25 bps |

## Key Findings

1. All ML models show weakly positive IC (0.005–0.015) but **no model's Sharpe is statistically significant** (all bootstrap 95% CIs include zero)
2. At 15 bps transaction cost, **all long-short strategies turn deeply negative**
3. No ML-vs-ML pair is significant under the Diebold-Mariano test (3/66 pairs significant, all involving passive benchmarks)
4. Strategy hyperparameters (rebalance frequency, top-K) shift Sharpe by >0.5 — **often more than model choice**
5. Regime decomposition reveals COVID-era profits drive most headline results

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (downloads data, trains models, generates reports)
python run_all.py

# Skip data download if you already have it
python run_all.py --skip-download
```

## Repository Structure

```
├── run_all.py              # 13-step pipeline orchestrator
├── requirements.txt        # Python dependencies
├── settings.yaml           # Default configuration
├── config/
│   ├── settings.yaml       # Pipeline settings (universe, split dates, etc.)
│   └── etf_universe.csv    # List of 50 ETF tickers
├── scripts/
│   ├── data_pipeline.py    # Data download (Stooq primary, yfinance fallback)
│   ├── feature_engineering.py  # 13 technical features + z-score normalisation
│   ├── split.py            # Walk-forward train/val/test split with embargo
│   ├── models.py           # 9 model classes with unified fit()/predict()
│   ├── backtest.py         # Backtest engine (long-short, long-only, benchmarks)
│   ├── metrics.py          # Sharpe, CAGR, IC/ICIR, bootstrap CI, DM test
│   └── report.py           # 8 tables + 9 figures generation
├── paper/
│   ├── quant_trading_survey_ESWA_ready.tex   # Survey paper LaTeX source
│   ├── quant_trading_survey_ESWA_ready.pdf   # Compiled PDF
│   └── references_eswa_core60_filled.bib     # Bibliography
├── figs/                   # All generated figures (PDF)
└── tables/                 # All result tables (CSV)
```

## Output Summary

### Tables (8)
| # | File | Content |
|---|------|---------|
| T1 | `table1_main_results.csv` | Sharpe, CAGR, IC, ICIR, bootstrap CI for all models |
| T2 | `table2_cost_sensitivity.csv` | Net Sharpe at 0/5/10/15/25 bps |
| T3 | `table3_regime_analysis.csv` | Per-regime (COVID, Recovery, Rate Hikes, Normalisation) metrics |
| T4 | `table4_feature_importance.csv` | Permutation importance rankings |
| T5 | `table5_longonly_vs_longshort.csv` | Long-short vs long-only comparison |
| T6 | `table6_dm_test.csv` | Pairwise Diebold-Mariano p-values |
| T7 | `table7_rebal_sensitivity.csv` | Sharpe vs rebalance frequency (1/5/10/20 days) |
| T8 | `table8_topk_sensitivity.csv` | Sharpe vs top-K (3/5/10/15/20) |

### Figures (9)
| # | File | Content |
|---|------|---------|
| F1 | Walk-forward timeline | Train/val/test split visualisation |
| F2 | Cost sensitivity curves | Net Sharpe vs transaction cost |
| F3 | Equity curves | Cumulative returns with drawdown + regime shading |
| F4 | Ranking heatmap | Monthly model performance ranks |
| F5 | Feature importance | Bar chart of permutation importance |
| F6 | Regime bars | Per-regime Sharpe comparison |
| F7 | Rebalance sensitivity | Sharpe vs rebalance frequency |
| F8 | Top-K sensitivity | Sharpe vs portfolio concentration |
| F9 | DM heatmap | Pairwise statistical significance |

## Requirements

- Python ≥ 3.10
- See `requirements.txt` for full dependency list
- ~6 minutes runtime on Apple M-series

## Citation

If you use this benchmark in your research, please cite the companion survey:

```bibtex
@article{zhang2025mltrading,
  title={Machine Learning for Quantitative Trading: Models, Data, Evaluation, and Practical Frontiers},
  author={Zhang, Yuchen},
  journal={Expert Systems with Applications},
  year={2025}
}
```

## License

MIT
