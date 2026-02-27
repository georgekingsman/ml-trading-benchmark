<p align="center">
  <h1 align="center">ML Trading Bench</h1>
  <p align="center">
    <strong>From Fragility to Robustness: Benchmarking &amp; Enhancing ML Models for Quantitative Trading</strong>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python 3.9+">
    <img src="https://img.shields.io/badge/Models-9+2_baselines-orange.svg" alt="Models">
    <img src="https://img.shields.io/badge/ETFs-50_US--listed-purple.svg" alt="ETFs">
  </p>
</p>

---

> **Paper:** *"From Fragility to Robustness: Benchmarking and Enhancing Machine Learning Models for Quantitative Trading under Adversarial Perturbations, Synthetic Stress, and Concept Drift"*
>
> **Author:** Zhang Yuchen — The University of Hong Kong

A unified evaluation protocol and open-source toolkit that combines a **reproducible trading benchmark** with a novel **algorithmic robustness analysis** framework. We evaluate 9 ML models + 2 passive baselines on 50 US-listed ETFs (2005–2024), producing **13 tables** and **17 figures** in a fully automated pipeline.

---

## ⚡ Quick Start

```bash
git clone https://github.com/georgekingsman/ml-trading-benchmark.git
cd ml-trading-benchmark/benchmark
pip install -r requirements.txt
python run_all.py          # → 13 tables + 17 figures in reports/
```

---

## 🔑 Key Findings

### Finding 1: The Profitability Illusion
Under realistic transaction costs (15 bps) and Benjamini–Hochberg multiple-testing corrections, **no ML model statistically outperforms a passive benchmark**.

### Finding 2: Extreme Fragility of Deep Models
Gradient-based attacks bounded within **0.1σ of historical feature noise** — perturbations *statistically indistinguishable from market noise* — cause catastrophic failure:

| Model | Sharpe (Clean) | Sharpe (Attacked) | Degradation |
|:------|:--------------:|:------------------:|:-----------:|
| LSTM  | 0.391          | −3.178             | **−913%**   |
| MLP   | 0.803          | −3.857             | **−580%**   |
| LightGBM | 0.290       | 0.270              | −6.9%       |

<p align="center">
  <img src="reports/figures/png/fig_collapse_curve.png" width="55%" alt="Collapse Curve">
</p>
<p align="center"><em>Performance collapse under increasing adversarial budget ε — deep models degrade catastrophically while tree/linear models remain stable.</em></p>

### Finding 3: The Regularisation Surprise
Adversarial training doesn't just defend — it **improves** clean-data performance for LSTM (+53% Sharpe):

| Model | Training | Clean Sharpe | Signal Stability (SSR) | Signal Flip Rate |
|:------|:---------|:------------:|:---------------------:|:----------------:|
| LSTM  | Standard      | 0.391  | 65.1%   | 34.9% |
| LSTM  | **Adversarial** | **0.600 (+53%)** | **74.8%** | **25.2%** |
| MLP   | Standard      | 0.803  | 85.7%   | 14.3% |
| MLP   | **Adversarial** | 0.698  | **99.1%** | **0.9%** |

> *Adversarial training acts as a powerful regulariser against the low signal-to-noise ratio of financial data — particularly for recurrent architectures prone to temporal noise accumulation.*

---

## 📊 Visual Highlights

<p align="center">
  <img src="reports/figures/png/fig3_cumulative_returns.png" width="48%" alt="Cumulative Returns">
  <img src="reports/figures/png/fig12_fuzzing_heatmap.png" width="48%" alt="Fuzzing Heatmap">
</p>
<p align="center"><em>Left: Cumulative returns across all models under 15 bps costs. Right: Synthetic market stress heatmap — model-specific fragility patterns invisible to historical backtests.</em></p>

<p align="center">
  <img src="reports/figures/png/fig16_adversarial_defense.png" width="48%" alt="Defense Results">
  <img src="reports/figures/png/fig17_robustness_frontier.png" width="48%" alt="Robustness Frontier">
</p>
<p align="center"><em>Left: Adversarial training defense effectiveness. Right: Robustness–performance frontier — the Pareto-optimal tradeoff across all models.</em></p>

---

## 🏗️ Pipeline Architecture

```
run_all.py                           ← One-click orchestrator
  │
  ├─ 1. Data Acquisition             ← 50 ETFs via yfinance (2005–2024)
  │     └── scripts/data_pipeline.py
  │
  ├─ 2. Feature Engineering          ← 13 technical features + forward returns
  │     └── scripts/feature_engineering.py
  │
  ├─ 3. Walk-Forward Split           ← Train/Val/Test with 5-day embargo
  │     └── scripts/split.py
  │
  ├─ 4. Model Training               ← 9 models + 2 passive baselines
  │     └── scripts/models.py
  │
  ├─ 5. Backtesting                  ← Top-K long-short, configurable costs
  │     └── scripts/backtest.py
  │
  ├─ 6. Evaluation & Statistical Tests  ← Sharpe, DM test, BH-FDR correction
  │     └── scripts/metrics.py + scripts/report.py
  │
  ├─ 7. Robustness Analysis          ← FGSM/PGD attacks, fuzzing, alpha decay
  │     └── run_robustness.py
  │
  └─ 8. Adversarial Defense          ← Min-max training + re-evaluation
        └── run_adversarial_defense.py
```

## 📋 Benchmark Protocol

| Component | Design Choice |
|:----------|:--------------|
| **Universe** | 50 US-listed ETFs — equity sectors, fixed income, commodities, currencies |
| **Period** | 2005-01-01 to 2024-12-31 (20 years) |
| **Split** | Walk-forward: Train 05–16 / Val 17–19 / Test 20–24, 5-day embargo |
| **Task** | 5-day forward return regression → top-K long-short signal |
| **Cost model** | `cost = (fee + slippage) × |Δposition|` — scenarios: 0, 5, 10, 15, 25 bps |
| **Statistical testing** | Diebold–Mariano test + Benjamini–Hochberg FDR correction (α = 0.05) |
| **Leakage control** | Rolling normalization, embargo gap, no future features, val-only tuning |

## 🤖 Models

| Category | Models |
|:---------|:-------|
| **Linear** | Linear Regression, Ridge, Logistic Regression |
| **Tree-based** | Random Forest, LightGBM |
| **Deep Learning** | MLP (tabular), LSTM (sequential) |
| **Passive Baselines** | Buy & Hold (SPY), Equal Weight |

## 🛡️ Robustness Analysis Suite

| Test | What It Measures |
|:-----|:-----------------|
| **FGSM / PGD attacks** | Adversarial fragility under gradient-based perturbations |
| **Epsilon sweep** | Collapse curves across ε ∈ {0.01, 0.05, 0.10, 0.15, 0.20, 0.30} × σ |
| **Synthetic fuzzing** | 5 stress scenarios: vol spike, correlation break, mean shift, fat tails, regime switch |
| **Label poisoning** | Robustness to corrupted training labels (5%–40% flip rates) |
| **Alpha decay** | Signal half-life estimation (2–5 day half-lives observed) |
| **Adversarial training** | Min-max defense following [Madry et al. (ICLR 2018)](https://arxiv.org/abs/1706.06083) |

## 📐 Novel Metrics Proposed

| Metric | Definition |
|:-------|:-----------|
| **Adversarial Sharpe Ratio** | Sharpe ratio under worst-case ε-bounded perturbation |
| **Signal Flip Rate (SFR)** | Fraction of predictions that change sign under attack |
| **Alpha Decay Half-Life** | Trading days until model's excess return decays to 50% |

---

## 📂 Output Structure

```
reports/
├── all_metrics.json                        # Machine-readable full results
├── robustness_metrics.json                 # Robustness analysis results
├── adversarial_defense_metrics.json        # Defense experiment results
├── tables/
│   ├── table1_main_results.{csv,tex}       # Main benchmark (9 models × 12 metrics)
│   ├── table2_cost_sensitivity.{csv,tex}   # Performance across 5 cost scenarios
│   ├── table3_regime_analysis.{csv,tex}    # Bull/bear/recovery decomposition
│   ├── table9_adversarial_robustness.{csv,tex}
│   ├── table13_adversarial_defense.{csv,tex}
│   └── ...                                 # 13 tables total (CSV + LaTeX)
└── figures/
    ├── fig1_walk_forward_timeline.pdf
    ├── fig3_cumulative_returns.pdf
    ├── fig_collapse_curve.pdf
    ├── fig12_fuzzing_heatmap.pdf
    └── ...                                 # 17 figures total (publication-ready PDF)
```

## ⚙️ Configuration

All parameters in [`config/settings.yaml`](config/settings.yaml):

```yaml
data:
  start_date: "2005-01-01"
  end_date: "2024-12-31"
split:
  method: "walk_forward"
  embargo_days: 5
backtest:
  top_k: 10
  rebalance_days: 5
  cost_scenarios_bps: [0, 5, 10, 15, 25]
```

## 🔄 Reproducing Results

```bash
python run_all.py                    # Full pipeline (≈ 20 min on M1 Mac)
python run_all.py --skip-download    # Use cached data
python run_robustness.py             # Robustness analysis only
python run_adversarial_defense.py    # Adversarial defense only
python verify_consistency.py         # Verify pipeline consistency
```

## 📝 Citation

```bibtex
@article{zhang2026fragility,
  title   = {From Fragility to Robustness: Benchmarking and Enhancing Machine
             Learning Models for Quantitative Trading under Adversarial
             Perturbations, Synthetic Stress, and Concept Drift},
  author  = {Zhang, Yuchen},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built at The University of Hong Kong · AI, Ethics and Society Programme</sub>
</p>
