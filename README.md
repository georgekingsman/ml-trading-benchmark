# ML Trading Bench

**Adversarial Training Regularises Financial Deep Learning: Evidence from a Reproducible Trading Benchmark**

Zhang Yuchen — The University of Hong Kong

---

## Overview

ML Trading Bench is a unified evaluation protocol and open-source toolkit for assessing machine learning models in cross-sectional quantitative trading. It combines a **reproducible trading benchmark** with a novel **algorithmic robustness analysis** framework, including adversarial perturbation, synthetic market fuzzing, and concept-drift diagnostics.

### Key Finding

We discover that adversarial training (Madry et al., 2018) acts as an **implicit regulariser** for deep learning models in quantitative trading:
- **LSTM**: adversarial training improves clean-data Sharpe by +53% (0.391 → 0.600) and Signal Stability Rate by +9.7 pp
- **MLP**: Signal Stability Rate improves from 85.7% to 99.1% (+13.4 pp)

This dual benefit — simultaneous robustness improvement and generalisation enhancement — is enabled by the low signal-to-noise ratio of financial data, where adversarial perturbations effectively denoise the optimisation landscape.

## Quick Start

The benchmark code is located in the `benchmark/` directory:

```bash
cd benchmark
pip install -r requirements.txt
python run_all.py                    # Standard benchmark (Tables 1-8)
python run_robustness.py             # Robustness analysis (Tables 9-12)
python run_adversarial_defense.py    # Defense experiment (Table 13)
```

See [benchmark/README.md](benchmark/README.md) for full documentation.

## Project Structure

```
benchmark/           # Main benchmark code and experiments
├── run_all.py       # One-click pipeline orchestrator
├── run_robustness.py
├── run_adversarial_defense.py
├── config/          # Settings and ETF universe
├── scripts/         # Pipeline components
├── models/          # Trained model checkpoints
└── reports/         # Generated tables, figures, and metrics
output/              # Paper source (LaTeX)
```

## Citation

```bibtex
@article{zhang2026adversarial,
  title   = {Adversarial Training Regularises Financial Deep Learning:
             Evidence from a Reproducible Trading Benchmark},
  author  = {Zhang, Yuchen},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License.
