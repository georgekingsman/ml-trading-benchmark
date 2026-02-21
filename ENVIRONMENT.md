# Environment Guide

## Tested Platforms

| Platform | Python | Runtime | Status |
|----------|--------|---------|--------|
| macOS 14.x (Apple M2) | 3.10.19 (Miniconda) | ~6 min | ✅ Fully tested |
| macOS 14.x (Apple M2) | 3.11.x (Homebrew) | ~6 min | ✅ Compatible |
| Ubuntu 22.04 (x86_64) | 3.10+ | ~8–12 min | ✅ Expected compatible |
| Windows 11 (x86_64) | 3.10+ | ~10–15 min | ⚠️ Untested; should work |

## Python Version

**Minimum:** Python 3.10 (required for `X | Y` union type syntax).

## Installation

### Option A: pip (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Option B: conda

```bash
conda create -n mlbench python=3.10 -y
conda activate mlbench
pip install -r requirements.txt
```

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥ 2.0 | Data manipulation |
| numpy | ≥ 1.24 | Numerical computing |
| scikit-learn | ≥ 1.3 | Traditional ML models |
| lightgbm | ≥ 4.0 | Gradient boosting |
| torch | ≥ 2.0 | MLP and LSTM models |
| yfinance | ≥ 0.2.31 | Data download fallback |
| scipy | ≥ 1.11 | Statistical tests |
| statsmodels | ≥ 0.14 | BH-FDR multiple testing correction |
| matplotlib | ≥ 3.7 | Plotting |
| seaborn | ≥ 0.12 | Statistical visualisation |
| tabulate | ≥ 0.9 | Console table formatting |

## Data Download Notes

- **Primary source:** Stooq (free, no API key needed, no rate limit)
- **Fallback:** yfinance (free, but rate-limited — the pipeline handles backoff automatically)
- First run downloads ~50 ETFs × 20 years of daily data (~15 MB total)
- Subsequent runs use the Parquet cache in `data/raw/`

## Known Issues

1. **yfinance rate limits**: If Stooq is unavailable, yfinance may throttle requests. The pipeline uses exponential backoff with 30s batch cooldowns. In rare cases, you may need to re-run with `--skip-download` after partial downloads complete.

2. **PyTorch on Apple Silicon**: `torch` installs the MPS-enabled build by default on M-series Macs. The pipeline automatically detects and uses MPS when available; no manual configuration needed.

3. **Reproducibility across platforms**: Due to floating-point differences across CPU architectures, exact numerical results may vary slightly (typically <0.01 in Sharpe). The qualitative conclusions are robust.

## Disk Space

- Raw data cache: ~50 MB
- Feature cache: ~100 MB
- Output (tables + figures): ~5 MB
- **Total:** ~155 MB
