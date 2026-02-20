"""
feature_engineering.py — Compute technical features & forward-return labels.

All features are computed using ONLY past data (no look-ahead leakage).

Usage:
    python scripts/feature_engineering.py --config config/settings.yaml
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ================================================================== #
#  Feature functions — each takes a per-ticker DataFrame (sorted by date)
# ================================================================== #

def _return(series: pd.Series, period: int) -> pd.Series:
    return series.pct_change(period)


def _volatility(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window).std() * np.sqrt(252)


def _momentum(series: pd.Series, window: int) -> pd.Series:
    return series.pct_change(window)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _ma_ratio(series: pd.Series, window: int) -> pd.Series:
    return series / series.rolling(window, min_periods=window).mean()


def _volume_ratio(volume: pd.Series, window: int) -> pd.Series:
    return volume / volume.rolling(window, min_periods=window).mean().clip(lower=1)


def _high_low_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return (high - low) / close.clip(lower=1e-8)


def _close_open_range(close: pd.Series, open_: pd.Series, prev_close: pd.Series) -> pd.Series:
    return (close - open_) / prev_close.clip(lower=1e-8)


# ================================================================== #
#  Build features per ticker
# ================================================================== #

def build_features_for_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features for one ticker's DataFrame (must have OHLCV + daily_return)."""
    df = df.sort_values("date").copy()
    c = df["close"]
    r = df["daily_return"]

    df["return_1d"]       = _return(c, 1)
    df["return_5d"]       = _return(c, 5)
    df["return_20d"]      = _return(c, 20)
    df["volatility_20d"]  = _volatility(r, 20)
    df["volatility_60d"]  = _volatility(r, 60)
    df["momentum_10d"]    = _momentum(c, 10)
    df["momentum_20d"]    = _momentum(c, 20)
    df["rsi_14"]          = _rsi(c, 14)
    df["ma_ratio_10"]     = _ma_ratio(c, 10)
    df["ma_ratio_50"]     = _ma_ratio(c, 50)
    df["volume_ratio_20d"] = _volume_ratio(df["volume"], 20)
    df["high_low_range"]  = _high_low_range(df["high"], df["low"], c)
    df["close_open_range"] = _close_open_range(c, df["open"], c.shift(1))

    return df


# ================================================================== #
#  Labels (forward returns — the prediction target)
# ================================================================== #

def build_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Add forward return columns. These are the targets for regression/classification."""
    df = df.sort_values(["ticker", "date"]).copy()
    for h in horizons:
        df[f"fwd_return_{h}d"] = (
            df.groupby("ticker")["close"]
            .transform(lambda x: x.pct_change(h).shift(-h))
        )
        # Direction label (for classification)
        df[f"fwd_direction_{h}d"] = (df[f"fwd_return_{h}d"] > 0).astype(int)
    return df


# ================================================================== #
#  Rolling normalisation (avoid look-ahead)
# ================================================================== #

FEATURE_COLS = [
    "return_1d", "return_5d", "return_20d",
    "volatility_20d", "volatility_60d",
    "momentum_10d", "momentum_20d",
    "rsi_14", "ma_ratio_10", "ma_ratio_50",
    "volume_ratio_20d", "high_low_range", "close_open_range",
]


def rolling_normalize(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Z-score normalise features using a trailing window (no future info)."""
    df = df.sort_values(["ticker", "date"]).copy()
    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        grp = df.groupby("ticker")[col]
        roll_mean = grp.transform(lambda x: x.rolling(window, min_periods=60).mean())
        roll_std  = grp.transform(lambda x: x.rolling(window, min_periods=60).std())
        df[col] = (df[col] - roll_mean) / roll_std.clip(lower=1e-8)
    return df


# ================================================================== #
#  Main
# ================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # Load panel
    panel_path = os.path.join(cfg["data"]["processed_dir"], "panel.parquet")
    panel = pd.read_parquet(panel_path)
    print(f"Loaded panel: {panel.shape}")

    # Build features per ticker
    print("Building features...")
    pieces = []
    for tkr, grp in tqdm(panel.groupby("ticker"), desc="Features"):
        pieces.append(build_features_for_ticker(grp))
    panel = pd.concat(pieces)

    # Build labels
    horizons = cfg["task"]["horizons"]
    panel = build_labels(panel, horizons)

    # Rolling normalisation
    norm_method = cfg["features"].get("normalize", "rolling")
    if norm_method == "rolling":
        win = cfg["features"].get("rolling_window", 252)
        print(f"Rolling z-score normalisation (window={win})...")
        panel = rolling_normalize(panel, window=win)

    # Drop rows where features/labels are NaN (warm-up period)
    primary_h = cfg["task"]["primary_horizon"]
    required = FEATURE_COLS + [f"fwd_return_{primary_h}d"]
    panel = panel.dropna(subset=required)

    # Save
    out_path = os.path.join("features", "features_panel.parquet")
    os.makedirs("features", exist_ok=True)
    panel.to_parquet(out_path, index=False)
    print(f"✓ Features saved: {out_path}  shape={panel.shape}")


if __name__ == "__main__":
    main()
