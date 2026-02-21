"""
data_pipeline.py — Download & process ETF data via yfinance.

Usage:
    python scripts/data_pipeline.py --config config/settings.yaml
"""

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_universe(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ------------------------------------------------------------------ #
#  Download — stooq (primary, no rate limit) + yfinance (fallback)
# ------------------------------------------------------------------ #

def _normalise_df(df: pd.DataFrame) -> pd.DataFrame | None:
    """Normalise any downloaded DataFrame to standard (date-indexed, lowercase OHLCV)."""
    if df is None or df.empty or len(df) < 100:
        return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Normalise column names
    col_map = {c: c.lower() if isinstance(c, str) else str(c).lower() for c in df.columns}
    df = df.rename(columns=col_map)
    df.index.name = "date"
    needed = ["open", "high", "low", "close", "volume"]
    if not all(c in df.columns for c in needed):
        return None
    df = df[needed].copy()
    df = df.dropna(how="all")
    if len(df) < 100:
        return None
    # Sort ascending by date (stooq returns descending)
    df = df.sort_index()
    return df


def _download_stooq(tkr: str, start: str, end: str) -> pd.DataFrame | None:
    """Download from stooq.com (no API key, no rate limit)."""
    try:
        url = (
            f"https://stooq.com/q/d/l/"
            f"?s={tkr.lower()}.us&d1={start.replace('-','')}&d2={end.replace('-','')}&i=d"
        )
        df = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
        return _normalise_df(df)
    except Exception:
        return None


def _download_yfinance_single(tkr: str, start: str, end: str, max_retries: int = 3) -> pd.DataFrame | None:
    """Download one ticker from yfinance with exponential backoff."""
    import time as _time
    for attempt in range(max_retries):
        try:
            raw = yf.download(tkr, start=start, end=end, progress=False, auto_adjust=True)
            if raw is None or raw.empty:
                return None
            return _normalise_df(raw)
        except Exception as e:
            err = str(e)
            if "Rate" in err or "Too Many" in err or "429" in err:
                wait = 10 * (2 ** attempt) + np.random.uniform(0, 5)
                print(f"  ⏳ {tkr}: rate limited, waiting {wait:.0f}s (attempt {attempt+1})")
                _time.sleep(wait)
                continue
            else:
                return None
    return None


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    raw_dir: str,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data: try stooq first (no rate limit), then yfinance."""
    import time as _time
    os.makedirs(raw_dir, exist_ok=True)
    all_data = {}
    to_download = []

    # Check cache first
    for tkr in tickers:
        out_path = os.path.join(raw_dir, f"{tkr}.parquet")
        if os.path.exists(out_path):
            all_data[tkr] = pd.read_parquet(out_path)
        else:
            to_download.append(tkr)

    if not to_download:
        print(f"✓ All {len(all_data)} tickers loaded from cache")
        return all_data

    # Pass 1: Stooq (no rate limit)
    print(f"Downloading {len(to_download)} tickers from stooq...")
    for tkr in tqdm(to_download, desc="Stooq"):
        df = _download_stooq(tkr, start, end)
        if df is not None:
            out_path = os.path.join(raw_dir, f"{tkr}.parquet")
            df.to_parquet(out_path)
            all_data[tkr] = df
        _time.sleep(0.3)  # polite

    # Pass 2: yfinance for any remaining (small batches with cooldown)
    still_missing = [t for t in to_download if t not in all_data]
    if still_missing:
        print(f"\n{len(still_missing)} tickers not on stooq, trying yfinance in small batches...")
        BATCH = 5
        for i in range(0, len(still_missing), BATCH):
            batch = still_missing[i:i+BATCH]
            if i > 0:
                print(f"  ⏳ Cooldown 30s between batches...")
                _time.sleep(30)
            try:
                batch_df = yf.download(
                    batch, start=start, end=end,
                    progress=False, auto_adjust=True, group_by="ticker", threads=False,
                )
                if batch_df is not None and not batch_df.empty:
                    for tkr in batch:
                        try:
                            if len(batch) == 1:
                                df = _normalise_df(batch_df)
                            else:
                                df = _normalise_df(batch_df[tkr])  # type: ignore[index]
                            if df is not None:
                                out_path = os.path.join(raw_dir, f"{tkr}.parquet")
                                df.to_parquet(out_path)
                                all_data[tkr] = df
                        except Exception:
                            pass
            except Exception:
                # Per-ticker fallback
                for tkr in batch:
                    _time.sleep(5)
                    df = _download_yfinance_single(tkr, start, end)
                    if df is not None:
                        out_path = os.path.join(raw_dir, f"{tkr}.parquet")
                        df.to_parquet(out_path)
                        all_data[tkr] = df

    failed = [t for t in tickers if t not in all_data]
    if failed:
        print(f"\n⚠ Failed tickers ({len(failed)}): {failed}")
    print(f"✓ Downloaded {len(all_data)} / {len(tickers)} tickers")
    return all_data


# ------------------------------------------------------------------ #
#  Processing (clean, forward-fill, compute returns)
# ------------------------------------------------------------------ #
def process_prices(
    all_data: dict[str, pd.DataFrame],
    processed_dir: str,
) -> pd.DataFrame:
    """
    Build a multi-asset panel: (date, ticker) → OHLCV + daily return.
    """
    os.makedirs(processed_dir, exist_ok=True)
    frames = []
    for tkr, df in all_data.items():
        df = df.copy()
        # Forward-fill missing, then drop if still NaN
        df = df.ffill().dropna()
        df["ticker"] = tkr
        df["daily_return"] = df["close"].pct_change()
        frames.append(df)

    panel = pd.concat(frames).reset_index()
    panel = panel.dropna(subset=["daily_return"])

    # Save
    out_path = os.path.join(processed_dir, "panel.parquet")
    panel.to_parquet(out_path, index=False)
    print(f"✓ Panel saved: {out_path}  shape={panel.shape}")

    # Also save data dictionary
    _save_data_dict(panel, processed_dir)
    return panel


def _save_data_dict(panel: pd.DataFrame, processed_dir: str):
    """Generate a human-readable data dictionary."""
    lines = [
        "# Data Dictionary",
        "",
        f"Tickers : {sorted(panel['ticker'].unique().tolist())}",
        f"Date range : {panel['date'].min()} – {panel['date'].max()}",
        f"Total rows : {len(panel):,}",
        "",
        "## Fields",
        "| Column       | Type    | Description                          |",
        "|-------------|---------|--------------------------------------|",
        "| date        | date    | Trading date                         |",
        "| open        | float64 | Adjusted open price                  |",
        "| high        | float64 | Adjusted high price                  |",
        "| low         | float64 | Adjusted low price                   |",
        "| close       | float64 | Adjusted close price                 |",
        "| volume      | float64 | Trading volume                       |",
        "| ticker      | str     | ETF ticker symbol                    |",
        "| daily_return| float64 | Close-to-close daily return          |",
        "",
        "## Missing data handling",
        "- Forward-fill applied to OHLCV before return computation",
        "- Rows still NaN after ffill are dropped",
    ]
    with open(os.path.join(processed_dir, "data_dictionary.md"), "w") as f:
        f.write("\n".join(lines))


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Download & process ETF data")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings YAML",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    universe = load_universe(cfg["universe_file"])
    tickers = universe["ticker"].tolist()

    dcfg = cfg["data"]
    all_data = download_prices(
        tickers,
        start=dcfg["start_date"],
        end=dcfg["end_date"],
        raw_dir=dcfg["raw_dir"],
    )

    process_prices(all_data, processed_dir=dcfg["processed_dir"])


if __name__ == "__main__":
    main()
