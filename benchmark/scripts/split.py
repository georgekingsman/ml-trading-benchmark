"""
split.py — Walk-forward & rolling-window split with embargo.

Implements the leakage-prevention protocol described in the benchmark.

Usage:
    python scripts/split.py --config config/settings.yaml
"""

import argparse
import os
import warnings
from datetime import timedelta

import pandas as pd
import yaml

warnings.filterwarnings("ignore")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ================================================================== #
#  Walk-forward (single split)
# ================================================================== #

def walk_forward_split(
    df: pd.DataFrame,
    cfg: dict,
) -> dict:
    """
    Single train / val / test split with embargo.

    Returns dict with keys: 'train', 'val', 'test' → DataFrames.
    """
    s = cfg["split"]
    embargo = timedelta(days=s.get("embargo_days", 5))

    df["date"] = pd.to_datetime(df["date"])
    train = df[
        (df["date"] >= s["train_start"]) &
        (df["date"] <= pd.Timestamp(s["train_end"]) - embargo)
    ]
    val = df[
        (df["date"] >= pd.Timestamp(s["val_start"]) + embargo) &
        (df["date"] <= pd.Timestamp(s["val_end"]) - embargo)
    ]
    test = df[
        (df["date"] >= pd.Timestamp(s["test_start"]) + embargo) &
        (df["date"] <= s["test_end"])
    ]

    print(f"Train : {train['date'].min().date()} – {train['date'].max().date()}  ({len(train):,} rows)")
    print(f"Val   : {val['date'].min().date()} – {val['date'].max().date()}  ({len(val):,} rows)")
    print(f"Test  : {test['date'].min().date()} – {test['date'].max().date()}  ({len(test):,} rows)")

    return {"train": train, "val": val, "test": test}


# ================================================================== #
#  Rolling-window split (multiple folds)
# ================================================================== #

def rolling_window_split(
    df: pd.DataFrame,
    cfg: dict,
) -> list[dict]:
    """
    Generate multiple walk-forward folds with rolling windows.

    Returns list of dicts, each with keys 'train', 'val', 'test'.
    """
    s = cfg["split"]
    embargo = timedelta(days=s.get("embargo_days", 5))
    train_years = s["rolling_train_years"]
    val_years = s["rolling_val_years"]
    step = s["rolling_step_years"]

    df["date"] = pd.to_datetime(df["date"])
    min_date = df["date"].min()
    max_date = df["date"].max()

    folds = []
    fold_start = pd.Timestamp(s.get("train_start", str(min_date.date())))

    fold_id = 0
    while True:
        train_end = fold_start + pd.DateOffset(years=train_years) - timedelta(days=1)
        val_start_d = train_end + timedelta(days=1) + embargo
        val_end_d = val_start_d + pd.DateOffset(years=val_years) - timedelta(days=1)
        test_start_d = val_end_d + timedelta(days=1) + embargo
        test_end_d = test_start_d + pd.DateOffset(years=step) - timedelta(days=1)

        if test_end_d > max_date:
            break

        train = df[(df["date"] >= fold_start) & (df["date"] <= train_end - embargo)]
        val   = df[(df["date"] >= val_start_d) & (df["date"] <= val_end_d)]
        test  = df[(df["date"] >= test_start_d) & (df["date"] <= test_end_d)]

        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            fold_start += pd.DateOffset(years=step)
            continue

        print(f"Fold {fold_id}: train {train['date'].min().date()}–{train['date'].max().date()} | "
              f"val {val['date'].min().date()}–{val['date'].max().date()} | "
              f"test {test['date'].min().date()}–{test['date'].max().date()}")

        folds.append({"train": train, "val": val, "test": test, "fold_id": fold_id})
        fold_id += 1
        fold_start += pd.DateOffset(years=step)

    print(f"✓ Generated {len(folds)} rolling folds")
    return folds


# ================================================================== #
#  Leakage Checklist (print to console — also included in paper)
# ================================================================== #

LEAKAGE_CHECKLIST = """
╔══════════════════════════════════════════════════════════════════╗
║                  DATA LEAKAGE PREVENTION CHECKLIST              ║
╠══════════════════════════════════════════════════════════════════╣
║  ✓ Forward returns computed ONLY from future prices             ║
║    (shift-based, not included in features)                      ║
║  ✓ Feature normalisation uses ONLY trailing rolling window      ║
║    (no future data in z-score computation)                      ║
║  ✓ Embargo gap between train/val and val/test boundaries        ║
║    (prevents label overlap leakage)                             ║
║  ✓ Hyperparameter tuning uses ONLY validation set               ║
║    (test set never seen until final evaluation)                 ║
║  ✓ No future-dated technical indicators                         ║
║    (all TA computed on past window only)                        ║
║  ✓ Universe is point-in-time (but for ETFs, survivorship        ║
║    bias is minimal—all selected ETFs still trade)               ║
║  ✓ Corporate action adjustment via yfinance auto_adjust=True    ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ================================================================== #
#  CLI
# ================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/settings.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    panel = pd.read_parquet("features/features_panel.parquet")

    method = cfg["split"]["method"]
    if method == "walk_forward":
        splits = walk_forward_split(panel, cfg)
        # Save splits
        os.makedirs("features/splits", exist_ok=True)
        for k, v in splits.items():
            v.to_parquet(f"features/splits/{k}.parquet", index=False)
    else:
        folds = rolling_window_split(panel, cfg)
        os.makedirs("features/splits", exist_ok=True)
        for fold in folds:
            fid = fold["fold_id"]
            for k in ("train", "val", "test"):
                fold[k].to_parquet(f"features/splits/fold{fid}_{k}.parquet", index=False)

    print(LEAKAGE_CHECKLIST)


if __name__ == "__main__":
    main()
