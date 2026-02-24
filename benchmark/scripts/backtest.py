"""
backtest.py — Standardised backtest engine with transaction cost model.

Strategy: Top-K Long-Short
    • Rank assets by predicted signal each rebalance date
    • Go long top-K, short bottom-K (equal weight)
    • Apply transaction costs proportional to turnover

Usage:
    python scripts/backtest.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ================================================================== #
#  Transaction cost model
# ================================================================== #

def compute_transaction_costs(
    weights_prev: np.ndarray,
    weights_new: np.ndarray,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> float:
    """
    cost = (fee + slippage) * sum(|Δweight|)
    """
    total_bps = fee_bps + slippage_bps
    turnover = np.sum(np.abs(weights_new - weights_prev))
    return turnover * total_bps / 10_000


# ================================================================== #
#  Top-K Long-Short Strategy
# ================================================================== #

def run_backtest(
    predictions_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    top_k: int = 10,
    rebalance_freq: int = 5,
    cost_bps: float = 15.0,
    slippage_bps: float = 5.0,
    initial_capital: float = 1_000_000,
) -> pd.DataFrame:
    """
    Run top-K long-short backtest.

    Parameters
    ----------
    predictions_df : DataFrame with columns ['date', 'ticker', 'prediction']
    returns_df     : DataFrame with columns ['date', 'ticker', 'daily_return']
    top_k          : Number of assets to go long/short
    rebalance_freq : Rebalance every N trading days
    cost_bps       : Total one-way cost in basis points (fee component)
    slippage_bps   : Slippage in basis points
    initial_capital: Starting capital

    Returns
    -------
    DataFrame with daily portfolio returns & cumulative performance.
    """
    # Merge
    df = predictions_df.merge(returns_df[["date", "ticker", "daily_return"]], on=["date", "ticker"])
    df = df.sort_values(["date", "ticker"])

    dates = sorted(df["date"].unique())
    tickers = sorted(df["ticker"].unique())
    n_assets = len(tickers)

    # Build weight and return matrices
    portfolio_returns = []
    portfolio_costs = []
    weights = np.zeros(n_assets)  # current target weights
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    rebal_counter = 0
    for date in dates:
        day_df = df[df["date"] == date].set_index("ticker")

        # Current day returns
        day_returns = np.zeros(n_assets)
        for tkr in tickers:
            if tkr in day_df.index:
                val = day_df.loc[tkr, "daily_return"]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                day_returns[ticker_to_idx[tkr]] = val

        # 1) Portfolio return = previous weights × today's returns
        port_return = np.sum(weights * day_returns)

        # 2) Drift weights after the day
        drifted = weights * (1 + day_returns)
        w_abs = np.sum(np.abs(drifted))
        if w_abs > 1e-10:
            drifted = drifted / w_abs * np.sum(np.abs(weights)).clip(min=1e-10)

        # 3) Rebalance?
        cost = 0.0
        if rebal_counter % rebalance_freq == 0:
            # Rank by prediction
            preds = np.full(n_assets, np.nan)
            for tkr in tickers:
                if tkr in day_df.index:
                    val = day_df.loc[tkr, "prediction"]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    preds[ticker_to_idx[tkr]] = val

            valid = ~np.isnan(preds)
            if valid.sum() >= 2 * top_k:
                valid_indices = np.where(valid)[0]
                ranked = valid_indices[np.argsort(preds[valid_indices])]

                weights_new = np.zeros(n_assets)
                # Long top-K (equal weight, sums to 1)
                long_idx = ranked[-top_k:]
                weights_new[long_idx] = 1.0 / top_k
                # Short bottom-K (equal weight, sums to -1)
                short_idx = ranked[:top_k]
                weights_new[short_idx] = -1.0 / top_k

                # Transaction cost based on turnover
                cost = compute_transaction_costs(
                    drifted, weights_new, fee_bps=cost_bps, slippage_bps=slippage_bps
                )
                weights = weights_new
            else:
                weights = drifted
        else:
            weights = drifted

        portfolio_returns.append(port_return)
        portfolio_costs.append(cost)
        rebal_counter += 1

    result = pd.DataFrame({
        "date": dates,
        "gross_return": portfolio_returns,
        "cost": portfolio_costs,
    })
    result["net_return"] = result["gross_return"] - result["cost"]
    result["cum_gross"] = (1 + result["gross_return"]).cumprod()
    result["cum_net"] = (1 + result["net_return"]).cumprod()
    result["cum_gross_pnl"] = result["cum_gross"] * initial_capital
    result["cum_net_pnl"] = result["cum_net"] * initial_capital

    return result


# ================================================================== #
#  Long-Only Top-K Strategy
# ================================================================== #

def run_backtest_long_only(
    predictions_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    top_k: int = 10,
    rebalance_freq: int = 5,
    cost_bps: float = 15.0,
    slippage_bps: float = 5.0,
    initial_capital: float = 1_000_000,
) -> pd.DataFrame:
    """
    Long-only top-K variant: buy top-K predicted assets equally weighted.
    Net exposure = +1 (fully invested long).
    """
    df = predictions_df.merge(
        returns_df[["date", "ticker", "daily_return"]], on=["date", "ticker"]
    )
    df = df.sort_values(["date", "ticker"])

    dates = sorted(df["date"].unique())
    tickers = sorted(df["ticker"].unique())
    n_assets = len(tickers)
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    portfolio_returns = []
    portfolio_costs = []
    weights = np.zeros(n_assets)
    rebal_counter = 0

    for date in dates:
        day_df = df[df["date"] == date].set_index("ticker")
        day_returns = np.zeros(n_assets)
        for tkr in tickers:
            if tkr in day_df.index:
                val = day_df.loc[tkr, "daily_return"]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                day_returns[ticker_to_idx[tkr]] = val

        port_return = np.sum(weights * day_returns)

        # Drift
        drifted = weights * (1 + day_returns)
        w_sum = np.sum(drifted)
        if w_sum > 1e-10:
            drifted = drifted / w_sum  # re-normalise to sum=1

        cost = 0.0
        if rebal_counter % rebalance_freq == 0:
            preds = np.full(n_assets, np.nan)
            for tkr in tickers:
                if tkr in day_df.index:
                    val = day_df.loc[tkr, "prediction"]
                    if isinstance(val, pd.Series):
                        val = val.iloc[0]
                    preds[ticker_to_idx[tkr]] = val

            valid = ~np.isnan(preds)
            if valid.sum() >= top_k:
                valid_indices = np.where(valid)[0]
                ranked = valid_indices[np.argsort(preds[valid_indices])]

                weights_new = np.zeros(n_assets)
                long_idx = ranked[-top_k:]
                weights_new[long_idx] = 1.0 / top_k

                cost = compute_transaction_costs(
                    drifted, weights_new, fee_bps=cost_bps, slippage_bps=slippage_bps
                )
                weights = weights_new
            else:
                weights = drifted
        else:
            weights = drifted

        portfolio_returns.append(port_return)
        portfolio_costs.append(cost)
        rebal_counter += 1

    result = pd.DataFrame({
        "date": dates,
        "gross_return": portfolio_returns,
        "cost": portfolio_costs,
    })
    result["net_return"] = result["gross_return"] - result["cost"]
    result["cum_gross"] = (1 + result["gross_return"]).cumprod()
    result["cum_net"] = (1 + result["net_return"]).cumprod()
    result["cum_gross_pnl"] = result["cum_gross"] * initial_capital
    result["cum_net_pnl"] = result["cum_net"] * initial_capital
    return result


# ================================================================== #
#  Passive Benchmarks: Buy-and-Hold SPY & Equal-Weight
# ================================================================== #

def run_buy_and_hold(
    returns_df: pd.DataFrame,
    ticker: str = "SPY",
    initial_capital: float = 1_000_000,
) -> pd.DataFrame:
    """Buy-and-hold a single asset (default: SPY)."""
    spy = returns_df[returns_df["ticker"] == ticker].sort_values("date").copy()
    spy = spy.rename(columns={"daily_return": "gross_return"})
    spy["cost"] = 0.0
    spy["net_return"] = spy["gross_return"]
    spy["cum_gross"] = (1 + spy["gross_return"]).cumprod()
    spy["cum_net"] = spy["cum_gross"]
    spy["cum_gross_pnl"] = spy["cum_gross"] * initial_capital
    spy["cum_net_pnl"] = spy["cum_net"] * initial_capital
    return spy[["date", "gross_return", "cost", "net_return",
                "cum_gross", "cum_net", "cum_gross_pnl", "cum_net_pnl"]].reset_index(drop=True)


def run_equal_weight(
    returns_df: pd.DataFrame,
    initial_capital: float = 1_000_000,
) -> pd.DataFrame:
    """Equal-weight (1/N) all assets, buy-and-hold (rebalanced daily by construction)."""
    ew = returns_df.groupby("date")["daily_return"].mean().reset_index()
    ew.columns = ["date", "gross_return"]
    ew = ew.sort_values("date")
    ew["cost"] = 0.0
    ew["net_return"] = ew["gross_return"]
    ew["cum_gross"] = (1 + ew["gross_return"]).cumprod()
    ew["cum_net"] = ew["cum_gross"]
    ew["cum_gross_pnl"] = ew["cum_gross"] * initial_capital
    ew["cum_net_pnl"] = ew["cum_net"] * initial_capital
    return ew[["date", "gross_return", "cost", "net_return",
               "cum_gross", "cum_net", "cum_gross_pnl", "cum_net_pnl"]].reset_index(drop=True)


# ================================================================== #
#  Multi-cost-scenario runner
# ================================================================== #

def run_cost_scenarios(
    predictions_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    cost_scenarios_bps: list[float],
    top_k: int = 10,
    rebalance_freq: int = 5,
    slippage_bps: float = 5.0,
) -> dict[float, pd.DataFrame]:
    """Run backtest under multiple cost scenarios."""
    results = {}
    for cost_bps in cost_scenarios_bps:
        results[cost_bps] = run_backtest(
            predictions_df, returns_df,
            top_k=top_k,
            rebalance_freq=rebalance_freq,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
        )
    return results
