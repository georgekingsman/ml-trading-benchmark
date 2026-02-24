"""
metrics.py — Unified performance metrics for the benchmark.

All metrics follow the same signature:
    metric(returns_series: pd.Series) → float

Portfolio-level metrics operate on daily return series.
Signal-level metrics operate on (prediction, actual) pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


# ================================================================== #
#  Portfolio-level metrics
# ================================================================== #

def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate."""
    cum = float((1 + returns).prod())  # type: ignore[arg-type]
    n_years = len(returns) / periods_per_year
    if n_years <= 0 or cum <= 0:
        return 0.0
    return float(cum ** (1.0 / n_years) - 1.0)


def sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio."""
    excess = returns - rf / periods_per_year
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (as a positive fraction)."""
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return float(-dd.min())


def calmar(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar ratio = CAGR / MaxDrawdown."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return cagr(returns, periods_per_year) / mdd


def turnover_from_costs(cost_series: pd.Series, total_bps: float) -> float:
    """
    Average daily turnover, inferred from cost series.
    turnover = cost / (total_bps / 10_000)
    """
    if total_bps == 0:
        return 0.0
    return float(cost_series.mean() / (total_bps / 10_000))


def net_return_after_costs(returns: pd.Series, costs: pd.Series) -> float:
    """Annualised net return after costs."""
    net = returns - costs
    return cagr(net)


def hit_rate(returns: pd.Series) -> float:
    """Fraction of positive-return days."""
    return float((returns > 0).mean())


# ================================================================== #
#  Signal-level metrics (cross-sectional)
# ================================================================== #

def information_coefficient(predictions: pd.Series, actuals: pd.Series) -> float:
    """Rank IC: Spearman correlation between predictions and actual returns."""
    if len(predictions) < 3:
        return 0.0
    result = stats.spearmanr(predictions, actuals)
    corr = float(result.statistic) if hasattr(result, 'statistic') else float(result[0])  # type: ignore[union-attr,arg-type]
    return corr if not np.isnan(corr) else 0.0


def compute_ic_icir(
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> tuple[float, float]:
    """
    Compute daily cross-sectional IC and ICIR.

    Parameters
    ----------
    pred_df  : DataFrame with columns [date, ticker, prediction]
    test_df  : DataFrame with columns [date, ticker, <target_col>]

    Returns
    -------
    (mean_IC, ICIR)
    """
    merged = pred_df.merge(
        test_df[["date", "ticker", target_col]],
        on=["date", "ticker"],
    )
    daily_ics = []
    for _, grp in merged.groupby("date"):
        if len(grp) >= 5:
            ic = information_coefficient(
                grp["prediction"], grp[target_col]
            )
            daily_ics.append(ic)

    daily_ics = np.array(daily_ics)
    if len(daily_ics) == 0:
        return 0.0, 0.0
    mean_ic = float(daily_ics.mean())
    icir = float(mean_ic / daily_ics.std()) if daily_ics.std() > 0 else 0.0
    return mean_ic, icir


# ================================================================== #
#  Bootstrap confidence intervals
# ================================================================== #

def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for annualised Sharpe ratio.

    Returns (sharpe_point, lower, upper).
    """
    rng = np.random.RandomState(seed)
    n = len(returns)
    boot_sharpes = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_ret = returns.iloc[idx]
        boot_sharpes.append(sharpe(boot_ret))

    boot_sharpes = np.array(boot_sharpes)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_sharpes, alpha * 100))
    upper = float(np.percentile(boot_sharpes, (1 - alpha) * 100))
    point = sharpe(returns)
    return point, lower, upper


# ================================================================== #
#  Regime detection & per-regime metrics
# ================================================================== #

def classify_regimes(
    test_df: pd.DataFrame,
    spy_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Classify each test date into regimes:
      - COVID crash (2020-02 to 2020-06)
      - Recovery rally (2020-07 to 2021-12)
      - Rate-hike / inflation (2022-01 to 2022-12)
      - Normalisation (2023-01 onward)
    Also classify by volatility regime (rolling 20d vol above/below median).
    """
    df = test_df[["date"]].drop_duplicates().copy()
    df["date"] = pd.to_datetime(df["date"])

    # Named regimes
    conditions = [
        (df["date"] >= "2020-02-01") & (df["date"] <= "2020-06-30"),
        (df["date"] >= "2020-07-01") & (df["date"] <= "2021-12-31"),
        (df["date"] >= "2022-01-01") & (df["date"] <= "2022-12-31"),
        (df["date"] >= "2023-01-01"),
    ]
    labels = ["COVID Crash", "Recovery", "Rate Hikes", "Normalisation"]
    df["regime"] = np.select(conditions, labels, default="Pre-2020")

    # Volatility regime (if SPY returns available)
    if spy_returns is not None:
        vol_20d = spy_returns.rolling(20).std() * np.sqrt(252)
        vol_median = vol_20d.median()
        df = df.set_index("date")
        df["vol_regime"] = "Low Vol"
        high_vol_dates = vol_20d[vol_20d > vol_median].index
        df.loc[df.index.isin(high_vol_dates), "vol_regime"] = "High Vol"
        df = df.reset_index()
    else:
        df["vol_regime"] = "Unknown"

    return df


def compute_regime_metrics(
    bt_result: pd.DataFrame,
    regime_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute metrics per regime period.

    Returns DataFrame: regime × metric.
    """
    bt = bt_result.copy()
    bt["date"] = pd.to_datetime(bt["date"])
    bt = bt.merge(regime_df[["date", "regime"]], on="date", how="left")

    rows = []
    for regime, grp in bt.groupby("regime"):
        if len(grp) < 20:
            continue
        rows.append({
            "Regime": regime,
            "Days": len(grp),
            "CAGR (gross) %": round(cagr(grp["gross_return"]) * 100, 2),
            "Sharpe (gross)": round(sharpe(grp["gross_return"]), 3),
            "Sharpe (net)": round(sharpe(grp["net_return"]), 3),
            "Max DD %": round(max_drawdown(grp["gross_return"]) * 100, 2),
        })
    return pd.DataFrame(rows)


# ================================================================== #
#  Feature importance (permutation-based, model-agnostic)
# ================================================================== #

def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance: measure drop in IC when each feature is shuffled.
    """
    rng = np.random.RandomState(seed)
    baseline_preds = model.predict(X)
    valid = ~np.isnan(baseline_preds)
    if valid.sum() < 10:
        return pd.DataFrame(columns=["feature", "importance", "std"])

    baseline_ic = information_coefficient(
        pd.Series(baseline_preds[valid]), pd.Series(y[valid])
    )

    results = []
    for i, fname in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_preds = model.predict(X_perm)
            valid_p = ~np.isnan(perm_preds)
            if valid_p.sum() < 10:
                drops.append(0)
                continue
            perm_ic = information_coefficient(
                pd.Series(perm_preds[valid_p]), pd.Series(y[valid_p])
            )
            drops.append(baseline_ic - perm_ic)
        results.append({
            "feature": fname,
            "importance": np.mean(drops),
            "std": np.std(drops),
        })

    df = pd.DataFrame(results).sort_values("importance", ascending=False)
    return df


# ================================================================== #
#  Diebold-Mariano test
# ================================================================== #

def diebold_mariano_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    h: int = 1,
) -> tuple[float, float]:
    """
    Two-sided Diebold-Mariano test comparing two return series.
    H0: E[L(e_a)] = E[L(e_b)] with quadratic loss on returns.

    Parameters
    ----------
    returns_a, returns_b : daily portfolio returns of model A and B
    h : forecast horizon (for Newey-West HAC correction)

    Returns
    -------
    (DM statistic, p-value)
    """
    d = np.asarray(returns_a.values, dtype=float) - np.asarray(returns_b.values, dtype=float)  # loss differential
    n = len(d)
    if n < 10:
        return 0.0, 1.0
    d_mean = d.mean()
    # HAC variance (Newey-West with bandwidth h-1)
    gamma_0 = np.sum((d - d_mean) ** 2) / n
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.sum((d[k:] - d_mean) * (d[:-k] - d_mean)) / n
        gamma_sum += 2 * gamma_k
    var_d = (gamma_0 + gamma_sum) / n
    if var_d <= 0:
        return 0.0, 1.0
    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def pairwise_dm_matrix(
    all_bt_results: dict[str, pd.DataFrame],
    return_col: str = "gross_return",
) -> pd.DataFrame:
    """
    Compute pairwise DM test p-values.
    Returns a symmetric DataFrame of p-values.
    """
    models = list(all_bt_results.keys())
    n = len(models)
    pvals = pd.DataFrame(np.ones((n, n)), index=models, columns=models)
    for i in range(n):
        for j in range(i + 1, n):
            ret_a = all_bt_results[models[i]][return_col]
            ret_b = all_bt_results[models[j]][return_col]
            # Align by minimum length
            min_len = min(len(ret_a), len(ret_b))
            _, p = diebold_mariano_test(
                ret_a.iloc[:min_len], ret_b.iloc[:min_len], h=5
            )
            pvals.iloc[i, j] = p
            pvals.iloc[j, i] = p
    return pvals


def apply_fdr_correction(
    dm_pvals: pd.DataFrame,
    alpha: float = 0.05,
    method: str = "fdr_bh",
) -> tuple[pd.DataFrame, int, int]:
    """
    Apply Benjamini–Hochberg (FDR) correction to pairwise DM p-values.

    Parameters
    ----------
    dm_pvals : symmetric DataFrame of raw p-values from pairwise_dm_matrix
    alpha    : significance level (default 0.05)
    method   : correction method for statsmodels multipletests
               'fdr_bh' = Benjamini-Hochberg, 'bonferroni' = Bonferroni

    Returns
    -------
    (adjusted_pvals DataFrame, n_sig_raw, n_sig_corrected)
    """
    models = dm_pvals.index.tolist()
    n = len(models)

    # Extract upper triangle p-values
    raw_ps = []
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            raw_ps.append(dm_pvals.iloc[i, j])
            pairs.append((i, j))

    raw_ps_arr = np.array(raw_ps)
    n_sig_raw = int((raw_ps_arr < alpha).sum())

    # Apply correction
    reject, adj_ps, _, _ = multipletests(raw_ps_arr, alpha=alpha, method=method)
    n_sig_corrected = int(reject.sum())

    # Build adjusted DataFrame
    adj_df = pd.DataFrame(np.ones((n, n)), index=models, columns=models)
    for idx, (i, j) in enumerate(pairs):
        adj_df.iloc[i, j] = adj_ps[idx]
        adj_df.iloc[j, i] = adj_ps[idx]
    for i in range(n):
        adj_df.iloc[i, i] = 1.0

    return adj_df, n_sig_raw, n_sig_corrected


# ================================================================== #
#  Aggregate: compute all metrics for one backtest result
# ================================================================== #

def compute_all_metrics(
    bt_result: pd.DataFrame,
    total_cost_bps: float = 15.0,
) -> dict[str, float]:
    """
    Compute all portfolio metrics from a backtest result DataFrame.

    Expects columns: gross_return, net_return, cost.
    """
    gross = bt_result["gross_return"]
    net = bt_result["net_return"]
    costs = bt_result["cost"]

    return {
        "CAGR (gross)":    round(cagr(gross) * 100, 2),
        "CAGR (net)":      round(cagr(net) * 100, 2),
        "Sharpe (gross)":  round(sharpe(gross), 3),
        "Sharpe (net)":    round(sharpe(net), 3),
        "Max DD":          round(max_drawdown(gross) * 100, 2),
        "Calmar":          round(calmar(gross), 3),
        "Hit Rate":        round(hit_rate(gross) * 100, 2),
        "Avg Turnover":    round(turnover_from_costs(costs, total_cost_bps), 4),
    }

