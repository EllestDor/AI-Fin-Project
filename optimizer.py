"""
optimizer.py — Portfolio optimization, Monte Carlo simulation, and walk-forward backtesting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

from data import RISK_FREE_RATE, TRADING_DAYS


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    weights:          pd.Series          # ticker → weight
    sharpe_ratio:     float
    expected_return:  float
    volatility:       float
    sim_std_devs:     np.ndarray
    sim_returns:      np.ndarray
    sim_sharpes:      np.ndarray
    max_sharpe_idx:   int


@dataclass
class WalkForwardResult:
    oos_returns:      pd.Series           # out-of-sample daily return series
    window_summary:   pd.DataFrame        # one row per window: weights + metrics
    cumulative_curve: pd.Series           # cumulative growth (starts at 1.0)
    cagr:             float
    ann_volatility:   float
    sharpe_ratio:     float
    max_drawdown:     float


# ---------------------------------------------------------------------------
# Monte Carlo portfolio optimizer
# ---------------------------------------------------------------------------

def run_monte_carlo(
    daily_returns: pd.DataFrame,
    tickers: list[str],
    sentiment_alpha: dict[str, float] | None = None,
    alpha_weight: float = 0.05,
    num_portfolios: int = 2000,
) -> OptimizationResult:
    """
    Randomly sample *num_portfolios* weight vectors and return the one
    that maximises the Sharpe ratio.

    *sentiment_alpha* maps ticker → VADER compound score; if provided,
    the expected return for each asset is nudged by score × alpha_weight.
    """
    df          = daily_returns[tickers].dropna()
    mean_ret    = df.mean() * TRADING_DAYS
    cov_mat     = df.cov()  * TRADING_DAYS
    n           = len(tickers)

    # Adjust expected returns with NLP sentiment
    adj_returns = mean_ret.copy()
    if sentiment_alpha:
        for ticker in tickers:
            adj_returns[ticker] += sentiment_alpha.get(ticker, 0.0) * alpha_weight

    # Dirichlet(1,…,1) = uniform on the simplex (avoids equal-weight bias)
    weights = np.random.dirichlet(np.ones(n), num_portfolios)

    port_ret = weights @ adj_returns.values
    port_var = np.einsum('ij,jk,ik->i', weights, cov_mat.values, weights)
    port_std = np.sqrt(port_var)
    port_sr  = (port_ret - RISK_FREE_RATE) / port_std

    best_idx    = int(np.argmax(port_sr))
    best_w      = pd.Series(weights[best_idx], index=tickers).sort_values()

    return OptimizationResult(
        weights          = best_w,
        sharpe_ratio     = float(port_sr[best_idx]),
        expected_return  = float(port_ret[best_idx]),
        volatility       = float(port_std[best_idx]),
        sim_std_devs     = port_std,
        sim_returns      = port_ret,
        sim_sharpes      = port_sr,
        max_sharpe_idx   = best_idx,
    )


# ---------------------------------------------------------------------------
# Walk-forward backtester
# ---------------------------------------------------------------------------

def walk_forward_backtest(
    daily_returns: pd.DataFrame,
    tickers: list[str],
    train_window: int = 126,   # ~6 months
    test_window:  int = 21,    # ~1 month
    num_portfolios: int = 1000,
) -> WalkForwardResult:
    """
    Out-of-sample walk-forward optimisation.

    Algorithm
    ---------
    1. Start at row 0; take rows [i : i+train_window] as the training set.
    2. Optimise portfolio weights (max Sharpe, Monte Carlo) on that window.
    3. Apply the resulting weights to the next *test_window* rows → OOS returns.
    4. Advance *i* by *test_window* and repeat until the data is exhausted.

    The test windows are non-overlapping and strictly out-of-sample — each
    window's weights were fitted only on data that preceded it.

    Parameters
    ----------
    daily_returns : DataFrame of daily asset returns (rows = dates, cols = tickers)
    tickers       : assets to include
    train_window  : number of trading days used to fit each optimization
    test_window   : number of trading days in each out-of-sample evaluation period
    num_portfolios: Monte Carlo draws per window (lower = faster)

    Returns
    -------
    WalkForwardResult with aggregated OOS metrics and per-window details.
    """
    df = daily_returns[tickers].dropna()
    n  = len(df)

    if n < train_window + test_window:
        raise ValueError(
            f"Need at least {train_window + test_window} rows of data "
            f"(train={train_window}, test={test_window}); got {n}."
        )

    oos_pieces:   list[pd.Series]       = []
    window_rows:  list[dict]            = []

    i = 0
    window_num = 1
    while i + train_window + test_window <= n:
        train_df = df.iloc[i : i + train_window]
        test_df  = df.iloc[i + train_window : i + train_window + test_window]

        # ── Optimise on training window ──────────────────────────────────
        mean_ret = train_df.mean() * TRADING_DAYS
        cov_mat  = train_df.cov()  * TRADING_DAYS
        n_assets = len(tickers)

        weights_mat = np.random.dirichlet(np.ones(n_assets), num_portfolios)
        port_ret    = weights_mat @ mean_ret.values
        port_var    = np.einsum('ij,jk,ik->i', weights_mat, cov_mat.values, weights_mat)
        port_std    = np.sqrt(port_var)
        port_sr     = (port_ret - RISK_FREE_RATE) / port_std

        best_w = weights_mat[np.argmax(port_sr)]

        # ── Evaluate on test window (OOS) ────────────────────────────────
        oos_daily = test_df.values @ best_w          # shape (test_window,)
        oos_series = pd.Series(oos_daily, index=test_df.index, name=f"w{window_num}")
        oos_pieces.append(oos_series)

        # Record per-window metadata
        row: dict = {
            "window":            window_num,
            "train_start":       train_df.index[0],
            "train_end":         train_df.index[-1],
            "test_start":        test_df.index[0],
            "test_end":          test_df.index[-1],
            "oos_return":        float((1 + oos_series).prod() - 1),
            "oos_sharpe":        float(
                oos_series.mean() / oos_series.std() * np.sqrt(TRADING_DAYS)
                if oos_series.std() > 0 else np.nan
            ),
        }
        for t, w in zip(tickers, best_w):
            row[t] = round(float(w), 4)
        window_rows.append(row)

        i += test_window
        window_num += 1

    # ── Aggregate OOS results ────────────────────────────────────────────
    full_oos = pd.concat(oos_pieces).sort_index()

    cum_curve    = (1 + full_oos).cumprod()
    total_return = float(cum_curve.iloc[-1]) - 1.0
    n_years      = len(full_oos) / TRADING_DAYS
    cagr         = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else np.nan

    ann_vol = float(full_oos.std() * np.sqrt(TRADING_DAYS))
    sharpe  = float((cagr - RISK_FREE_RATE) / ann_vol) if ann_vol > 0 else np.nan

    # Max drawdown
    peak        = cum_curve.cummax()
    drawdowns   = (cum_curve - peak) / peak
    max_drawdown = float(drawdowns.min())

    window_summary = pd.DataFrame(window_rows).set_index("window")

    return WalkForwardResult(
        oos_returns      = full_oos,
        window_summary   = window_summary,
        cumulative_curve = cum_curve,
        cagr             = cagr,
        ann_volatility   = ann_vol,
        sharpe_ratio     = sharpe,
        max_drawdown     = max_drawdown,
    )
