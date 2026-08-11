"""Performance metrics: returns, volatility, ratios, drawdown, turnover."""

from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


def total_return(wealth: np.ndarray | pd.Series, initial_capital: float) -> float:
    """Final wealth minus one, relative to ``initial_capital``."""
    wealth = np.asarray(wealth, dtype=float)
    if not len(wealth):
        return 0.0
    return float(wealth[-1] / initial_capital - 1.0)


def cagr(wealth: np.ndarray | pd.Series, initial_capital: float, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Compounded annual growth rate over the realized daily periods."""
    wealth = np.asarray(wealth, dtype=float)
    periods = max(len(wealth) - 1, 0)
    if not periods or wealth[-1] <= 0:
        return -1.0
    return float((wealth[-1] / initial_capital) ** (periods_per_year / periods) - 1.0)


def annualized_volatility(returns: np.ndarray | pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized standard deviation of daily returns."""
    returns = np.asarray(returns, dtype=float)
    if not len(returns):
        return 0.0
    return float(returns.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe(returns: np.ndarray | pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized Sharpe ratio (zero excess-return baseline)."""
    returns = np.asarray(returns, dtype=float)
    if not len(returns):
        return 0.0
    std = returns.std(ddof=0)
    if std <= 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def sortino(returns: np.ndarray | pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualized Sortino ratio using downside (negative) volatility."""
    returns = np.asarray(returns, dtype=float)
    downside = returns[returns < 0]
    if not len(downside):
        return 0.0
    std = downside.std(ddof=0)
    if std <= 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(wealth: np.ndarray | pd.Series) -> float:
    """Maximum peak-to-trough decline of a wealth path (negative or zero)."""
    wealth = np.asarray(wealth, dtype=float)
    if not len(wealth):
        return 0.0
    return float((wealth / np.maximum.accumulate(wealth) - 1.0).min())


def time_under_water(wealth: np.ndarray | pd.Series) -> int:
    """Longest consecutive stretch of trading days below the running peak."""
    wealth = np.asarray(wealth, dtype=float)
    if not len(wealth):
        return 0
    running_max = np.maximum.accumulate(wealth)
    longest = 0
    current = 0
    for below in wealth < running_max:
        current = current + 1 if below else 0
        longest = max(longest, current)
    return longest


def turnover_fraction(ledger: pd.DataFrame, initial_capital: float) -> float:
    """Total one-way dollar turnover relative to ``initial_capital``."""
    if not len(ledger):
        return 0.0
    return float(ledger["turnover"].sum() / initial_capital)


def trade_count(ledger: pd.DataFrame) -> int:
    """Number of executed trade events."""
    return int(len(ledger)) if len(ledger) else 0


def hit_rate(ledger: pd.DataFrame) -> float:
    """Fraction of closed position segments that were profitable."""
    sells = ledger.loc[ledger["side"] == "sell"] if len(ledger) else ledger
    if not len(sells):
        return 0.0
    return float((sells["gross_return"] > 0).mean())


def avg_win_loss(ledger: pd.DataFrame) -> tuple[float, float]:
    """Mean per-share return of winning and losing closed segments."""
    sells = ledger.loc[ledger["side"] == "sell"] if len(ledger) else ledger
    wins = sells.loc[sells["gross_return"] > 0]
    losses = sells.loc[sells["gross_return"] < 0]
    return (
        float(wins["gross_return"].mean()) if len(wins) else 0.0,
        float(losses["gross_return"].mean()) if len(losses) else 0.0,
    )


def exposure(positions: pd.Series) -> float:
    """Mean absolute target position across all rows."""
    if not len(positions):
        return 0.0
    return float(np.abs(positions.to_numpy(dtype=float)).mean())


def compute_metrics(
    wealth: np.ndarray | pd.Series,
    returns: np.ndarray | pd.Series,
    ledger: pd.DataFrame,
    effective_positions: pd.Series,
    initial_capital: float,
) -> dict[str, float]:
    """Compute the full metric set for one simulation scenario."""
    avg_win, avg_loss = avg_win_loss(ledger)
    return {
        "initial_capital": float(initial_capital),
        "final_wealth": float(np.asarray(wealth, dtype=float)[-1]) if len(np.asarray(wealth, dtype=float)) else 0.0,
        "total_return": total_return(wealth, initial_capital),
        "cagr": cagr(wealth, initial_capital),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "maximum_drawdown": max_drawdown(wealth),
        "time_under_water_days": time_under_water(wealth),
        "turnover": turnover_fraction(ledger, initial_capital),
        "trade_count": trade_count(ledger),
        "hit_rate": hit_rate(ledger),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "exposure": exposure(effective_positions),
    }
