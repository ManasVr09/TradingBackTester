"""Portfolio construction: equal-weight allocation and per-stock aggregation.

The pooled path is the exact sum of the per-stock accounts — each ticker is
seeded with an equal share of ``initial_capital`` and compounds on its own
timeline. Pooling therefore never compounds interleaved rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import max_drawdown, total_return


def equal_weights(tickers: list[str]) -> pd.Series:
    """Return an equal-weight vector over ``tickers`` (empty -> empty Series)."""
    if not tickers:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(tickers), index=list(tickers))


def allocate_capital(initial_capital: float, tickers: list[str]) -> float:
    """Per-ticker starting capital under an equal-weight split."""
    return float(initial_capital) / max(len(tickers), 1)


def pooled_wealth(per_stock: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    """Pool per-stock wealth into one equal-weight account.

    ``per_stock`` has columns ``ticker, date, wealth``. The pooled wealth at a
    date is the sum of the per-stock wealths; the pooled return is the
    percentage change of that sum. Dates where a ticker has no data yet
    contribute zero (its account is not yet deployed).
    """
    pivoted = per_stock.pivot(index="date", columns="ticker", values="wealth")
    pivoted = pivoted.sort_index().fillna(0.0)
    summed = pivoted.sum(axis=1)
    out = pd.DataFrame({"date": pivoted.index, "pooled_wealth": summed.to_numpy()})
    out["pooled_return"] = out["pooled_wealth"].pct_change().fillna(0.0)
    return out


def per_stock_summary(
    per_stock: pd.DataFrame,
    ledger: pd.DataFrame,
    positions: pd.Series,
    capital: float,
) -> pd.DataFrame:
    """Aggregate per-stock results from wealth, ledger, and position paths."""
    rows = []
    for ticker, group in per_stock.groupby("ticker", sort=False):
        wealth = group["wealth"].to_numpy(dtype=float)
        ticker_ledger = ledger.loc[ledger["ticker"] == ticker]
        ticker_positions = positions.loc[ticker] if ticker in positions.index.get_level_values(0) else pd.Series(dtype=float)
        rows.append({
            "ticker": ticker,
            "total_return": total_return(wealth, capital),
            "maximum_drawdown": max_drawdown(wealth),
            "trades": int((ticker_ledger["side"] == "buy").sum()),
            "exposure": float(np.abs(ticker_positions.to_numpy()).mean()) if len(ticker_positions) else 0.0,
            "turnover": float(ticker_ledger["turnover"].sum() / capital) if len(ticker_ledger) else 0.0,
        })
    return pd.DataFrame(rows)
