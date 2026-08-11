"""Technical indicators computed on cached price data (never the network).

All indicators are backward-looking and vectorized: the value at row ``t``
depends only on rows ``<= t``, so there is no look-ahead.
"""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average of ``series`` over ``window`` rows."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with span ``span`` (adjust=False)."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False).mean()
    result = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    result = result.where(avg_loss > 0, 100.0)
    result = result.where((avg_loss > 0) | (avg_gain > 0), 50.0)
    return result.where(avg_gain.notna(), pd.NA)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram as a DataFrame."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": macd_line - signal_line})


def atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing of the true range."""
    high, low, close = frame["high"], frame["low"], frame["close"]
    previous_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False).mean()
