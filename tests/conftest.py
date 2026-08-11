"""Shared pytest fixtures for the trading backtester test suite."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """A 60-row single-ticker frame with a RangeIndex, like a cached frame."""
    dates = pd.bdate_range("2024-01-01", periods=60)
    closes = [100.0 + 0.5 * i for i in range(60)]
    return pd.DataFrame({
        "date": dates,
        "open": [c - 0.2 for c in closes],
        "high": [c + 0.5 for c in closes],
        "low": [c - 0.8 for c in closes],
        "close": closes,
        "volume": [1_000_000] * 60,
    })
