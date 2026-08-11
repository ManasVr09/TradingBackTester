"""Portfolio tests: allocation, pooled path, per-stock aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.portfolio import allocate_capital, equal_weights, per_stock_summary, pooled_wealth


def test_equal_weights_split_evenly():
    weights = equal_weights(["A", "B", "C"])
    assert list(weights) == [1 / 3, 1 / 3, 1 / 3]
    assert equal_weights([]).empty
    assert allocate_capital(1000.0, ["A", "B"]) == 500.0
    assert allocate_capital(1000.0, ["A"]) == 1000.0


def test_pooled_path_is_sum_of_per_stock_paths():
    per_stock = pd.DataFrame({
        "ticker": ["A", "A", "A", "B", "B", "B"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"] * 2),
        "wealth": [100.0, 110.0, 121.0, 100.0, 105.0, 115.5],
    })
    pooled = pooled_wealth(per_stock, initial_capital=200.0)
    assert list(pooled["pooled_wealth"]) == pytest.approx([200.0, 215.0, 236.5])
    # Equal-weight pooled path equals the average of the normalized per-stock paths.
    assert list(pooled["pooled_wealth"] / 200.0) == pytest.approx([1.0, 1.075, 1.1825])
    assert list(pooled["pooled_return"]) == pytest.approx([0.0, 0.075, 0.1])


def test_pooled_path_is_not_interleaved_compounding():
    per_stock = pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"] * 2),
        "wealth": [100.0, 121.0, 100.0, 115.5],
    })
    pooled = pooled_wealth(per_stock, initial_capital=200.0)
    interleaved = (121.0 / 100.0) * (115.5 / 100.0) - 1.0
    assert pooled["pooled_wealth"].iloc[-1] / 200.0 - 1.0 == pytest.approx(0.1825)
    assert abs(interleaved - 0.1825) > 0.01


def test_pooled_wealth_with_differing_ticker_starts():
    per_stock = pd.DataFrame({
        "ticker": ["A", "A", "B"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03"]),
        "wealth": [100.0, 110.0, 105.0],
    })
    pooled = pooled_wealth(per_stock, initial_capital=200.0)
    assert list(pooled["pooled_wealth"]) == pytest.approx([100.0, 215.0])


def test_per_stock_summary_columns():
    per_stock = pd.DataFrame({
        "ticker": ["A", "A"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "wealth": [100.0, 110.0],
    })
    ledger = pd.DataFrame({
        "ticker": ["A", "A"], "side": ["buy", "sell"],
        "turnover": [100.0, 110.0],
    })
    positions = pd.Series(
        [0.0, 1.0], index=pd.MultiIndex.from_tuples([("A", pd.Timestamp("2024-01-02")), ("A", pd.Timestamp("2024-01-03"))]),
    )
    summary = per_stock_summary(per_stock, ledger, positions, capital=100.0)
    assert list(summary.columns) == ["ticker", "total_return", "maximum_drawdown", "trades", "exposure", "turnover"]
    assert summary.iloc[0]["total_return"] == pytest.approx(0.1)
    assert summary.iloc[0]["trades"] == 1
    assert summary.iloc[0]["turnover"] == pytest.approx(2.1)
