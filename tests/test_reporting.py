"""Reporting tests: BacktestReport round-trips and config snapshot preservation."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.backtest import BacktestConfig, simulate
from src.reporting import BacktestReport, load_report, report_to_dict, write_report
from src.strategies import SmaCrossover


def make_prices():
    rows = []
    for ticker, base in (("AAA", 100.0), ("BBB", 50.0)):
        for j, day in enumerate(pd.bdate_range("2024-01-02", periods=30)):
            close = base + j * 0.5
            rows.append({
                "ticker": ticker, "date": day, "open": close - 0.2, "high": close + 0.5,
                "low": close - 0.8, "close": close, "volume": 1e6,
            })
    return pd.DataFrame(rows)


def make_config():
    return BacktestConfig(
        entry_rule="next_open",
        exit_rule="same_day_close",
        initial_capital=10000.0,
        max_position=1.0,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
        cost_scenarios_bps=(0.0, 5.0, 10.0, 25.0),
    )


def test_report_round_trips_through_disk(tmp_path):
    prices = make_prices()
    positions = SmaCrossover(fast=3, slow=10).positions(prices)
    metadata = {"retrieved_at_utc": "2024-02-01T00:00:00+00:00", "source": "yahoo_finance"}
    report = simulate(prices, positions, make_config(), data_metadata=metadata)

    directory = tmp_path / "report"
    write_report(report, directory)
    loaded = load_report(directory)

    assert loaded.strategy_name == report.strategy_name
    pd.testing.assert_frame_equal(loaded.equity_curve, report.equity_curve)
    pd.testing.assert_frame_equal(loaded.trade_ledger, report.trade_ledger)
    pd.testing.assert_frame_equal(loaded.per_stock, report.per_stock)
    assert loaded.metrics == report.metrics
    assert loaded.benchmark_metrics == report.benchmark_metrics
    assert loaded.config_snapshot == report.config_snapshot
    assert loaded.data_metadata == report.data_metadata


def test_config_snapshot_is_json_serializable_and_preserved():
    prices = make_prices()
    report = simulate(prices, SmaCrossover(fast=3, slow=10).positions(prices), make_config())
    payload = report_to_dict(report)
    dumped = json.dumps(payload, sort_keys=True)
    restored = json.loads(dumped)
    assert restored["config_snapshot"]["backtest"]["entry_rule"] == "next_open"
    assert restored["config_snapshot"]["backtest"]["cost_scenarios_bps"] == [0.0, 5.0, 10.0, 25.0]
    assert restored["config_snapshot"]["strategy"] == "sma_crossover_3_10"
    assert set(restored["config_snapshot"]["tickers"]) == {"AAA", "BBB"}
    assert report.config_snapshot["backtest"]["initial_capital"] == 10000.0


def test_report_is_immutable():
    prices = make_prices()
    report = simulate(prices, SmaCrossover(fast=3, slow=10).positions(prices), make_config())
    with pytest.raises(Exception):
        report.config_snapshot = {}  # type: ignore[misc]
    with pytest.raises(Exception):
        report.equity_curve = pd.DataFrame()  # type: ignore[misc]
