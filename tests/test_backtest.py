"""Backtest engine tests: hand-computed fixtures, timing, costs, invariance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import BacktestConfig, simulate
from src.strategies import BuyAndHold, SmaCrossover


def make_prices(tickers: list[str], dates: list[str], opens: list[float], closes: list[float]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        for day, open_price, close_price in zip(dates, opens, closes):
            rows.append({
                "ticker": ticker, "date": pd.Timestamp(day), "open": open_price,
                "high": max(open_price, close_price) + 0.5, "low": min(open_price, close_price) - 0.5,
                "close": close_price, "volume": 1e6,
            })
    return pd.DataFrame(rows)


def buy_and_hold_positions(prices: pd.DataFrame) -> pd.Series:
    return BuyAndHold().positions(prices)


def config(**overrides) -> BacktestConfig:
    values = dict(
        entry_rule="next_open",
        exit_rule="same_day_close",
        initial_capital=1000.0,
        max_position=1.0,
        transaction_cost_bps=5.0,
        slippage_bps=2.0,
        cost_scenarios_bps=(0.0, 5.0, 10.0, 25.0),
    )
    values.update(overrides)
    return BacktestConfig(**values)


@pytest.fixture
def three_day_prices() -> pd.DataFrame:
    return make_prices(["AAA"], ["2024-01-02", "2024-01-03", "2024-01-04"], [100.0, 100.0, 110.0], [100.0, 110.0, 121.0])


def test_buy_and_hold_matches_hand_calc_gross(three_day_prices):
    report = simulate(three_day_prices, buy_and_hold_positions(three_day_prices), config(transaction_cost_bps=0.0, slippage_bps=0.0))
    # Signal at close(2024-01-02) -> buy at open(2024-01-03) = 100; hold to close(2024-01-04) = 121.
    assert report.metrics[0.0]["final_wealth"] == pytest.approx(1210.0)
    assert report.metrics[0.0]["total_return"] == pytest.approx(0.21)
    assert len(report.trade_ledger) == 2  # one entry + one terminal liquidation


def test_buy_and_hold_matches_hand_calc_with_terminal_cost(three_day_prices):
    report = simulate(three_day_prices, buy_and_hold_positions(three_day_prices), config(transaction_cost_bps=50.0, slippage_bps=0.0))
    ledger = report.trade_ledger
    assert len(ledger) == 2
    entry, exit_ = ledger.iloc[0], ledger.iloc[1]
    assert entry["side"] == "buy" and entry["entry_price"] == pytest.approx(100.0)
    assert entry["qty"] == pytest.approx(10.0)
    assert entry["turnover"] == pytest.approx(1000.0)
    assert entry["cost"] == pytest.approx(5.0)
    assert exit_["side"] == "sell" and exit_["exit_price"] == pytest.approx(121.0)
    assert exit_["cost"] == pytest.approx(6.05)
    assert report.metrics[50.0]["final_wealth"] == pytest.approx(1198.95)
    assert report.metrics[50.0]["total_return"] == pytest.approx(1198.95 / 1000.0 - 1.0)


def test_ledger_prices_reconcile_with_equity(three_day_prices):
    report = simulate(three_day_prices, buy_and_hold_positions(three_day_prices), config(transaction_cost_bps=10.0, slippage_bps=0.0))
    wealth = _replay_from_ledger(three_day_prices, report.trade_ledger)
    final_pooled = report.equity_curve.loc[report.equity_curve["path"] == "buy_and_hold", "wealth"].iloc[-1]
    assert wealth == pytest.approx(final_pooled)


def _replay_from_ledger(prices: pd.DataFrame, ledger: pd.DataFrame) -> float:
    """Recompute final cash+shares from raw ledger prices, costs, and qty."""
    cash = 1000.0
    shares = 0.0
    for _, event in ledger.sort_values("date").iterrows():
        if event["side"] == "buy":
            cash -= event["qty"] * event["entry_price"] + event["cost"]
            shares += event["qty"]
        else:
            cash += event["qty"] * event["exit_price"] - event["cost"]
            shares -= event["qty"]
    return cash + shares * float(prices["close"].iloc[-1])


def test_zero_signal_stays_in_cash_without_costs(three_day_prices):
    positions = pd.Series(0.0, index=pd.MultiIndex.from_frame(three_day_prices[["ticker", "date"]]), name="flat")
    report = simulate(three_day_prices, positions, config(transaction_cost_bps=25.0, slippage_bps=10.0))
    assert len(report.trade_ledger) == 0
    assert report.metrics[25.0]["final_wealth"] == pytest.approx(1000.0)
    assert report.metrics[25.0]["turnover"] == 0.0
    assert report.metrics[25.0]["trade_count"] == 0


def test_higher_cost_cannot_increase_wealth(three_day_prices):
    positions = buy_and_hold_positions(three_day_prices)
    final_wealths = []
    for bps in (0.0, 5.0, 10.0, 25.0):
        report = simulate(three_day_prices, positions, config(transaction_cost_bps=bps, slippage_bps=0.0))
        final_wealths.append(report.metrics[float(bps)]["final_wealth"])
    assert final_wealths == sorted(final_wealths, reverse=True)
    assert final_wealths[0] > final_wealths[-1]


def test_reordered_rows_give_identical_results(three_day_prices):
    positions = SmaCrossover(fast=2, slow=3).positions(three_day_prices)
    shuffled = three_day_prices.sample(frac=1.0, random_state=7).reset_index(drop=True)
    base = simulate(three_day_prices, positions, config(transaction_cost_bps=10.0))
    reordered = simulate(shuffled, positions, config(transaction_cost_bps=10.0))
    base_path = base.equity_curve.loc[base.equity_curve["path"] == "sma_crossover_2_3", "wealth"].to_numpy()
    reorder_path = reordered.equity_curve.loc[reordered.equity_curve["path"] == "sma_crossover_2_3", "wealth"].to_numpy()
    np.testing.assert_allclose(base_path, reorder_path)
    pd.testing.assert_frame_equal(base.trade_ledger, reordered.trade_ledger)


def test_duplicate_ticker_date_raises(three_day_prices):
    duplicated = pd.concat([three_day_prices, three_day_prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        simulate(duplicated, buy_and_hold_positions(three_day_prices), config())


def test_signal_enters_at_next_open_not_same_day(three_day_prices):
    report = simulate(three_day_prices, buy_and_hold_positions(three_day_prices), config())
    first_trade = report.trade_ledger.iloc[0]
    assert first_trade["date"] == pd.Timestamp("2024-01-03")
    assert first_trade["signal_date"] == pd.Timestamp("2024-01-02")
    assert first_trade["entry_price"] == pytest.approx(100.0)


def test_trade_price_uses_open_of_execution_day():
    prices = make_prices(["AAA"], ["2024-01-02", "2024-01-03"], [100.0, 100.0], [100.0, 110.0])
    positions = buy_and_hold_positions(prices)
    base = simulate(prices, positions, config())

    open_shifted = prices.copy()
    open_shifted.loc[open_shifted["date"] == "2024-01-02", "open"] = 999.0  # day 1 open: no trade happens there
    shifted = simulate(open_shifted, positions, config())

    open_executed = prices.copy()
    open_executed.loc[open_executed["date"] == "2024-01-03", "open"] = 999.0  # execution open
    executed = simulate(open_executed, positions, config())

    assert base.metrics[5.0]["final_wealth"] == shifted.metrics[5.0]["final_wealth"]
    assert executed.metrics[5.0]["final_wealth"] != base.metrics[5.0]["final_wealth"]


def test_multi_ticker_equal_split_pooled_path(three_day_prices):
    prices = make_prices(["AAA", "BBB"], ["2024-01-02", "2024-01-03", "2024-01-04"], [100.0, 100.0, 110.0], [100.0, 110.0, 121.0])
    positions = buy_and_hold_positions(prices)
    report = simulate(prices, positions, config(initial_capital=1000.0, transaction_cost_bps=0.0, slippage_bps=0.0))
    pooled = report.equity_curve.loc[report.equity_curve["path"] == "buy_and_hold"]
    # Two accounts of 500, each growing 100 -> 121: pooled final = 500*1.21 + 500*1.21 = 1210.
    assert pooled["wealth"].iloc[-1] == pytest.approx(1210.0)
    per_ticker_paths = report.equity_curve[report.equity_curve["path"].str.startswith("buy_and_hold::")]
    assert set(per_ticker_paths["path"]) == {"buy_and_hold::AAA", "buy_and_hold::BBB"}
    assert per_ticker_paths["wealth"].iloc[-1] == pytest.approx(605.0)


def test_benchmark_renders_for_every_strategy(three_day_prices):
    report = simulate(three_day_prices, SmaCrossover(fast=2, slow=3).positions(three_day_prices), config())
    paths = set(report.equity_curve["path"])
    assert "buy_and_hold_benchmark" in paths
    assert "sma_crossover_2_3" in paths
    assert report.benchmark_metrics["total_return"] == pytest.approx(0.21)


def test_cost_scenario_metrics_all_present(three_day_prices):
    report = simulate(three_day_prices, buy_and_hold_positions(three_day_prices), config())
    assert set(report.metrics) == {0.0, 5.0, 10.0, 25.0}
    assert report.metrics[0.0]["total_return"] >= report.metrics[25.0]["total_return"]


def test_missing_columns_and_empty_input_raise(three_day_prices):
    with pytest.raises(ValueError, match="missing columns"):
        simulate(three_day_prices.drop(columns=["open"]), buy_and_hold_positions(three_day_prices), config())
    with pytest.raises(ValueError, match="No price rows"):
        simulate(three_day_prices.iloc[0:0], buy_and_hold_positions(three_day_prices), config())


def test_positions_must_cover_all_rows(three_day_prices):
    positions = pd.Series(1.0, index=pd.MultiIndex.from_frame(three_day_prices[["ticker", "date"]].iloc[1:]), name="partial")
    with pytest.raises(ValueError, match="do not cover"):
        simulate(three_day_prices, positions, config())


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        config(entry_rule="market")
    with pytest.raises(ValueError):
        config(initial_capital=0.0)
    with pytest.raises(ValueError):
        config(max_position=1.5)
    with pytest.raises(ValueError):
        config(transaction_cost_bps=-1.0)
