"""Strategy tests: determinism, registry, and no-look-ahead signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies import (
    STRATEGY_REGISTRY,
    AtrBreakout,
    BuyAndHold,
    EmaCrossover,
    MacdMomentum,
    RsiMeanReversion,
    SmaCrossover,
    Strategy,
    build_strategy,
    register,
)
from src.indicators import atr, ema, macd, rsi, sma


def make_prices(closes_per_ticker: dict[str, list[float]]) -> pd.DataFrame:
    rows = []
    for ticker, closes in closes_per_ticker.items():
        dates = pd.bdate_range("2024-01-01", periods=len(closes))
        rows.extend({
            "ticker": ticker, "date": day, "open": close, "high": close + 1.0,
            "low": close - 1.0, "close": close, "volume": 1e6,
        } for close, day in zip(closes, dates))
    return pd.DataFrame(rows)


def test_sma_crossover_deterministic_and_binary():
    prices = make_prices({"AAA": [float(i) for i in range(1, 31)]})
    first = SmaCrossover(fast=3, slow=10).positions(prices)
    second = SmaCrossover(fast=3, slow=10).positions(prices)
    pd.testing.assert_series_equal(first, second)
    assert set(first.unique()) <= {0.0, 1.0}
    assert first.name == "sma_crossover_3_10"
    assert first.index.nlevels == 2


def test_ema_crossover_signals():
    prices = make_prices({"AAA": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    positions = EmaCrossover(fast=2, slow=5).positions(prices)
    assert set(positions.unique()) <= {0.0, 1.0}
    assert positions.name == "ema_crossover_2_5"


def test_allow_short_produces_short_positions():
    prices = make_prices({"AAA": [float(i) for i in range(1, 31)]})
    long_only = SmaCrossover(fast=3, slow=10).positions(prices)
    shortable = SmaCrossover(fast=3, slow=10, allow_short=True).positions(prices)
    assert set(shortable.unique()) <= {-1.0, 1.0}
    assert set(long_only.unique()) <= {0.0, 1.0}
    assert shortable.name == "sma_crossover_3_10_short"


def test_buy_and_hold_is_always_long():
    prices = make_prices({"AAA": [1.0, 2.0, 3.0], "BBB": [5.0, 6.0, 7.0]})
    positions = BuyAndHold().positions(prices)
    assert (positions == 1.0).all()
    assert positions.name == "buy_and_hold"


def test_signal_at_t_uses_no_data_after_t():
    closes = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    prices = make_prices({"AAA": closes})
    original = SmaCrossover(fast=3, slow=5).positions(prices)
    perturbed_prices = prices.copy()
    perturbed_prices.loc[perturbed_prices["date"] > pd.Timestamp("2024-01-15"), "close"] = 999.0
    perturbed = SmaCrossover(fast=3, slow=5).positions(perturbed_prices)
    cutoff = pd.Timestamp("2024-01-15")
    original_at_t = original[original.index.get_level_values(1) <= cutoff]
    perturbed_at_t = perturbed[perturbed.index.get_level_values(1) <= cutoff]
    pd.testing.assert_series_equal(original_at_t, perturbed_at_t)


def test_registry_builds_strategies_from_specs():
    for spec in (
        {"type": "sma_crossover", "fast": 10, "slow": 30},
        {"type": "ema_crossover", "fast": 5, "slow": 20, "allow_short": True},
        {"type": "buy_and_hold"},
    ):
        strategy = build_strategy(spec)
        assert isinstance(strategy.name, str) and strategy.name
    with pytest.raises(ValueError, match="Unknown strategy type"):
        build_strategy({"type": "not_a_strategy"})
    with pytest.raises(ValueError, match="type"):
        build_strategy({"fast": 10})


def test_invalid_windows_rejected():
    with pytest.raises(ValueError):
        SmaCrossover(fast=50, slow=20)
    with pytest.raises(ValueError):
        SmaCrossover(fast=0, slow=20)
    with pytest.raises(ValueError):
        build_strategy({"type": "ema_crossover", "fast": 26, "slow": 12})


def test_builtins_registered_under_class_names():
    assert STRATEGY_REGISTRY["sma_crossover"] is SmaCrossover
    assert STRATEGY_REGISTRY["ema_crossover"] is EmaCrossover
    assert STRATEGY_REGISTRY["buy_and_hold"] is BuyAndHold
    assert STRATEGY_REGISTRY["rsi_mean_reversion"] is RsiMeanReversion
    assert STRATEGY_REGISTRY["macd_momentum"] is MacdMomentum
    assert STRATEGY_REGISTRY["atr_breakout"] is AtrBreakout


def rsi_fixture_closes() -> list[float]:
    """Flat, then a steep crash (RSI dives below 30), then a strong recovery."""
    flat = [100.0] * 20
    crash = [96.0, 90.0, 84.0, 78.0, 73.0, 69.0, 66.0, 64.0, 63.0, 62.0]
    recovery = [65.0, 69.0, 74.0, 80.0, 87.0, 93.0, 98.0, 102.0, 105.0, 108.0, 111.0, 113.0, 115.0, 117.0, 118.0]
    return flat + crash + recovery


def expected_rsi_state(values, oversold, overbought, exit_mid, allow_short) -> list[int]:
    """Independent reimplementation of the documented RSI state machine."""
    state = 0
    out: list[int] = []
    for value in values:
        if pd.isna(value):
            out.append(0)
            continue
        if state == 0:
            if value < oversold:
                state = 1
            elif allow_short and value > overbought:
                state = -1
        elif state == 1:
            if value >= exit_mid:
                state = 0
        elif state == -1:
            if value <= exit_mid:
                state = 0
        out.append(state)
    return out


def test_rsi_mean_reversion():
    prices = make_prices({"AAA": rsi_fixture_closes()})
    positions = RsiMeanReversion().positions(prices)
    r = rsi(prices["close"], 14)
    expected = expected_rsi_state(r, 30, 70, 50, False)
    assert list(positions) == expected
    assert positions.iloc[:14].eq(0).all()  # warmup rows flat
    assert set(positions.unique()) <= {0, 1}
    assert any(positions == 1)  # an entry happened
    assert positions.name == "position"


def test_rsi_mean_reversion_short_path():
    prices = make_prices({"AAA": rsi_fixture_closes()})
    positions = RsiMeanReversion(allow_short=True).positions(prices)
    r = rsi(prices["close"], 14)
    expected = expected_rsi_state(r, 30, 70, 50, True)
    assert list(positions) == expected
    assert -1 in set(positions.unique())  # the overbought recovery region goes short


def test_rsi_mean_reversion_no_look_ahead_and_tiny_frame():
    prices = make_prices({"AAA": rsi_fixture_closes()})
    base = RsiMeanReversion().positions(prices)
    perturbed = prices.copy()
    perturbed.loc[perturbed.index[-5:], "close"] = 999.0
    after = RsiMeanReversion().positions(perturbed)
    assert list(base.iloc[:-5]) == list(after.iloc[:-5])
    tiny = make_prices({"AAA": [100.0] * 5})
    assert list(RsiMeanReversion().positions(tiny)) == [0] * 5


def macd_fixture_closes() -> list[float]:
    """Flat, strong rally (MACD crosses above signal), then a decline."""
    flat = [100.0] * 30
    rally = [100 + 3.0 * i for i in range(1, 26)]
    decline = [174 - 2.5 * i for i in range(1, 21)]
    return flat + rally + decline


def expected_macd_state(prices: pd.DataFrame, allow_short: bool) -> list[int]:
    m = macd(prices["close"], 12, 26, 9)
    above = (m["macd"] > m["macd_signal"]).fillna(False)
    if allow_short:
        below = (m["macd"] < m["macd_signal"]).fillna(False)
        return np.where(above, 1, np.where(below, -1, 0)).astype(int).tolist()
    return above.astype(int).tolist()


def test_macd_momentum():
    prices = make_prices({"AAA": macd_fixture_closes()})
    positions = MacdMomentum().positions(prices)
    assert list(positions) == expected_macd_state(prices, False)
    m = macd(prices["close"], 12, 26, 9)
    entry = m.index[m["macd"] > m["macd_signal"]][0]
    assert positions.iloc[entry] == 1
    assert positions.iloc[:entry].eq(0).all()
    assert set(positions.unique()) <= {0, 1}


def test_macd_momentum_short_path():
    prices = make_prices({"AAA": macd_fixture_closes()})
    positions = MacdMomentum(allow_short=True).positions(prices)
    assert list(positions) == expected_macd_state(prices, True)
    assert -1 in set(positions.unique())


def test_macd_momentum_warmup_tiny_frame_and_no_look_ahead():
    prices = make_prices({"AAA": macd_fixture_closes()})
    m = macd(prices["close"], 12, 26, 9)
    warmup = m.index[m["macd"].notna()][0]
    positions = MacdMomentum().positions(prices)
    assert positions.iloc[:warmup].eq(0).all()
    base = list(positions)
    perturbed = prices.copy()
    perturbed.loc[perturbed.index[-5:], "close"] = 999.0
    assert base[:-5] == list(MacdMomentum().positions(perturbed))[:-5]
    tiny = make_prices({"AAA": [100.0] * 5})
    assert list(MacdMomentum().positions(tiny)) == [0] * 5


def test_macd_momentum_uses_indicator_output_faithfully():
    prices = make_prices({"AAA": macd_fixture_closes()})
    lines = macd(prices["close"], 12, 26, 9)
    reference = ema(prices["close"], 12) - ema(prices["close"], 26)
    pd.testing.assert_series_equal(lines["macd"], reference, check_names=False, check_exact=False, rtol=1e-9, atol=1e-6)


def atr_breakout_fixture_closes() -> list[float]:
    """Oscillating range (stable ATR), a breakout above the upper band, a decline below the midline and lower band."""
    oscillate = [100.0 + (1.0 if i % 2 else -1.0) for i in range(25)]
    breakout = [
        100.5, 102.0, 104.0, 106.5, 108.0, 107.5, 106.0, 104.5, 103.0,
        101.5, 100.0, 98.5, 97.0, 96.0, 95.5, 94.5, 93.0, 91.0, 89.5,
    ]
    return oscillate + breakout


def atr_band_series(prices: pd.DataFrame, sma_window=20, atr_window=14, k=2.0, include_current_bar=False):
    mid = sma(prices["close"], sma_window)
    spread = atr(prices[["high", "low", "close"]], atr_window)
    if not include_current_bar:
        mid = mid.shift(1)
        spread = spread.shift(1)
    return mid, mid + k * spread, mid - k * spread


def expected_atr_state(prices: pd.DataFrame, exit_rule, allow_short, k=2.0) -> list[int]:
    mid, up, down = atr_band_series(prices, k=k)
    state = 0
    out: list[int] = []
    closes = prices["close"].to_numpy()
    for t in range(len(prices)):
        if pd.isna(up.iloc[t]):
            out.append(0)
            continue
        if state == 0:
            if closes[t] > up.iloc[t]:
                state = 1
            elif allow_short and closes[t] < down.iloc[t]:
                state = -1
        elif state == 1:
            level = mid.iloc[t] if exit_rule == "midline" else down.iloc[t]
            if closes[t] <= level:
                state = 0
        elif state == -1:
            level = mid.iloc[t] if exit_rule == "midline" else up.iloc[t]
            if closes[t] >= level:
                state = 0
        out.append(state)
    return out


def test_atr_breakout_midline():
    prices = make_prices({"AAA": atr_breakout_fixture_closes()})
    positions = AtrBreakout().positions(prices)
    expected = expected_atr_state(prices, "midline", False)
    assert list(positions) == expected
    assert positions.iloc[:21].eq(0).all()  # warmup = max(20, 14) + 1
    entry = next(i for i, value in enumerate(expected) if value == 1)
    assert prices["close"].iloc[entry] > (sma(prices["close"], 20).shift(1) + 2.0 * atr(prices[["high", "low", "close"]], 14).shift(1)).iloc[entry]
    assert set(positions.unique()) <= {0, 1}


def test_atr_breakout_lower_band_holds_longer():
    prices = make_prices({"AAA": atr_breakout_fixture_closes()})
    midline = list(AtrBreakout(exit_rule="midline").positions(prices))
    lower = list(AtrBreakout(exit_rule="lower_band").positions(prices))
    expected_lower = expected_atr_state(prices, "lower_band", False)
    assert lower == expected_lower
    exit_midline = max(i for i, value in enumerate(midline) if value == 1)
    exit_lower = max(i for i, value in enumerate(lower) if value == 1)
    assert exit_lower > exit_midline


def test_atr_breakout_short_path():
    prices = make_prices({"AAA": atr_breakout_fixture_closes()})
    positions = AtrBreakout(allow_short=True).positions(prices)
    assert list(positions) == expected_atr_state(prices, "midline", True)
    assert -1 in set(positions.unique())


def test_atr_breakout_no_look_ahead_and_tiny_frame():
    prices = make_prices({"AAA": atr_breakout_fixture_closes()})
    base = list(AtrBreakout().positions(prices))
    perturbed = prices.copy()
    perturbed.loc[perturbed.index[-5:], "close"] = 999.0
    after = list(AtrBreakout().positions(perturbed))
    assert base[:-5] == after[:-5]  # prior bars used only prior closes
    tiny = make_prices({"AAA": [100.0] * 5})
    assert list(AtrBreakout(sma_window=20).positions(tiny)) == [0] * 5


def test_atr_breakout_k_sensitivity_monotonic():
    prices = make_prices({"AAA": atr_breakout_fixture_closes()})
    entries_wide = {i for i, value in enumerate(AtrBreakout(k=4.0).positions(prices)) if value == 1}
    entries_narrow = {i for i, value in enumerate(AtrBreakout(k=2.0).positions(prices)) if value == 1}
    assert entries_wide <= entries_narrow


def test_new_strategies_reject_invalid_params():
    with pytest.raises(ValueError):
        RsiMeanReversion(oversold=60, exit_mid=50)
    with pytest.raises(ValueError):
        MacdMomentum(fast=26, slow=12)
    with pytest.raises(ValueError):
        AtrBreakout(exit_rule="upper_band")
    with pytest.raises(ValueError):
        AtrBreakout(k=-1.0)


def test_register_decorator_registers_custom_class():
    @register
    class MomentumSqueeze(Strategy):
        name = "momentum_squeeze"
        params = {"window": 10}

        def positions(self, prices: pd.DataFrame) -> pd.Series:
            return pd.Series(1.0, index=prices.index)

    assert STRATEGY_REGISTRY["momentum_squeeze"] is MomentumSqueeze
    STRATEGY_REGISTRY.pop("momentum_squeeze")


def test_register_never_overwrites_existing_name():
    @register
    class DupBuiltin(Strategy):
        name = "sma_crossover"  # collision with the built-in

        def positions(self, prices: pd.DataFrame) -> pd.Series:
            return pd.Series(0.0, index=prices.index)

    assert STRATEGY_REGISTRY["sma_crossover"] is SmaCrossover
    assert not any(
        key == "sma_crossover" and value is not SmaCrossover for key, value in STRATEGY_REGISTRY.items()
    )


def test_custom_strategy_with_params_and_int_positions_runs_in_engine():
    from src.backtest import BacktestConfig, simulate

    @register
    class ThresholdStrategy(Strategy):
        name = "threshold_strategy"
        params = {"threshold": 105.0}

        def positions(self, prices: pd.DataFrame) -> pd.Series:
            return (prices["close"] > self.params["threshold"]).astype(int)

    prices = make_prices({"AAA": [float(i) for i in range(1, 31)]})
    cfg = BacktestConfig(
        entry_rule="next_open", exit_rule="same_day_close", initial_capital=1000.0,
        max_position=1.0, transaction_cost_bps=0.0, slippage_bps=0.0,
    )
    report = simulate(prices, ThresholdStrategy().positions(prices), cfg, strategy_name="threshold_strategy")
    assert report.strategy_name == "threshold_strategy"
    assert set(report.equity_curve["path"]) >= {"threshold_strategy", "buy_and_hold_benchmark"}
    STRATEGY_REGISTRY.pop("threshold_strategy")
