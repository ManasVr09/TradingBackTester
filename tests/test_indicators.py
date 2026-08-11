"""Indicator tests: hand-computed fixtures and no-look-ahead checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.indicators import atr, ema, macd, rsi, sma


def test_sma_matches_hand_computed_values():
    closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = sma(closes, 3)
    assert result.iloc[0] != result.iloc[0]  # NaN for the warmup window
    assert result.iloc[:2].isna().all()
    assert list(result.iloc[2:]) == pytest.approx([2.0, 3.0, 4.0])


def test_ema_matches_hand_computed_values():
    closes = pd.Series([1.0, 2.0, 3.0])
    result = ema(closes, 2)  # alpha = 2 / (2 + 1) = 2/3
    assert result.iloc[0] == pytest.approx(1.0)
    assert result.iloc[1] == pytest.approx(1.0 + (2.0 / 3.0) * 1.0)
    assert result.iloc[2] == pytest.approx(1.6666667 + (2.0 / 3.0) * (3.0 - 1.6666667))


def test_rsi_matches_hand_computed_wilder_values():
    closes = pd.Series([1.0, 2.0, 1.0, 2.0, 1.0])
    result = rsi(closes, 2)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(100.0)
    assert result.iloc[2] == pytest.approx(50.0)
    assert result.iloc[3] == pytest.approx(75.0)
    assert result.iloc[4] == pytest.approx(37.5)


def test_macd_is_ema_difference():
    closes = pd.Series(np.linspace(10.0, 20.0, 40))
    out = macd(closes, fast=12, slow=26, signal=9)
    assert list(out.columns) == ["macd", "macd_signal", "macd_hist"]
    expected_line = ema(closes, 12) - ema(closes, 26)
    pd.testing.assert_series_equal(out["macd"], expected_line, check_names=False)
    pd.testing.assert_series_equal(out["macd_signal"], expected_line.ewm(span=9, adjust=False).mean(), check_names=False)
    assert (out["macd_hist"] == out["macd"] - out["macd_signal"]).all()


def test_atr_matches_hand_computed_wilder_values():
    frame = pd.DataFrame({
        "high": [10.0, 11.0, 12.0, 12.5, 13.0],
        "low": [9.0, 9.5, 10.0, 10.5, 11.0],
        "close": [9.5, 10.5, 11.5, 12.0, 12.5],
    })
    true_ranges = [1.0, 1.5, 2.0, 2.0, 2.0]
    result = atr(frame, 2)
    # Wilder smoothing with alpha = 1/2, seeded at the first true range.
    expected = [1.0, 1.25, 1.625, 1.8125, 1.90625]
    assert list(result) == pytest.approx(expected)
    assert true_ranges == [1.0, 1.5, 2.0, 2.0, 2.0]


def test_indicators_have_no_look_ahead():
    closes = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    original = closes.copy()
    before = sma(closes, 3).copy()
    perturbed = closes.copy()
    perturbed.iloc[5] = 999.0
    after = sma(perturbed, 3)
    pd.testing.assert_series_equal(before.iloc[:4], after.iloc[:4])
    assert before.iloc[4] == after.iloc[4]

    rsi_before = rsi(original, 3).copy()
    rsi_after = rsi(perturbed, 3)
    pd.testing.assert_series_equal(rsi_before.iloc[:4], rsi_after.iloc[:4])

    macd_before = macd(original, 12, 26, 9)
    macd_after = macd(perturbed, 12, 26, 9)
    pd.testing.assert_series_equal(macd_before["macd"].iloc[:4], macd_after["macd"].iloc[:4])
