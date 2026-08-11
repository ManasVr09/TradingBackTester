"""Metrics tests on known equity/return series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    annualized_volatility,
    avg_win_loss,
    cagr,
    compute_metrics,
    exposure,
    hit_rate,
    max_drawdown,
    sharpe,
    sortino,
    time_under_water,
    total_return,
    turnover_fraction,
)


def test_sharpe_and_volatility_on_known_returns():
    returns = np.array([0.01, -0.02, 0.03])
    mean = 0.02 / 3.0
    std = np.sqrt(((returns - mean) ** 2).mean())
    assert sharpe(returns) == pytest.approx(mean / std * np.sqrt(252), rel=1e-9)
    assert annualized_volatility(returns) == pytest.approx(std * np.sqrt(252), rel=1e-9)
    assert sharpe(np.zeros(5)) == 0.0
    assert sharpe(np.array([])) == 0.0


def test_sortino_uses_downside_only():
    returns = np.array([0.01, -0.02, 0.03, -0.01])
    downside = returns[returns < 0]
    assert sortino(returns) == pytest.approx(returns.mean() / downside.std(ddof=0) * np.sqrt(252), rel=1e-9)
    assert sortino(np.array([0.01, 0.02])) == 0.0


def test_max_drawdown_and_time_under_water():
    wealth = np.array([100.0, 110.0, 105.0, 103.0, 112.0, 109.0])
    assert max_drawdown(wealth) == pytest.approx(103.0 / 110.0 - 1.0, rel=1e-9)
    assert time_under_water(wealth) == 2  # 105, 103 under the 110 peak; then 112 recovers; 109 dips again
    assert max_drawdown(np.array([100.0])) == 0.0
    assert time_under_water(np.array([100.0])) == 0


def test_returns_and_cagr():
    assert total_return(np.array([1000.0, 1100.0]), 1000.0) == pytest.approx(0.1)
    assert cagr(np.array([1000.0, 1100.0]), 1000.0, periods_per_year=12) == pytest.approx(1.1**12 - 1.0)
    assert cagr(np.array([1000.0]), 1000.0) == -1.0


def test_turnover_hit_rate_and_avg_win_loss():
    ledger = pd.DataFrame({
        "side": ["buy", "sell", "sell"],
        "turnover": [1000.0, 500.0, 200.0],
        "gross_return": [np.nan, 0.05, -0.02],
    })
    assert turnover_fraction(ledger, 1000.0) == pytest.approx(1.7)
    assert hit_rate(ledger) == pytest.approx(0.5)
    assert avg_win_loss(ledger) == (0.05, -0.02)
    assert turnover_fraction(pd.DataFrame(), 1000.0) == 0.0
    assert hit_rate(pd.DataFrame()) == 0.0


def test_exposure_and_compute_metrics_smoke():
    positions = pd.Series([0.0, 1.0, 1.0, 0.0])
    assert exposure(positions) == pytest.approx(0.5)
    ledger = pd.DataFrame({
        "side": ["buy", "sell"],
        "turnover": [1000.0, 1100.0],
        "gross_return": [np.nan, 0.1],
    })
    wealth = np.array([1000.0, 1050.0, 1020.0])
    returns = np.array([0.0, 0.05, -0.0285714])
    metrics = compute_metrics(wealth, returns, ledger, positions, 1000.0)
    for key in ("total_return", "cagr", "annualized_volatility", "sharpe", "sortino",
                "maximum_drawdown", "time_under_water_days", "turnover", "trade_count",
                "hit_rate", "avg_win", "avg_loss", "exposure", "final_wealth", "initial_capital"):
        assert key in metrics
    assert metrics["total_return"] == pytest.approx(0.02)
    assert metrics["trade_count"] == 2
