"""Dashboard tests via Streamlit AppTest: renders, no silent network, disclaimer."""

from __future__ import annotations

import hashlib
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
ROOT = APP.parents[1]
CACHE_DIR = ROOT / "data" / "cache"
METADATA_PATH = ROOT / "data" / "metadata.json"
CUSTOM_DIR = ROOT / "strategies" / "custom"
ALL_TICKERS = ["AAPL", "MSFT", "AMZN", "META", "NVDA", "SPY"]
FROZEN_TODAY = date(2024, 3, 1)

VALID_STRATEGY = b'''from __future__ import annotations
import pandas as pd
from src.strategies import Strategy, register
from src.indicators import sma

@register
class DashboardTestStrategy(Strategy):
    name = "dashboard_test_strategy"

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        fast = sma(prices["close"], 5)
        slow = sma(prices["close"], 10)
        return (fast > slow).astype(int).fillna(0).astype(int)
'''

INVALID_STRATEGY = b'''from __future__ import annotations
import pandas as pd
from src.strategies import Strategy, register

@register
class DashboardPeek(Strategy):
    name = "dashboard_peek"

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        pos = prices["close"].shift(-1).gt(prices["close"]).astype(int)
        return pos.fillna(0).astype(int)
'''


@pytest.fixture(autouse=True)
def clean_global_registry():
    """Restore the global STRATEGY_REGISTRY after each test."""
    from src.strategies import STRATEGY_REGISTRY

    before = set(STRATEGY_REGISTRY)
    yield
    for key in set(STRATEGY_REGISTRY) - before:
        STRATEGY_REGISTRY.pop(key)


@pytest.fixture(autouse=True)
def clean_custom_files():
    """Remove only the strategy files this test module creates."""
    yield
    for path in CUSTOM_DIR.glob("dashboard_*.py"):
        path.unlink(missing_ok=True)


@pytest.fixture
def cached_market(monkeypatch):
    """A synthetic, freshness-skipping cache in the real project data dir."""
    import src.data as data

    monkeypatch.setattr(data, "_today", lambda: FROZEN_TODAY)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(end=pd.Timestamp("2024-02-29"), periods=60)
    for i, ticker in enumerate(ALL_TICKERS):
        base = 100.0 + 10.0 * i
        rows = [{
            "date": day.strftime("%Y-%m-%d"),
            "open": base + j, "high": base + j + 1.0, "low": base + j - 1.0,
            "close": base + j + 0.5, "volume": 1_000_000,
            "retrieved_at_utc": "2024-02-01T00:00:00+00:00",
        } for j, day in enumerate(dates)]
        pd.DataFrame(rows).to_csv(CACHE_DIR / f"{ticker}.csv", index=False)
    yield
    for ticker in ALL_TICKERS:
        (CACHE_DIR / f"{ticker}.csv").unlink(missing_ok=True)
    METADATA_PATH.unlink(missing_ok=True)


def test_dashboard_renders_with_disclaimer(cached_market):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert any("Trading backtester" in title.value for title in app.title)
    assert any("point-in-time" in warning.value for warning in app.warning)
    assert any("not investment advice" in warning.value for warning in app.warning)
    assert any("Run backtest" == button.label for button in app.button)
    assert any("Refresh cache from Yahoo" == button.label for button in app.button)


def test_run_backtest_button_produces_metrics(cached_market):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    run_button = next(button for button in app.button if button.key == "run_button")
    run_button.click().run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Net return" in labels
    assert "Buy-and-hold return" in labels
    assert "Max drawdown" in labels


def test_refresh_button_does_not_contact_network_when_cache_fresh(cached_market, monkeypatch):
    import src.data as data

    def boom(*args, **kwargs):
        raise AssertionError("Refresh must not call yf.download on a fresh cache")

    monkeypatch.setattr(data.yf, "download", boom)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    refresh_button = next(button for button in app.button if button.key == "refresh_button")
    refresh_button.click().run()
    assert not app.exception
    assert any("Cache refreshed" in caption.value for caption in app.caption)


def test_dashboard_without_cache_shows_guidance():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert any("No cached data" in error.value for error in app.error)
    assert any("Run backtest" == button.label for button in app.button)


def test_dashboard_never_runs_backtest_on_load(cached_market):
    """A plain load must not create a report (immutable-report contract)."""
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert "No backtest run yet" in " ".join(info.value for info in app.info)


def strategy_selectbox(app):
    return next(selectbox for selectbox in app.selectbox if selectbox.key == "strategy_key")


def test_dashboard_renders_without_custom_dir(cached_market):
    existed = CUSTOM_DIR.exists()
    if existed:
        shutil.rmtree(CUSTOM_DIR)
    try:
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        assert not app.exception
        assert any("point-in-time" in warning.value for warning in app.warning)
        assert strategy_selectbox(app).value == "sma_crossover"
    finally:
        CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
        (CUSTOM_DIR / ".gitkeep").touch()


def test_all_builtin_strategies_in_dropdown(cached_market):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    options = list(strategy_selectbox(app).options)
    for label in ("SMA crossover", "EMA crossover", "RSI mean reversion", "MACD momentum", "ATR breakout"):
        assert label in options
    assert not any("buy_and_hold" in option for option in options)


def test_new_builtin_runs_from_dropdown(cached_market):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    strategy_selectbox(app).set_value("atr_breakout").run()
    run_button = next(button for button in app.button if button.key == "run_button")
    run_button.click().run()
    assert not app.exception
    assert not app.error
    report = app.session_state["report"]
    assert report.config_snapshot["strategy"] == "atr_breakout"
    assert report.config_snapshot["strategy_params"]["k"] == 2.0
    assert report.config_snapshot["strategy_source"]["builtin"] is True
    assert report.config_snapshot["strategy_source"]["file_name"] == "src/strategies.py"
    assert len(report.config_snapshot["strategy_source"]["sha256"]) == 64
    assert any(metric.label == "Net return" for metric in app.metric)


def test_valid_upload_appears_in_dropdown_and_runs(cached_market):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    before = list(strategy_selectbox(app).options)
    uploader = next(upload for upload in app.file_uploader if upload.key == "custom_upload")
    uploader.set_value(("dashboard_test_strategy.py", VALID_STRATEGY, "text/x-python")).run()
    assert not app.exception
    selectbox = strategy_selectbox(app)
    assert "custom: dashboard_test_strategy" in selectbox.options
    assert selectbox.value == "dashboard_test_strategy"  # auto-selected
    assert any("All gates passed" in success.value for success in app.success)

    run_button = next(button for button in app.button if button.key == "run_button")
    run_button.click().run()
    assert not app.exception
    labels = [metric.label for metric in app.metric]
    assert "Net return" in labels
    report = app.session_state["report"]
    assert report.config_snapshot["strategy"] == "dashboard_test_strategy"
    assert report.config_snapshot["strategy_source"]["sha256"] == hashlib.sha256(VALID_STRATEGY).hexdigest()
    assert report.config_snapshot["strategy_source"]["file_name"] == "dashboard_test_strategy.py"


def test_invalid_upload_shows_checklist_and_keeps_dropdown(cached_market):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    before = list(strategy_selectbox(app).options)
    uploader = next(upload for upload in app.file_uploader if upload.key == "custom_upload")
    uploader.set_value(("dashboard_peek.py", INVALID_STRATEGY, "text/x-python")).run()
    assert not app.exception
    assert any("Validation failed" in error.value for error in app.error)
    assert any("look-ahead" in markdown.value for markdown in app.markdown)
    assert "custom: dashboard_peek" not in strategy_selectbox(app).options
    assert list(strategy_selectbox(app).options) == before
    assert "session_registry" not in app.session_state or "dashboard_peek" not in app.session_state["session_registry"]


def test_saved_strategy_auto_registers_on_next_start(cached_market):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    uploader = next(upload for upload in app.file_uploader if upload.key == "custom_upload")
    uploader.set_value(("dashboard_saved_strategy.py", VALID_STRATEGY, "text/x-python")).run()
    save_button = next(button for button in app.button if button.key == "save_custom")
    save_button.click().run()
    assert not app.exception
    assert (CUSTOM_DIR / "dashboard_saved_strategy.py").exists()

    fresh = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not fresh.exception
    assert "custom: dashboard_test_strategy" in strategy_selectbox(fresh).options


def test_disclaimer_present_with_custom_view(cached_market):
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    uploader = next(upload for upload in app.file_uploader if upload.key == "custom_upload")
    uploader.set_value(("dashboard_test_strategy.py", VALID_STRATEGY, "text/x-python")).run()
    assert not app.exception
    assert any("point-in-time" in warning.value for warning in app.warning)
    assert any("not investment advice" in warning.value for warning in app.warning)


def test_inactive_tabs_do_not_read_cache_status(cached_market, monkeypatch):
    import src.data as data
    def unexpected(*args, **kwargs):
        raise AssertionError("The hidden Data Cache tab must not read every CSV")
    monkeypatch.setattr(data, "cache_status", unexpected)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    next(button for button in app.button if button.key == "run_button").click().run()
    assert not app.exception
    assert len([el for el in app if el.type in ("vega_lite_chart", "arrow_vega_lite_chart")]) == 1
    assert len(app.dataframe) == 1  # Cost table only, not hidden ticker/ledger tables.
    frozen = app.session_state["report"]
    app.session_state["report_view"] = "Trade Ledger"
    app.run()
    assert not app.exception
    assert not [el for el in app if el.type in ("vega_lite_chart", "arrow_vega_lite_chart")]
    assert app.dataframe[0].value.equals(frozen.trade_ledger)
    app.session_state["report_view"] = "Strategy config"
    app.run()
    assert not app.exception
    assert app.session_state["report"].metrics == frozen.metrics


def test_cache_file_change_invalidates_date_bounds(cached_market):
    import streamlit as st
    st.cache_data.clear()
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    path = CACHE_DIR / "AAPL.csv"
    frame = pd.read_csv(path)
    extra = frame.iloc[[-1]].copy()
    extra["date"] = "2024-03-05"
    pd.concat([frame, extra], ignore_index=True).to_csv(path, index=False)
    app.run()
    assert not app.exception
    assert next(widget for widget in app.date_input if widget.key == "end_date").proto.max == "2024-03-05"
