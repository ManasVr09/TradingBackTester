"""Heavy coverage of the data layer: scraper, cache, incremental refresh."""

from __future__ import annotations

import copy
import json
from datetime import date

import pandas as pd
import pytest

import src.data as data
from src.config import load_config

UNIVERSE = ["AAPL", "MSFT", "AMZN", "META", "NVDA"]
ALL_TICKERS = UNIVERSE + ["SPY"]


class Clock:
    """Controllable now()/today() so freshness and timestamps are deterministic."""

    def __init__(self, today: str) -> None:
        self._today = date.fromisoformat(today)
        self._ticks = 0

    def today(self) -> date:
        return self._today

    def now_iso(self) -> str:
        self._ticks += 1
        return f"{self._today}T00:00:{self._ticks:02d}+00:00"


def make_prices(tickers: list[str], start: str = "2024-01-01", end: str = "2024-01-31") -> pd.DataFrame:
    """Deterministic synthetic OHLCV for a universe of tickers."""
    dates = pd.bdate_range(start, end)
    rows = []
    for i, ticker in enumerate(tickers):
        base = 100.0 + 10.0 * i
        for j, day in enumerate(dates):
            open_price = base + j
            rows.append({
                "ticker": ticker, "date": day, "open": open_price,
                "high": open_price + 1.0, "low": open_price - 1.0,
                "close": open_price + 0.5, "volume": 1_000_000,
            })
    return pd.DataFrame(rows)


def install_stub(monkeypatch, prices: pd.DataFrame, calls: list[str] | None = None):
    """Replace yf.download with a stub serving ``prices`` filtered by range."""
    def stub_download(ticker, start=None, end=None, auto_adjust=True, actions=False, progress=False, group_by="column", **kwargs):
        if calls is not None:
            calls.append(str(ticker))
        subset = prices.loc[
            (prices["ticker"] == str(ticker))
            & (prices["date"] >= pd.Timestamp(start))
            & (prices["date"] < pd.Timestamp(end))
        ]
        frame = subset.set_index("date")[["open", "high", "low", "close", "volume"]].copy()
        return frame

    monkeypatch.setattr(data.yf, "download", stub_download)


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Project config with an isolated cache root and no rate-limit sleeps."""
    monkeypatch.setattr(data, "ROOT", tmp_path)
    cfg = copy.deepcopy(load_config())
    cfg["cache"]["rate_limit_seconds"] = 0.0
    cfg["cache"]["backoff_seconds"] = 0.0
    cfg["cache"]["retry_attempts"] = 1
    return cfg


@pytest.fixture
def clock(monkeypatch):
    clock = Clock("2024-02-01")
    monkeypatch.setattr(data, "_today", clock.today)
    monkeypatch.setattr(data, "_now_iso", clock.now_iso)
    return clock


def test_full_download_populates_cache_and_metadata(config, monkeypatch, clock):
    prices = make_prices(ALL_TICKERS)
    install_stub(monkeypatch, prices)

    frames = data.ensure_data(config)

    assert set(frames) == set(ALL_TICKERS)
    for ticker in ALL_TICKERS:
        path = data.cache_path(ticker, config)
        assert path.exists()
        cached = data.load_cached(ticker, config)
        assert list(cached.columns) == data.CACHE_COLUMNS
        assert cached["date"].is_monotonic_increasing
        expected = prices.loc[prices["ticker"] == ticker].set_index("date").sort_index()
        assert list(cached["close"]) == list(expected["close"])

    meta = json.loads(data.metadata_path(config).read_text())
    assert meta["source"] == "yahoo_finance"
    assert set(meta["tickers"]) == set(ALL_TICKERS)
    for ticker in ALL_TICKERS:
        assert meta["per_ticker"][ticker]["rows"] == 23
        assert meta["per_ticker"][ticker]["last_date"] == "2024-01-31"


def test_warm_cache_triggers_zero_network_requests(config, monkeypatch, clock):
    prices = make_prices(ALL_TICKERS)
    install_stub(monkeypatch, prices)
    first = data.ensure_data(config)

    def boom(*args, **kwargs):
        raise AssertionError("yf.download must not be called on a warm cache")

    monkeypatch.setattr(data.yf, "download", boom)
    second = data.ensure_data(config)

    assert set(first) == set(second)
    for ticker in ALL_TICKERS:
        pd.testing.assert_frame_equal(first[ticker], second[ticker])


def test_incremental_append_adds_only_new_rows(config, monkeypatch, clock):
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-01-31"))
    first = data.ensure_data(config)
    january_rows = {ticker: len(frame) for ticker, frame in first.items()}

    clock._today = date.fromisoformat("2024-03-01")
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-02-29"))
    second = data.ensure_data(config)

    for ticker in ALL_TICKERS:
        frame = second[ticker]
        assert len(frame) == january_rows[ticker] + 21
        assert not frame["date"].duplicated().any()
        assert frame["date"].is_monotonic_increasing
        old = first[ticker]["retrieved_at_utc"].iloc[0]
        new = second[ticker]["retrieved_at_utc"].iloc[-1]
        assert old.startswith("2024-02-01")
        assert new.startswith("2024-03-01")
        assert old != new
        assert len(second[ticker]["retrieved_at_utc"].unique()) == 2


def test_stale_cache_refresh_never_duplicates(config, monkeypatch, clock):
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-01-31"))
    first = data.ensure_data(config)
    clock._today = date.fromisoformat("2024-03-01")
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-03-02"))
    second = data.ensure_data(config)
    combined = make_prices(ALL_TICKERS, end="2024-03-02")
    for ticker in ALL_TICKERS:
        expected = len(combined.loc[combined["ticker"] == ticker])
        assert len(second[ticker]) == expected
        assert len(second[ticker]) == len(second[ticker]["date"].unique())


def test_freshness_skips_network_when_last_date_recent(config, monkeypatch, clock):
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-01-31"))
    data.ensure_data(config)
    clock._today = date.fromisoformat("2024-02-01")
    calls: list[str] = []
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-01-31"), calls)
    data.ensure_data(config)
    assert calls == []


def test_duplicate_date_in_cache_raises(config, monkeypatch, clock):
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-01-10"))
    data.ensure_data(config)
    path = data.cache_path("AAPL", config)
    duplicated = pd.concat([
        pd.read_csv(path),
        pd.read_csv(path).tail(1),
    ], ignore_index=True)
    duplicated.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate dates"):
        data.load_cached("AAPL", config)


@pytest.mark.parametrize("mutation", [
    ("close", 0, float("nan")),
    ("open", 1, -5.0),
    ("low", 0, 0.0),
    ("volume", 1, -1),
    ("high", 2, float("nan")),
])
def test_validate_prices_rejects_corruption(config, mutation):
    column, row, value = mutation
    frame = make_prices(["AAPL"], end="2024-01-05").drop(columns=["ticker"])
    frame.iloc[row, frame.columns.get_loc(column)] = value
    with pytest.raises(ValueError):
        data.validate_prices(frame, "AAPL")


def test_validate_prices_rejects_missing_column_and_empty():
    frame = make_prices(["AAPL"], end="2024-01-05").drop(columns=["ticker"])
    with pytest.raises(ValueError, match="missing required columns"):
        data.validate_prices(frame.drop(columns=["volume"]), "AAPL")
    with pytest.raises(ValueError, match="empty price frame"):
        data.validate_prices(frame.iloc[0:0], "AAPL")


def test_corrupted_cache_csv_fails_loudly_not_silently(config, monkeypatch, clock):
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-01-10"))
    data.ensure_data(config)
    path = data.cache_path("AAPL", config)
    broken = pd.read_csv(path)
    broken.loc[0, "close"] = float("nan")
    broken.to_csv(path, index=False)
    with pytest.raises(ValueError, match="NaN"):
        data.load_cached("AAPL", config)


def test_stale_cache_on_error_policy(config, monkeypatch, clock):
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-01-31"))
    first = data.ensure_data(config)
    clock._today = date.fromisoformat("2024-03-01")

    def boom(*args, **kwargs):
        raise RuntimeError("Yahoo is down")

    monkeypatch.setattr(data.yf, "download", boom)
    with pytest.raises(RuntimeError, match="Yahoo is down"):
        data.ensure_data(config)

    for ticker in ALL_TICKERS:
        before = data.cache_path(ticker, config).read_bytes()
        assert data.cache_path(ticker, config).read_bytes() == before

    config["cache"]["allow_stale_cache_on_error"] = True
    stale = data.ensure_data(config)
    for ticker in ALL_TICKERS:
        pd.testing.assert_frame_equal(stale[ticker], first[ticker])


def test_retries_then_succeeds_and_then_raises(config, monkeypatch, clock):
    prices = make_prices(["AAPL"], end="2024-01-10")
    failures = {"n": 0}

    def flaky(ticker, start=None, end=None, auto_adjust=True, actions=False, progress=False, group_by="column", **kwargs):
        failures["n"] += 1
        if failures["n"] < 3:
            raise RuntimeError("transient")
        return prices.set_index("date")[["open", "high", "low", "close", "volume"]]

    monkeypatch.setattr(data.yf, "download", flaky)
    frame = data.download_ticker("AAPL", "2024-01-01", "2024-01-10", retry_attempts=3, backoff_seconds=0.0)
    assert len(frame) == 8

    def always_down(*args, **kwargs):
        raise RuntimeError("hard failure")

    monkeypatch.setattr(data.yf, "download", always_down)
    with pytest.raises(RuntimeError, match="hard failure"):
        data.download_ticker("AAPL", "2024-01-01", "2024-01-10", retry_attempts=2, backoff_seconds=0.0)


def test_empty_download_fails_loudly(config, monkeypatch, clock):
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def stub_empty(ticker, start=None, end=None, **kwargs):
        return empty

    monkeypatch.setattr(data.yf, "download", stub_empty)
    with pytest.raises(RuntimeError, match="no data"):
        data.download_ticker("AAPL", "2024-01-01", "2024-01-10", retry_attempts=1)


def _layout_a() -> pd.DataFrame:
    """yfinance group_by='column' layout: (Price, Ticker) columns."""
    dates = pd.bdate_range("2024-01-02", "2024-01-04")
    columns = pd.MultiIndex.from_tuples([
        ("Open", "AAPL"), ("High", "AAPL"), ("Low", "AAPL"), ("Close", "AAPL"), ("Volume", "AAPL"),
        ("Open", "MSFT"), ("High", "MSFT"), ("Low", "MSFT"), ("Close", "MSFT"), ("Volume", "MSFT"),
    ])
    values = [[101 + j, 102 + j, 99 + j, 100.5 + j, 1e6, 201 + j, 202 + j, 199 + j, 200.5 + j, 2e6] for j in range(3)]
    return pd.DataFrame(values, index=dates, columns=columns)


def _layout_b() -> pd.DataFrame:
    """yfinance group_by='ticker' layout: (Ticker, Price) columns."""
    dates = pd.bdate_range("2024-01-02", "2024-01-04")
    columns = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Open", "High", "Low", "Close", "Volume"]])
    values = [[101 + j, 102 + j, 99 + j, 100.5 + j, 1e6, 201 + j, 202 + j, 199 + j, 200.5 + j, 2e6] for j in range(3)]
    return pd.DataFrame(values, index=dates, columns=columns)


def test_normalize_handles_both_yf_multi_ticker_layouts():
    for raw in (_layout_a(), _layout_b()):
        tidy = data._normalize_yf(raw, "AAPL")
        assert list(tidy.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert len(tidy) == 3
        assert list(tidy["close"]) == [100.5, 101.5, 102.5]
        assert tidy["date"].is_monotonic_increasing


def test_normalize_drops_adj_close_and_handles_single_layout():
    dates = pd.bdate_range("2024-01-02", "2024-01-04")
    frame = pd.DataFrame({
        "Open": [101.0, 102.0, 103.0], "High": [102.0, 103.0, 104.0],
        "Low": [99.0, 100.0, 101.0], "Close": [100.5, 101.5, 102.5],
        "Adj Close": [99.5, 100.5, 101.5], "Volume": [1e6, 1e6, 1e6],
    }, index=dates)
    tidy = data._normalize_yf(frame, "AAPL")
    assert "adj_close" not in tidy.columns
    assert "close" in tidy.columns
    assert list(tidy["close"]) == [100.5, 101.5, 102.5]


def test_incremental_with_nothing_new_leaves_cache_untouched(config, monkeypatch, clock):
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-01-31"))
    data.ensure_data(config)
    path = data.cache_path("AAPL", config)
    before = path.read_bytes()
    clock._today = date.fromisoformat("2024-03-01")
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-01-31"))
    data.ensure_data(config)
    assert path.read_bytes() == before


def test_last_cached_date_and_cache_status(config, monkeypatch, clock):
    install_stub(monkeypatch, make_prices(ALL_TICKERS, end="2024-01-31"))
    data.ensure_data(config)
    assert data.last_cached_date("AAPL", config) == pd.Timestamp("2024-01-31")
    assert data.last_cached_date("UNKNOWN", config) is None
    status = data.cache_status(config)
    assert list(status["ticker"]) == ALL_TICKERS
    assert (status["rows"] == 23).all()
