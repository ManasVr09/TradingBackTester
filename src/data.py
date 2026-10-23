"""Data layer: Yahoo Finance scraper, per-ticker CSV cache, incremental refresh.

Guarantees: scrape once, replay forever from CSV, never re-download data
already on disk. The scraper fails loudly on bad data and never overwrites a
good cache with partial or empty data.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .config import ROOT


def __getattr__(name: str):
    """Load the network client only for refreshes (and keep the public yf alias)."""
    if name == "yf":
        import yfinance
        globals()["yf"] = yfinance
        return yfinance
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

logger = logging.getLogger(__name__)

CACHE_COLUMNS = ["date", "open", "high", "low", "close", "volume", "retrieved_at_utc"]
PRICE_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
_PRICE_NAMES = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}


def _today() -> date:
    """Return the current UTC date (isolated for test control)."""
    return datetime.now(timezone.utc).date()


def _now_iso() -> str:
    """Return the current UTC timestamp as an ISO string (test-controlled)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_end(end_exclusive: str) -> str:
    """Resolve an exclusive end date; an empty value means tomorrow (today + 1)."""
    if end_exclusive:
        return end_exclusive
    return (_today() + timedelta(days=1)).isoformat()


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a frame to path via a temp file + atomic rename, dates as YYYY-MM-DD."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    frame.assign(date=frame["date"].dt.strftime("%Y-%m-%d")).to_csv(tmp, index=False)
    os.replace(tmp, path)


def _write_json_atomic(payload: dict, path: Path) -> None:
    """Write a JSON payload atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _normalize_yf(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert a yfinance download into a tidy long-format OHLCV frame.

    Handles both column layouts yfinance returns: a MultiIndex of
    ``(Price, Ticker)`` (group_by="column") or ``(Ticker, Price)``
    (group_by="ticker"), and the single-level layout for one ticker.
    ``adj_close`` is dropped because downloads run with ``auto_adjust=True``.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = set(frame.columns.get_level_values(0))
        price_first = bool(level0 & _PRICE_NAMES)
        frame = frame.xs(ticker, axis=1, level=1 if price_first else 0)
    tidy = frame.copy()
    tidy.columns = [str(column).lower().replace(" ", "_") for column in tidy.columns]
    tidy = tidy.drop(columns=["adj_close"], errors="ignore")
    missing = set(PRICE_COLUMNS[1:]) - set(tidy.columns)
    if missing:
        raise ValueError(f"{ticker}: missing columns after normalization: {sorted(missing)}")
    tidy = tidy.reset_index()
    tidy = tidy.rename(columns={tidy.columns[0]: "date"})
    tidy["date"] = pd.to_datetime(tidy["date"]).dt.tz_localize(None).dt.normalize()
    tidy = tidy.dropna(subset=["date"])
    tidy = tidy[PRICE_COLUMNS]
    return tidy.sort_values("date").reset_index(drop=True)


def validate_prices(frame: pd.DataFrame, ticker: str) -> None:
    """Raise ValueError on any structural or value corruption in a price frame."""
    missing = set(PRICE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{ticker}: missing required columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{ticker}: empty price frame")
    if frame["date"].isna().any():
        raise ValueError(f"{ticker}: NaT dates present")
    if frame.duplicated("date").any():
        raise ValueError(f"{ticker}: duplicate dates")
    if frame[PRICE_COLUMNS[1:]].isna().any().any():
        raise ValueError(f"{ticker}: NaN OHLCV values present")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{ticker}: non-positive OHLC price")
    if (frame["volume"] < 0).any():
        raise ValueError(f"{ticker}: negative volume")


def download_ticker(
    ticker: str,
    start: str,
    end: str,
    auto_adjust: bool = True,
    actions: bool = False,
    retry_attempts: int = 3,
    backoff_seconds: float = 2.0,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Download one ticker from Yahoo Finance with retry/backoff and validation.

    With ``allow_empty=False`` (the default for full downloads) an empty
    response fails loudly; incremental refreshes pass ``allow_empty=True``
    because a fully-caught-up cache legitimately yields nothing new.
    """
    last_error: Exception | None = None
    for attempt in range(int(retry_attempts)):
        try:
            raw = __getattr__("yf").download(
                ticker,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
                actions=actions,
                progress=False,
                group_by="column",
            )
            tidy = _normalize_yf(raw, ticker)
            if tidy.empty:
                if allow_empty:
                    return tidy
                raise RuntimeError("Yahoo Finance returned no data.")
            tidy["retrieved_at_utc"] = _now_iso()
            validate_prices(tidy, ticker)
            return tidy
        except Exception as exc:  # noqa: BLE001 - any failure triggers retry
            last_error = exc
            if attempt < int(retry_attempts) - 1:
                time.sleep(float(backoff_seconds) * (2 ** attempt))
    raise RuntimeError(f"Failed to download {ticker} after {retry_attempts} attempts: {last_error}")


def cache_path(ticker: str, config: dict) -> Path:
    """Return the CSV cache path for a ticker."""
    return ROOT / config["cache"]["dir"] / f"{ticker.upper()}.csv"


def metadata_path(config: dict) -> Path:
    """Return the cache metadata path (sibling of the cache directory)."""
    return Path(ROOT / config["cache"]["dir"]).parent / "metadata.json"


def load_cached(ticker: str, config: dict) -> pd.DataFrame:
    """Load a ticker's cached CSV, parse types, sort, and validate it."""
    path = cache_path(ticker, config)
    if not path.exists():
        raise FileNotFoundError(f"No cache for {ticker} at {path}")
    frame = pd.read_csv(path)
    missing = set(CACHE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{ticker}: cache missing columns {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame = frame.sort_values("date").reset_index(drop=True)
    validate_prices(frame, ticker)
    return frame


def last_cached_date(ticker: str, config: dict) -> pd.Timestamp | None:
    """Return the most recent cached date for a ticker, or None if not cached."""
    path = cache_path(ticker, config)
    if not path.exists():
        return None
    frame = load_cached(ticker, config)
    return frame["date"].max() if len(frame) else None


def refresh_ticker(ticker: str, start: str, end: str, config: dict, force: bool = False) -> pd.DataFrame:
    """Incrementally refresh one ticker and return the full cached frame.

    No CSV exists -> full download. Cache fresh -> zero network requests.
    Cache stale -> download only dates after the last cached date, drop
    duplicates, append with a fresh ``retrieved_at_utc``, write atomically.
    """
    cache_cfg = config["cache"]
    path = cache_path(ticker, config)
    cached = load_cached(ticker, config) if path.exists() else None

    if cached is not None and len(cached):
        last = cached["date"].max().date()
        fresh_cutoff = _today() - timedelta(days=int(cache_cfg["freshness_days"]))
        if not force and last >= fresh_cutoff:
            logger.info("%s: cache fresh through %s; skipping network", ticker, last)
            return cached
        download_start = (cached["date"].max() + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        download_start = start

    try:
        new_rows = download_ticker(
            ticker,
            download_start,
            end,
            retry_attempts=int(cache_cfg["retry_attempts"]),
            backoff_seconds=float(cache_cfg["backoff_seconds"]),
            allow_empty=True,
        )
    except Exception as exc:  # noqa: BLE001 - policy decides whether to swallow
        if cached is not None and cache_cfg["allow_stale_cache_on_error"]:
            logger.warning("%s: refresh failed (%s); serving stale cache", ticker, exc)
            return cached
        raise

    if len(new_rows) == 0:
        if cached is None:
            raise RuntimeError(f"No data returned for {ticker} in range {download_start}..{end}")
        logger.info("%s: nothing new to append; cache unchanged", ticker)
        return cached

    combined = pd.concat([cached, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    validate_prices(combined, ticker)
    _write_csv_atomic(combined, path)
    return combined


def ensure_data(config: dict, force: bool = False) -> dict[str, pd.DataFrame]:
    """Orchestrate downloads/refreshes for every configured ticker + benchmark.

    Returns a mapping of ticker -> full cached DataFrame. Never makes network
    requests for tickers whose cache is fresh unless ``force`` is set.
    """
    cache_cfg = config["cache"]
    tickers = list(dict.fromkeys(config["tickers"]["universe"] + [config["tickers"]["benchmark"]]))
    start = config["data"]["start"]
    end = _resolve_end(config["data"]["end_exclusive"])
    frames: dict[str, pd.DataFrame] = {}
    for index, ticker in enumerate(tickers):
        if index:
            time.sleep(float(cache_cfg["rate_limit_seconds"]))
        frames[ticker] = refresh_ticker(ticker, start, end, config, force=force)
    _write_metadata(config, frames, start, end)
    return frames


def _write_metadata(config: dict, frames: dict[str, pd.DataFrame], start: str, end: str) -> None:
    """Write retrieval metadata for the current cache state."""
    per_ticker = {
        ticker: {
            "last_date": str(frame["date"].max().date()) if len(frame) else None,
            "rows": len(frame),
        }
        for ticker, frame in frames.items()
    }
    payload = {
        "retrieved_at_utc": _now_iso(),
        "source": config["data"]["source"],
        "start": start,
        "end_exclusive": end,
        "tickers": list(frames),
        "per_ticker": per_ticker,
    }
    _write_json_atomic(payload, metadata_path(config))


def load_metadata(config: dict) -> dict:
    """Load the cache metadata file, or an empty dict if absent."""
    path = metadata_path(config)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cache_status(config: dict) -> pd.DataFrame:
    """Return a freshness report over the cache, one row per configured ticker."""
    tickers = list(dict.fromkeys(config["tickers"]["universe"] + [config["tickers"]["benchmark"]]))
    freshness_days = int(config["cache"]["freshness_days"])
    rows = []
    for ticker in tickers:
        path = cache_path(ticker, config)
        if not path.exists():
            rows.append({"ticker": ticker, "rows": 0, "last_date": None, "fresh": False, "file_mtime_utc": None})
            continue
        frame = load_cached(ticker, config)
        last = frame["date"].max().date()
        rows.append({
            "ticker": ticker,
            "rows": len(frame),
            "last_date": str(last),
            "fresh": last >= _today() - timedelta(days=freshness_days),
            "file_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        })
    return pd.DataFrame(rows)
