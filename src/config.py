"""Shared configuration handling for the trading backtester."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
_REQUIRED_SECTIONS = ("data", "tickers", "cache", "portfolio", "execution", "strategies")


def load_config(path: str | Path | None = None) -> dict:
    """Load and return the project configuration as a plain dict."""
    config_path = Path(path) if path else ROOT / "configs" / "project.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path.resolve())
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Validate the structure and value ranges of a project config, raising on violation."""
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping.")
    for section in _REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(f"Missing config section: {section}")

    data = config["data"]
    for key in ("source", "granularity", "start"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"data.{key} must be a non-empty string")
    if data["granularity"] != "1d":
        raise ValueError("Only 1d granularity is supported.")
    start = datetime.strptime(data["start"], "%Y-%m-%d").date()
    end_exclusive = data.get("end_exclusive") or None
    if end_exclusive:
        end = datetime.strptime(end_exclusive, "%Y-%m-%d").date()
        if end <= start:
            raise ValueError("data.end_exclusive must be after data.start")
    if data.get("auto_adjust") is not True:
        raise ValueError("data.auto_adjust must be true.")

    tickers = config["tickers"]
    if not isinstance(tickers.get("universe"), list) or not tickers["universe"]:
        raise ValueError("tickers.universe must be a non-empty list")
    if not all(isinstance(t, str) and t for t in tickers["universe"]):
        raise ValueError("tickers.universe must contain only non-empty strings")
    if not isinstance(tickers.get("benchmark"), str) or not tickers["benchmark"]:
        raise ValueError("tickers.benchmark must be a non-empty string")

    cache = config["cache"]
    for key in ("dir", "format", "mode"):
        if not isinstance(cache.get(key), str) or not cache[key]:
            raise ValueError(f"cache.{key} must be a non-empty string")
    if cache["format"] != "csv" or cache["mode"] != "incremental_append":
        raise ValueError("Only csv incremental_append caching is supported.")
    if int(cache["retry_attempts"]) < 1:
        raise ValueError("cache.retry_attempts must be >= 1")
    for key in ("backoff_seconds", "rate_limit_seconds", "freshness_days"):
        if float(cache[key]) < 0:
            raise ValueError(f"cache.{key} must be >= 0")
    if not isinstance(cache["allow_stale_cache_on_error"], bool):
        raise ValueError("cache.allow_stale_cache_on_error must be a boolean")

    portfolio = config["portfolio"]
    if float(portfolio["initial_capital"]) <= 0:
        raise ValueError("portfolio.initial_capital must be positive")
    if portfolio["allocation"] != "equal_weight":
        raise ValueError("Only equal_weight allocation is supported.")
    if not 0 < float(portfolio["max_position"]) <= 1:
        raise ValueError("portfolio.max_position must be in (0, 1]")

    execution = config["execution"]
    if execution["entry_rule"] != "next_open" or execution["exit_rule"] != "same_day_close":
        raise ValueError("Only next_open/same_day_close execution is supported.")
    for key in ("transaction_cost_bps", "slippage_bps"):
        if float(execution[key]) < 0:
            raise ValueError(f"execution.{key} must be >= 0")
    scenarios = execution.get("cost_scenarios_bps")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("execution.cost_scenarios_bps must be a non-empty list")
    if any(float(value) < 0 for value in scenarios):
        raise ValueError("execution.cost_scenarios_bps values must be >= 0")

    strategies = config["strategies"]
    enabled = strategies.get("enabled")
    if not isinstance(enabled, list) or not enabled:
        raise ValueError("strategies.enabled must be a non-empty list")
    for spec in enabled:
        if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
            raise ValueError("Each enabled strategy must have a type")

    if strategies.get("benchmark") != "buy_and_hold":
        raise ValueError("strategies.benchmark must be buy_and_hold")
