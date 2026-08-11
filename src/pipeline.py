"""Orchestration: config -> data -> positions -> backtest -> immutable reports.

The pipeline reuses the local CSV cache by default (zero Yahoo requests on a
warm cache); ``--refresh`` explicitly updates the cache first.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, simulate
from .config import ROOT, load_config, validate_config
from .data import ensure_data, load_metadata
from .reporting import BacktestReport, write_report
from .strategies import Strategy, build_strategy
from .strategy_loader import register_saved_strategies


def _backtest_config(config: dict, transaction_cost_bps: float | None = None) -> BacktestConfig:
    """Build an execution config from the project YAML (optional cost override)."""
    return BacktestConfig(
        entry_rule=config["execution"]["entry_rule"],
        exit_rule=config["execution"]["exit_rule"],
        initial_capital=float(config["portfolio"]["initial_capital"]),
        max_position=float(config["portfolio"]["max_position"]),
        transaction_cost_bps=float(transaction_cost_bps if transaction_cost_bps is not None
                                   else config["execution"]["transaction_cost_bps"]),
        slippage_bps=float(config["execution"]["slippage_bps"]),
        cost_scenarios_bps=tuple(float(value) for value in config["execution"]["cost_scenarios_bps"]),
    )


def build_report(
    config: dict,
    prices: dict[str, pd.DataFrame],
    strategy: Strategy,
    metadata: dict | None = None,
    transaction_cost_bps: float | None = None,
) -> BacktestReport:
    """Run one strategy over the cached prices and freeze a report.

    Provenance is recorded in the config snapshot so every run is
    reproducible: custom strategies carry their file name, sha256, upload
    timestamp, and validation gates; built-ins record the sha256 of
    ``src/strategies.py``. The ``params`` actually used (post-widget) are
    stored under ``strategy_params``.
    """
    frame = pd.concat(
        [ticker_frame.assign(ticker=ticker) for ticker, ticker_frame in prices.items()],
        ignore_index=True,
    )
    positions = strategy.positions(frame)
    key = getattr(strategy.__class__, "registry_key", None) or strategy.name
    positions.name = key
    source = getattr(strategy.__class__, "strategy_source", None)
    if source is None:
        strategies_path = Path(__file__).resolve().parent / "strategies.py"
        source = {
            "file_name": "src/strategies.py",
            "sha256": hashlib.sha256(strategies_path.read_bytes()).hexdigest(),
            "builtin": True,
        }
    params_used = dict(getattr(strategy, "params", {}) or {})
    return simulate(
        frame,
        positions,
        _backtest_config(config, transaction_cost_bps),
        data_metadata=metadata,
        strategy_name=key,
        strategy_source=source,
        strategy_params=params_used,
    )


def build_reports(config: dict, prices: dict[str, pd.DataFrame], metadata: dict | None = None) -> list[BacktestReport]:
    """Build one report per strategy declared in ``strategies.enabled``."""
    return [build_report(config, prices, build_strategy(spec), metadata) for spec in config["strategies"]["enabled"]]


def _print_summary(reports: list[BacktestReport]) -> None:
    print("\nStrategy                              Net return  Sharpe  Max DD   Trades  Benchmark")
    for report in reports:
        base = report.metrics[float(report.config_snapshot["backtest"]["transaction_cost_bps"])]
        benchmark = report.benchmark_metrics
        print(
            f"{report.strategy_name:<36} {base['total_return']:>9.1%} {base['sharpe']:>6.2f} "
            f"{base['maximum_drawdown']:>7.1%} {base['trade_count']:>6d}  {benchmark['total_return']:>9.1%}"
        )


def run(config_path: str | None = None, refresh: bool = False) -> list[BacktestReport]:
    """Load config, ensure data, run every enabled strategy, persist reports."""
    config = load_config(config_path)
    validate_config(config)
    if refresh:
        print("Refreshing cache from Yahoo Finance...")
    prices = ensure_data(config, force=refresh)
    metadata = load_metadata(config)
    print(f"Cache ready: {len(prices)} tickers, metadata present: {bool(metadata)}")
    saved = register_saved_strategies(ROOT / "strategies" / "custom", next(iter(prices.values())) if prices else None)
    if saved:
        print(f"Registered saved custom strategies: {sorted(saved)}")
    reports = build_reports(config, prices, metadata)
    for report in reports:
        write_report(report, ROOT / "data" / "results" / report.strategy_name)
    _print_summary(reports)
    print(f"\nReports written under {ROOT / 'data' / 'results'}")
    print("Run the dashboard: streamlit run dashboard/app.py")
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trading backtester pipeline.")
    parser.add_argument("--config", default=None, help="Path to a YAML project configuration.")
    parser.add_argument("--refresh", action="store_true", help="Refresh the Yahoo cache before backtesting.")
    args = parser.parse_args()
    run(args.config, refresh=args.refresh)
