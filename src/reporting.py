"""Immutable report artifacts the dashboard consumes.

A ``BacktestReport`` is frozen at the end of a simulation; the dashboard never
re-runs the backtest behind the scenes. Reports can be serialized to a
directory and loaded back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class BacktestReport:
    """A complete, immutable result of one backtest run."""

    strategy_name: str
    equity_curve: pd.DataFrame
    trade_ledger: pd.DataFrame
    per_stock: pd.DataFrame
    metrics: dict[float, dict[str, float]]
    benchmark_metrics: dict[str, float]
    config_snapshot: dict
    data_metadata: dict


def report_to_dict(report: BacktestReport) -> dict:
    """Serialize everything except the DataFrames to JSON-safe types."""
    return {
        "strategy_name": report.strategy_name,
        "metrics": {str(bps): values for bps, values in sorted(report.metrics.items())},
        "benchmark_metrics": dict(report.benchmark_metrics),
        "config_snapshot": dict(report.config_snapshot),
        "data_metadata": dict(report.data_metadata),
    }


def write_report(report: BacktestReport, directory: Path | str) -> None:
    """Write a report's DataFrames and JSON payload to ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report.equity_curve.to_csv(directory / "equity_curve.csv", index=False)
    report.trade_ledger.to_csv(directory / "trade_ledger.csv", index=False)
    report.per_stock.to_csv(directory / "per_stock.csv", index=False)
    (directory / "report.json").write_text(
        json.dumps(report_to_dict(report), indent=2, sort_keys=True), encoding="utf-8",
    )


def load_report(directory: Path | str) -> BacktestReport:
    """Load a report previously written by ``write_report``."""
    directory = Path(directory)
    payload = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    equity_curve = pd.read_csv(directory / "equity_curve.csv", parse_dates=["date"])
    trade_ledger = pd.read_csv(directory / "trade_ledger.csv", parse_dates=["date", "signal_date", "entry_date"])
    per_stock = pd.read_csv(directory / "per_stock.csv")
    return BacktestReport(
        strategy_name=payload["strategy_name"],
        equity_curve=equity_curve,
        trade_ledger=trade_ledger,
        per_stock=per_stock,
        metrics={float(bps): values for bps, values in payload["metrics"].items()},
        benchmark_metrics=dict(payload["benchmark_metrics"]),
        config_snapshot=dict(payload["config_snapshot"]),
        data_metadata=dict(payload["data_metadata"]),
    )
