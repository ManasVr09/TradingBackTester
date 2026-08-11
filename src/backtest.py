"""Cost-aware multi-asset backtest engine.

Pure functions, no Streamlit dependency. Signals formed from ``close(t)``
execute at ``open(t+1)``; positions exit at the same-day close. Each ticker
runs as an independent cash account seeded with an equal share of
``initial_capital``; the pooled path is the exact sum of the per-stock
accounts (never interleaved compounding). Terminal liquidation is an
explicit closing trade on the last date.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .metrics import compute_metrics, max_drawdown
from .portfolio import allocate_capital, per_stock_summary, pooled_wealth
from .reporting import BacktestReport

_REQUIRED_COLUMNS = {"ticker", "date", "open", "close"}
_EMPTY_THRESHOLD = 1e-12
_ORDER = ["ticker", "date", "signal_date", "side", "entry_date", "entry_price", "exit_price",
          "qty", "turnover", "cost", "gross_return", "position_after"]


@dataclass(frozen=True)
class BacktestConfig:
    """Execution configuration for one backtest run."""

    entry_rule: str
    exit_rule: str
    initial_capital: float
    max_position: float
    transaction_cost_bps: float
    slippage_bps: float
    cost_scenarios_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0)

    def __post_init__(self) -> None:
        if self.entry_rule != "next_open" or self.exit_rule != "same_day_close":
            raise ValueError("Only next_open entry / same_day_close exit is supported.")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive.")
        if not 0 < self.max_position <= 1:
            raise ValueError("max_position must be in (0, 1].")
        if self.transaction_cost_bps < 0 or self.slippage_bps < 0:
            raise ValueError("Costs cannot be negative.")
        if not self.cost_scenarios_bps:
            raise ValueError("cost_scenarios_bps cannot be empty.")
        if any(value < 0 for value in self.cost_scenarios_bps):
            raise ValueError("cost_scenarios_bps values cannot be negative.")


@dataclass
class _TickerResult:
    ticker: str
    dates: pd.Series
    wealth: np.ndarray
    events: list[dict]


def _align_positions(prices: pd.DataFrame, positions: pd.Series | pd.DataFrame) -> pd.Series:
    """Normalize positions into a Series indexed by (ticker, date) covering all rows.

    Two accepted layouts: a Series indexed by a ``(ticker, date)`` MultiIndex
    (built-in strategies) or a Series whose index equals ``prices.index``
    (custom strategies, aligned positionally). Discrete positions are cast
    to ``int`` by the engine (NaN is treated as flat).
    """
    if isinstance(positions, pd.DataFrame):
        missing = {"ticker", "date", "position"} - set(positions.columns)
        if missing:
            raise ValueError(f"positions missing columns: {sorted(missing)}")
        pos = positions.set_index(["ticker", "date"])["position"]
    else:
        pos = positions
    idx = pd.MultiIndex.from_frame(prices[["ticker", "date"]])
    if isinstance(pos.index, pd.MultiIndex):
        aligned = pos.reindex(idx)
        if aligned.isna().any():
            raise ValueError("positions do not cover every (ticker, date) row")
    else:
        if len(pos) != len(prices):
            raise ValueError(f"positions length {len(pos)} does not match prices rows {len(prices)}")
        aligned = pd.Series(pos.to_numpy(dtype=float), index=idx)
    return aligned.fillna(0.0).astype(int).astype(float).rename("position")


def _simulate_ticker(
    ticker: str,
    dates: np.ndarray,
    opens: np.ndarray,
    closes: np.ndarray,
    effective: np.ndarray,
    capital: float,
    cost_rate: float,
) -> _TickerResult:
    """Simulate one ticker with exact dollar accounting and FIFO segments.

    Sizing is fixed at trade time and shares are held until the target
    changes: when the target moves, the desired share count is recomputed
    from current open-time wealth; between target changes the position
    drifts and no trades occur.
    """
    cash = float(capital)
    shares = 0.0
    prev_target = 0.0
    wealth = np.empty(len(dates))
    events: list[dict] = []
    open_segments: list[dict] = []

    for i in range(len(dates)):
        date = dates[i]
        open_price = float(opens[i])
        close_price = float(closes[i])
        target = float(effective[i])

        if abs(target - prev_target) > _EMPTY_THRESHOLD:
            wealth_at_open = cash + shares * open_price
            target_shares = target * wealth_at_open / open_price
            delta = target_shares - shares
            if abs(delta) > _EMPTY_THRESHOLD:
                turnover = abs(delta) * open_price
                cost = turnover * cost_rate
                cash -= delta * open_price + cost
                signal_date = pd.Timestamp(dates[i - 1]) if i else pd.NaT
                if delta > 0:
                    events.append({
                        "ticker": ticker, "date": date, "signal_date": signal_date, "side": "buy",
                        "entry_date": date, "entry_price": open_price, "exit_price": np.nan,
                        "qty": delta, "turnover": turnover, "cost": cost,
                        "gross_return": np.nan, "position_after": target,
                    })
                    open_segments.append({"entry_date": date, "entry_price": open_price, "qty": delta})
                else:
                    remaining = -delta
                    while remaining > _EMPTY_THRESHOLD and open_segments:
                        segment = open_segments[0]
                        taken = min(segment["qty"], remaining)
                        segment["qty"] -= taken
                        remaining -= taken
                        if segment["qty"] <= _EMPTY_THRESHOLD:
                            open_segments.pop(0)
                        events.append({
                            "ticker": ticker, "date": date, "signal_date": signal_date, "side": "sell",
                            "entry_date": segment["entry_date"], "entry_price": segment["entry_price"],
                            "exit_price": open_price, "qty": taken, "turnover": taken * open_price,
                            "cost": cost * (taken / abs(delta)),
                            "gross_return": open_price / segment["entry_price"] - 1.0,
                            "position_after": target,
                        })
                shares = target_shares
            prev_target = target
        wealth[i] = cash + shares * close_price

    if abs(shares) > _EMPTY_THRESHOLD:
        last_price = float(closes[-1])
        turnover = abs(shares) * last_price
        cost = turnover * cost_rate
        while open_segments:
            segment = open_segments.pop(0)
            events.append({
                "ticker": ticker, "date": dates[-1], "signal_date": pd.Timestamp(dates[-2]) if len(dates) > 1 else pd.NaT,
                "side": "sell", "entry_date": segment["entry_date"], "entry_price": segment["entry_price"],
                "exit_price": last_price, "qty": segment["qty"], "turnover": segment["qty"] * last_price,
                "cost": cost * (segment["qty"] / abs(shares)),
                "gross_return": last_price / segment["entry_price"] - 1.0,
                "position_after": 0.0,
            })
        # wealth[-1] already prices the shares at the final close; the sale
        # converts them to cash, so only the liquidation cost is subtracted.
        wealth[-1] -= cost

    return _TickerResult(ticker=ticker, dates=pd.Series(pd.to_datetime(dates)), wealth=wealth, events=events)


def simulate(
    prices: pd.DataFrame,
    positions: pd.Series | pd.DataFrame,
    config: BacktestConfig,
    data_metadata: dict | None = None,
    strategy_name: str | None = None,
    strategy_source: dict | None = None,
    strategy_params: dict | None = None,
) -> BacktestReport:
    """Run the cost-aware simulation and freeze a ``BacktestReport``.

    The input is sorted defensively; duplicate ``(ticker, date)`` rows raise.
    Metrics are produced for every scenario in ``cost_scenarios_bps``; the
    stored trade ledger reflects the base ``transaction_cost_bps`` scenario.
    ``strategy_source`` (provenance for custom strategies) and the
    ``strategy_params`` actually used are recorded in the report's config
    snapshot.
    """
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No price rows to backtest.")
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Backtest input missing columns: {sorted(missing)}")
    if frame.duplicated(["ticker", "date"]).any():
        raise ValueError("Duplicate (ticker, date) rows are not allowed.")
    if (frame[["open", "close"]] <= 0).any().any():
        raise ValueError("Open and close prices must be positive.")

    aligned = _align_positions(frame, positions)
    if strategy_name:
        resolved_name = strategy_name
    else:
        resolved_name = getattr(positions, "name", None) or "strategy"
    effective = aligned.groupby(level=0).shift(1).fillna(0.0) * config.max_position

    tickers = list(frame["ticker"].unique())
    capital = allocate_capital(config.initial_capital, tickers)
    base_cost_rate = (config.transaction_cost_bps + config.slippage_bps) / 10_000.0

    base_results: dict[str, _TickerResult] = {}
    scenario_metrics: dict[float, dict[str, float]] = {}
    scenarios = sorted(set(config.cost_scenarios_bps) | {float(config.transaction_cost_bps)})
    for bps in scenarios:
        cost_rate = (float(bps) + config.slippage_bps) / 10_000.0
        per_ticker: dict[str, _TickerResult] = {}
        for ticker in tickers:
            rows = frame.loc[frame["ticker"] == ticker]
            per_ticker[ticker] = _simulate_ticker(
                ticker,
                rows["date"].to_numpy(),
                rows["open"].to_numpy(dtype=float),
                rows["close"].to_numpy(dtype=float),
                effective.loc[ticker].to_numpy(dtype=float),
                capital,
                cost_rate,
            )
        wealth_df = pd.DataFrame({
            "ticker": [t for t in tickers for _ in per_ticker[t].wealth],
            "date": pd.to_datetime([d for t in tickers for d in per_ticker[t].dates]),
            "wealth": [w for t in tickers for w in per_ticker[t].wealth],
        })
        pooled = pooled_wealth(wealth_df, config.initial_capital)
        ledger_df = pd.DataFrame([event for t in tickers for event in per_ticker[t].events], columns=_ORDER)
        metric_positions = effective.reset_index().set_index(["ticker", "date"])["position"]
        scenario_metrics[float(bps)] = compute_metrics(
            wealth=pooled["pooled_wealth"].to_numpy(),
            returns=pooled["pooled_return"].to_numpy(),
            ledger=ledger_df,
            effective_positions=metric_positions,
            initial_capital=config.initial_capital,
        )
        if float(bps) == float(config.transaction_cost_bps):
            base_results = per_ticker

    ledger = pd.DataFrame([event for t in tickers for event in base_results[t].events], columns=_ORDER)
    ledger = ledger.sort_values(["ticker", "date", "side"]).reset_index(drop=True)
    wealth_df = pd.DataFrame({
        "ticker": [t for t in tickers for _ in base_results[t].wealth],
        "date": pd.to_datetime([d for t in tickers for d in base_results[t].dates]),
        "wealth": [w for t in tickers for w in base_results[t].wealth],
    })
    pooled = pooled_wealth(wealth_df, config.initial_capital)
    metric_positions = effective.reset_index().set_index(["ticker", "date"])["position"]

    benchmark = _benchmark_report(frame, config)
    per_stock = per_stock_summary(wealth_df, ledger, metric_positions, capital)

    paths: list[tuple[str, np.ndarray]] = []
    equity_pieces: list[pd.DataFrame] = []
    for ticker in tickers:
        result = base_results[ticker]
        equity_pieces.append(pd.DataFrame({
            "date": pd.to_datetime(result.dates),
            "path": f"{resolved_name}::{ticker}",
            "wealth": result.wealth,
        }))
    equity_pieces.append(pd.DataFrame({
        "date": pooled["date"],
        "path": resolved_name,
        "wealth": pooled["pooled_wealth"].to_numpy(),
    }))
    equity_pieces.append(benchmark.equity_curve)
    equity_curve = pd.concat(equity_pieces, ignore_index=True)
    equity_curve["date"] = pd.to_datetime(equity_curve["date"])

    snapshot = {
        "backtest": {key: list(value) if isinstance(value, tuple) else value for key, value in asdict(config).items()},
        "tickers": tickers,
        "strategy": resolved_name,
    }
    if strategy_source:
        snapshot["strategy_source"] = dict(strategy_source)
    if strategy_params:
        snapshot["strategy_params"] = dict(strategy_params)
    return BacktestReport(
        strategy_name=resolved_name,
        equity_curve=equity_curve,
        trade_ledger=ledger,
        per_stock=per_stock,
        metrics=scenario_metrics,
        benchmark_metrics=benchmark.metrics,
        config_snapshot=snapshot,
        data_metadata=data_metadata or {},
    )


@dataclass
class _Benchmark:
    equity_curve: pd.DataFrame
    metrics: dict[str, float]


def _benchmark_report(frame: pd.DataFrame, config: BacktestConfig) -> _Benchmark:
    """Equal-weight buy-and-hold benchmark, simulated gross of costs.

    Uses the same timing convention as every strategy: the first session has
    no signal, so the benchmark buys at the second session's open and is
    liquidated at the final close.
    """
    tickers = list(frame["ticker"].unique())
    capital = allocate_capital(config.initial_capital, tickers)
    results: dict[str, _TickerResult] = {}
    for ticker in tickers:
        rows = frame.loc[frame["ticker"] == ticker]
        effective = np.ones(len(rows)) * config.max_position
        effective[0] = 0.0
        results[ticker] = _simulate_ticker(
            ticker, rows["date"].to_numpy(), rows["open"].to_numpy(dtype=float),
            rows["close"].to_numpy(dtype=float), effective, capital, 0.0,
        )
    wealth_df = pd.DataFrame({
        "ticker": [t for t in tickers for _ in results[t].wealth],
        "date": pd.to_datetime([d for t in tickers for d in results[t].dates]),
        "wealth": [w for t in tickers for w in results[t].wealth],
    })
    pooled = pooled_wealth(wealth_df, config.initial_capital)
    ledger = pd.DataFrame([event for t in tickers for event in results[t].events], columns=_ORDER)
    positions = pd.Series(
        config.max_position, index=pd.MultiIndex.from_frame(frame[["ticker", "date"]]), name="buy_and_hold",
    )
    positions = positions.groupby(level=0).shift(1).fillna(0.0)
    equity = pd.DataFrame({
        "date": pd.to_datetime([d for d in results[tickers[0]].dates]),
        "path": "buy_and_hold_benchmark",
        "wealth": pooled["pooled_wealth"].to_numpy(),
    })
    metrics = compute_metrics(
        wealth=pooled["pooled_wealth"].to_numpy(),
        returns=pooled["pooled_return"].to_numpy(),
        ledger=ledger,
        effective_positions=positions,
        initial_capital=config.initial_capital,
    )
    return _Benchmark(equity_curve=equity, metrics=metrics)
