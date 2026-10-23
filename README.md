# Trading Backtester

An educational multi-asset trading backtester: it scrapes daily OHLCV from Yahoo Finance, persists it to a per-ticker CSV cache, replays runs from that cache (a warm cache triggers zero network requests), runs pluggable strategies over an equal-weight portfolio, and exposes results through a Streamlit dashboard.

**Educational research tool only. Yahoo Finance data is not point-in-time**: historical prices can be revised after the fact, delisted tickers vanish, and a backtest cannot know yesterday's revisions. Results are historical simulations, not investment advice. **Do not use for portfolio action.**

## Quick start

Python 3.11+ recommended.

```bash
cd trading_backtester
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline            # warm cache: zero Yahoo requests
python -m src.pipeline --refresh  # explicit cache refresh from Yahoo
streamlit run dashboard/app.py
```

## Project structure

```text
trading_backtester/
  configs/project.yaml          # tickers, date range, costs, strategy params
  src/
    config.py                   # ROOT, load_config, validate_config
    data.py                     # Yahoo scraper + CSV cache + incremental refresh  (core)
    indicators.py               # SMA, EMA, RSI, MACD, ATR
    strategies.py               # Strategy interface + SMA/EMA crossover, registry
    backtest.py                 # engine, ledger, cost-aware simulation
    portfolio.py                # equal-weight allocation + pooling
    metrics.py                  # returns, vol, Sharpe, Sortino, drawdown, turnover
    reporting.py                # frozen BacktestReport dataclass + serialization
    pipeline.py                 # config -> data -> backtest -> report
  data/
    cache/{TICKER}.csv          # per-ticker daily OHLCV (gitignored)
    metadata.json               # retrieval timestamps and row counts (gitignored)
  dashboard/app.py              # Streamlit UI consuming immutable BacktestReport
  tests/                        # pytest
```

## Data layer (the core)

- Per-ticker CSV schema: `date,open,high,low,close,volume,retrieved_at_utc` (dates written as `YYYY-MM-DD`).
- **Incremental append rule**: no CSV → full download; cache fresh (`last_date >= today - freshness_days`) → zero network; otherwise download only dates after `last_date`, drop duplicates, append with a fresh `retrieved_at_utc`, and write atomically (temp file + rename).
- **Resilience**: retry with exponential backoff, rate limiting between tickers, and loud failures on empty frames, missing columns, duplicate dates, NaN OHLCV, non-positive prices, or negative volume. A failed refresh never overwrites a good cache; by default it raises (or serves the stale cache when `allow_stale_cache_on_error: true`, never silently).
- `ensure_data` is idempotent: a second call on a warm cache makes **zero** `yf.download` calls (enforced by tests).

## Execution contract

Signals are formed from `close(t)` and enter at `open(t+1)`; positions exit at the same-day close. Indicators are strictly backward-looking (value at `t` uses only rows `<= t`). The engine executes each ticker as an independent cash account seeded with an equal share of `initial_capital`; the pooled path is the exact sum of the per-stock accounts (never interleaved compounding). Terminal liquidation on the last date is an explicit closing trade with costs. One-way cost = `transaction_cost_bps + slippage_bps` charged on absolute turnover.

`AtrBreakout` compares today's close to **yesterday's** channel: the SMA midline and ATR are shifted one bar, so `band[t]` never contains `close[t]` itself — the textbook Donchian look-ahead trap. `SmaCrossover`, `EmaCrossover`, `MacdMomentum`, and `RsiMeanReversion` follow the same prior-information discipline. None of the built-ins model borrow costs for shorts; enabling `allow_short` skews net results optimistically.

## Dashboard

Cache-first: charts and backtests read `data/cache/*.csv` (loaded via `@st.cache_data` keyed by file mtime). Yahoo is contacted **only** through the explicit "Refresh cache from Yahoo" button. "Run backtest" freezes an immutable `BacktestReport` from cache data; the UI never silently re-runs a backtest.

## Tests

```bash
pytest
```

Coverage: scraper/cache/incremental append (heavy), indicator fixtures, strategy determinism and no-look-ahead, hand-computed backtest accounting including terminal liquidation, ledger-price reconciliation, monotonic cost sensitivity, row-reorder invariance, pooled-path construction, metric fixtures, report round-trips, and the dashboard via Streamlit `AppTest`.

## Limitations

- Yahoo Finance is not an institutional market-data source; adjusted OHLC reconstruction is approximate.
- Data is not point-in-time and the dashboard can show results computed from data that has since been revised.
- Daily bars and a small universe limit the conclusions; costs, taxes, borrow, market impact, and capacity are only partially modeled.
- Results are historical simulations for education only — not investment advice.

## Performance

See the [performance notes](PERFORMANCE.md) for caching, lazy loading, and preservation guarantees.
