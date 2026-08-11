# Trading Backtester — Implementation Plan

## Goal

A Python trading-backtester that scrapes daily OHLCV from Yahoo Finance, persists it to a per-ticker CSV cache, replays runs from the cache for performance, runs pluggable strategies over a multi-asset portfolio, and exposes results through a Streamlit dashboard. The data layer (scraping + incremental CSV caching) is the core focus and must be the most heavily tested component.

## Conventions to follow (from sibling projects)

- Package layout: `src/` modules + `dashboard/app.py`, `configs/project.yaml`, `tests/`, top-level `requirements.txt`, `README.md`, `.gitignore`.
- Typed config loaded via `src/config.py` with a `ROOT` path constant (mirrors `stock-direction-classifier/src/config.py`).
- Docstrings on every module; `from __future__ import annotations` at the top of source files.
- pytest under `tests/`; dashboard test uses Streamlit `AppTest`.
- README must disclose that Yahoo data is not point-in-time and that results are educational, not advisory (mirrors the "do not use for portfolio action" disclaimer).

## Project structure

```text
trading_backtester/
  README.md
  requirements.txt
  .gitignore
  IMPLEMENTATION_PLAN.md          # this file
  configs/
    project.yaml                   # tickers, date range, costs, strategy params
  src/
    __init__.py
    config.py                      # ROOT, load_config, validate config
    data.py                        # Yahoo scraper + CSV cache + incremental refresh  *** core focus ***
    indicators.py                  # SMA, EMA, RSI, MACD, ATR
    strategies.py                  # Strategy interface + built-ins (SMA/EMA crossover)
    backtest.py                    # engine, ledger, cost-aware simulation
    portfolio.py                   # multi-asset allocation + per-stock aggregation
    metrics.py                     # returns, vol, Sharpe, Sortino, drawdown, turnover
    reporting.py                   # build BacktestReport (equity curve, ledger, metrics)
    pipeline.py                    # orchestrate: load config -> data -> backtest -> report
  data/
    cache/                         # per-ticker CSVs (gitignored)
    metadata.json                  # last retrieval timestamps per ticker
  dashboard/
    app.py                          # Streamlit UI consuming immutable BacktestReport objects
    requirements.txt
  tests/
    test_data.py                   # scraper + cache + incremental append  *** heavy ***
    test_indicators.py
    test_strategies.py
    test_backtest.py
    test_portfolio.py
    test_metrics.py
    test_reporting.py
    test_dashboard.py              # Streamlit AppTest
```

## Dependencies (`requirements.txt`)

```
numpy>=1.26,<3
pandas>=2.1,<3
yfinance>=0.2.40,<1
pyyaml>=6,<7
streamlit>=1.35,<2
altair>=5.4,<7
pytest>=8,<9
```

Root `requirements.txt` already covers numpy/pandas/pyarrow/scikit-learn/streamlit/altair; this project's requirements.txt should pin yfinance and pyyaml and inherit the shared versions for the rest.

## configs/project.yaml

```yaml
project_name: trading_backtester
research_status: educational
data:
  source: yahoo_finance
  granularity: 1d
  start: "2016-01-01"
  end_exclusive: ""        # empty = up to today
  auto_adjust: true
  actions: false
tickers:
  universe:
    - AAPL
    - MSFT
    - AMZN
    - META
    - NVDA
  benchmark: SPY
cache:
  dir: data/cache
  format: csv
  mode: incremental_append
  retry_attempts: 3
  backoff_seconds: 2
  rate_limit_seconds: 1.0
  allow_stale_cache_on_error: false
  freshness_days: 1
portfolio:
  initial_capital: 100000.0
  allocation: equal_weight
  max_position: 1.0
execution:
  entry_rule: next_open
  exit_rule: same_day_close
  transaction_cost_bps: 5.0
  slippage_bps: 2.0
  cost_scenarios_bps: [0.0, 5.0, 10.0, 25.0]
strategies:
  enabled:
    - type: sma_crossover
      fast: 20
      slow: 50
    - type: ema_crossover
      fast: 12
      slow: 26
  benchmark: buy_and_hold
```

## Phase 1 — Data layer (core focus)

`src/data.py` is the centerpiece. It must guarantee: scrape once, replay forever from CSV, never re-download data already on disk.

### 1.1 CSV cache schema

Per-ticker file `data/cache/{TICKER}.csv`, long-day rows, columns:

```
date,open,high,low,close,volume,retrieved_at_utc
```

- `date` is tz-naive daily timestamp; write as `YYYY-MM-DD`.
- `retrieved_at_utc` records when each row was fetched (ISO UTC) — supports freshness checks and reproducibility audits.
- One metadata file `data/metadata.json` with: `retrieved_at_utc`, `source`, `start`, `end_exclusive`, `tickers`, per-ticker `last_date` and `rows`.

### 1.2 Functions

```python
def download_ticker(ticker, start, end, auto_adjust=True, actions=False) -> pd.DataFrame
def _normalize_yf(frame, ticker) -> pd.DataFrame            # tidy long format, lowercase cols
def validate_prices(frame, ticker) -> None                 # OHLC positivity, dup date, NaN
def cache_path(ticker, config) -> Path
def load_cached(ticker, config) -> pd.DataFrame            # read CSV, parse types
def last_cached_date(ticker, config) -> Timestamp | None
def refresh_ticker(ticker, start, end, ...) -> pd.DataFrame # incremental append
def ensure_data(config) -> dict[str, pd.DataFrame]          # orchestrator used by pipeline
```

### 1.3 Incremental append rule

For each ticker:
1. If no CSV exists → full download `start..end`, write CSV.
2. If CSV exists → `last_date = max(date)`; if `last_date >= today - freshness_days` → skip network.
3. Else → `yf.download(ticker, start=last_date+1d, end=end)`, drop rows already present to avoid duplicates, append with fresh `retrieved_at_utc`, write back via atomic temp-file + rename.

### 1.4 Resilience (the scraper must not silently produce bad data)

- Retry with exponential backoff (`retry_attempts`, `backoff_seconds`).
- Rate-limit between tickers (`rate_limit_seconds`).
- Fail loudly on: empty frame, missing required columns, duplicate dates, NaN close/volume, non-positive open/high/low/close, negative volume.
- Never mask a scrape failure behind stale cache: if a refresh errors and `allow_stale_cache_on_error` is `false` (default), raise. When the flag is `true`, log the failure and return the existing cache with a warning, but never overwrite cache with partial/empty data.
- Validate yfinance's multi-ticker column layout defensively (it changes between versions); normalize lowercase/snake_case columns; drop `adj_close` since `auto_adjust=True`.

### 1.5 Acceptance criteria (Phase 1)

- `ensure_data` with an empty cache populates per-ticker CSVs and writes `metadata.json`.
- A second `ensure_data` call makes **zero** network requests (monkeypatch `yf.download` to assert) and returns identical DataFrames.
- Appending a hand-injected newer row via `yf.download` stub adds only the new rows; row count and `date` ordering stay correct.
- Corrupting a CSV (NaN close) causes `validate_prices` to raise, not silently load.
- The `.gitignore` excludes `data/cache/*.csv` and `data/metadata.json`.

## Phase 2 — Indicators (compute-on-cache, never re-scrape)

`src/indicators.py`: SMA, EMA, RSI (Wilder), MACD, ATR. All operate on a cached DataFrame, never read network. Vectorized pandas, return Series aligned to the input index. No look-ahead — indicators only use rows `<= t`.

## Phase 3 — Strategies

`src/strategies.py` defines a Protocol/interface:

```python
class Strategy(Protocol):
    name: str
    def positions(self, prices: pd.DataFrame) -> pd.Series:  # -1/0/+1 per date
        ...
```

Built-ins:
- `SmaCrossover(fast, slow)` — long when `sma_fast > sma_slow`, else flat (no shorts by default; expose `allow_short` flag).
- `EmaCrossover(fast, slow)`.
- `BuyAndHold` — held as a baseline benchmark only; reported alongside every strategy, never the headline.

Pluggability: instantiate any strategy from `configs/project.yaml` via a small registry (`STRATEGY_REGISTRY[type](**params)`). Custom strategies subclass the interface and register.

Signal-timing contract (match sibling project rigor): signals formed from `close(t)` enter at `open(t+1)`; exit at same-day `close`. Document the information cutoff in the README. No same-day OHLCV look-ahead.

## Phase 4 — Backtest engine

`src/backtest.py` pure function, no Streamlit dependency:

```python
@dataclass
class BacktestConfig:
    entry_rule: str
    exit_rule: str
    initial_capital: float
    max_position: float
    transaction_cost_bps: float
    slippage_bps: float
    cost_scenarios_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0)

def simulate(prices, positions, config) -> BacktestReport
```

- Per ticker per day: compute target position from `positions`; trade only on changed target; charge one-way `transaction_cost_bps + slippage_bps` on absolute turnover.
- Track a trade ledger: `ticker, date, side, entry_price, exit_price, qty, turnover, cost`.
- Terminal liquidation at the last date: explicit closing trade with costs.
- Reordering input rows must not change results (sort defensively, duplicate `(ticker, date)` must raise).

`src/portfolio.py`: equal-weight allocation across selected tickers; per-stock compounded equity AND an equal-weight pooled path computed separately (do NOT compound interleaved rows — sibling project flagged this error explicitly).

## Phase 5 — Metrics and reporting

`src/metrics.py`:
- Total return, CAGR, annualized vol (252-day), Sharpe, Sortino.
- Max drawdown, time under water.
- Turnover, trade count, hit rate, avg win/loss, exposure.
- Report gross + net at every `cost_scenarios_bps`.
- Equal-weight pooled portfolio plus per-stock; chart buy-and-hold benchmark on the same axes.

`src/reporting.py`: freeze a `BacktestReport` dataclass carrying `equity_curve`, `trade_ledger`, `metrics`, `config_snapshot`, `data_metadata`. The dashboard consumes this immutable object — UI never re-runs the backtest.

## Phase 6 — Streamlit dashboard

`dashboard/app.py` consumes one or more `BacktestReport` objects produced by `pipeline.py`:

- Title/run-config sidebar: select tickers, date range, strategy, params, cost scenario. Buttons: **Refresh cache from Yahoo** (writes CSVs), **Run backtest**.
- Cache-first: the dashboard reads `data/cache/*.csv` for charting and parameter sweeps; Yahoo is only contacted via the explicit manual refresh button — never silently in a callback.
- Use modern Streamlit APIs: `st.dataframe`, `st.altair_chart`, `st.metric`, cache data loads with `@st.cache_data` keyed by file mtime. Avoid deprecated `use_container_width` (use `width`), prefer native theming over injected CSS (sibling project flagged this).
- Tabs: Overview (equity curve vs buy-and-hold, key metrics), Per-Ticker, Trade Ledger, Data Cache status (last refresh, rows per ticker, freshness), Strategy params.
- Prominent disclaimer banner: educational, Yahoo data not point-in-time, not investment advice.

## Phase 7 — Tests (pytest)

Heavy coverage on `data.py`:

- **test_data.py** (most important):
  - full download populates cache + metadata;
  - second call issues zero `yf.download` calls (stub);
  - incremental append adds only new rows, no duplicates;
  - duplicate date in cache raises;
  - NaN/invalid values raise `validate_prices`;
  - stale-cache-on-error behavior controlled by `allow_stale_cache_on_error` flag;
  - multi-ticker column normalization for both layouts yfinance returns.
- test_indicators.py: SMA/EMA/RSI/MACD/ATR values match hand-computed fixtures; no look-ahead.
- test_strategies.py: SMA crossover signals deterministic; signal at `t` uses no `>t` data.
- test_backtest.py: buy-and-hold fixture matches hand calc including terminal liquidation cost; zero-signal == cash less no costs; increasing costs can't increase net wealth; row reorder invariance.
- test_portfolio.py: pooled path equals average of per-stock paths (not compounded interleaved).
- test_metrics.py: Sharpe, drawdown, turnover on a known equity series.
- test_reporting.py: `BacktestReport` round-trips; config snapshot preserved.
- test_dashboard.py: Streamlit `AppTest` — renders without error, refresh button doesn't fire network when cache fresh, disclaimer present.

## Acceptance gates (project is "done" when all hold)

1. `ensure_data` is idempotent: a warm cache triggers zero Yahoo requests.
2. Incremental refresh appends only new rows with fresh `retrieved_at_utc`.
3. Backtest gross/net results reconcile exactly to the stored trade ledger prices.
4. Buy-and-hold benchmark renders on the equity curve for every selected strategy.
5. Net result decreases monotonically as `transaction_cost_bps` increases.
6. `pytest -q` is green; `data/cache/*.csv` is gitignored.
7. Dashboard never makes a network call except via the explicit manual **Refresh cache** button.

## .gitignore

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
data/cache/*.csv
data/metadata.json
.DS_Store
```