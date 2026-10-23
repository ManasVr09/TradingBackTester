"""Streamlit dashboard for the trading backtester.

Cache-first: charts and backtests read ``data/cache/*.csv``; Yahoo is only
contacted through the explicit "Refresh cache from Yahoo" button. The UI
consumes immutable ``BacktestReport`` objects produced by ``src.pipeline``;
it never silently re-runs a backtest.

Custom strategies: a user may upload a ``.py`` file subclassing
``src.strategies.Strategy``. It is validated against the six-gate contract
(no look-ahead, discrete positions, no forbidden imports, ...) before it can
run; valid uploads may be saved to ``strategies/custom/`` and auto-register
on the next app start.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data import cache_path, cache_status, ensure_data, load_cached, load_metadata  # noqa: E402
from src.strategies import STRATEGY_REGISTRY  # noqa: E402
from src.strategy_loader import (  # noqa: E402
    GATES,
    StrategyValidationError,
    load_and_validate,
    register_custom,
    register_saved_strategies,
    safe_strategy_filename,
)

st.set_page_config(page_title="Trading backtester", page_icon=":material/query_stats:", layout="wide")

CONFIG = load_config()
CUSTOM_DIR = ROOT / "strategies" / "custom"

DISCLAIMER = (
    "Educational research tool only. Yahoo Finance data is not point-in-time: historical "
    "prices can be revised, delisted tickers vanish, and today's backtest cannot know "
    "yesterday's revisions. Results are historical simulations, not investment advice. "
    "Do not use for portfolio action."
)

_BUILTIN_LABELS = {
    "sma_crossover": "SMA crossover",
    "ema_crossover": "EMA crossover",
    "rsi_mean_reversion": "RSI mean reversion",
    "macd_momentum": "MACD momentum",
    "atr_breakout": "ATR breakout",
}
_STRING_PARAM_OPTIONS = {("atr_breakout", "exit_rule"): ["midline", "lower_band"]}


def readable_table(frame, *, hide_index=True):
    columns = {
        name: st.column_config.Column(str(name), width="large" if name in
            ("Strategy", "strategy", "path", "model", "retrieved_at_utc") else "medium")
        for name in frame.columns
    }
    st.dataframe(frame, hide_index=hide_index, width="stretch", row_height=44, column_config=columns)


def result_card(label, value, *, text=False, help=None, width=250):
    """Wrap full labels and text outcomes; reserve large metric type for short numbers."""
    with st.container(border=True, width=width):
        st.markdown(label, help=help)
        if text or len(str(value)) > 8:
            st.subheader(str(value), anchor=False)
        else:
            st.metric(label, value, label_visibility="collapsed")


def available_tickers() -> list[str]:
    """Configured tickers that currently have a cache file."""
    configured = list(dict.fromkeys(CONFIG["tickers"]["universe"] + [CONFIG["tickers"]["benchmark"]]))
    return [ticker for ticker in configured if cache_path(ticker, CONFIG).exists()]


def cache_signatures(tickers: list[str]) -> tuple:
    """Paths, nanosecond mtimes, and sizes invalidate cached reads after a refresh."""
    signatures = []
    for ticker in tickers:
        path = cache_path(ticker, CONFIG)
        stat = path.stat()
        signatures.append((ticker, str(path.resolve()), stat.st_mtime_ns, stat.st_size))
    return tuple(signatures)


@st.cache_data(max_entries=16, show_spinner=False)
def load_cache_frames(signatures: tuple) -> dict[str, pd.DataFrame]:
    """Load validated cached frames, keyed by file mtimes so edits invalidate."""
    return {ticker: load_cached(ticker, CONFIG) for ticker, _path, _mtime, _size in signatures}


def sample_prices() -> pd.DataFrame | None:
    """A single-ticker cached frame used to validate uploaded strategies."""
    available = available_tickers()
    if not available:
        return None
    frames = load_cache_frames(cache_signatures(available[:1]))
    return next(iter(frames.values()))


@st.cache_data(max_entries=16, show_spinner=False)
def cached_date_bounds(signatures: tuple):
    frames = load_cache_frames(signatures)
    return (min(frame["date"].min() for frame in frames.values()).date(),
            max(frame["date"].max() for frame in frames.values()).date())


def strategy_specs() -> list[dict]:
    return list(CONFIG["strategies"]["enabled"])


def custom_keys() -> list[str]:
    """Registry keys for custom strategies: session uploads first, then saved."""
    session_keys = sorted(st.session_state.get("session_registry", {}))
    global_keys = sorted(
        key for key, cls in STRATEGY_REGISTRY.items() if hasattr(cls, "strategy_source")
    )
    return [key for key in dict.fromkeys(session_keys + global_keys)]


def strategy_options() -> list[str]:
    """Internal selection keys: built-in registry keys first, then customs."""
    builtins = [key for key in STRATEGY_REGISTRY if key in _BUILTIN_LABELS]
    return [key for key in dict.fromkeys(builtins + custom_keys())]


def format_strategy_option(key: str) -> str:
    if key in _BUILTIN_LABELS:
        return _BUILTIN_LABELS[key]
    return f"custom: {key}"


def builtin_defaults(key: str) -> dict:
    """Declared params for a built-in, overridden by the config spec if present."""
    cls = STRATEGY_REGISTRY[key]
    defaults = dict(getattr(cls, "params", {}) or {})
    for spec in strategy_specs():
        if spec.get("type") == key:
            for param, value in spec.items():
                if param != "type":
                    defaults.setdefault(param, value)
    return defaults


def _param_value(key: str, param: str, default) -> object:
    widget_key = f"param_{key}_{param}"
    value = st.session_state.get(widget_key, default)
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, str):
        return str(value)
    if isinstance(default, int):
        return int(value)
    return float(value)


def selected_custom_class() -> type | None:
    key = st.session_state["strategy_key"]
    if key in _BUILTIN_LABELS:
        return None
    return st.session_state.get("session_registry", {}).get(key) or STRATEGY_REGISTRY.get(key)


def build_strategy_from_sidebar() -> object:
    """Instantiate the strategy configured in the sidebar with its params."""
    key = st.session_state["strategy_key"]
    if key in _BUILTIN_LABELS:
        defaults = builtin_defaults(key)
        values = {param: _param_value(key, param, default) for param, default in defaults.items()}
        return STRATEGY_REGISTRY[key](**values)
    cls = selected_custom_class()
    if cls is None:
        raise ValueError(f"Unknown custom strategy: {key}")
    instance = cls()
    instance.params = dict(getattr(cls, "params", {}) or {})
    for param in list(instance.params):
        value = st.session_state.get(f"custom_param_{param}", instance.params[param])
        if isinstance(instance.params[param], bool):
            instance.params[param] = bool(value)
        elif isinstance(instance.params[param], int):
            instance.params[param] = int(value)
        else:
            instance.params[param] = float(value)
    return instance


def run_backtest() -> None:
    """Build an immutable report from cache data and the sidebar configuration."""
    tickers = list(st.session_state.get("tickers", []))
    if not tickers:
        st.session_state["run_error"] = "Select at least one ticker."
        return
    frames = load_cache_frames(cache_signatures(tickers))
    start = st.session_state.get("start_date")
    end = st.session_state.get("end_date")
    if start is not None and end is not None:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        frames = {ticker: frame.loc[(frame["date"] >= start) & (frame["date"] <= end)] for ticker, frame in frames.items()}
    try:
        from src.pipeline import build_report

        report = build_report(
            CONFIG,
            frames,
            build_strategy_from_sidebar(),
            metadata=load_metadata(CONFIG),
            transaction_cost_bps=float(st.session_state["cost_bps"]),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.session_state["run_error"] = str(exc)
        return
    st.session_state["report"] = report
    st.session_state["run_error"] = None


def refresh_cache() -> None:
    """Explicit user-requested refresh; the only path that contacts Yahoo."""
    try:
        frames = ensure_data(CONFIG, force=bool(st.session_state.get("force_refresh", False)))
        rows = {ticker: len(frame) for ticker, frame in frames.items()}
        st.session_state["refresh_result"] = f"Cache refreshed: {len(frames)} tickers, rows {rows}"
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.session_state["refresh_result"] = f"Refresh failed: {exc}"


def scan_saved_customs() -> None:
    """Register strategies saved under ``strategies/custom/`` once per session."""
    if "saved_customs" in st.session_state:
        return
    if not CUSTOM_DIR.exists() or not any(CUSTOM_DIR.glob("*.py")):
        st.session_state["saved_customs"] = {}
        return
    sample = sample_prices()
    registered = register_saved_strategies(CUSTOM_DIR, sample)
    st.session_state["saved_customs"] = {
        key: (cls.strategy_source or {}).get("sha256", "") for key, cls in registered.items()
    }


def handle_upload(uploaded) -> None:
    """Validate an uploaded strategy file and surface the gate checklist."""
    file_bytes = uploaded.getvalue()
    sha = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("upload_sha") == sha:
        return
    st.session_state["upload_sha"] = sha
    sample = sample_prices()
    if sample is None:
        st.session_state["upload_status"] = {
            "kind": "fail",
            "failures": ["No cached data to validate against — refresh the cache first."],
        }
        return
    for key, saved_sha in st.session_state.get("saved_customs", {}).items():
        if saved_sha == sha:
            st.session_state["strategy_key"] = key
            st.session_state["upload_status"] = {"kind": "dup", "key": key}
            return
    tmp = Path(tempfile.mkstemp(suffix=".py")[1])
    try:
        tmp.write_bytes(file_bytes)
        cls = load_and_validate(tmp, sample)
        key = register_custom(cls, tmp, registry=st.session_state.setdefault("session_registry", {}))
        cls.strategy_source["file_name"] = safe_strategy_filename(uploaded.name)
        st.session_state["strategy_key"] = key
        st.session_state["upload_status"] = {
            "kind": "ok",
            "key": key,
            "gates": list(getattr(cls, "validation_gates", GATES)),
        }
        st.session_state["pending_save"] = {
            "key": key,
            "file_name": safe_strategy_filename(uploaded.name),
            "sha256": sha,
            "bytes": file_bytes,
        }
    except StrategyValidationError as exc:
        st.session_state["upload_status"] = {"kind": "fail", "failures": list(exc.failures)}
    finally:
        tmp.unlink(missing_ok=True)


def save_custom() -> None:
    """Persist the pending upload to ``strategies/custom/``."""
    pending = st.session_state.get("pending_save")
    if pending is None:
        return
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    target = CUSTOM_DIR / pending["file_name"]
    target.write_bytes(pending["bytes"])
    st.session_state.setdefault("saved_customs", {})[pending["key"]] = pending["sha256"]
    st.session_state["upload_status"] = {"kind": "saved", "path": str(target)}
    st.session_state["pending_save"] = None


def pct(value: float) -> str:
    return f"{value:.1%}"


def overview(report) -> None:
    import altair as alt

    st.subheader("Equal-weight portfolio vs buy-and-hold")
    equity = report.equity_curve
    plot = equity.loc[equity["path"].isin([report.strategy_name, "buy_and_hold_benchmark"])]
    chart = alt.Chart(plot[["date", "path", "wealth"]]).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("wealth:Q", title="Wealth ($)"),
        color=alt.Color("path:N", title="Path"),
        tooltip=["date", "path", "wealth"],
    )
    st.altair_chart(chart, width="stretch", height=400)
    base_bps = float(report.config_snapshot["backtest"]["transaction_cost_bps"])
    base = report.metrics[base_bps]
    benchmark = report.benchmark_metrics
    with st.container(horizontal=True):
        result_card('Net return', pct(base['total_return']))
        result_card('CAGR', pct(base['cagr']))
        result_card('Sharpe', f"{base['sharpe']:.2f}")
        result_card('Max drawdown', pct(base['maximum_drawdown']))
        result_card('Trades', f"{base['trade_count']:,}")
        result_card('Buy-and-hold return', pct(benchmark['total_return']))
    st.subheader("Cost sensitivity")
    table = pd.DataFrame(report.metrics).T.rename_axis("cost_bps").reset_index()
    readable_table(table, hide_index=True)
    st.caption(
        f"Execution clock: signal from close(t) enters at open(t+1); exits at the same-day close; "
        f"terminal liquidation on the last date. All rows are net of one-way "
        f"transaction cost + slippage for that scenario."
    )


def per_ticker(report) -> None:
    import altair as alt

    st.subheader("Per-ticker equity paths")
    equity = report.equity_curve
    ticker_paths = equity.loc[equity["path"].str.startswith(f"{report.strategy_name}::")]
    benchmark = equity.loc[equity["path"] == "buy_and_hold_benchmark"]
    plot = pd.concat([ticker_paths, benchmark], ignore_index=True)
    chart = alt.Chart(plot[["date", "path", "wealth"]]).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("wealth:Q", title="Wealth ($)"),
        color=alt.Color("path:N", title="Path"),
        tooltip=["date", "path", "wealth"],
    )
    st.altair_chart(chart, width="stretch", height=400)
    readable_table(report.per_stock, hide_index=True)


def trade_ledger(report) -> None:
    st.subheader("Trade ledger")
    readable_table(report.trade_ledger, hide_index=True)
    st.caption("Prices are raw market fills; costs are charged separately per event.")


def data_cache() -> None:
    st.subheader("Cache status")
    readable_table(cache_status(CONFIG), hide_index=True)
    metadata = load_metadata(CONFIG)
    if metadata:
        st.json(metadata, expanded=False)
    else:
        st.info("No metadata yet — run 'Refresh cache from Yahoo' once.")


def strategy_config(report) -> None:
    st.subheader("Frozen run configuration")
    st.json(report.config_snapshot, expanded=True)


scan_saved_customs()
report: object | None = st.session_state.get("report")

with st.sidebar:
    st.header("Run configuration")
    available = available_tickers()
    if not available:
        st.error("No cached data yet. Click 'Refresh cache from Yahoo' to download.")
    tickers = st.multiselect("Tickers", options=available, default=available, key="tickers")

    st.subheader("Custom strategy")
    uploaded = st.file_uploader("Upload a strategy (.py)", type="py", key="custom_upload")
    if uploaded is not None:
        handle_upload(uploaded)
    upload_status = st.session_state.get("upload_status")
    if upload_status:
        if upload_status["kind"] == "ok":
            st.success("All gates passed: " + ", ".join(upload_status["gates"]))
        elif upload_status["kind"] == "dup":
            st.info(f"Already saved; selected custom: {upload_status['key']}")
        elif upload_status["kind"] == "saved":
            st.success(f"Saved to {upload_status['path']}")
        else:
            st.error("Validation failed:")
            for failure in upload_status.get("failures", []):
                st.markdown(f"- {failure}")
    if st.session_state.get("pending_save") is not None:
        st.button("Save to strategies/custom/", key="save_custom", on_click=save_custom)

    options = strategy_options()
    default_index = 0
    st.selectbox(
        "Active strategy",
        options,
        index=default_index,
        format_func=format_strategy_option,
        key="strategy_key",
    )
    selected_key = st.session_state["strategy_key"]
    if selected_key in _BUILTIN_LABELS:
        for param, default in builtin_defaults(selected_key).items():
            widget_key = f"param_{selected_key}_{param}"
            if isinstance(default, bool):
                st.checkbox(param, value=default, key=widget_key)
            elif isinstance(default, str):
                choices = _STRING_PARAM_OPTIONS.get((selected_key, param)) or [default]
                st.selectbox(param, choices, index=choices.index(default) if default in choices else 0, key=widget_key)
            elif isinstance(default, int):
                st.number_input(param, value=int(default), step=1, key=widget_key)
            else:
                st.number_input(param, value=float(default), key=widget_key)
    else:
        cls = selected_custom_class()
        if cls is not None and getattr(cls, "params", None):
            with st.expander("Params"):
                for param, default in (getattr(cls, "params", {}) or {}).items():
                    if isinstance(default, bool):
                        st.checkbox(param, value=default, key=f"custom_param_{param}")
                    elif isinstance(default, int):
                        st.number_input(param, value=int(default), step=1, key=f"custom_param_{param}")
                    else:
                        st.number_input(param, value=float(default), key=f"custom_param_{param}")
    if available:
        min_date, max_date = cached_date_bounds(cache_signatures(available))
        st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date, key="start_date")
        st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date, key="end_date")
    costs = CONFIG["execution"]["cost_scenarios_bps"]
    st.selectbox(
        "Base transaction cost (bps)",
        costs,
        index=costs.index(CONFIG["execution"]["transaction_cost_bps"]),
        key="cost_bps",
    )
    st.button("Run backtest", type="primary", key="run_button", on_click=run_backtest)
    st.divider()
    st.checkbox("Force re-download on refresh", value=False, key="force_refresh")
    st.button("Refresh cache from Yahoo", key="refresh_button", on_click=refresh_cache)
    if st.session_state.get("refresh_result"):
        st.caption(st.session_state["refresh_result"])
    if st.session_state.get("run_error"):
        st.error(st.session_state["run_error"])
    st.caption("Cache-first: Yahoo is contacted only by the refresh button.")

st.title("Trading backtester")
st.caption("Multi-asset daily backtesting with a per-ticker CSV cache")
st.warning(DISCLAIMER, icon=":material/warning:")

tab_labels = ["Overview", "Per-Ticker", "Trade Ledger", "Data Cache", "Strategy config"]
tabs = st.tabs(tab_labels, on_change="rerun", key="report_view")
renderers = [overview, per_ticker, trade_ledger, None, strategy_config]
empty_messages = [
    "No backtest run yet — choose a configuration and click 'Run backtest' in the sidebar.",
    "Run a backtest to see per-ticker paths.",
    "Run a backtest to inspect the trade ledger.",
    "", "Run a backtest to freeze a configuration snapshot.",
]
for tab, render, empty_message in zip(tabs, renderers, empty_messages):
    if tab.open:
        with tab:
            if render is None:
                data_cache()
            elif report is None:
                st.info(empty_message)
            else:
                render(report)
