"""Pluggable trading strategies producing per-date target positions.

Signal-timing contract: a signal formed from ``close(t)`` enters at
``open(t+1)``. Strategies here return the raw daily signal aligned to the
signal date ``t``; the backtest engine shifts signals by one day so the
execution happens at the next session's open. No same-day OHLCV is visible
to the signal at ``t``.

Custom strategies subclass :class:`Strategy` and are loaded by
``src.strategy_loader``; their ``name`` is the registry key, auto-prefixed
with ``custom/`` on collision with a built-in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .indicators import atr, ema, macd, rsi, sma


class Strategy(ABC):
    """Contract every strategy (built-in or custom) must implement.

    - ``name``: unique non-empty string used as the registry key; a custom
      strategy whose name collides with a built-in is auto-prefixed
      ``custom/{name}``.
    - ``params``: dict of tunables surfaced in the dashboard sidebar.
    - ``positions(prices)``: Series of -1/0/+1 per row, index equal to
      ``prices.index``. The value at row ``t`` must depend only on rows
      ``<= t`` (no look-ahead).
    """

    name: str
    params: dict = {}

    @abstractmethod
    def positions(self, prices: pd.DataFrame) -> pd.Series:
        """Position per date in {-1, 0, +1}. Index == prices.index.

        Value at ``t`` must depend only on rows ``<= t`` (no look-ahead).
        """
        raise NotImplementedError


STRATEGY_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Register a Strategy subclass under its ``name``.

    A duplicate name does not overwrite the existing entry; the strategy
    loader withdraws decorator registrations for custom modules and
    re-registers them through ``register_custom`` so collision prefixing
    applies.
    """
    cls._decorator_registered = True  # type: ignore[attr-defined]
    name = getattr(cls, "name", None)
    if isinstance(name, str) and name and name not in STRATEGY_REGISTRY:
        STRATEGY_REGISTRY[name] = cls
    return cls


def _per_ticker(prices: pd.DataFrame, func) -> pd.Series:
    """Apply ``func`` to each ticker's rows and concatenate into one Series.

    Custom positions must be index-equal to ``prices.index``; grouping by
    ticker is required so state machines and ``shift`` calls never leak
    across ticker boundaries. A frame without a ``ticker`` column is treated
    as a single-ticker frame.
    """
    if "ticker" not in prices.columns:
        return func(prices)
    pieces = [func(group.sort_values("date")) for _, group in prices.groupby("ticker", sort=False)]
    return pd.concat(pieces).reindex(prices.index).rename("position")


def _cross_positions(prices: pd.DataFrame, fast: int, slow: int, allow_short: bool, kind: str) -> pd.Series:
    """Long (or short) when the fast average is above the slow average."""
    if fast <= 0 or slow <= 0 or fast >= slow:
        raise ValueError("Strategy requires 0 < fast < slow.")
    name = f"{kind}_crossover_{fast}_{slow}" + ("_short" if allow_short else "")
    pieces: list[pd.Series] = []
    for ticker, group in prices.groupby("ticker", sort=False):
        grp = group.sort_values("date").set_index("date")
        fast_avg = sma(grp["close"], fast) if kind == "sma" else ema(grp["close"], fast)
        slow_avg = sma(grp["close"], slow) if kind == "sma" else ema(grp["close"], slow)
        long = (fast_avg > slow_avg).fillna(False).to_numpy()
        if allow_short:
            position = np.where(long, 1.0, -1.0)
        else:
            position = np.where(long, 1.0, 0.0)
        pieces.append(pd.Series(position, index=pd.MultiIndex.from_product([[ticker], grp.index]), name=name))
    return pd.concat(pieces).sort_index()


@register
class SmaCrossover(Strategy):
    """Long when ``sma(fast) > sma(slow)``, flat otherwise (short optional)."""

    name = "sma_crossover"
    params = {"fast": 20, "slow": 50, "allow_short": False}

    def __init__(self, fast: int = 20, slow: int = 50, allow_short: bool = False) -> None:
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("SmaCrossover requires 0 < fast < slow.")
        self.fast = int(fast)
        self.slow = int(slow)
        self.allow_short = bool(allow_short)
        self.params = {"fast": self.fast, "slow": self.slow, "allow_short": self.allow_short}
        self.name = f"sma_crossover_{self.fast}_{self.slow}" + ("_short" if self.allow_short else "")

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        """Return the daily long/flat (or long/short) target per ticker."""
        return _cross_positions(prices, self.fast, self.slow, self.allow_short, "sma")


@register
class EmaCrossover(Strategy):
    """Long when ``ema(fast) > ema(slow)``, flat otherwise (short optional)."""

    name = "ema_crossover"
    params = {"fast": 12, "slow": 26, "allow_short": False}

    def __init__(self, fast: int = 12, slow: int = 26, allow_short: bool = False) -> None:
        if fast <= 0 or slow <= 0 or fast >= slow:
            raise ValueError("EmaCrossover requires 0 < fast < slow.")
        self.fast = int(fast)
        self.slow = int(slow)
        self.allow_short = bool(allow_short)
        self.params = {"fast": self.fast, "slow": self.slow, "allow_short": self.allow_short}
        self.name = f"ema_crossover_{self.fast}_{self.slow}" + ("_short" if self.allow_short else "")

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        """Return the daily long/flat (or long/short) target per ticker."""
        return _cross_positions(prices, self.fast, self.slow, self.allow_short, "ema")


@register
class RsiMeanReversion(Strategy):
    """Mean reversion on Wilder RSI: long oversold, exit at the midpoint.

    Long when ``rsi < oversold``; flat when ``rsi >= exit_mid``. With
    ``allow_short``, short when ``rsi > overbought`` and cover when
    ``rsi <= exit_mid``. A state machine holds the position until the exit
    threshold is crossed, so the position reflects "held long until the
    midpoint" rather than "long only on the trigger bar". Borrow costs for
    shorts are not modelled by the engine, so enabling ``allow_short``
    skews net results optimistically.
    """

    name = "rsi_mean_reversion"
    params = {"rsi_window": 14, "oversold": 30, "overbought": 70, "exit_mid": 50, "allow_short": False}

    def __init__(
        self,
        rsi_window: int = 14,
        oversold: int = 30,
        overbought: int = 70,
        exit_mid: int = 50,
        allow_short: bool = False,
    ) -> None:
        if rsi_window <= 0 or not 0 < oversold < exit_mid < overbought < 100:
            raise ValueError("RsiMeanReversion requires 0 < oversold < exit_mid < overbought < 100.")
        self.rsi_window = int(rsi_window)
        self.oversold = int(oversold)
        self.overbought = int(overbought)
        self.exit_mid = int(exit_mid)
        self.allow_short = bool(allow_short)
        self.params = {
            "rsi_window": self.rsi_window,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "exit_mid": self.exit_mid,
            "allow_short": self.allow_short,
        }

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        """Return the daily long/flat (or long/short) target per ticker."""
        return _per_ticker(prices, self._positions_for_ticker)

    def _positions_for_ticker(self, group: pd.DataFrame) -> pd.Series:
        values = rsi(group["close"], self.rsi_window).to_numpy()
        pos = np.zeros(len(group), dtype=int)
        state = 0
        for t in range(len(group)):
            value = values[t]
            if np.isnan(value):
                continue  # warmup rows stay flat
            if state == 0:
                if value < self.oversold:
                    state = 1
                elif self.allow_short and value > self.overbought:
                    state = -1
            elif state == 1:
                if value >= self.exit_mid:
                    state = 0
            elif state == -1:
                if value <= self.exit_mid:
                    state = 0
            pos[t] = state
        return pd.Series(pos, index=group.index, name="position")


@register
class MacdMomentum(Strategy):
    """Momentum on MACD: long while MACD sits above its signal line.

    Long when ``macd > signal``; flat (or short, with ``allow_short``) when
    ``macd < signal``. This is a pure-state rule — no crossover detection is
    required, the held-until-opposite form avoids whipsaws. Borrow costs for
    shorts are not modelled by the engine, so enabling ``allow_short``
    skews net results optimistically.
    """

    name = "macd_momentum"
    params = {"fast": 12, "slow": 26, "signal": 9, "allow_short": False}

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, allow_short: bool = False) -> None:
        if fast <= 0 or slow <= 0 or signal <= 0 or fast >= slow:
            raise ValueError("MacdMomentum requires 0 < fast < slow and signal > 0.")
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)
        self.allow_short = bool(allow_short)
        self.params = {
            "fast": self.fast, "slow": self.slow, "signal": self.signal, "allow_short": self.allow_short,
        }

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        """Return the daily long/flat (or long/short) target per ticker."""
        return _per_ticker(prices, self._positions_for_ticker)

    def _positions_for_ticker(self, group: pd.DataFrame) -> pd.Series:
        lines = macd(group["close"], self.fast, self.slow, self.signal)
        above = (lines["macd"] > lines["macd_signal"]).fillna(False).to_numpy()
        if self.allow_short:
            below = (lines["macd"] < lines["macd_signal"]).fillna(False).to_numpy()
            pos = np.where(above, 1, np.where(below, -1, 0))
        else:
            pos = np.where(above, 1, 0)
        return pd.Series(pos.astype(int), index=group.index, name="position")


@register
class AtrBreakout(Strategy):
    """Volatility break-out: close above a prior-bar ATR channel -> long.

    An SMA midline plus/minus ``k`` ATR forms the channel. The band at bar
    ``t`` is built from data up to bar ``t-1`` (``mid`` and ``atr`` are
    shifted by one bar) and compared against ``close[t]`` — the textbook
    "yesterday's band, today's close" rule that keeps the entry condition
    free of the Donchian look-ahead trap. ``exit_rule="midline"`` exits when
    ``close <= mid``; ``"lower_band"`` exits when ``close <= down`` and
    holds longer. With ``allow_short``, a close below the lower band is a
    short. Borrow costs for shorts are not modelled by the engine, so
    enabling ``allow_short`` skews net results optimistically.
    """

    name = "atr_breakout"
    params = {
        "sma_window": 20,
        "atr_window": 14,
        "k": 2.0,
        "exit_rule": "midline",
        "allow_short": False,
        "include_current_bar": False,
    }

    def __init__(
        self,
        sma_window: int = 20,
        atr_window: int = 14,
        k: float = 2.0,
        exit_rule: str = "midline",
        allow_short: bool = False,
        include_current_bar: bool = False,
    ) -> None:
        if sma_window <= 0 or atr_window <= 0 or k < 0:
            raise ValueError("AtrBreakout requires positive windows and k >= 0.")
        if exit_rule not in ("midline", "lower_band"):
            raise ValueError("AtrBreakout exit_rule must be 'midline' or 'lower_band'.")
        self.sma_window = int(sma_window)
        self.atr_window = int(atr_window)
        self.k = float(k)
        self.exit_rule = str(exit_rule)
        self.allow_short = bool(allow_short)
        self.include_current_bar = bool(include_current_bar)
        self.params = {
            "sma_window": self.sma_window,
            "atr_window": self.atr_window,
            "k": self.k,
            "exit_rule": self.exit_rule,
            "allow_short": self.allow_short,
            "include_current_bar": self.include_current_bar,
        }

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        """Return the daily long/flat (or long/short) target per ticker."""
        return _per_ticker(prices, self._positions_for_ticker)

    def _positions_for_ticker(self, group: pd.DataFrame) -> pd.Series:
        close = group["close"]
        mid = sma(close, self.sma_window)
        spread = atr(group[["high", "low", "close"]], self.atr_window)
        if not self.include_current_bar:
            mid = mid.shift(1)
            spread = spread.shift(1)
        up = mid + self.k * spread
        down = mid - self.k * spread
        closes = close.to_numpy()
        ups = up.to_numpy()
        downs = down.to_numpy()
        mids = mid.to_numpy()
        pos = np.zeros(len(group), dtype=int)
        state = 0
        for t in range(len(group)):
            if np.isnan(ups[t]):
                continue  # warmup rows stay flat
            if state == 0:
                if closes[t] > ups[t]:
                    state = 1
                elif self.allow_short and closes[t] < downs[t]:
                    state = -1
            elif state == 1:
                level = mids[t] if self.exit_rule == "midline" else downs[t]
                if closes[t] <= level:
                    state = 0
            elif state == -1:
                level = mids[t] if self.exit_rule == "midline" else ups[t]
                if closes[t] >= level:
                    state = 0
            pos[t] = state
        return pd.Series(pos, index=group.index, name="position")


@register
class BuyAndHold(Strategy):
    """Always-long baseline used as the benchmark, never the headline."""

    name = "buy_and_hold"

    def __init__(self) -> None:
        pass

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        """Return an all-long target for every (ticker, date)."""
        idx = pd.MultiIndex.from_frame(prices[["ticker", "date"]].drop_duplicates())
        return pd.Series(1.0, index=idx, name=self.name)


def build_strategy(spec: dict) -> Strategy:
    """Instantiate a strategy from a config spec like ``{"type": ..., **params}``."""
    if not isinstance(spec, dict) or not isinstance(spec.get("type"), str):
        raise ValueError("Strategy spec must be a dict with a type")
    kind = spec["type"]
    if kind not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy type: {kind}. Available: {sorted(STRATEGY_REGISTRY)}")
    params = {key: value for key, value in spec.items() if key != "type"}
    return STRATEGY_REGISTRY[kind](**params)
