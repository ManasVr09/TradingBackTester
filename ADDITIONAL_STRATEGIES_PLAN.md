# Additional Built-In Strategies — Plan

Supplements `IMPLEMENTATION_PLAN.md` and `CUSTOM_STRATEGY_PLAN.md`. Read those first for the `Strategy` interface, the `STRATEGY_REGISTRY` registration pattern, the signal-timing contract, and the indicators available in `src/indicators.py`.

## Goal

Add three more built-in strategies to `src/strategies.py` alongside the existing `SmaCrossover`, `EmaCrossover`, and `BuyAndHold`. The new strategies exercise the remaining built-in indicators (RSI, MACD, ATR) so every indicator module has at least one native consumer, and they span the three main strategy families a beginner quant needs: mean reversion, momentum, and volatility break-out. Each ships with `params` declared for the dashboard sidebar, obeys the no-look-ahead contract, and is covered by a dedicated test file.

## Strategy family map after this addition

| Strategy | Family | Indicators used | Position universe |
|---|---|---|---|
| `SmaCrossover` | trend-following | SMA | {-1, 0, +1} |
| `EmaCrossover` | trend-following | EMA | {-1, 0, +1} |
| `RsiMeanReversion` | mean reversion | RSI | {-1, 0, +1} |
| `MacdMomentum` | momentum | MACD | {-1, 0, +1} |
| `AtrBreakout` | volatility break-out | SMA, ATR | {-1, 0, +1} |
| `BuyAndHold` | baseline | none | {+1} |

Nobody has all-zero positions by default; nobody uses fractional sizing (the locked decision from the custom-strategy plan carries over — discrete `{-1, 0, +1}` only).

## Shared conventions (apply to every new strategy)

- Live in `src/strategies.py`. Register with `@register` so they appear in `STRATEGY_REGISTRY` at import and in the dashboard dropdown automatically.
- Declare `params: dict` with the tunable knobs and sensible defaults; the dashboard sidebar renders these (number inputs keyed by name) for any selected strategy.
- `positions(prices)` returns a `pd.Series` of `int`, index-equal to `prices.index`, named `"position"`. Cast via `.astype(int)` at the end so float comparisons never leak `0.0`/`1.0` into the output.
- Warmup rows (where an indicator is NaN because there are not enough observations) map to position `0` (flat). Use `fillna(0)` after computing the boolean long/short mask. Every strategy must degrade gracefully on short history — the shared `graceful-degradation` rule from the custom-strategy plan applies to built-ins too.
- Signal timing follows the existing `IMPLEMENTATION_PLAN.md` contract: signals formed from `close(t)` enter at `open(t+1)`, exit at `close(t+1)`. The position series returned by `positions()` describes the target held over the `t → t+1` interval; the backtest engine turns it into trades on the executable clock. No strategy reads `prices.iloc[t+1:]`.
- `allow_short: false` is the default for all three (long/flat only). Expose `allow_short` as a `params` boolean; when `true`, the short-rule output flips to `-1` instead of `0`. Document that shorts borrow cost is not modelled in the v1 engine, so enabling shorts skews net results optimistically — the README should repeat this caveat.

## Strategy 1 — `RsiMeanReversion`

### Thesis

RSI (Wilder, 14-day default) measures recent up-move vs down-move magnitude. When RSI falls below a low threshold the asset is "oversold" and is statistically likely to revert; when it rises above a high threshold it is "overbought" and likely to pull back. The strategy goes long on oversold, exits on a neutral midpoint, and optionally shorts on overbought.

### Parameters

```python
class RsiMeanReversion(Strategy):
    name = "rsi_mean_reversion"
    params = {
        "rsi_window": 14,
        "oversold": 30,
        "overbought": 70,
        "exit_mid": 50,
        "allow_short": False,
    }
```

### Rules

- `rsi = rsi(prices["close"], rsi_window)`.
- Long when `rsi < oversold`.
- Exit (to flat) when `rsi >= exit_mid`.
- If `allow_short`: short when `rsi > overbought`, cover when `rsi <= exit_mid`.
- If not `allow_short`: the overbought region simply holds no position (flat).

### Position logic (pseudo)

```python
rsi = rsi(prices["close"], w)
pos = pd.Series(0, index=prices.index, name="position")
pos[rsi < oversold] = 1
if allow_short:
    pos[rsi > overbought] = -1
# exit_mid handled implicitly: positions only set on threshold crossings,
# and otherwise stay at the previous bar's value via ffill of *changes*.
```

Implementation note: use `pd.Series(...)` seeded at 0, set entries only on threshold-cross bars, then `ffill` the state forward bar-by-bar so the position reflects "held long until exit_mid is reached", not "long only on the bar the condition fired". Alternatively, track a small state machine:

```python
state = 0
for t in range(len(prices)):
    r = rsi.iloc[t]
    if state == 0:
        if r < oversold: state = 1
        elif allow_short and r > overbought: state = -1
    elif state == 1:
        if r >= exit_mid: state = 0
    elif state == -1:
        if r <= exit_mid: state = 0
    pos.iloc[t] = state
```

Either form is acceptable; the test must prove warmup rows are 0, entries happen only when the threshold is crossed, and exits happen only when `exit_mid` is crossed. Choose the explicit state-machine form when in doubt — the loop makes the no-look-ahead property obvious to a reviewer.

### Edge cases / warmup

- `rsi_window` warmup: first `rsi_window` rows of `rsi` are NaN → those rows are position `0`.
- A bar where `rsi` is exactly `oversold` (boundary) counts as long (the `<` is `<` strict; document whether equality is overbought or neutral; recommend `<` for `oversold` and `>` for `overbought`, with `>=`/`<=` for exits, so a flat RSI of exactly 50 with no cross does nothing).
- On very short history (e.g. 5 rows) the strategy returns all-zeros and does not raise.

### Tests (`test_strategies.py::test_rsi_mean_reversion`)

- Construct a hand-built `prices` frame where RSI crosses oversold then returns to the midpoint; assert the position sequence matches the expected `1 ... 1 0 0 ...` pattern exactly.
- `allow_short=False` → position never equals `-1`.
- `allow_short=True` → overbought region produces `-1`.
- Warmup rows (first `rsi_window`) are all `0`.
- Perturb the last 5 rows of `close` and assert positions for all earlier rows are byte-identical (no look-ahead).
- Tiny frame (5 rows) → no exception, all-zeros returned.

## Strategy 2 — `MacdMomentum`

### Thesis

MACD (12, 26, 9 default) is the difference of a fast and slow EMA; its signal line is an EMA of MACD. Crossovers of MACD over its signal line are classic momentum entries. The strategy goes long on a bullish crossover (MACD crosses above signal) and exits on a bearish crossover; optionally shorts on the bearish crossover.

### Parameters

```python
class MacdMomentum(Strategy):
    name = "macd_momentum"
    params = {
        "fast": 12,
        "slow": 26,
        "signal": 9,
        "allow_short": False,
    }
```

### Rules

- `m, sig = macd(prices["close"], fast, slow, signal)` (returns two Series).
- Long when `m > sig` (i.e. MACD above its signal line).
- Exit (to flat) when `m <= sig`.
- If `allow_short`: short when `m < sig`.

### Position logic

```python
m, sig = macd(prices["close"], fast, slow, signal)
long_mask = (m > sig).fillna(False)
pos = long_mask.astype(int)
if allow_short:
    short_mask = (m < sig).fillna(False)
    pos = pos.where(~short_mask, -1)
return pos.astype(int).rename("position")
```

This is a pure-state rule (long while above, flat/short while below), so no `ffill` state machine needed. The MACD indicator itself handles warmup (NaN until enough data for both EMAs); those NaN rows flow through as `fillna(False)` → position `0`.

### Edge cases / warmup

- Warmup is `slow + signal` rows roughly; first warmup rows are position `0`.
- A flat MACD-equals-signal bar counts as not-long (`>` strict); document this so the user knows a sideways market keeps the strategy flat.
- Crossover detection is *not* required — the rule "MACD > signal" already produces entry/exit at the right bars; a separate "cross-on-this-bar" rule is a stricter variant (long only on the bar of the cross, exit on the bar of the opposite cross, flat in between) — spec it as an alternative `mode` param if desired, but the default should be the held-until-opposite form (simpler, fewer whipsaws, easier to test).

### Tests (`test_strategies.py::test_macd_momentum`)

- Hand-built frame where MACD crosses above signal at bar `k` and below at bar `j` → positions are `0 ... 0 1 ... 1 0 ... 0` with the transitions exactly at `k` and `j`.
- `allow_short=False` → no `-1` values appear.
- `allow_short=True` → bars after the bearish crossover (until the next bullish one) are `-1`.
- Warmup rows are `0`.
- Perturb last 5 rows of `close`; assert earlier positions unchanged.
- Tiny frame → no exception, all-zeros.
- Hand-computed MACD on a 40-bar fixture matches a reference value to 6 decimal places (this validates that the strategy uses the indicator output faithfully, not a recalculated EMA).

## Strategy 3 — `AtrBreakout`

### Thesis

A volatility channel: an SMA midline plus/minus `k` ATR defines an upper and lower band. A close above the upper band is a break-out entry; a close below the midline or below the lower band (per `exit_rule`) is the exit. Optionally a break below the lower band is a short. This exercises both SMA and ATR and is the textbook "Donchian / Keltner-style" channel break-out.

### Parameters

```python
class AtrBreakout(Strategy):
    name = "atr_breakout"
    params = {
        "sma_window": 20,
        "atr_window": 14,
        "k": 2.0,                 # band width in ATR units
        "exit_rule": "midline",   # "midline" or "lower_band"
        "allow_short": False,
    }
```

### Rules

- `mid = sma(prices["close"], sma_window)`.
- `atr = atr(prices["high"], prices["low"], prices["close"], atr_window)`.
- Upper band `up = mid + k * atr`; lower band `down = mid - k * atr`.
- Long when `close > up` (close of the current bar exceeds the upper band computed from data up to the previous bar — see look-ahead note).
- Exit per `exit_rule`: `"midline"` exits when `close <= mid`; `"lower_band"` exits when `close <= down`.
- If `allow_short`: short when `close < down`, cover when `close >= mid` (`midline`) or `close >= up` (`upper_band` symmetric).

### Look-ahead subtlety (must be handled explicitly)

The bands at bar `t` must use only data up to and including bar `t`, *not* the close of bar `t` itself for the band computation, because the entry condition compares `close[t]` to `band[t]`. Two valid implementations:

1. **Use prior-bar bands**: `mid_t = sma.iloc[t-1]`, `atr_t = atr.iloc[t-1]`, then long if `close[t] > mid_t + k * atr_t`. This is the cleanest "yesterday's band, today's close" rule and matches the no-look-ahead contract without ambiguity. Mark this as the recommended reference implementation.
2. **Same-bar bands with SMA**: a standard SMA*including* bar `t` is fine because SMA is causal (uses `close[t]` and earlier), but it makes the comparison `close[t] > sma[t] + k*atr[t]` circular (the value being compared is part of the band). It still respects no-look-ahead but is weaker economically. Spec option 1 as the default and offer option 2 via an `include_current_bar: bool = False` param.

The README and the docstring must call this out — it is the textbook Donchian look-ahead trap and the sibling `stock-direction-classifier` project was bitten by exactly this class of bug (using `close(t)` features for a `close(t)→close(t+1)` return).

### Position logic

```python
mid = sma(prices["close"], sma_window).shift(1)   # prior bar's SMA
a = atr(prices["high"], prices["low"], prices["close"], atr_window).shift(1)
up = mid + k * a
down = mid - k * a

c = prices["close"]
state = 0
pos = pd.Series(0, index=prices.index, name="position")
for t in range(len(prices)):
    if pd.isna(up.iloc[t]):
        continue
    if state == 0:
        if c.iloc[t] > up.iloc[t]: state = 1
        elif allow_short and c.iloc[t] < down.iloc[t]: state = -1
    elif state == 1:
        exit_level = mid.iloc[t] if exit_rule == "midline" else down.iloc[t]
        if c.iloc[t] <= exit_level: state = 0
    elif state == -1:
        cover_level = mid.iloc[t] if exit_rule == "midline" else up.iloc[t]
        if c.iloc[t] >= cover_level: state = 0
    pos.iloc[t] = state
return pos.astype(int)
```

The `.shift(1)` on `mid` and `atr` is the no-look-ahead guarantee: the band is built from data up to bar `t-1`, and compared to `close[t]`. The explicit state machine makes the held-until-exit behavior obvious.

### Edge cases / warmup

- Warmup is `max(sma_window, atr_window) + 1` rows (the `+1` from the `.shift(1)`). Those rows are position `0`.
- `k = 0` degenerates to a pure SMA-cross strategy; document that this is valid but redundant with `SmaCrossover`, and the dashboard may warn the user.
- A wide band (`k = 5`) → very few entries; strategy should still be valid and return mostly-zeros.
- `exit_rule = "lower_band"` produces longer holds than `"midline"`; tests should cover both.
- On a commodity-style gap where `close[t]` gaps over both bands simultaneously, the long entry fires (step `state = 0 → 1`); the next bar's exit logic handles the gap.

### Tests (`test_strategies.py::test_atr_breakout`)

- Hand-built prices that break above the upper band at bar `k` and back below the midline at bar `j` → positions `0 ... 0 1 ... 1 0 ... 0` with transitions at `k` and `j`. Verify against a hand-calculated expected series using `exit_rule="midline"`.
- Same fixture with `exit_rule="lower_band"` → the position stays `1` longer, only exiting when `close` falls below the lower band at a later bar `j'`.
- `allow_short=True` with a downside breakout at bar `p` and a cover at the midline at bar `q` → positions `-1` over `[p, q)`.
- Warmup rows are `0`.
- **No look-ahead**: a strategy that accidentally omits the `.shift(1)` (comparing `close[t]` to `sma[t]` where `sma[t]` includes `close[t]`) is detectable by the perturbation test — perturbing `close[t]` changes `sma[t]` and so changes whether `close[t] > sma[t] + k*atr[t]` *for the same bar* in a way that depends on the new close. The test should construct such a case and assert the *prior* bar's position is unchanged (since prior bars only used prior closes). The recommended `.shift(1)` implementation passes; an unshifted implementation fails this test.
- Tiny frame (5 rows, `sma_window=20`) → no exception, all-zeros.
- `k=2` vs `k=4` on the same fixture: `k=4` produces a strict superset-or-equal of zero positions (fewer or equal entries) — assert the entry bars of `k=4` are a subset of those of `k=2`.

## Integration with the dashboard and pipeline

- All three strategies register at import via `@register`, so they appear in the strategy dropdown automatically, grouped under the built-ins heading, ahead of any uploaded customs.
- The `params` dict for the selected strategy is rendered as `st.number_input` / `st.checkbox` widgets in the sidebar, keyed by name, with the declared default. Changing a param re-runs the backtest (debounced by Streamlit's natural rerun semantics; the existing `IMPLEMENTATION_PLAN.md` cache-first approach applies).
- The existing buy-and-hold benchmark is overlaid on every strategy's equity chart regardless of which one is selected — no change to that behavior.
- `BacktestReport.config_snapshot` records the strategy `name`, `params` actually used (post-widget), and the code commit / sha256 of `src/strategies.py` so a saved run is reproducible. This applies to built-ins the same way it applies to uploaded customs per `CUSTOM_STRATEGY_PLAN.md`.

## Tests summary

Extend `tests/test_strategies.py` (do not create a new file — these are built-ins, not a separate module). New test functions:

- `test_rsi_mean_reversion` — long-only path, short path, warmup, no-look-ahead, tiny frame.
- `test_macd_momentum` — crossover entry/exit, short path, warmup, no-look-ahead, tiny frame, MACD value fidelity.
- `test_atr_breakout` — midline vs lower-band exits, short path, warmup, no-look-ahead via `.shift(1)`, tiny frame, `k` sensitivity monotonicity.

Each test uses a small deterministic fixture built inline (no network, no CSV) so it runs in milliseconds and is stable across environments. A shared `tests/conftest.py` fixture already provides `sample_prices` per the base plan; reuse it where the warmup is short and build a longer fixture (`tests/fixtures/` or inline `pd.DataFrame`) where the indicator windows demand it.

## Acceptance gates for this feature

1. All three strategies appear in the dashboard dropdown with no extra configuration and accept the declared `params` from the sidebar.
2. Each strategy's positions reconcile exactly with a hand-calculated expected series on a deterministic fixture.
3. The no-look-ahead perturbation test passes for each strategy (perturbing last 5 rows of `close` leaves all earlier positions byte-identical).
4. Each strategy degrades to all-zeros on a frame shorter than its warmup, without raising.
5. `AtrBreakout` with `exit_rule="lower_band"` holds longer than with `exit_rule="midline"` on the same fixture.
6. `allow_short=True` produces `-1` positions for each strategy's documented short condition.
7. Buy-and-hold remains the only baseline benchmark; the three new strategies are selectable trade strategies, not baselines.
8. `pytest -q` is green with the three new test functions added.