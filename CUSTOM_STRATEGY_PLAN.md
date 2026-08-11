# Custom Strategy Import — Plan

Supplements `IMPLEMENTATION_PLAN.md`. Read that first for the project layout, the `Strategy` interface baseline, and the Yahoo → CSV → cache → backtest → Streamlit architecture this feature extends.

## Goal

Let a dashboard user author a trading strategy as a `.py` file subclassing the `Strategy` interface, upload it via `st.file_uploader`, validate it against the contract (no look-ahead, discrete positions, correct index/length, no forbidden imports), optionally persist it to `strategies/custom/`, and select it from the strategy dropdown alongside built-ins. No DSL, no in-app code editor, no arbitrary indicator authoring — custom strategies compose built-in indicators from `src/indicators.py` only.

## Design decisions (locked)

| Decision | Choice |
|---|---|
| Definition format | Python file upload only (`.py` subclassing `Strategy`) |
| Authoring surface | `st.file_uploader` — no in-dashboard code editor |
| Persistence | Uploaded strategies may be saved to `strategies/custom/` and auto-register on next app start |
| Position sizing | Discrete `{-1, 0, +1}` only — no fractional sizing |
| Indicators | Custom strategies may call built-ins from `src/indicators.py` but not define their own |
| Name collisions | Prefix the custom with `custom/{name}` so built-ins and customs coexist without forcing renames |
| `strategies/custom/` in git | **Not** gitignored by default — users may version their strategies. Add a `.gitkeep` + commented hint in `.gitignore` for users who want privacy. |

## Files added / changed

```text
src/
  strategies.py             # Strategy ABC, built-ins, STRATEGY_REGISTRY
  strategy_loader.py        # NEW: load + validate + register uploaded .py files
  pipeline.py               # extend to accept custom strategies + custom-origin provenance
strategies/
  custom/                    # NEW: persisted user strategies
    README.md               # the Strategy contract + authoring guide + examples (checked in)
    .gitkeep               # keep dir in git even when empty
dashboard/
  app.py                     # file_uploader + validation feedback + save button + dropdown integration
tests/
  test_strategy_loader.py    # NEW (heavy — runs user code)
  test_strategies.py         # extend with custom-strategy edge cases
  test_dashboard.py          # extend with upload flow AppTest
```

## The `Strategy` contract (lives in `src/strategies.py`)

```python
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    name: str
    params: dict = {}            # tunables surfaced in the dashboard sidebar

    @abstractmethod
    def positions(self, prices: pd.DataFrame) -> pd.Series:
        """Position per date in {-1, 0, +1}. Index == prices.index.
        Value at t must depend only on rows <= t (no look-ahead)."""

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {}

def register(cls: type[Strategy]) -> type[Strategy]:
    STRATEGY_REGISTRY[cls.name] = cls
    return cls

# built-ins registered via @register at import
```

Built-ins register at import; `strategy_loader` registers custom ones after validation. The dropdown shows built-ins and customs separately (`"SMA crossover (20,50)"` vs `"custom: momentum_reversal"`). Collisions are resolved by prefixing the custom with `custom/{name}`, so a built-in `SMA crossover` and a user file also named `sma_crossover` both appear without forcing a rename.

## Authoring guide (strategies/custom/README.md content)

The markdown handed to your agent should include this section verbatim — it is what a user reads to author their first strategy.

### How to author a custom strategy

1. Subclass `src.strategies.Strategy`.
2. Set a unique `name` (string). Collisions with a built-in are auto-prefixed `custom/{name}`.
3. Implement `positions(prices) -> pd.Series` returning one of `-1`, `0`, `+1` per row.
4. Use only indicators from `src/indicators.py` — no inline math on `prices` that skips the indicators module. This keeps the contract auditable and lets the validator trust the no-look-ahead property of indicators.
5. Imports allowed: `pandas`, `numpy`, and `src.indicators` only. Anything else is rejected at load time.
6. The value at row `t` may only read `prices.iloc[:t+1]`. The validator perturbs the last rows and checks earlier positions are unchanged.
7. Degrade gracefully on insufficient history (fewer rows than your longest indicator window) — return all-zeros, do not raise. The backtest treats NaN/0 as flat.

### Copy-paste template

```python
from __future__ import annotations
import pandas as pd
from src.strategies import Strategy, register
from src.indicators import sma, rsi

@register
class MyStrategy(Strategy):
    name = "my_strategy"
    params = {"fast": 20, "slow": 50, "rsi_window": 14}

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        fast = sma(prices["close"], self.params["fast"])
        slow = sma(prices["close"], self.params["slow"])
        r = rsi(prices["close"], self.params["rsi_window"])
        long_today = (fast > slow) & (r < 70)
        pos = long_today.astype(int).rename("position")
        # warmup: NaN -> 0
        pos = pos.fillna(0).astype(int)
        return pos
```

### Worked example — SMA crossover with RSI filter

```python
from __future__ import annotations
import pandas as pd
from src.strategies import Strategy, register
from src.indicators import sma, rsi

@register
class SmaRsiFilter(Strategy):
    name = "sma_rsi_filter"
    params = {"fast": 20, "slow": 50, "rsi_window": 14, "rsi_cap": 70}

    def positions(self, prices: pd.DataFrame) -> pd.Series:
        f = sma(prices["close"], self.params["fast"])
        s = sma(prices["close"], self.params["slow"])
        r = rsi(prices["close"], self.params["rsi_window"])
        pos = ((f > s) & (r < self.params["rsi_cap"])).astype(int)
        return pos.fillna(0).astype(int).rename("position")
```

## `src/strategy_loader.py` — the validator

```python
from __future__ import annotations
import ast
import hashlib
import importlib.util
from pathlib import Path
import pandas as pd
from src.strategies import Strategy, STRATEGY_REGISTRY

class StrategyValidationError(Exception):
    """Raised when an uploaded strategy fails one or more validation gates.
    The `failures` attribute carries the list of gate failure messages."""

def load_strategy_file(path: Path) -> type[Strategy]:
    """Import a .py file, find its Strategy subclass, return it.
    Raises StrategyValidationError on any contract violation."""

def validate_strategy(cls: type[Strategy], sample_prices: pd.DataFrame) -> None:
    """Run every gate below; collect all failures and raise once with the list."""

def register_custom(cls: type[Strategy], source_path: Path) -> str:
    """Register into runtime registry with a custom/ prefix on name collision.
    Returns the registry key used."""
```

### Validation gates (all must pass before the strategy enters the registry)

Run every gate, collect failures, raise once with the full list so the user sees the whole picture rather than fixing one error at a time.

1. **Interface** — the file defines exactly one `class` subclassing `Strategy`. The class sets `name` as a non-empty string. If `name` collides with a built-in or an already-registered custom, prefix with `custom/{name}` (the registry key is the prefixed form; the original `name` attribute is preserved on the class).

2. **Return shape** — `positions(sample_prices)` returns a `pd.Series` with `len == len(sample_prices)` and `Index.equals(sample_prices.index)`. Catch any exception from the call and report it as a gate failure.

3. **Discrete positions** — `result.isin({-1, 0, 1}).all()`. Floats like `1.0` are accepted and coerced to `int`; anything else (e.g., `0.5`, `2`, `NaN` surviving into the output) is rejected. After validation the registered positions must be cast to `int` by the backtest engine.

4. **No look-ahead** — make a deep copy of `sample_prices`, perturb the last 5 rows of `close` by a large factor (e.g., `* 10.0`), re-run `positions`, and assert positions for all rows *before* the last 5 are byte-identical (compare via `.astype(int).eq(...).all()`). This catches any `shift(-k)` or future-row read. Re-run with a perturbation of `+ 1000.0` too, in case the strategy compares to an absolute level. (Same invariant the sibling `stock-direction-classifier` project mandates.)

5. **No forbidden imports** — parse the source with `ast`, walk `Import` and `ImportFrom` nodes. Allowed: `pandas` (and alias `pd`), `numpy` (and alias `np`), `src.indicators`, `src.strategies`, `__future__`. Reject `os`, `subprocess`, `socket`, `open`, `requests`, `urllib`, `pathlib`, `shutil`, `sys`, `ctypes`, `pickle`, `builtins`, and any `*` wildcard import. Raise on the first forbidden import before even instantiating the class — defense in depth, not the only safety boundary.

6. **Graceful degradation** — call `positions()` on a tiny frame (e.g., 3 rows, smaller than the longest built-in indicator window the strategy is likely to use). Must not throw; may return all-zeros. Strategies that raise on short history are unusable in the first warmup days of a long backtest, so reject them.

Failures are collected into `StrategyValidationError.failures: list[str]` and surfaced to the user in the dashboard as a checklist, not a single cryptic traceback.

## Dashboard integration (dashboard/app.py)

Sidebar additions, rendered above the existing strategy dropdown:

```text
┌─ Custom strategy ──────────────────┐
│ [+ Upload .py]            <- st.file_uploader(accept=".py")
│                           │
│ Validation:               │
│  ✓ interface               │
│  ✓ return shape            │
│  ✓ discrete positions      │
│  ✓ no look-ahead           │
│  ✓ no forbidden imports    │
│  ✓ graceful degradation   │
│  ✗ name collision with built-in
│     -> registered as custom:foo
│                           │
│ [Save to strategies/custom/]   <- only shown after a valid upload
│                           │
│ Active strategy: [SMA ▾]   │  built-ins + customs grouped
│ Params: fast 20 slow 50   │
│ [Run backtest]            │
└───────────────────────────┘
```

Behavior:

- **Upload** → read bytes, write to a temp file, call `load_strategy_file` → `validate_strategy` against a sample prices frame built from the cached data. Show the gate checklist. Invalid: nothing is registered, user sees the failure list, the dropdown is unchanged. Valid: `register_custom` puts it in an in-memory `session_registry` (a dict separate from `STRATEGY_REGISTRY` so session uploads don't pollute the persistent registry until saved), auto-selects it in the dropdown, and reveals the **Save to `strategies/custom/`** button.
- **Save** → writes the file to `strategies/custom/{safe_name}.py` where `safe_name` is derived from the uploaded filename (sanitized, no path traversal) — not the class `name`, so two classes with the same `name` can coexist on disk with different filenames. On the next app start, `pipeline` scans `strategies/custom/*.py`, validates each, and registers survivors into the persistent `STRATEGY_REGISTRY` (collisions auto-prefixed).
- **Provenance** — every custom run records in `BacktestReport.config_snapshot`: the file name, sha256 of the uploaded bytes, upload timestamp, and the validation gate results. This makes a run reproducible and auditable.
- **Dropdown grouping** — built-ins first, then a `custom/` heading listing session-uploaded and saved-to-disk strategies. A saved custom that is also uploaded again is de-duplicated by sha256.
- **Refresh button unchanged** — the existing **Refresh cache from Yahoo** button only touches the CSV layer; strategy upload never calls Yahoo.

## Tests (test_strategy_loader.py) — heavy, since this runs user code

Write each test with a real on-disk `.py` temp file (via `tmp_path`) so the AST gate and import path exercise the real loader.

- Valid file with a proper `Strategy` subclass → loads, validates, and registers. The registry key equals the class `name` (no prefix when there is no collision).
- File with no `Strategy` subclass → raises `StrategyValidationError` with a message naming the missing contract.
- `positions()` returns a `pd.Series` with wrong length or wrong index → raises.
- `positions()` returns floats out of `{-1, 0, 1}` (e.g., `0.5`, `2.0`) → discrete-positions gate raises.
- A strategy that reads `prices["close"].shift(-1)` → look-ahead gate raises (perturbing last rows changes earlier positions).
- A strategy whose indicator warmup raises on a tiny frame → graceful-degradation gate raises.
- A strategy importing `os` → AST gate raises before the class is instantiated; assert the forbidden module name appears in the failure message.
- A strategy importing `requests` or `urllib` → same AST gate.
- Name collision with a built-in (`name = "sma_crossover"`) → not rejected; registered as `custom/sma_crossover`; the class `name` attribute is unchanged.
- Two files in `strategies/custom/` defining the same `name` → second one logs a warning and is skipped or auto-prefixed (not crashed); assert the app still starts.
- Saving an uploaded valid file → it appears in `strategies/custom/` and is auto-registered on a simulated reload without re-uploading.
- sha256 of the uploaded bytes recorded in the run report matches `hashlib.sha256(file_bytes).hexdigest()`.

## Dashboard test (extend test_dashboard.py with AppTest)

- Dashboard renders without `strategies/custom/` existing (empty dir is fine).
- `AppTest` simulates a file upload of a valid strategy → the dropdown contains `custom/...`.
- `AppTest` simulates a file upload of an invalid strategy (look-ahead) → the validation checklist shows the failure and the dropdown is unchanged.
- Disclaimer banner still present on the custom-strategy view.

## `strategies/custom/.gitkeep` and `.gitignore` note

- `.gitkeep` so the directory is tracked when empty.
- Add a **commented** line to `.gitignore`:
  ```
  # Uncomment to keep custom strategies out of git:
  # strategies/custom/*.py
  ```
  This lets users opt into privacy without forcing it by default.

## Acceptance gates for this feature

1. A freshly authored `.py` file implementing the contract uploads, validates, runs, and appears in the strategy dropdown within one click.
2. A strategy that peeks at future rows is rejected before it can run.
3. A strategy importing `os`/`subprocess`/`requests`/`urllib` is rejected before it can run.
4. A strategy that raises on short history is rejected by the graceful-degradation gate.
5. Saved-to-`custom/` strategies auto-register on the next app start without user action.
6. Every custom run records the file name, sha256, and upload timestamp in `BacktestReport.config_snapshot` for reproducibility.
7. Name collisions between a custom and a built-in do not crash; the custom is auto-prefixed `custom/{name}`.
8. `pytest -q` is green; `test_strategy_loader.py` covers all six gates and the dashboard `AppTest` upload flow.