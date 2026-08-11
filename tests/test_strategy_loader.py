"""Heavy tests for the custom strategy loader — these run user code."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.strategies import STRATEGY_REGISTRY, Strategy, register
from src.strategy_loader import (
    StrategyValidationError,
    load_and_validate,
    load_strategy_file,
    register_custom,
    register_saved_strategies,
    safe_strategy_filename,
    validate_strategy,
)

VALID_BODY = """fast = sma(prices["close"], 10)
slow = sma(prices["close"], 20)
pos = (fast > slow).astype(int)
return pos.fillna(0).astype(int).rename("position")"""

TEMPLATE = '''from __future__ import annotations
import pandas as pd
from src.strategies import Strategy, register
from src.indicators import sma

@register
class {cls_name}(Strategy):
    name = "{name}"

    def positions(self, prices: pd.DataFrame) -> pd.Series:
{body}
'''


def write_strategy(
    tmp_path: Path,
    name: str = "valid_strategy",
    body: str = VALID_BODY,
    cls_name: str = "ValidStrategy",
    filename: str | None = None,
) -> Path:
    path = tmp_path / (filename or f"{name}.py")
    path.write_text(TEMPLATE.format(cls_name=cls_name, name=name, body=_indent(body)), encoding="utf-8")
    return path


def _indent(body: str) -> str:
    return "\n".join(f"        {line}" if line else line for line in body.splitlines())


@pytest.fixture(autouse=True)
def clean_registry():
    """Restore the global registry after each test."""
    before = set(STRATEGY_REGISTRY)
    yield
    for key in set(STRATEGY_REGISTRY) - before:
        STRATEGY_REGISTRY.pop(key)


def test_valid_file_loads_validates_and_registers(tmp_path, sample_prices):
    path = write_strategy(tmp_path)
    cls = load_and_validate(path, sample_prices)
    assert getattr(cls, "validation_gates") == [
        "interface", "return_shape", "discrete_positions", "no_look_ahead",
        "no_forbidden_imports", "graceful_degradation",
    ]
    key = register_custom(cls, path)
    assert key == "valid_strategy"  # no prefix without a collision
    assert STRATEGY_REGISTRY[key] is cls
    positions = cls().positions(sample_prices)
    assert set(positions.unique()) <= {-1, 0, 1}


def test_file_without_strategy_subclass_raises(tmp_path):
    path = tmp_path / "no_subclass.py"
    path.write_text("import pandas as pd\nx = 1\n", encoding="utf-8")
    with pytest.raises(StrategyValidationError, match="exactly one Strategy subclass"):
        load_strategy_file(path)


def test_wrong_length_raises(tmp_path, sample_prices):
    path = tmp_path / "short.py"
    path.write_text(
        TEMPLATE.format(
            cls_name="ShortStrategy", name="short",
            body=_indent('return pd.Series([0, 1, 0], index=prices.index[:3])'),
        ),
        encoding="utf-8",
    )
    with pytest.raises(StrategyValidationError, match="length mismatch"):
        validate_strategy(load_strategy_file(path), sample_prices)


def test_wrong_index_raises(tmp_path, sample_prices):
    path = tmp_path / "bad_index.py"
    path.write_text(
        TEMPLATE.format(
            cls_name="BadIndex", name="bad_index",
            body=_indent('return pd.Series([0] * len(prices), index=prices["date"])'),
        ),
        encoding="utf-8",
    )
    with pytest.raises(StrategyValidationError, match="index mismatch"):
        validate_strategy(load_strategy_file(path), sample_prices)


@pytest.mark.parametrize("fractional", ["0.5", "2.0", "float('nan')"])
def test_fractional_positions_rejected(tmp_path, sample_prices, fractional):
    path = tmp_path / f"fractional_{fractional.replace('.', '').replace('(', '').replace(')', '').replace('nan', 'nan')}.py"
    path.write_text(
        TEMPLATE.format(
            cls_name="Fractional", name="fractional",
            body=_indent(f"return pd.Series([{fractional}] * len(prices))"),
        ),
        encoding="utf-8",
    )
    with pytest.raises(StrategyValidationError, match="discrete"):
        validate_strategy(load_strategy_file(path), sample_prices)


def test_look_ahead_shift_rejected(tmp_path, sample_prices):
    path = tmp_path / "peek.py"
    body = (
        'pos = prices["close"].shift(-1).gt(prices["close"]).astype(int).fillna(0)\n'
        'return pos.rename("position")'
    )
    path.write_text(
        TEMPLATE.format(cls_name="Peek", name="peek", body=_indent(body)),
        encoding="utf-8",
    )
    with pytest.raises(StrategyValidationError, match="look-ahead"):
        validate_strategy(load_strategy_file(path), sample_prices)


def test_raises_on_tiny_frame_rejected(tmp_path, sample_prices):
    path = tmp_path / "needs_history.py"
    body = (
        'if len(prices) < 10:\n'
        '    raise ValueError("need more history")\n'
        'return pd.Series([0] * len(prices))'
    )
    path.write_text(
        TEMPLATE.format(cls_name="NeedsHistory", name="needs_history", body=_indent(body)),
        encoding="utf-8",
    )
    with pytest.raises(StrategyValidationError, match="tiny"):
        validate_strategy(load_strategy_file(path), sample_prices)


@pytest.mark.parametrize("bad_import", ["os", "subprocess", "socket", "requests", "urllib", "pathlib", "sys", "pickle"])
def test_forbidden_import_rejected_before_instantiation(tmp_path, sample_prices, bad_import):
    path = tmp_path / f"bad_{bad_import}.py"
    path.write_text(
        f"import {bad_import}\n"
        + TEMPLATE.format(cls_name="BadImport", name="bad_import", body=_indent(VALID_BODY)),
        encoding="utf-8",
    )
    with pytest.raises(StrategyValidationError) as excinfo:
        load_strategy_file(path)
    assert bad_import in str(excinfo.value)


def test_wildcard_import_rejected(tmp_path):
    path = tmp_path / "wildcard.py"
    path.write_text(
        "from src.indicators import *\n"
        + TEMPLATE.format(cls_name="Wild", name="wild", body=_indent(VALID_BODY)),
        encoding="utf-8",
    )
    with pytest.raises(StrategyValidationError, match="wildcard"):
        load_strategy_file(path)


def test_relative_import_rejected(tmp_path):
    path = tmp_path / "relative.py"
    path.write_text(
        "from . import x\n"
        + TEMPLATE.format(cls_name="Rel", name="rel", body=_indent(VALID_BODY)),
        encoding="utf-8",
    )
    with pytest.raises(StrategyValidationError, match="relative imports"):
        load_strategy_file(path)


def test_name_collision_with_builtin_is_prefixed(tmp_path, sample_prices):
    path = write_strategy(tmp_path, name="sma_crossover", cls_name="Collision")
    cls = load_and_validate(path, sample_prices)
    key = register_custom(cls, path)
    assert key == "custom/sma_crossover"
    assert cls.name == "sma_crossover"  # the class attribute is preserved
    assert STRATEGY_REGISTRY["sma_crossover"] is not cls  # built-in untouched


def test_two_files_same_name_auto_prefix(tmp_path, sample_prices):
    write_strategy(tmp_path, name="dual", cls_name="DualOne", filename="dual_one.py")
    write_strategy(tmp_path, name="dual", cls_name="DualTwo", filename="dual_two.py")
    registered = register_saved_strategies(tmp_path, sample_prices)
    assert set(registered) == {"dual", "custom/dual"}
    assert registered["dual"] is not registered["custom/dual"]


def test_saved_strategy_auto_registers_on_reload(tmp_path, sample_prices):
    saved_dir = tmp_path / "strategies" / "custom"
    saved_dir.mkdir(parents=True)
    write_strategy(saved_dir, name="persisted_one")
    registered = register_saved_strategies(saved_dir, sample_prices)
    assert "persisted_one" in registered
    cls = registered["persisted_one"]
    assert cls.strategy_source["file_name"] == "persisted_one.py"
    assert cls.strategy_source["registry_key"] == "persisted_one"


def test_missing_directory_returns_empty(tmp_path):
    assert register_saved_strategies(tmp_path / "nope", None) == {}


def test_sha256_recorded_matches_upload_bytes(tmp_path, sample_prices):
    path = write_strategy(tmp_path)
    file_bytes = path.read_bytes()
    cls = load_and_validate(path, sample_prices)
    key = register_custom(cls, path)
    assert cls.strategy_source["sha256"] == hashlib.sha256(file_bytes).hexdigest()
    assert cls.strategy_source["registry_key"] == key
    assert isinstance(cls.strategy_source["uploaded_at"], str)


def test_safe_strategy_filename_sanitizes():
    assert safe_strategy_filename("my strategy.py") == "my_strategy.py"
    assert safe_strategy_filename("../../evil.py") == "evil.py"
    assert safe_strategy_filename("momentum") == "momentum.py"
    assert safe_strategy_filename("a/b/c.txt") == "c.txt.py"


def test_engine_runs_a_loaded_custom_strategy(tmp_path, sample_prices):
    from src.backtest import BacktestConfig, simulate

    path = write_strategy(tmp_path)
    cls = load_and_validate(path, sample_prices)
    frame = sample_prices.assign(ticker="AAA")
    positions = cls().positions(frame)
    cfg = BacktestConfig(
        entry_rule="next_open", exit_rule="same_day_close", initial_capital=1000.0,
        max_position=1.0, transaction_cost_bps=0.0, slippage_bps=0.0,
    )
    report = simulate(frame, positions, cfg, strategy_name="valid_strategy")
    assert report.strategy_name == "valid_strategy"
    assert report.equity_curve["wealth"].notna().all()


def test_engine_treats_nan_positions_as_flat(sample_prices):
    from src.backtest import BacktestConfig, simulate

    frame = sample_prices.assign(ticker="AAA")
    positions = pd.Series(float("nan"), index=frame.index, name="nan_strategy")
    cfg = BacktestConfig(
        entry_rule="next_open", exit_rule="same_day_close", initial_capital=1000.0,
        max_position=1.0, transaction_cost_bps=25.0, slippage_bps=10.0,
    )
    report = simulate(frame, positions, cfg)
    assert report.metrics[25.0]["trade_count"] == 0
    assert report.metrics[25.0]["final_wealth"] == pytest.approx(1000.0)


def test_register_decorator_does_not_pollute_registry_via_loader(tmp_path, sample_prices):
    """@register in a custom file is withdrawn by the loader until register_custom."""
    path = write_strategy(tmp_path)
    before = set(STRATEGY_REGISTRY)
    load_strategy_file(path)
    assert set(STRATEGY_REGISTRY) == before
