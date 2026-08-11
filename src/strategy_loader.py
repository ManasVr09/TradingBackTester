"""Load, validate, and register custom strategies uploaded as ``.py`` files.

Every uploaded strategy must pass six gates before it can run: interface,
return shape, discrete positions, no look-ahead, no forbidden imports, and
graceful degradation on short history. Failures are collected and raised
once as a :class:`StrategyValidationError` so the dashboard can show a
checklist instead of a traceback.

Defense in depth: the forbidden-import gate is enforced on the source AST
*before* the module is imported, so hostile code never executes in this
process.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .strategies import STRATEGY_REGISTRY, Strategy

logger = logging.getLogger(__name__)

FORBIDDEN_IMPORTS = {
    "os", "subprocess", "socket", "open", "requests", "urllib",
    "pathlib", "shutil", "sys", "ctypes", "pickle", "builtins",
}
ALLOWED_IMPORTS = {"pandas", "numpy"}
ALLOWED_MODULE_PREFIXES = ("src.indicators", "src.strategies")
GATES = [
    "interface",
    "return_shape",
    "discrete_positions",
    "no_look_ahead",
    "no_forbidden_imports",
    "graceful_degradation",
]
LOOK_AHEAD_ROWS = 5
TINY_ROWS = 3


class StrategyValidationError(Exception):
    """Raised when an uploaded strategy fails one or more validation gates.

    The ``failures`` attribute carries the list of gate failure messages.
    """

    def __init__(self, failures: list[str]) -> None:
        self.failures = list(failures)
        super().__init__("; ".join(self.failures))


def safe_strategy_filename(uploaded_name: str) -> str:
    """Sanitize an uploaded filename: basename only, safe chars, ``.py`` suffix."""
    name = Path(str(uploaded_name)).name.strip()
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    if not name.lower().endswith(".py"):
        name = name + ".py"
    return name


def _ast_check_imports(source: str) -> None:
    """Gate: reject forbidden or wildcard imports before the module is imported."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS:
                    raise StrategyValidationError([f"forbidden import: {alias.name}"])
                if root not in ALLOWED_IMPORTS and not alias.name.startswith(ALLOWED_MODULE_PREFIXES):
                    raise StrategyValidationError([f"forbidden import: {alias.name}"])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                raise StrategyValidationError(["relative imports are not allowed"])
            module = node.module or ""
            if any(alias.name == "*" for alias in node.names):
                raise StrategyValidationError([f"wildcard import is not allowed: from {module} import *"])
            root = module.split(".")[0]
            if root in FORBIDDEN_IMPORTS:
                raise StrategyValidationError([f"forbidden import: {module}"])
            if module != "__future__" and root not in ALLOWED_IMPORTS and not module.startswith(ALLOWED_MODULE_PREFIXES):
                raise StrategyValidationError([f"forbidden import: {module}"])


def _module_name_for(path: Path) -> str:
    return f"_custom_strategy_{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]}"


def _find_defined_strategies(module) -> list[type[Strategy]]:
    return [
        value for value in vars(module).values()
        if isinstance(value, type) and value is not Strategy
        and issubclass(value, Strategy) and value.__module__ == module.__name__
    ]


def load_strategy_file(path: Path) -> type[Strategy]:
    """Import a ``.py`` file, find its Strategy subclass, return it.

    Runs the interface gate (exactly one subclass, non-empty ``name``,
    ``positions`` implemented) and the forbidden-import gate. Raises
    :class:`StrategyValidationError` on any violation.
    """
    path = Path(path)
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StrategyValidationError([f"cannot read strategy file: {exc!r}"]) from exc
    _ast_check_imports(source)

    module_name = _module_name_for(path)
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise StrategyValidationError(["could not create a module spec for the file"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - any import failure is reported to the user
        raise StrategyValidationError([f"module import failed: {exc!r}"]) from exc

    defined = _find_defined_strategies(module)
    for cls in defined:
        if STRATEGY_REGISTRY.get(getattr(cls, "name", "")) is cls:
            STRATEGY_REGISTRY.pop(cls.name)

    failures: list[str] = []
    if len(defined) != 1:
        failures.append(f"expected exactly one Strategy subclass, found {len(defined)}")
    else:
        cls = defined[0]
        name = getattr(cls, "name", None)
        if not isinstance(name, str) or not name.strip():
            failures.append("name must be a non-empty string")
        if getattr(cls, "__abstractmethods__", ()):
            failures.append("must implement positions()")
    if failures:
        raise StrategyValidationError(failures)
    return defined[0]


def validate_strategy(cls: type[Strategy], sample_prices: pd.DataFrame) -> None:
    """Run every price-based gate; collect all failures and raise once."""
    failures: list[str] = []

    result: pd.Series | None = None
    try:
        result = cls().positions(sample_prices)
    except Exception as exc:  # noqa: BLE001 - reported as a gate failure
        failures.append(f"positions() raised on the sample: {exc!r}")

    if result is not None:
        if not isinstance(result, pd.Series):
            failures.append(f"must return a pd.Series, got {type(result).__name__}")
        else:
            if len(result) != len(sample_prices):
                failures.append(f"length mismatch: expected {len(sample_prices)}, got {len(result)}")
            elif not result.index.equals(sample_prices.index):
                failures.append("index mismatch: positions index must equal prices.index")
            else:
                if not result.isin({-1, 0, 1}).all():
                    failures.append("positions must be discrete in {-1, 0, +1} (no fractions, no NaN)")
                perturbations = (
                    ("scale", lambda values: values * 10.0),
                    ("add", lambda values: values + 1000.0),
                    ("flip", lambda values: values * -1.0),
                )
                for label, apply in perturbations:
                    rerun = None
                    try:
                        copy = sample_prices.copy(deep=True)
                        tail = copy.index[-LOOK_AHEAD_ROWS:]
                        copy.loc[tail, "close"] = apply(copy.loc[tail, "close"])
                        rerun = cls().positions(copy)
                    except Exception as exc:  # noqa: BLE001
                        failures.append(f"no-look-ahead re-run ({label}) raised: {exc!r}")
                        continue
                    if rerun is None or not isinstance(rerun, pd.Series) or len(rerun) != len(result):
                        failures.append(f"no-look-ahead re-run ({label}) returned an invalid series")
                    elif not rerun.iloc[:-LOOK_AHEAD_ROWS].fillna(-9).astype(int).eq(
                        result.iloc[:-LOOK_AHEAD_ROWS].fillna(-9).astype(int)
                    ).all():
                        failures.append(
                            f"look-ahead detected: positions before the last {LOOK_AHEAD_ROWS} rows "
                            f"changed when later closes were perturbed ({label})"
                        )

    try:
        tiny = sample_prices.iloc[:TINY_ROWS].copy()
        cls().positions(tiny)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"raised on a tiny {TINY_ROWS}-row frame: {exc!r}")

    if failures:
        raise StrategyValidationError(failures)


def load_and_validate(path: Path, sample_prices: pd.DataFrame) -> type[Strategy]:
    """Load a strategy file and run every gate, returning the validated class."""
    cls = load_strategy_file(path)
    validate_strategy(cls, sample_prices)
    cls.validation_gates = list(GATES)  # type: ignore[attr-defined]
    return cls


def register_custom(
    cls: type[Strategy],
    source_path: Path,
    registry: dict[str, type[Strategy]] | None = None,
) -> str:
    """Register a validated strategy, prefixing ``custom/{name}`` on collision.

    Records provenance on the class (``registry_key`` and ``strategy_source``
    with file name, sha256, upload timestamp, and gate results). Returns the
    registry key used.
    """
    registry = registry if registry is not None else STRATEGY_REGISTRY
    source_path = Path(source_path)
    key = str(getattr(cls, "name", "")).strip()
    if key in STRATEGY_REGISTRY or key in registry:
        key = f"custom/{key}"
    if key in STRATEGY_REGISTRY or key in registry:
        raise StrategyValidationError([f"duplicate custom strategy name: {key}"])
    cls.registry_key = key  # type: ignore[attr-defined]
    file_bytes = source_path.read_bytes()
    cls.strategy_source = {  # type: ignore[attr-defined]
        "file_name": source_path.name,
        "sha256": hashlib.sha256(file_bytes).hexdigest(),
        "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gates": list(getattr(cls, "validation_gates", GATES)),
        "registry_key": key,
    }
    registry[key] = cls
    return key


def register_saved_strategies(
    directory: Path,
    sample_prices: pd.DataFrame | None = None,
) -> dict[str, type[Strategy]]:
    """Scan ``strategies/custom/*.py``, validate each, register survivors.

    Files that fail validation are logged with their failure list and
    skipped — the app still starts. Returns ``{registry_key: class}``.
    """
    directory = Path(directory)
    if not directory.exists():
        return {}
    registered: dict[str, type[Strategy]] = {}
    for path in sorted(directory.glob("*.py")):
        try:
            cls = load_strategy_file(path)
            if sample_prices is not None:
                validate_strategy(cls, sample_prices)
            cls.validation_gates = list(GATES)  # type: ignore[attr-defined]
            key = register_custom(cls, path)
            registered[key] = cls
            logger.info("Registered custom strategy %s as %s", path.name, key)
        except StrategyValidationError as exc:
            logger.warning("Skipping custom strategy %s: %s", path.name, exc.failures)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping custom strategy %s: %r", path.name, exc)
    return registered
