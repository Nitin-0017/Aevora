"""Validated configuration; importing/loading never reads data or creates files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math
import tomllib


@dataclass(frozen=True)
class Settings:
    project_root: Path
    dataset: str
    raw_data_dir: Path
    artifact_dir: Path
    random_seed: int
    sequence_window: int
    validation_fraction: float
    models: dict[str, dict[str, Any]]
    rul_cap: int | None = None
    rolling_window: int = 5

    @property
    def train_path(self) -> Path:
        return self.raw_data_dir / f"train_{self.dataset}.txt"

    @property
    def test_path(self) -> Path:
        return self.raw_data_dir / f"test_{self.dataset}.txt"

    @property
    def test_targets_path(self) -> Path:
        return self.raw_data_dir / f"RUL_{self.dataset}.txt"


def _number(value: Any, name: str, *, integer: bool = False, minimum: float = 0) -> None:
    expected = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise ValueError(f"{name} must be {'an integer' if integer else 'numeric'}")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty path string")
    return (base / value).resolve()


def load_config(path: str | Path) -> Settings:
    """Resolve paths against an explicit config location, independent of cwd."""
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)
    expected = {"project_root", "dataset", "raw_data_dir", "artifact_dir",
                "random_seed", "sequence_window", "validation_fraction", "models"}
    optional = {"rul_cap", "rolling_window"}
    if set(raw) - optional != expected:
        raise ValueError(f"Config keys mismatch: missing={expected - set(raw)}, unknown={set(raw) - expected - optional}")
    if raw["dataset"] != "FD001":
        raise ValueError("Only FD001 is supported in the initial project")
    if "rul_cap" in raw:
        _number(raw["rul_cap"], "rul_cap", integer=True, minimum=1)
    _number(raw.get("rolling_window", 5), "rolling_window", integer=True, minimum=2)
    _number(raw["random_seed"], "random_seed", integer=True)
    _number(raw["sequence_window"], "sequence_window", integer=True, minimum=1)
    _number(raw["validation_fraction"], "validation_fraction")
    if not 0 < raw["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    models = raw["models"]
    schemas = {
        "baseline": {"strategy"},
        "ridge": {"alpha"},
        "gradient_boosting": {"learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization"},
        "lstm": {"hidden_size", "num_layers", "batch_size", "learning_rate", "weight_decay", "max_epochs", "patience", "device"},
    }
    if not isinstance(models, dict) or set(models) != set(schemas):
        raise ValueError("models must define baseline, ridge, gradient_boosting, and lstm")
    # Optional Phase 5 knobs preserve older configuration files.
    lstm_defaults = {"dropout": 0.1, "gradient_clip": 1.0, "min_delta": 0.0, "num_threads": 2}
    if isinstance(models["lstm"], dict):
        for key, value in lstm_defaults.items():
            models["lstm"].setdefault(key, value)
    schemas["lstm"] |= set(lstm_defaults)
    integer_keys = {"max_iter", "max_leaf_nodes", "hidden_size", "num_layers", "batch_size", "max_epochs", "patience", "num_threads"}
    for name, keys in schemas.items():
        if not isinstance(models[name], dict) or set(models[name]) != keys:
            raise ValueError(f"Invalid settings for {name}")
        for key, value in models[name].items():
            if key in {"strategy", "device"}:
                continue
            minimum = 1 if key in integer_keys else (0.000000000001 if key == "learning_rate" else 0)
            if key == "max_leaf_nodes":
                minimum = 2
            _number(value, f"{name}.{key}", integer=key in integer_keys, minimum=minimum)
    if not 0 <= models["lstm"]["dropout"] < 1 or models["lstm"]["gradient_clip"] <= 0:
        raise ValueError("LSTM dropout must be in [0,1) and gradient_clip positive")
    if models["baseline"]["strategy"] != "median":
        raise ValueError("The initial baseline strategy must be median")
    if models["lstm"]["device"] != "cpu":
        raise ValueError("The initial reproducible configuration uses cpu")
    root = _path(config_path.parent, raw["project_root"], "project_root")
    return Settings(root, raw["dataset"],
                    _path(root, raw["raw_data_dir"], "raw_data_dir"),
                    _path(root, raw["artifact_dir"], "artifact_dir"),
                    raw["random_seed"], raw["sequence_window"],
                    raw["validation_fraction"], models, rul_cap=raw.get("rul_cap"),
                    rolling_window=raw.get("rolling_window", 5))
