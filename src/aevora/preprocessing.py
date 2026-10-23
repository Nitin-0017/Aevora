"""Engine splits and immutable, training-fitted preprocessing state."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .data.validation import SENSORS, validate_trajectory

META = ["unit_id", "cycle"]


def ordered_history(frame: pd.DataFrame, features: list[str] | tuple[str, ...]) -> pd.DataFrame:
    """Copy and sort complete or partial histories; reject gaps and bad values."""
    if frame.empty or not frame.columns.is_unique:
        raise ValueError("History must be nonempty with unique column names")
    if not features or len(set(features)) != len(features) or any(not f.startswith("sensor_") for f in features):
        raise ValueError("Features must be unique sensor feature names; IDs and targets are metadata")
    required = META + list(features)
    if not set(required).issubset(frame.columns):
        raise ValueError(f"Missing required columns: {set(required) - set(frame.columns)}")
    try:
        values = frame[required].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("History contains nonnumeric values") from exc
    if not np.isfinite(values).all():
        raise ValueError("History contains missing or nonfinite values")
    ids = values[:, :2]
    if np.any((ids <= 0) | (ids != np.floor(ids)) | (ids > 2**53 - 1)):
        raise ValueError("Engine IDs and cycles must be positive integers")
    result = pd.DataFrame(values, columns=required)
    result[META] = result[META].astype("int64")
    if result.duplicated(META).any():
        raise ValueError("Duplicate engine-cycle keys")
    result = result.sort_values(META, kind="stable").reset_index(drop=True)
    if result.groupby("unit_id").cycle.diff().dropna().ne(1).any():
        raise ValueError("History contains missing cycles; windows require consecutive observations")
    return result


def split_engines(frame: pd.DataFrame, *, seed: int, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_trajectory(frame)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    units = np.sort(frame.unit_id.unique())
    count = math.ceil(len(units) * validation_fraction)
    if not 1 <= count < len(units):
        raise ValueError("Split must leave at least one engine in each partition")
    validation_ids = np.random.default_rng(seed).permutation(units)[:count]
    mask = frame.unit_id.isin(validation_ids)
    return frame.loc[~mask].copy(), frame.loc[mask].copy()


def select_sensors(training: pd.DataFrame) -> list[str]:
    """Drop exact constants using only the supplied training partition."""
    validate_trajectory(training)
    selected = [name for name in SENSORS if training[name].nunique() > 1]
    if not selected:
        raise ValueError("All training sensors are constant")
    return selected


@dataclass(frozen=True)
class FeatureScaler:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    training_engine_ids: tuple[int, ...]
    training_rows: int
    version: int = 1

    def __post_init__(self):
        n = len(self.feature_names)
        if self.version != 1 or not n or len(set(self.feature_names)) != n:
            raise ValueError("Invalid scaler version or feature schema")
        if any(not name.startswith("sensor_") for name in self.feature_names):
            raise ValueError("Only sensor features may be scaled")
        if len(self.means) != n or len(self.scales) != n:
            raise ValueError("Scaler state dimensions disagree")
        if not np.isfinite(self.means).all() or not np.isfinite(self.scales).all() or min(self.scales) <= 0:
            raise ValueError("Scaler statistics must be finite with positive scales")
        if not self.training_engine_ids or self.training_rows < 1:
            raise ValueError("Scaler training provenance is missing")

    @classmethod
    def fit(cls, training: pd.DataFrame, features: list[str], *, expected_training_ids: list[int]):
        ordered = ordered_history(training, features)
        units = tuple(sorted(int(x) for x in ordered.unit_id.unique()))
        if units != tuple(sorted(expected_training_ids)):
            raise ValueError("Scaler fit engines do not match the declared training partition")
        values = ordered[features].to_numpy()
        means = values.mean(axis=0)
        scales = values.std(axis=0, ddof=0)
        scales[scales == 0] = 1
        return cls(tuple(features), tuple(means), tuple(scales), units, len(ordered))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = ordered_history(frame, self.feature_names)
        values = (result[list(self.feature_names)].to_numpy() - self.means) / self.scales
        if not np.isfinite(values).all():
            raise ValueError("Scaling produced nonfinite values")
        result[list(self.feature_names)] = values
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | Path):
        raw = json.loads(Path(path).read_text())
        for key in ["feature_names", "means", "scales", "training_engine_ids"]:
            raw[key] = tuple(raw[key])
        return cls(**raw)
