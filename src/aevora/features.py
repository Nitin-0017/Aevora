"""Historical sensor features, calculated before training-fitted scaling."""

import numpy as np
import pandas as pd
from .data.validation import SENSORS
from .preprocessing import META, ordered_history


def _slope(values: np.ndarray) -> float:
    if len(values) == 1:
        return 0.
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    return float(np.dot(x, values - values.mean()) / np.dot(x, x))


def sensor_features(frame: pd.DataFrame, sensors: list[str], rolling_window: int = 5) -> pd.DataFrame:
    if isinstance(rolling_window, bool) or not isinstance(rolling_window, int) or rolling_window < 2:
        raise ValueError("rolling_window must be an integer >= 2")
    if not set(sensors).issubset(SENSORS):
        raise ValueError("Select raw sensor columns only")
    ordered = ordered_history(frame, sensors)
    columns = {name: ordered[name] for name in META}
    for sensor in sensors:
        grouped = ordered.groupby("unit_id", sort=False)[sensor]
        columns[sensor] = ordered[sensor]
        columns[f"{sensor}_mean_{rolling_window}"] = grouped.transform(
            lambda s: s.rolling(rolling_window, min_periods=1).mean())
        columns[f"{sensor}_std_{rolling_window}"] = grouped.transform(
            lambda s: s.rolling(rolling_window, min_periods=1).std(ddof=0))
        columns[f"{sensor}_change_1"] = grouped.diff().fillna(0.)
        columns[f"{sensor}_trend_{rolling_window}"] = grouped.transform(
            lambda s: s.rolling(rolling_window, min_periods=1).apply(_slope, raw=True))
    result = pd.DataFrame(columns)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Feature calculation produced nonfinite values")
    return result
