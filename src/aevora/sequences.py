"""Chronological full windows with explicit endpoint metadata."""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from .preprocessing import META, ordered_history


@dataclass
class Windows:
    x: np.ndarray
    y: np.ndarray | None
    endpoints: pd.DataFrame
    skipped_engine_ids: list[int]


def make_windows(frame: pd.DataFrame, features: list[str], length: int = 20,
                 targets: pd.DataFrame | None = None, *, latest_only: bool = False) -> Windows:
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise ValueError("Window length must be a positive integer")
    ordered = ordered_history(frame, features)
    if targets is not None:
        if not targets.columns.is_unique or not set(META + ["rul"]).issubset(targets.columns):
            raise ValueError("Targets require unique columns with engine, cycle and rul")
        t = targets[META + ["rul"]]
        try:
            vals = t.to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("Targets must be numeric") from exc
        if not np.isfinite(vals).all() or (vals[:, 2] < 0).any() or (vals != np.floor(vals)).any():
            raise ValueError("Target keys and RUL must be finite integers with nonnegative RUL")
        if t.duplicated(META).any():
            raise ValueError("Duplicate target keys")
        if set(map(tuple, t[META].to_numpy())) != set(map(tuple, ordered[META].to_numpy())):
            raise ValueError("Target keys must exactly match history keys")
        ordered = ordered.merge(t, on=META, how="left", validate="one_to_one", sort=False)
    arrays, ys, keys, skipped = [], [], [], []
    for unit, group in ordered.groupby("unit_id", sort=True):
        if len(group) < length:
            skipped.append(int(unit))
            continue
        values = group[features].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Features exceed finite float32 range")
        ends = [len(group) - 1] if latest_only else range(length - 1, len(group))
        for end in ends:
            arrays.append(values[end - length + 1:end + 1])
            keys.append((int(unit), int(group.cycle.iloc[end])))
            if targets is not None:
                ys.append(float(group.rul.iloc[end]))
    if latest_only and skipped:
        raise ValueError(f"Insufficient history for engines {skipped}; need {length} cycles, no padding")
    x = np.stack(arrays) if arrays else np.empty((0, length, len(features)), dtype=np.float32)
    return Windows(x, None if targets is None else np.asarray(ys, dtype=float),
                   pd.DataFrame(keys, columns=META).astype("int64"), skipped)
