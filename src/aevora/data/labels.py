"""Targets for confirmed run-to-failure histories, never truncated test data."""

import pandas as pd
from .validation import validate_trajectory


def training_rul(frame: pd.DataFrame, *, run_to_failure: bool, cap: int | None = None) -> pd.DataFrame:
    """Return metadata and uncapped/capped targets in original row/index order.

    The caller must attest these are complete histories. Sensor values cannot
    prove provenance; use the official training split only.
    """
    if run_to_failure is not True:
        raise ValueError("RUL labeling requires confirmed run-to-failure training histories")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError("cap must be a positive integer or None")
    validate_trajectory(frame)
    result = frame[["unit_id", "cycle"]].copy()
    result["rul_uncapped"] = (frame.groupby("unit_id")["cycle"].transform("max") - frame.cycle).astype("int64")
    result["rul"] = result.rul_uncapped if cap is None else result.rul_uncapped.clip(upper=cap)
    return result
