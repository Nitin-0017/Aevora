"""Non-mutating FD001 trajectory quality reports."""

from typing import Any
import numpy as np
import pandas as pd

SENSORS = [f"sensor_{i}" for i in range(1, 22)]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
COLUMNS = ["unit_id", "cycle", *SETTINGS, *SENSORS]


class DataValidationError(ValueError):
    """Invalid data, with optional machine-readable diagnostics."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


def quality_report(frame: pd.DataFrame, near_constant_fraction: float = 0.99) -> dict[str, Any]:
    """Near-constant = one exact recorded value occupies >= threshold of rows.

    This is a descriptive frequency heuristic, not a feature-removal decision.
    Cycles must start at 1 and advance by 1 within each engine in input order.
    Engines may appear in any order; no rows are sorted or repaired here.
    """
    if not 0 < near_constant_fraction <= 1:
        raise ValueError("near_constant_fraction must be in (0, 1]")
    report: dict[str, Any] = {
        "rows": len(frame), "columns": len(frame.columns), "errors": [],
        "near_constant_fraction": near_constant_fraction,
    }
    errors = report["errors"]
    if list(frame.columns) != COLUMNS:
        errors.append("Expected exactly the 26 named C-MAPSS columns in schema order")
        report["valid"] = False
        return report
    if frame.empty:
        errors.append("Dataset is empty")
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    missing = frame.isna()
    malformed = numeric.isna() & ~missing
    values = numeric.to_numpy(dtype=float)
    report["missing_values"] = int(missing.to_numpy().sum())
    report["malformed_values"] = int(malformed.to_numpy().sum())
    report["nonfinite_values"] = int((~np.isfinite(values)).sum())
    for field in ("missing_values", "malformed_values", "nonfinite_values"):
        if report[field]:
            errors.append(f"{field}: {report[field]}")
    invalid = {}
    for name in ("unit_id", "cycle"):
        v = numeric[name].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            bad = ~np.isfinite(v) | (v <= 0) | (v != np.floor(v)) | (v > 2**53 - 1)
        invalid[name] = int(bad.sum())
    report["invalid_identifiers_and_cycles"] = invalid
    if any(invalid.values()):
        errors.append(f"IDs and cycles must be positive, exactly representable integers: {invalid}")
    report["duplicate_engine_cycle_records"] = int(frame.duplicated(["unit_id", "cycle"]).sum())
    if report["duplicate_engine_cycle_records"]:
        errors.append("Duplicate engine-cycle records")
    report.update(unique_engines=0, cycles_per_engine={}, min_cycle=None, max_cycle=None,
                  chronological_errors=[], continuity_errors=[])
    if not any(invalid.values()) and not frame.empty:
        groups = numeric.groupby("unit_id", sort=True)["cycle"]
        report["unique_engines"] = int(groups.ngroups)
        report["cycles_per_engine"] = {str(int(k)): int(len(v)) for k, v in groups}
        report["min_cycle"] = int(numeric.cycle.min())
        report["max_cycle"] = int(numeric.cycle.max())
        for unit, cycles in groups:
            v = cycles.to_numpy()
            if np.any(np.diff(v) <= 0):
                report["chronological_errors"].append(int(unit))
            if v[0] != 1 or np.any(np.diff(np.sort(v)) != 1):
                report["continuity_errors"].append(int(unit))
        if report["chronological_errors"]:
            errors.append("Cycles are not strictly increasing in input order")
        if report["continuity_errors"]:
            errors.append("Cycles must start at 1 without gaps or duplicates")
    report["constant_sensors"] = []
    report["near_constant_sensors"] = []
    for sensor in SENSORS:
        series = numeric[sensor]
        if len(series) and np.isfinite(series.to_numpy()).all():
            counts = series.value_counts(normalize=True)
            if len(counts) == 1:
                report["constant_sensors"].append(sensor)
            elif counts.iloc[0] >= near_constant_fraction:
                report["near_constant_sensors"].append(sensor)
    report["valid"] = not errors
    return report


def validate_trajectory(frame: pd.DataFrame) -> dict[str, Any]:
    report = quality_report(frame)
    if not report["valid"]:
        raise DataValidationError("; ".join(report["errors"]), report)
    return report
