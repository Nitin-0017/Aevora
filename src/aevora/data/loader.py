"""Strict whitespace parsing with physical line numbers in parse errors."""

from pathlib import Path
import numpy as np
import pandas as pd
from aevora.config import Settings
from .validation import COLUMNS, DataValidationError, validate_trajectory


def _read_numeric(path: str | Path, width: int) -> list[list[float]]:
    path = Path(path)
    rows = []
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                tokens = line.split()
                if len(tokens) != width:
                    raise DataValidationError(f"{path}: line {number}: expected {width} columns, got {len(tokens)}")
                try:
                    rows.append([float(token) for token in tokens])
                except ValueError as exc:
                    raise DataValidationError(f"{path}: line {number}: malformed numeric value") from exc
    except OSError as exc:
        raise OSError(f"Cannot read dataset {path}: {exc}") from exc
    if not rows:
        raise DataValidationError(f"{path}: empty file")
    return rows


def load_trajectory(path: str | Path) -> pd.DataFrame:
    frame = pd.DataFrame(_read_numeric(path, 26), columns=COLUMNS)
    validate_trajectory(frame)
    frame[["unit_id", "cycle"]] = frame[["unit_id", "cycle"]].astype("int64")
    return frame


def load_split(settings: Settings, split: str = "train") -> pd.DataFrame:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    return load_trajectory(settings.train_path if split == "train" else settings.test_path)


def load_rul_targets(path: str | Path, test: pd.DataFrame) -> pd.Series:
    """Explicit opt-in for final evaluation. Never called by Phase 2 EDA.

    NASA targets map to unit 1, 2, ..., N, not first appearance in the file.
    Reject subsets/noncontiguous IDs, whose target alignment is ambiguous.
    """
    validate_trajectory(test)
    ids = sorted(int(value) for value in test.unit_id.unique())
    if ids != list(range(1, len(ids) + 1)):
        raise DataValidationError("Target alignment requires complete test engine IDs 1..N")
    values = np.asarray(_read_numeric(path, 1), dtype=float).ravel()
    with np.errstate(invalid="ignore"):
        invalid = ~np.isfinite(values) | (values < 0) | (values != np.floor(values)) | (values > 2**53 - 1)
    if invalid.any():
        raise DataValidationError("RUL targets must be finite nonnegative integers")
    if len(values) != len(ids):
        raise DataValidationError(f"RUL target count {len(values)} does not match {len(ids)} test engines")
    return pd.Series(values.astype("int64"), index=pd.Index(ids, name="unit_id"), name="rul")
