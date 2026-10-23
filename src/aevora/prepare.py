"""Prepare official training histories only; no estimator training or test access."""

from dataclasses import asdict
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from .config import Settings, load_config
from .data.labels import training_rul
from .data.loader import load_split
from .features import sensor_features
from .preprocessing import META, FeatureScaler, ordered_history, select_sensors, split_engines
from .sequences import Windows, make_windows


def prepare_data(settings: Settings, *, save_arrays: bool = False) -> dict:
    raw = load_split(settings, "train")
    train, valid = split_engines(raw, seed=settings.random_seed, validation_fraction=settings.validation_fraction)
    train_ids = sorted(int(x) for x in train.unit_id.unique())
    valid_ids = sorted(int(x) for x in valid.unit_id.unique())
    sensors = select_sensors(train)
    train_features = sensor_features(train, sensors, settings.rolling_window)
    valid_features = sensor_features(valid, sensors, settings.rolling_window)
    features = [name for name in train_features if name not in META]
    scaler = FeatureScaler.fit(train_features, features, expected_training_ids=train_ids)
    transformed = {"train": scaler.transform(train_features), "validation": scaler.transform(valid_features)}
    labels = {name: training_rul(frame, run_to_failure=True, cap=settings.rul_cap)
              for name, frame in [("train", train), ("validation", valid)]}
    windows = {name: make_windows(frame, features, settings.sequence_window, labels[name])
               for name, frame in transformed.items()}
    config = asdict(settings)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    manifest = {
        "version": 1, "source_sha256": hashlib.sha256(settings.train_path.read_bytes()).hexdigest(),
        "config": config, "training_engine_ids": train_ids, "validation_engine_ids": valid_ids,
        "selected_sensors": sensors, "feature_names": features,
        "sensor_selection": "exact constants dropped using training engines only",
        "scaler_fit_engine_ids": list(scaler.training_engine_ids), "scaler_fit_rows": scaler.training_rows,
        "rows": {name: len(frame) for name, frame in transformed.items()},
        "window_shapes": {name: list(batch.x.shape) for name, batch in windows.items()},
        "skipped_engine_ids": {name: batch.skipped_engine_ids for name, batch in windows.items()},
        "arrays_saved": save_arrays,
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas"]},
    }
    manifest["versions"]["python"] = platform.python_version()
    output = settings.artifact_dir / "prepared" / settings.dataset
    output.mkdir(parents=True, exist_ok=True)
    scaler.save(output / "scaler.json")
    for name, batch in windows.items():
        labels[name].to_csv(output / f"{name}_labels.csv", index=False)
        batch.endpoints.to_csv(output / f"{name}_endpoints.csv", index=False)
        if save_arrays:
            np.savez_compressed(output / f"{name}_windows.npz", x=batch.x, y=batch.y,
                                unit_id=batch.endpoints.unit_id.to_numpy(), cycle=batch.endpoints.cycle.to_numpy())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def prepare_inference(history: pd.DataFrame, manifest_path: str | Path) -> Windows:
    """Use saved sensor selection/scaling; pass available history through now only."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest["version"] != 1:
        raise ValueError("Unsupported preparation manifest version")
    sensors = manifest["selected_sensors"]
    ordered = ordered_history(history, sensors)
    if ordered.groupby("unit_id").cycle.min().ne(1).any():
        raise ValueError("Provide complete available history from cycle 1 for consistent rolling context")
    scaler = FeatureScaler.load(path.parent / "scaler.json")
    if list(scaler.feature_names) != manifest["feature_names"] or list(scaler.training_engine_ids) != manifest["training_engine_ids"]:
        raise ValueError("Scaler and manifest do not match")
    features = sensor_features(ordered, sensors, manifest["config"]["rolling_window"])
    return make_windows(scaler.transform(features), list(scaler.feature_names),
                        manifest["config"]["sequence_window"], latest_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--save-arrays", action="store_true", help="Save compressed arrays under ignored artifacts")
    args = parser.parse_args()
    try:
        report = prepare_data(load_config(args.config), save_arrays=args.save_arrays)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")
    print(json.dumps({key: report[key] for key in ["rows", "window_shapes", "skipped_engine_ids", "arrays_saved"]}, indent=2))


if __name__ == "__main__":
    main()
