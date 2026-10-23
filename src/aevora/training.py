"""Train classical models using verified Phase 3 artifacts; never read official test files."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from .classical import Predictor, summarize
from .config import load_config
from .data.labels import training_rul
from .data.loader import load_split
from .features import sensor_features
from .preprocessing import FeatureScaler, META, split_engines, select_sensors
from .sequences import make_windows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.ndim != 1 or prediction.shape != y.shape or not len(y):
        raise ValueError("Targets and predictions must be aligned nonempty vectors")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite evaluation values")
    residual = prediction - y
    return {"mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.mean(residual ** 2)))}


def load_prepared(settings):
    """Verify saved windows against raw TRAIN histories and frozen scaler, without refitting."""
    directory = settings.artifact_dir / "prepared" / settings.dataset
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["version"] != 1 or not manifest["arrays_saved"]:
        raise ValueError("Run Phase 3 preparation with --save-arrays first")
    for key in ("dataset", "random_seed", "validation_fraction", "rolling_window", "sequence_window", "rul_cap"):
        if manifest["config"].get(key) != getattr(settings, key):
            raise ValueError(f"Prepared configuration differs: {key}")
    if digest(settings.train_path) != manifest["source_sha256"]:
        raise ValueError("Training source changed since preparation")
    scaler = FeatureScaler.load(directory / "scaler.json")
    names = manifest["feature_names"]
    if (list(scaler.feature_names) != names
            or list(scaler.training_engine_ids) != manifest["training_engine_ids"]
            or manifest["scaler_fit_engine_ids"] != manifest["training_engine_ids"]
            or scaler.training_rows != manifest["scaler_fit_rows"]):
        raise ValueError("Saved scaler provenance mismatch")
    if set(manifest["training_engine_ids"]) & set(manifest["validation_engine_ids"]):
        raise ValueError("Training and validation engines overlap")
    raw = load_split(settings, "train")
    train, valid = split_engines(raw, seed=settings.random_seed, validation_fraction=settings.validation_fraction)
    if select_sensors(train) != manifest["selected_sensors"] or len(train) != scaler.training_rows:
        raise ValueError("Training sensor selection or row count mismatch")
    batches = {}
    for name, frame, id_key in [("train", train, "training_engine_ids"), ("validation", valid, "validation_engine_ids")]:
        if sorted(frame.unit_id.unique()) != manifest[id_key]:
            raise ValueError("Saved engine split differs from configuration")
        features = sensor_features(frame, manifest["selected_sensors"], settings.rolling_window)
        labels = training_rul(frame, run_to_failure=True, cap=settings.rul_cap)
        expected = make_windows(scaler.transform(features), names, settings.sequence_window, labels)
        with np.load(directory / f"{name}_windows.npz", allow_pickle=False) as archive:
            x, y = archive["x"], archive["y"]
            endpoints = pd.DataFrame({key: archive[key] for key in META})
        if (list(x.shape) != manifest["window_shapes"][name]
                or not np.array_equal(x, expected.x) or not np.array_equal(y, expected.y)
                or not endpoints.equals(expected.endpoints)
                or not endpoints.equals(pd.read_csv(directory / f"{name}_endpoints.csv"))) :
            raise ValueError(f"{name} saved windows or target alignment mismatch")
        saved_labels = pd.read_csv(directory / f"{name}_labels.csv")
        if not saved_labels.equals(labels.reset_index(drop=True)):
            raise ValueError(f"{name} saved labels mismatch")
        if not len(y):
            raise ValueError(f"{name} has no usable windows")
        batches[name] = expected
    return manifest, batches


def candidates(name, settings):
    if name == "baseline":
        return [DummyRegressor(strategy=settings.models["baseline"]["strategy"])]
    if name == "ridge":
        # Small regularization sweep anchored to the central configuration.
        alphas = sorted(set(settings.models["ridge"]["alpha"] * factor for factor in (1, 10, 100)))
        return [Ridge(alpha=alpha, solver="svd") for alpha in alphas]
    if name == "gradient_boosting":
        parameters = dict(settings.models[name])
        leaves = parameters.pop("max_leaf_nodes")
        # Fixed iteration budget avoids an internal random split of overlapping windows.
        return [HistGradientBoostingRegressor(**parameters, max_leaf_nodes=n,
                    early_stopping=False, random_state=settings.random_seed)
                for n in sorted(set([max(2, leaves // 2), leaves]))]
    raise ValueError(f"Unsupported model: {name}")


def train_model(settings, name, prepared=None):
    manifest, batches = prepared if prepared is not None else load_prepared(settings)
    features = manifest["feature_names"]
    x = {key: summarize(batch.x, features, settings.sequence_window) for key, batch in batches.items()}
    trials, fitted = [], []
    for estimator in candidates(name, settings):
        start = time.perf_counter()
        estimator.fit(x["train"], batches["train"].y)
        seconds = time.perf_counter() - start
        score = metrics(batches["validation"].y, estimator.predict(x["validation"]))
        trials.append({"parameters": estimator.get_params(), "fit_seconds": seconds, **score})
        fitted.append(estimator)
    best = min(range(len(trials)), key=lambda i: trials[i]["rmse"])
    model = Predictor(fitted[best], features, settings.sequence_window)
    output = settings.artifact_dir / "models" / settings.dataset / name
    model.save(output)
    prep_dir = output / "preprocessing"
    prep_dir.mkdir(exist_ok=True)
    source_dir = settings.artifact_dir / "prepared" / settings.dataset
    shutil.copyfile(source_dir / "scaler.json", prep_dir / "scaler.json")
    portable = dict(manifest)
    portable["config"] = {key: value for key, value in manifest["config"].items()
                          if key not in {"project_root", "raw_data_dir", "artifact_dir"}}
    (prep_dir / "manifest.json").write_text(json.dumps(portable, indent=2) + "\n")
    prediction = model.predict(batches["validation"].x, features)
    np.testing.assert_array_equal(prediction, Predictor.load(output).predict(batches["validation"].x, features))
    results = batches["validation"].endpoints.copy()
    results["rul"] = batches["validation"].y
    results["predicted_rul"] = prediction
    results.to_csv(output / "validation_predictions.csv", index=False)
    report = {"model": name, "selection": "lowest development-validation RMSE; first candidate wins ties",
              "selected_trial": best, "trials": trials, **metrics(batches["validation"].y, prediction),
              "fit_seconds": trials[best]["fit_seconds"], "search_fit_seconds": sum(t["fit_seconds"] for t in trials),
              "training_observations": len(batches["train"].y), "validation_observations": len(prediction),
              "validation_engines": int(results.unit_id.nunique()), "input_dimensions": x["train"].shape[1],
              "model_config": settings.models[name], "seed": settings.random_seed,
              "source_sha256": manifest["source_sha256"],
              "prepared_sha256": {p.name: digest(p) for p in sorted(source_dir.iterdir()) if p.is_file()},
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn")}}
    (output / "metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["baseline", "ridge", "gradient_boosting"], required=True)
    args = parser.parse_args()
    print(json.dumps(train_model(load_config(args.config), args.model), indent=2))


if __name__ == "__main__":
    main()
