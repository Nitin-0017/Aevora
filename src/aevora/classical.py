"""Fixed historical summaries and portable, trusted-local model bundles."""

import json
from pathlib import Path
import pickle
import numpy as np
from .prepare import prepare_inference


def summary_names(features):
    return [f"{stat}::{name}" for stat in ("last", "mean", "std") for name in features]


def summarize(x, features, length):
    x = np.asarray(x)
    if len(features) != len(set(features)) or not features:
        raise ValueError("Features must be unique and nonempty")
    if x.ndim != 3 or x.shape[1:] != (length, len(features)) or not len(x):
        raise ValueError("Expected nonempty windows with matching history and feature dimensions")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite window input")
    # Float64 accumulation avoids roundoff in long/near-constant histories.
    return np.concatenate([x[:, -1, :], x.mean(axis=1, dtype=np.float64),
                           x.std(axis=1, dtype=np.float64, ddof=0)], axis=1)


class Predictor:
    def __init__(self, estimator, features, length):
        self.estimator = estimator
        self.features = tuple(features)
        self.length = length

    def predict(self, x, features):
        if tuple(features) != self.features:
            raise ValueError("Feature order differs from the trained schema")
        return self.estimator.predict(summarize(x, self.features, self.length))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "estimator.pkl").open("wb") as stream:
            pickle.dump(self, stream)
        schema = {"version": 1, "window_length": self.length, "features": self.features,
                  "representation": "last_mean_population_std", "vector_features": summary_names(self.features)}
        (directory / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")

    @classmethod
    def load(cls, directory):
        """Load only trusted local bundles: pickle can execute arbitrary code."""
        directory = Path(directory)
        with (directory / "estimator.pkl").open("rb") as stream:
            model = pickle.load(stream)
        schema = json.loads((directory / "schema.json").read_text())
        if (schema["version"] != 1 or schema["features"] != list(model.features)
                or schema["window_length"] != model.length
                or schema["representation"] != "last_mean_population_std"
                or schema["vector_features"] != summary_names(model.features)):
            raise ValueError("Saved model schema mismatch")
        return model


def predict_history(directory, history):
    """Return endpoint keys and predictions for each engine's latest available cycle."""
    directory = Path(directory)
    model = Predictor.load(directory)
    manifest_path = directory / "preprocessing/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    batch = prepare_inference(history, manifest_path)
    result = batch.endpoints.copy()
    result["predicted_rul"] = model.predict(batch.x, manifest["feature_names"])
    return result
