"""Small CPU LSTM and schema-checked sequence inference; no data access on import."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .prepare import prepare_inference


class RULLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.1,
                 target_mean=0., target_scale=1.):
        super().__init__()
        if min(input_size, hidden_size, num_layers) < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid LSTM architecture")
        if not np.isfinite([target_mean, target_scale]).all() or target_scale <= 0:
            raise ValueError("Invalid training target scale")
        self.architecture = dict(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=num_layers, dropout=dropout)
        self.recurrent = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_scale", torch.tensor(float(target_scale)))

    def forward(self, x):
        if x.ndim != 3 or x.shape[2] != self.architecture["input_size"] or min(x.shape[:2]) < 1:
            raise ValueError("Expected nonempty [batch, timesteps, features] tensor")
        sequence, _ = self.recurrent(x)
        return self.head(self.dropout(sequence[:, -1])).squeeze(-1) * self.target_scale + self.target_mean


class LSTMPredictor:
    def __init__(self, model, features, length, *, batch_size=64, preprocessing_sha256=None):
        if (not features or len(features) != len(set(features))
                or len(features) != model.architecture["input_size"] or length < 1 or batch_size < 1):
            raise ValueError("Invalid sequence schema")
        self.model = model.cpu().eval()
        self.features = tuple(features)
        self.length = length
        self.batch_size = batch_size
        self.preprocessing_sha256 = preprocessing_sha256 or {}

    def predict(self, x, features):
        if tuple(features) != self.features:
            raise ValueError("Feature order differs from the trained schema")
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (self.length, len(self.features)) or not len(x):
            raise ValueError("Invalid sequence shape")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite sequence input")
        self.model.eval()
        with torch.inference_mode():
            result = np.concatenate([self.model(torch.from_numpy(np.ascontiguousarray(x[i:i+self.batch_size]))).numpy()
                                     for i in range(0, len(x), self.batch_size)])
        if result.shape != (len(x),) or not np.isfinite(result).all():
            raise ValueError("Invalid model predictions")
        return result.astype(np.float64)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        schema = {"version": 1, "architecture": self.model.architecture,
                  "features": list(self.features), "sequence_length": self.length,
                  "batch_size": self.batch_size, "preprocessing_sha256": self.preprocessing_sha256}
        torch.save({"schema": schema, "state_dict": self.model.state_dict()}, directory / "best.pt")
        (directory / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        schema = json.loads((directory / "schema.json").read_text())
        if schema != saved["schema"] or schema["version"] != 1:
            raise ValueError("Checkpoint schema mismatch")
        for name, expected in schema["preprocessing_sha256"].items():
            if name not in {"manifest.json", "scaler.json"}:
                raise ValueError("Unknown preprocessing artifact")
            actual = hashlib.sha256((directory / "preprocessing" / name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError("Preprocessing checksum mismatch")
        model = RULLSTM(**schema["architecture"])
        model.load_state_dict(saved["state_dict"], strict=True)
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()) or model.target_scale <= 0:
            raise ValueError("Invalid checkpoint weights or scaling")
        return cls(model, schema["features"], schema["sequence_length"], batch_size=schema["batch_size"],
                   preprocessing_sha256=schema["preprocessing_sha256"])


def predict_history(directory, history):
    directory = Path(directory)
    predictor = LSTMPredictor.load(directory)
    path = directory / "preprocessing/manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["config"]["sequence_window"] != predictor.length:
        raise ValueError("Preprocessing sequence length differs")
    batch = prepare_inference(history, path)
    result = batch.endpoints.copy()
    result["predicted_rul"] = predictor.predict(batch.x, manifest["feature_names"])
    return result
