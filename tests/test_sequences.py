from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from aevora.config import load_config
from aevora.data.labels import training_rul
from aevora.features import sensor_features
from aevora.prepare import prepare_data, prepare_inference
from aevora.sequences import make_windows
from test_labels import histories


class SequenceTests(unittest.TestCase):
    def test_shapes_labels_sorting_and_boundaries(self):
        frame = histories((4, 3))
        frame["sensor_1"] = [1, 2, 3, 4, 101, 102, 103]
        labels = training_rul(frame, run_to_failure=True)
        batch = make_windows(frame.iloc[::-1], ["sensor_1"], 3, labels.iloc[::-1])
        self.assertEqual(batch.x.shape, (3, 3, 1))
        np.testing.assert_array_equal(batch.x[:, :, 0], [[1, 2, 3], [2, 3, 4], [101, 102, 103]])
        np.testing.assert_array_equal(batch.y, [1, 0, 0])
        self.assertEqual(batch.endpoints.to_records(index=False).tolist(), [(1, 3), (1, 4), (2, 3)])

    def test_short_history_and_inference_latest(self):
        frame = histories((2, 4))
        batch = make_windows(frame, ["sensor_1"], 3)
        self.assertEqual(batch.skipped_engine_ids, [1])
        self.assertIsNone(batch.y)
        empty = make_windows(frame, ["sensor_1"], 20)
        self.assertEqual(empty.x.shape, (0, 20, 1))
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            make_windows(frame, ["sensor_1"], 3, latest_only=True)
        latest = make_windows(frame[frame.unit_id == 2], ["sensor_1"], 3, latest_only=True)
        self.assertEqual(latest.endpoints.cycle.tolist(), [4])

    def test_no_future_information(self):
        frame = histories((6, 6))
        first = sensor_features(frame, ["sensor_1"], 3)
        later = frame.copy()
        later.loc[later.cycle > 4, "sensor_1"] = 999
        second = sensor_features(later, ["sensor_1"], 3)
        features = first.columns[2:].tolist()
        a, b = [make_windows(f, features, 3) for f in [first, second]]
        mask = a.endpoints.cycle <= 4
        np.testing.assert_array_equal(a.x[mask], b.x[mask])

    def test_invalid_labels_and_window(self):
        frame = histories()
        labels = training_rul(frame, run_to_failure=True)
        for bad in [labels.iloc[:-1], pd.concat([labels, labels.iloc[:1]]), labels.assign(rul=-1), labels.assign(rul=np.inf)]:
            with self.assertRaises(ValueError):
                make_windows(frame, ["sensor_1"], 2, bad)
        with self.assertRaises(ValueError):
            make_windows(frame, ["sensor_1"], 0)

    def test_end_to_end_saved_inference_and_training_only_access(self):
        default = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = replace(default, raw_data_dir=root / "raw", artifact_dir=root / "out", sequence_window=3)
            settings.raw_data_dir.mkdir()
            frame = histories((5,) * 5)
            frame.to_csv(settings.train_path, sep=" ", header=False, index=False)
            from aevora.prepare import load_split
            with patch("aevora.prepare.load_split", wraps=load_split) as loader:
                manifest = prepare_data(settings, save_arrays=True)
                loader.assert_called_once_with(settings, "train")
            self.assertFalse(settings.test_path.exists())
            self.assertFalse(settings.test_targets_path.exists())
            self.assertEqual(manifest["scaler_fit_engine_ids"], manifest["training_engine_ids"])
            self.assertFalse(set(manifest["training_engine_ids"]) & set(manifest["validation_engine_ids"]))
            output = settings.artifact_dir / "prepared/FD001"
            saved = np.load(output / "validation_windows.npz", allow_pickle=False)
            valid = frame[frame.unit_id.isin(manifest["validation_engine_ids"])]
            inference = prepare_inference(valid, output / "manifest.json")
            np.testing.assert_allclose(inference.x, saved["x"][-1:])
            self.assertIsNone(inference.y)
            # Changes restricted to validation must not affect fitted state.
            original_state = (output / "scaler.json").read_text()
            frame.loc[frame.unit_id.isin(manifest["validation_engine_ids"]), "sensor_1"] += 1000
            frame.to_csv(settings.train_path, sep=" ", header=False, index=False)
            prepare_data(settings)
            self.assertEqual(original_state, (output / "scaler.json").read_text())
            # Same saved scaler, past prefix: last historical window must match.
            prefix = valid[valid.cycle <= 4]
            pred_input = prepare_inference(prefix, output / "manifest.json")
            np.testing.assert_allclose(pred_input.x, saved["x"][1:2])
            with self.assertRaises(ValueError):
                prepare_inference(valid[valid.cycle >= 2], output / "manifest.json")
