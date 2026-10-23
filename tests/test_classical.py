from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sklearn.dummy import DummyRegressor
from aevora.classical import Predictor, summarize, predict_history
from aevora.config import load_config
from aevora.prepare import prepare_data
from aevora.training import load_prepared, train_model, metrics
from test_labels import histories


class ClassicalTests(unittest.TestCase):
    def test_summary_order_and_invalid_inputs(self):
        x = np.arange(12).reshape(2, 3, 2)
        actual = summarize(x, ["a", "b"], 3)
        self.assertEqual(actual.shape, (2, 6))
        np.testing.assert_allclose(actual[0], [4, 5, 2, 3, np.std([0, 2, 4]), np.std([1, 3, 5])])
        for bad in (x[:, :2], x.reshape(2, 6), x * np.nan, x[:0]):
            with self.assertRaises(ValueError):
                summarize(bad, ["a", "b"], 3)

    def test_median_reload_and_schema(self):
        x = np.arange(12).reshape(2, 3, 2)
        estimator = DummyRegressor(strategy="median").fit(np.ones((3, 6)), [0, 4, 100])
        model = Predictor(estimator, ["a", "b"], 3)
        np.testing.assert_array_equal(model.predict(x, ["a", "b"]), [4, 4])
        with self.assertRaisesRegex(ValueError, "order"):
            model.predict(x, ["b", "a"])
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp)
            np.testing.assert_array_equal(Predictor.load(tmp).predict(x, ["a", "b"]), [4, 4])

    def test_metrics_reject_misalignment(self):
        self.assertEqual(metrics([0, 2], [1, 1]), {"mae": 1., "rmse": 1.})
        for bad in ([1], [[1], [2]], [1, np.nan]):
            with self.assertRaises(ValueError):
                metrics([0, 2], bad)

    def test_verified_artifacts_inference_and_corrupt_target(self):
        settings = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
        with tempfile.TemporaryDirectory() as tmp:
            settings = replace(settings, raw_data_dir=Path(tmp) / "raw", artifact_dir=Path(tmp) / "out", sequence_window=3)
            settings.raw_data_dir.mkdir()
            frame = histories((5,) * 5)
            frame.to_csv(settings.train_path, sep=" ", header=False, index=False)
            prepare_data(settings, save_arrays=True)
            manifest, batches = load_prepared(settings)
            report = train_model(settings, "baseline", (manifest, batches))
            self.assertEqual(report["validation_observations"], 3)
            output = settings.artifact_dir / "models/FD001/baseline"
            pred = predict_history(output, frame[frame.unit_id.isin(manifest["validation_engine_ids"])])
            self.assertEqual(len(pred), 1)
            self.assertEqual(pred.predicted_rul.iloc[0], np.median(batches["train"].y))
            self.assertFalse(settings.test_path.exists())
            self.assertFalse(settings.test_targets_path.exists())
            path = settings.artifact_dir / "prepared/FD001/validation_windows.npz"
            with np.load(path) as a:
                values = dict(a)
            values["y"] = values["y"][::-1]
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, "alignment"):
                load_prepared(settings)
