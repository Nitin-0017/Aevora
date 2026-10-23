from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd
from aevora.classical import Predictor, predict_history
from aevora.config import load_config
from aevora.evaluation import align_predictions, error_tables, compare_models, MODELS
from aevora.prepare import prepare_data
from aevora.training import load_prepared, train_model
from test_labels import histories


class EvaluationTests(unittest.TestCase):
    def test_alignment_rejects_reordered_or_changed_targets(self):
        reference = pd.DataFrame({"unit_id": [1, 1], "cycle": [3, 4], "rul": [1., 0.]})
        candidate = reference.assign(predicted_rul=[1., 0.])
        align_predictions(reference, candidate)
        for bad in [candidate.iloc[::-1], candidate.assign(rul=[0., 1.]),
                    candidate.iloc[:1], pd.concat([candidate, candidate.iloc[:1]])]:
            with self.assertRaisesRegex(ValueError, "alignment"):
                align_predictions(reference, bad)

    def test_bands_and_engine_errors(self):
        frame = pd.DataFrame({"unit_id": [1, 1, 2, 2, 2], "rul": [0., 30., 31., 100., 101.],
                              "predicted_rul": [1., 31., 33., 102., 104.]})
        engines, bands = error_tables(frame)
        self.assertEqual([b["observations"] for b in bands], [2, 2, 1])
        self.assertEqual([b["mae"] for b in bands], [1, 2, 3])
        self.assertEqual([e["observations"] for e in engines], [2, 3])
        self.assertIsNone(error_tables(frame.iloc[:1])[1][1]["mae"])

    def test_all_models_same_endpoints_portable_inference_and_report(self):
        default = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_config = {k: dict(v) for k, v in default.models.items()}
            model_config["gradient_boosting"]["max_iter"] = 3
            settings = replace(default, raw_data_dir=root / "raw", artifact_dir=root / "out",
                               sequence_window=3, models=model_config)
            settings.raw_data_dir.mkdir()
            frame = histories((12,) * 10)
            frame.to_csv(settings.train_path, sep=" ", header=False, index=False)
            manifest = prepare_data(settings, save_arrays=True)
            prepared = load_prepared(settings)
            for name in MODELS:
                train_model(settings, name, prepared)
                source = settings.artifact_dir / "models/FD001" / name
                relocated = root / "relocated" / name
                shutil.copytree(source, relocated)
                # One available prefix per validation engine, never future rows.
                prefix = frame[frame.unit_id.isin(manifest["validation_engine_ids"]) & (frame.cycle <= 8)]
                actual = predict_history(relocated, prefix)
                batch = prepared[1]["validation"]
                mask = batch.endpoints.cycle == 8
                expected = Predictor.load(source).predict(batch.x[mask], manifest["feature_names"])
                np.testing.assert_allclose(actual.predicted_rul, expected, rtol=0, atol=1e-12)
                self.assertNotIn(str(root), (relocated / "preprocessing/manifest.json").read_text())
            report = compare_models(settings)
            self.assertEqual([m["observations"] for m in report["models"]], [20, 20, 20])
            output = settings.artifact_dir / "reports/FD001/validation"
            for filename in ("metrics.json", "metrics.csv", "errors_by_engine.csv", "errors_by_rul_band.csv",
                             "actual_vs_predicted.png", "residual_distributions.png", "absolute_error_by_engine.png"):
                self.assertGreater((output / filename).stat().st_size, 0)
            self.assertFalse(settings.test_path.exists())
            self.assertFalse(settings.test_targets_path.exists())
            path = settings.artifact_dir / "prepared/FD001/manifest.json"
            manifest["validation_engine_ids"] = manifest["training_engine_ids"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_prepared(settings)
