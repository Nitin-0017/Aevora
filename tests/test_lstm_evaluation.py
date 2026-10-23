from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from aevora.config import load_config
from aevora.lstm_training import train_lstm
from aevora.lstm_evaluation import evaluate_lstm
from aevora.prepare import prepare_data
from test_labels import histories


class LSTMEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        default = load_config(Path(__file__).resolve().parents[1] / 'configs/default.toml')
        models = {k: dict(v) for k, v in default.models.items()}
        models['lstm'].update(hidden_size=4, max_epochs=2, batch_size=4)
        self.settings = replace(default, raw_data_dir=root / 'raw', artifact_dir=root / 'out',
                                sequence_window=3, models=models)
        self.settings.raw_data_dir.mkdir()
        histories((8,) * 5).to_csv(self.settings.train_path, sep=' ', header=False, index=False)
        prepare_data(self.settings, save_arrays=True)
        self.report = train_lstm(self.settings)

    def test_report_matches_checkpoint_and_plots_exist(self):
        report = evaluate_lstm(self.settings)
        self.assertEqual(report['observations'], 6)
        self.assertEqual(report['engines'], 1)
        self.assertAlmostEqual(report['mae'], self.report['mae'])
        self.assertAlmostEqual(report['rmse'], self.report['rmse'])
        output = self.settings.artifact_dir / 'reports/FD001/lstm_validation'
        for name in ['training_curves.png', 'actual_vs_predicted.png', 'residual_distributions.png',
                     'absolute_error_by_engine.png', 'errors_by_engine.csv', 'errors_by_rul_band.csv']:
            self.assertGreater((output / name).stat().st_size, 0)

    def test_reject_corrupt_predictions_and_preparation_provenance(self):
        directory = self.settings.artifact_dir / 'models/FD001/lstm'
        path = directory / 'validation_predictions.csv'
        original = path.read_text()
        frame = pd.read_csv(path)
        frame['rul'] += 1
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'alignment'):
            evaluate_lstm(self.settings)
        path.write_text(original)
        metadata = json.loads((directory / 'metadata.json').read_text())
        metadata['checkpoint_sha256'] = 'bad'
        (directory / 'metadata.json').write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            evaluate_lstm(self.settings)

    def test_four_model_comparison_has_identical_counts_and_correct_metrics(self):
        from aevora.evaluation import compare_models, MODELS
        from aevora.training import train_model, metrics
        self.settings.models['gradient_boosting']['max_iter'] = 3
        for name in MODELS:
            train_model(self.settings, name)
        report = compare_models(self.settings, include_lstm=True)
        self.assertEqual([row['model'] for row in report['models']], [*MODELS, 'lstm'])
        for row in report['models']:
            self.assertEqual((row['observations'], row['engines']), (6, 1))
            path = self.settings.artifact_dir / 'models/FD001' / row['model'] / 'validation_predictions.csv'
            predictions = pd.read_csv(path)
            score = metrics(predictions.rul.to_numpy(), predictions.predicted_rul.to_numpy())
            self.assertAlmostEqual(row['mae'], score['mae'])
            self.assertAlmostEqual(row['rmse'], score['rmse'])
        self.assertEqual(report['models'][-1]['timing_scope'], 'epoch training plus validation')
        self.assertFalse((self.settings.artifact_dir / 'reports/FD001/validation').exists())
        self.assertTrue((self.settings.artifact_dir / 'reports/FD001/validation_with_lstm/metrics.json').exists())
