from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from aevora.audit import run_audit, input_fingerprints
from aevora.audit_stability import perturb_sensors
from aevora.config import load_config
from aevora.prepare import prepare_data
from aevora.training import train_model
from aevora.lstm_training import train_lstm
from test_labels import histories


class AuditTests(unittest.TestCase):
    def test_noise_seed_scale_and_no_input_mutation(self):
        raw = histories((5, 5))
        before = raw.copy()
        a = perturb_sensors(raw, ['sensor_1'], [10.], 42)
        b = perturb_sensors(raw, ['sensor_1'], [10.], 42)
        pd.testing.assert_frame_equal(a, b)
        pd.testing.assert_frame_equal(raw, before)
        np.testing.assert_array_equal(a.sensor_2, raw.sensor_2)
        expected = np.random.default_rng(42).normal(size=(10, 1))[:, 0] * .1
        np.testing.assert_allclose(a.sensor_1 - raw.sensor_1, expected)
        np.testing.assert_array_equal(perturb_sensors(raw, ['sensor_1'], [10.], 42, fraction=0).sensor_1, raw.sensor_1)
        with self.assertRaises(ValueError):
            perturb_sensors(raw, ['sensor_1'], [0.], 42)

    def test_full_audit_reproducible_no_refit_or_test_access(self):
        default = load_config(Path(__file__).resolve().parents[1] / 'configs/default.toml')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = {k: dict(v) for k,v in default.models.items()}
            models['lstm'].update(hidden_size=4, max_epochs=1, batch_size=4)
            models['gradient_boosting']['max_iter'] = 2
            settings = replace(default, artifact_dir=root/'artifacts', raw_data_dir=root/'raw',
                               sequence_window=3, models=models)
            settings.raw_data_dir.mkdir()
            histories((8,)*5).to_csv(settings.train_path, sep=' ', header=False, index=False)
            prepare_data(settings, save_arrays=True)
            for name in ('baseline', 'ridge', 'gradient_boosting'):
                train_model(settings, name)
            train_lstm(settings)
            before = input_fingerprints(settings)
            from aevora.data.loader import load_split
            def train_only(config, split):
                self.assertEqual(split, 'train')
                return load_split(config, split)
            with patch('aevora.training.load_split', side_effect=train_only), \
                 patch('aevora.audit_stability.load_split', side_effect=train_only), \
                 patch('aevora.preprocessing.FeatureScaler.fit', side_effect=AssertionError('Refit forbidden')), \
                 patch('sklearn.linear_model.Ridge.fit', side_effect=AssertionError('Refit forbidden')), \
                 patch('sklearn.ensemble.HistGradientBoostingRegressor.fit', side_effect=AssertionError('Refit forbidden')), \
                 patch('torch.optim.AdamW.step', side_effect=AssertionError('Training forbidden')):
                first = run_audit(settings)
                output = settings.artifact_dir/'reports/FD001/reliability'
                json_before = (output/'audit.json').read_bytes()
                second = run_audit(settings)
            self.assertEqual(first, second)
            self.assertEqual(json_before, (output/'audit.json').read_bytes())
            self.assertEqual(before, input_fingerprints(settings))
            self.assertTrue(all(row['rejected'] for row in first['tables']['input_contracts']))
            self.assertEqual(first['tables']['stability'][0]['mean_absolute_prediction_change'], 0.)
            self.assertFalse(settings.test_path.exists())
            self.assertFalse(settings.test_targets_path.exists())
            for name in ('residuals_vs_rul.png', 'absolute_error_ecdf.png', 'errors_by_rul_band.png',
                         'largest_errors.png', 'error_vs_history_length.png'):
                self.assertGreater((output/name).stat().st_size, 0)
