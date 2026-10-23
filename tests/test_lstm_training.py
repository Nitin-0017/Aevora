from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from aevora.config import load_config
from aevora.lstm import LSTMPredictor, predict_history
from aevora.lstm_training import EarlyStopping, fit_lstm, train_lstm
from aevora.prepare import prepare_data
from aevora.sequences import Windows
from test_labels import histories


def small_batches():
    rng = np.random.default_rng(42)
    return [Windows(rng.normal(size=(8, 3, 2)).astype('float32'), np.arange(8, dtype=float),
                    pd.DataFrame({'unit_id': [unit] * 8, 'cycle': np.arange(3, 11)}), []) for unit in (1, 2)]


class LSTMTrainingTests(unittest.TestCase):
    def test_early_stop_snapshot_restoration_and_ties(self):
        model = torch.nn.Linear(1, 1)
        stop = EarlyStopping(2, min_delta=.1)
        with torch.no_grad():
            model.weight.fill_(1)
        self.assertFalse(stop.update(5., model, 1))
        with torch.no_grad():
            model.weight.fill_(2)
        self.assertFalse(stop.update(4., model, 2))
        with torch.no_grad():
            model.weight.fill_(9)
        self.assertFalse(stop.update(4., model, 3))
        self.assertTrue(stop.update(3.95, model, 4))
        stop.restore(model)
        self.assertEqual(model.weight.item(), 2.)
        self.assertEqual(stop.best_epoch, 2)
        with self.assertRaises(ValueError):
            stop.update(float('nan'), model, 5)

    def test_deterministic_training_and_overlap_rejection(self):
        config = dict(load_config(Path(__file__).resolve().parents[1] / 'configs/default.toml').models['lstm'])
        config.update(hidden_size=4, max_epochs=3, batch_size=4)
        train, valid = small_batches()
        a, history_a, _ = fit_lstm(train, valid, ['a', 'b'], config, 42)
        b, history_b, _ = fit_lstm(train, valid, ['a', 'b'], config, 42)
        self.assertEqual(history_a, history_b)
        np.testing.assert_array_equal(a.predict(valid.x, ['a', 'b']), b.predict(valid.x, ['a', 'b']))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            fit_lstm(train, train, ['a', 'b'], config, 42)

    def test_loop_restores_best_and_stops(self):
        config = dict(load_config(Path(__file__).resolve().parents[1] / 'configs/default.toml').models['lstm'])
        config.update(hidden_size=4, max_epochs=8, patience=2, batch_size=4)
        train, valid = small_batches()
        snapshots = []
        original = EarlyStopping.update
        def record(stop, loss, model, epoch):
            snapshots.append({k: v.clone() for k, v in model.state_dict().items()})
            return original(stop, loss, model, epoch)
        # Force losses 0,1,4 to test the actual training loop's stop/restore wiring.
        with patch.object(LSTMPredictor, 'predict', side_effect=[valid.y, valid.y + 1, valid.y + 2]), \
             patch.object(EarlyStopping, 'update', record):
            predictor, history, outcome = fit_lstm(train, valid, ['a', 'b'], config, 42)
        self.assertEqual(len(history), 3)
        self.assertTrue(outcome['early_stopped'])
        self.assertEqual(outcome['best_epoch'], 1)
        for key, value in predictor.model.state_dict().items():
            torch.testing.assert_close(value, snapshots[0][key], rtol=0, atol=0)

    def test_training_bundle_inference_and_training_only_access(self):
        default = load_config(Path(__file__).resolve().parents[1] / 'configs/default.toml')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = {k: dict(v) for k, v in default.models.items()}
            models['lstm'].update(hidden_size=4, max_epochs=2, batch_size=4)
            settings = replace(default, raw_data_dir=root / 'raw', artifact_dir=root / 'out',
                               sequence_window=3, models=models)
            settings.raw_data_dir.mkdir()
            frame = histories((8,) * 5)
            frame.to_csv(settings.train_path, sep=' ', header=False, index=False)
            manifest = prepare_data(settings, save_arrays=True)
            from aevora.training import load_split
            with patch('aevora.training.load_split', wraps=load_split) as loader:
                report = train_lstm(settings)
                loader.assert_called_once_with(settings, 'train')
            self.assertEqual(report['validation_shape'], [6, 3, 105])
            self.assertFalse(settings.test_targets_path.exists())
            directory = settings.artifact_dir / 'models/FD001/lstm'
            valid = frame[frame.unit_id.isin(manifest['validation_engine_ids']) & (frame.cycle <= 6)]
            result = predict_history(directory, valid)
            with np.load(settings.artifact_dir / 'prepared/FD001/validation_windows.npz') as saved:
                expected = LSTMPredictor.load(directory).predict(saved['x'][saved['cycle'] == 6], manifest['feature_names'])
            np.testing.assert_allclose(result.predicted_rul, expected, rtol=1e-6, atol=1e-5)
            scaler = directory / 'preprocessing/scaler.json'
            scaler.write_text(scaler.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                LSTMPredictor.load(directory)
