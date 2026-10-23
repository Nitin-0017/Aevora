import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from aevora.config import load_config
from aevora.lstm import RULLSTM, LSTMPredictor


class LSTMTests(unittest.TestCase):
    def test_shapes_finite_predictions_and_fixed_initialization(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = RULLSTM(3, hidden_size=4, target_mean=50, target_scale=10)
        x = np.random.default_rng(1).normal(size=(7, 20, 3)).astype('float32')
        p = LSTMPredictor(model, ['a', 'b', 'c'], 20, batch_size=3)
        result = p.predict(x, p.features)
        self.assertEqual(result.shape, (7,))
        self.assertTrue(np.isfinite(result).all())
        torch.manual_seed(42)
        other = LSTMPredictor(RULLSTM(3, 4, target_mean=50, target_scale=10), p.features, 20, batch_size=3)
        np.testing.assert_array_equal(result, other.predict(x, p.features))
        for bad in [x[:, :19], x[:, :, :2], x[:0], x * np.nan, x.reshape(7, 60)]:
            with self.assertRaises(ValueError):
                p.predict(bad, p.features)
        with self.assertRaisesRegex(ValueError, 'order'):
            p.predict(x, p.features[::-1])

    def test_checkpoint_round_trip_and_schema_tampering(self):
        torch.manual_seed(42)
        predictor = LSTMPredictor(RULLSTM(2, 4, target_mean=8, target_scale=3), ['a', 'b'], 3)
        x = np.ones((5, 3, 2), dtype='float32')
        with tempfile.TemporaryDirectory() as tmp:
            predictor.save(tmp)
            np.testing.assert_array_equal(predictor.predict(x, predictor.features),
                                          LSTMPredictor.load(tmp).predict(x, predictor.features))
            path = Path(tmp) / 'schema.json'
            schema = json.loads(path.read_text()); schema['features'].reverse()
            path.write_text(json.dumps(schema))
            with self.assertRaisesRegex(ValueError, 'schema'):
                LSTMPredictor.load(tmp)

    def test_config_rejects_invalid_training_controls(self):
        text = (Path(__file__).resolve().parents[1] / 'configs/default.toml').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.toml'
            for old, new in [('dropout = 0.1', 'dropout = 1'), ('gradient_clip = 1.0', 'gradient_clip = 0'),
                             ('min_delta = 0.0', 'min_delta = -1'), ('num_threads = 2', 'num_threads = 0')]:
                path.write_text(text.replace(old, new))
                with self.assertRaises(ValueError):
                    load_config(path)
