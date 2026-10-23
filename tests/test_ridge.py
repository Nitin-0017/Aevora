from pathlib import Path
import tempfile
import unittest
import numpy as np
from aevora.classical import Predictor, summarize
from aevora.config import load_config
from aevora.training import candidates


class RidgeTests(unittest.TestCase):
    def test_search_and_reproducible_predictions(self):
        settings = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
        models = candidates("ridge", settings)
        self.assertEqual([model.alpha for model in models], [1, 10, 100])
        x = np.random.default_rng(42).normal(size=(40, 3, 2))
        y = 10 + 2 * x[:, -1, 0]
        vector = summarize(x, ["a", "b"], 3)
        for estimator in models:
            estimator.fit(vector, y)
            model = Predictor(estimator, ["a", "b"], 3)
            predictions = model.predict(x, ["a", "b"])
            self.assertEqual(predictions.shape, (40,))
            with tempfile.TemporaryDirectory() as tmp:
                model.save(tmp)
                np.testing.assert_array_equal(predictions, Predictor.load(tmp).predict(x, ["a", "b"]))
            estimator.fit(vector, y)
            np.testing.assert_array_equal(predictions, model.predict(x, ["a", "b"]))
