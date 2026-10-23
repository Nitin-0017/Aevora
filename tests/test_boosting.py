from pathlib import Path
import tempfile
import unittest
import numpy as np
from aevora.classical import Predictor, summarize
from aevora.config import load_config
from aevora.training import candidates


class BoostingTests(unittest.TestCase):
    def test_engine_safe_search_and_reload(self):
        settings = load_config(Path(__file__).resolve().parents[1] / "configs/default.toml")
        models = candidates("gradient_boosting", settings)
        self.assertEqual([m.max_leaf_nodes for m in models], [7, 15])
        x = np.random.default_rng(42).normal(size=(80, 3, 2))
        y = 10 + x[:, -1, 0] ** 2
        vector = summarize(x, ["a", "b"], 3)
        for estimator in models:
            self.assertFalse(estimator.early_stopping)
            self.assertEqual(estimator.random_state, 42)
            estimator.set_params(max_iter=5).fit(vector, y)
            model = Predictor(estimator, ["a", "b"], 3)
            prediction = model.predict(x, ["a", "b"])
            self.assertEqual(prediction.shape, (80,))
            with tempfile.TemporaryDirectory() as tmp:
                model.save(tmp)
                np.testing.assert_array_equal(prediction, Predictor.load(tmp).predict(x, ["a", "b"]))
            estimator.fit(vector, y)
            np.testing.assert_array_equal(prediction, model.predict(x, ["a", "b"]))
