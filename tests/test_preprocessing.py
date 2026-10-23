import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from aevora.preprocessing import FeatureScaler, select_sensors, split_engines
from test_labels import histories


class PreprocessingTests(unittest.TestCase):
    def test_split_disjoint_reproducible_and_order_independent(self):
        frame = histories((3,) * 10)
        train, valid = split_engines(frame, seed=42, validation_fraction=.2)
        train2, valid2 = split_engines(frame.iloc[::-1].sort_values(["unit_id", "cycle"], ascending=[False, True]), seed=42, validation_fraction=.2)
        self.assertEqual(train.unit_id.nunique(), 8)
        self.assertEqual(valid.unit_id.nunique(), 2)
        self.assertFalse(set(train.unit_id) & set(valid.unit_id))
        self.assertEqual(set(valid.unit_id), set(valid2.unit_id))
        self.assertEqual(set(train.unit_id), set(train2.unit_id))
        self.assertEqual(len(train) + len(valid), len(frame))

    def test_fit_boundaries_column_order_and_roundtrip(self):
        train = histories((3,))
        train["sensor_2"] = 9.
        valid = histories((3,))
        valid["unit_id"] = 2
        valid["sensor_1"] += 1000
        selected = select_sensors(train)
        self.assertNotIn("sensor_2", selected)
        scaler = FeatureScaler.fit(train, selected, expected_training_ids=[1])
        state = repr(scaler)
        self.assertAlmostEqual(scaler.means[0], 2)
        output = scaler.transform(valid[valid.columns[::-1]])
        self.assertGreater(output.sensor_1.min(), 100)
        self.assertEqual(repr(scaler), state)
        self.assertEqual(output.unit_id.tolist(), [2, 2, 2])
        self.assertEqual(output.cycle.tolist(), [1, 2, 3])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scaler.json"
            scaler.save(path)
            pd.testing.assert_frame_equal(output, FeatureScaler.load(path).transform(valid))
        with self.assertRaises(ValueError):
            FeatureScaler.fit(pd.concat([train, valid]), selected, expected_training_ids=[1])

    def test_invalid_transform_and_identifiers(self):
        frame = histories((3,))
        scaler = FeatureScaler.fit(frame, ["sensor_1"], expected_training_ids=[1])
        for bad in [frame.drop(columns="sensor_1"), frame.assign(sensor_1=np.nan), frame.assign(cycle=0)]:
            with self.assertRaises(ValueError):
                scaler.transform(bad)
        for features in [["rul"], ["unit_id"], ["sensor_1", "sensor_1"]]:
            with self.assertRaises(ValueError):
                FeatureScaler.fit(frame, features, expected_training_ids=[1])
        with self.assertRaises(ValueError):
            split_engines(frame, seed=42, validation_fraction=.2)

    def test_zero_variance_scale_is_finite(self):
        frame = histories((3,)).assign(sensor_1=5.)
        scaler = FeatureScaler.fit(frame, ["sensor_1"], expected_training_ids=[1])
        np.testing.assert_array_equal(scaler.transform(frame).sensor_1, [0, 0, 0])
