import unittest
import numpy as np
import pandas as pd
from aevora.features import sensor_features
from test_labels import histories


class FeatureTests(unittest.TestCase):
    def test_hand_calculated_statistics_and_engine_boundaries(self):
        frame = histories((3, 2))
        frame["sensor_1"] = [1, 3, 5, 100, 104]
        out = sensor_features(frame.iloc[::-1], ["sensor_1"], 3)
        np.testing.assert_allclose(out.sensor_1_mean_3, [1, 2, 3, 100, 102])
        np.testing.assert_allclose(out.sensor_1_std_3, [0, 1, np.sqrt(8/3), 0, 2])
        np.testing.assert_allclose(out.sensor_1_change_1, [0, 2, 2, 0, 4])
        np.testing.assert_allclose(out.sensor_1_trend_3, [0, 2, 2, 0, 4])
        self.assertEqual(out.shape, (5, 7))

    def test_future_changes_do_not_affect_past(self):
        frame = histories((6, 6))
        original = sensor_features(frame, ["sensor_1"], 3)
        changed = frame.copy()
        changed.loc[changed.cycle > 3, "sensor_1"] = 9000
        altered = sensor_features(changed, ["sensor_1"], 3)
        pd.testing.assert_frame_equal(original[original.cycle <= 3], altered[altered.cycle <= 3])
        prefix = sensor_features(frame[frame.cycle <= 3], ["sensor_1"], 3)
        pd.testing.assert_frame_equal(prefix, original[original.cycle <= 3].reset_index(drop=True))

    def test_invalid_history_and_window(self):
        frame = histories()
        for window in [0, 1, True, 2.5]:
            with self.assertRaises(ValueError):
                sensor_features(frame, ["sensor_1"], window)
        for bad in [frame.assign(sensor_1=np.inf), frame.drop(index=1), pd.concat([frame, frame.iloc[:1]])]:
            with self.assertRaises(ValueError):
                sensor_features(bad, ["sensor_1"])
        with self.assertRaises(ValueError):
            sensor_features(frame, ["cycle"])
