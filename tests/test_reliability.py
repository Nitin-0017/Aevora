import unittest
import numpy as np
import pandas as pd
from aevora.reliability import describe_errors, engine_metrics, engine_variability


def example():
    return pd.DataFrame({'unit_id': [1, 1, 2], 'cycle': [20, 21, 20],
                         'rul': [10., 5., 0.], 'predicted_rul': [12., 2., 0.]})


class ReliabilityTests(unittest.TestCase):
    def test_signed_metrics_known_example(self):
        result = describe_errors(example())
        self.assertAlmostEqual(result['mae'], 5/3)
        self.assertAlmostEqual(result['rmse'], np.sqrt(13/3))
        self.assertAlmostEqual(result['mean_signed_error'], -1/3)
        self.assertEqual(result['median_signed_error'], 0.)
        self.assertEqual(result['max_absolute_error'], 3.)
        for key in ('underprediction_fraction', 'overprediction_fraction', 'exact_fraction'):
            self.assertAlmostEqual(result[key], 1/3)

    def test_empty_single_and_nonfinite_groups(self):
        empty = describe_errors(example().iloc[:0])
        self.assertEqual(empty['observations'], 0)
        self.assertIsNone(empty['mae'])
        self.assertIsNone(describe_errors(example().iloc[:1])['residual_std'])
        for bad in (example().assign(predicted_rul=np.nan), example().assign(rul=np.inf),
                    pd.concat([example(), example().iloc[:1]]), example().assign(rul=-1)):
            with self.assertRaises(ValueError):
                describe_errors(bad)

    def test_engine_grouping_and_benchmark_differences(self):
        frame = example()
        result = engine_metrics({'gradient_boosting': frame, 'other': frame.assign(predicted_rul=frame.rul)})
        self.assertEqual(result.observations.tolist(), [2, 1, 2, 1])
        self.assertEqual(result.mae_minus_boosted.tolist(), [0, 0, -2.5, 0])
        spread = engine_variability(result)
        self.assertEqual(spread.iloc[0].best_engine, 2)
        self.assertEqual(spread.iloc[0].worst_engine, 1)
        self.assertEqual(spread.iloc[0].engine_mae_std, 1.25)
