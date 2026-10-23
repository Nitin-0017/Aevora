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

    def test_rul_boundaries_counts_and_empty_bands(self):
        from aevora.reliability import grouped_errors
        frame = pd.DataFrame({'unit_id': [1]*5, 'cycle': [20, 50, 51, 100, 101],
                              'rul': [0., 30., 31., 100., 101.], 'predicted_rul': [1., 31., 32., 101., 102.]})
        groups = grouped_errors({'model': frame})
        self.assertEqual(groups.observations.tolist(), [2, 2, 1, 2, 2, 1])
        self.assertEqual(groups.mean_signed_error.tolist(), [1.] * 6)
        empty = grouped_errors({'model': frame.iloc[:1]})
        self.assertTrue(pd.isna(empty.iloc[1].mae))
        self.assertEqual(empty.iloc[1].engines, 0)

    def test_tables_preserve_alignment_and_rank_extremes(self):
        from aevora.reliability import build_tables
        frames = {'gradient_boosting': example(), 'other': example()}
        result = build_tables(frames)
        self.assertEqual(result['largest_errors'].iloc[0].cycle, 21)
        self.assertEqual(result['per_engine'].last_observed_cycle.tolist(), [21, 20, 21, 20])
        frames['other'] = example().iloc[::-1]
        with self.assertRaisesRegex(ValueError, 'alignment'):
            build_tables(frames)
