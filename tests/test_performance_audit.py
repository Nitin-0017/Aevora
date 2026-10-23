"""Analytical checks for extended audit metrics; no model training."""
import unittest
import numpy as np
import pandas as pd
from aevora.performance_audit import regression_summary, improvement, summarize_frames


class PerformanceAuditTests(unittest.TestCase):
    def test_known_metrics_and_inclusive_tolerances(self):
        result = regression_summary([0, 10, 20, 30], [5, 0, 40, 0])
        self.assertEqual(result['mae'], 16.25)
        self.assertAlmostEqual(result['rmse'], np.sqrt(1425 / 4))
        self.assertAlmostEqual(result['r2'], 1 - 1425 / 500)
        self.assertEqual(result['mean_signed_error'], -3.75)
        self.assertEqual(result['median_absolute_error'], 15)
        self.assertEqual(result['max_absolute_error'], 30)
        self.assertEqual(result['p90_absolute_error'], 27)
        self.assertAlmostEqual(result['p95_absolute_error'], 28.5)
        for tolerance, percentage in zip((5, 10, 20, 30), (25, 50, 75, 100)):
            self.assertEqual(result[f'within_{tolerance}_pct'], percentage)

    def test_empty_constant_singleton(self):
        self.assertIsNone(regression_summary([], [])['mae'])
        self.assertIsNone(regression_summary([], [])['within_10_pct'])
        self.assertIsNone(regression_summary([2], [2])['r2'])
        self.assertIsNone(regression_summary([2, 2], [2, 2])['r2'])
        self.assertEqual(regression_summary([1, 2], [1, 2])['r2'], 1)

    def test_invalid_predictions_are_not_silently_dropped(self):
        result = regression_summary([0, 0, 0, 0], [-1, np.nan, np.inf, -np.inf])
        self.assertEqual(result['negative_pct'], 25)
        self.assertEqual(result['nonfinite_pct'], 75)
        self.assertEqual(result['within_5_pct'], 25)
        self.assertIsNone(result['mae'])
        with self.assertRaises(ValueError):
            regression_summary([np.nan], [1])
        with self.assertRaises(ValueError):
            regression_summary([1, 2], [1])

    def test_comparison_alignment_group_coverage_and_improvements(self):
        def frame(offset):
            return pd.DataFrame({'unit_id': [1, 1, 2], 'cycle': [20, 21, 20],
                                 'rul': [0., 31., 101.], 'predicted_rul': np.array([0., 31., 101.]) + offset})
        frames = {'baseline': frame(10), 'ridge': frame(5), 'gradient_boosting': frame(2), 'lstm': frame(4)}
        tables = summarize_frames(frames)
        scores = tables['metrics'].set_index('model')
        self.assertEqual(scores.loc['gradient_boosting', 'mae_improvement_vs_baseline_pct'], 80)
        self.assertEqual(scores.loc['gradient_boosting', 'best_alternative'], 'lstm')
        self.assertEqual(scores.loc['gradient_boosting', 'mae_improvement_vs_best_alternative_pct'], 50)
        self.assertEqual(scores.loc['lstm', 'mae_improvement_vs_best_alternative_pct'], -100)
        self.assertIsNone(improvement(0, 0))
        self.assertTrue((tables['rul_bands'].groupby('model').observations.sum() == 3).all())
        self.assertTrue((tables['per_engine'].groupby('model').observations.sum() == 3).all())
        frames['ridge'] = frames['ridge'].iloc[::-1]
        with self.assertRaises(ValueError):
            summarize_frames(frames)
