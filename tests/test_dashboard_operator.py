import unittest
import numpy as np
from aevora.dashboard.operator import operator_summary,operating_hours,RUL_EXPLANATION,GUIDANCE_LABEL


class OperatorPresentationTests(unittest.TestCase):
    def test_status_boundaries_and_recommendations(self):
        for value,status,action in [(0,'Needs review','Schedule'),(30,'Needs review','Schedule'),
            (30.001,'Monitor','routine monitoring'),(100,'Monitor','routine monitoring'),
            (100.001,'Within demonstration range','not a safety clearance')]:
            result=operator_summary(value)
            self.assertEqual(result['status'],status)
            self.assertIn(action,result['action'])
            self.assertTrue(result['icon'])
        self.assertEqual(operator_summary(6,5,10)['status'],'Monitor')
        with self.assertRaises(ValueError): operator_summary(1,10,5)

    def test_invalid_is_never_reassuring(self):
        for value in (None,np.nan,np.inf,-np.inf,-.001,-100):
            result=operator_summary(value)
            self.assertEqual(result['status'],'Unable to assess')
            self.assertIsNone(result['display_prediction'])
            self.assertEqual(result['formatted_cycles'],'Unavailable')
            self.assertIn('Do not rely',result['action'])
        self.assertEqual(operator_summary(-.001)['raw_prediction'],-.001)

    def test_formatting_does_not_change_raw_or_boundaries(self):
        result=operator_summary(30.001)
        self.assertEqual(result['formatted_cycles'],'30.00')
        self.assertEqual(result['raw_prediction'],30.001)
        self.assertEqual(result['status'],'Monitor')
        self.assertEqual(result['display_prediction'],30.001)

    def test_hours_require_supplied_valid_rate(self):
        self.assertEqual(operating_hours(100,20),5)
        self.assertEqual(operating_hours(0,20),0)
        self.assertIsNone(operating_hours(100))
        for prediction in (None,np.nan,np.inf,-1):
            self.assertIsNone(operating_hours(prediction,20))
        for rate in (0,-1,np.nan,np.inf,True):
            with self.assertRaises(ValueError): operating_hours(100,rate)

    def test_explanations_are_decision_support(self):
        self.assertIn('operating cycles',RUL_EXPLANATION)
        self.assertIn("model's estimate",RUL_EXPLANATION)
        self.assertIn('not a validated operational procedure',GUIDANCE_LABEL)
