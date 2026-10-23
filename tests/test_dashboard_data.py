import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from aevora.dashboard.data import risk_state, engine_history, read_json, ArtifactError, DEMO_LABEL, VALIDATION_LABEL


class DashboardDataTests(unittest.TestCase):
    def test_risk_boundaries_and_raw_output(self):
        for value, expected in [(0,'high'),(30,'high'),(30.1,'elevated'),(100,'elevated'),(100.1,'neutral'),(-2,'invalid')]:
            result=risk_state(value)
            self.assertEqual(result['level'],expected)
            self.assertEqual(result['raw_prediction'],value)
            self.assertEqual(result['display_prediction'],value)
        self.assertEqual(risk_state(10,5,10)['level'],'elevated')
        for value in (None,np.nan,np.inf,-np.inf):
            self.assertIsNone(risk_state(value)['display_prediction'])
        with self.assertRaises(ValueError): risk_state(1,30,20)

    def test_engine_and_cycle_filter(self):
        raw=pd.DataFrame({'unit_id':[1,1,2], 'cycle':[2,1,1], 'sensor_1':[5.,4.,8.]})
        result=engine_history(raw,1,1)
        self.assertEqual(result.sensor_1.tolist(),[4.])
        self.assertEqual(len(raw),3)
        with self.assertRaises(ValueError): engine_history(raw,99,1)
        with self.assertRaises(ValueError): engine_history(raw,1,0)

    def test_missing_artifact_and_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ArtifactError,'Restore'):
                read_json(Path(tmp)/'missing.json')
        self.assertIn('not live',DEMO_LABEL)
        self.assertIn('used for model selection',VALIDATION_LABEL)
