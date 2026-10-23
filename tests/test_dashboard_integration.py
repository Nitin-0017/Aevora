"""Read-only checks over local artifacts; skip explicitly on source-only clones."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from aevora.audit import input_fingerprints
from aevora.dashboard.data import (load_snapshot,load_predictor,fleet_predictions,engine_history,
                                  infer_history,check_saved_metrics,ArtifactError,cache_token)
from aevora.prepare import prepare_inference

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/default.toml'


class DashboardIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (ROOT/'artifacts/reports/FD001/performance_audit/audit.json').exists():
            raise unittest.SkipTest('Matching local model/data/audit bundle required; no training/download fallback')
        original_open=Path.open
        def guarded_open(path,*args,**kwargs):
            if path.name.startswith(('RUL_FD001','test_FD001')):
                raise AssertionError('Official test access forbidden')
            return original_open(path,*args,**kwargs)
        with patch.object(Path,'open',guarded_open), \
             patch('aevora.preprocessing.FeatureScaler.fit',side_effect=AssertionError('Scaler refit forbidden')), \
             patch('sklearn.linear_model.Ridge.fit',side_effect=AssertionError('Model training forbidden')), \
             patch('sklearn.ensemble.HistGradientBoostingRegressor.fit',side_effect=AssertionError('Model training forbidden')), \
             patch('torch.optim.AdamW.step',side_effect=AssertionError('Neural training forbidden')):
            cls.snapshot=load_snapshot(CONFIG)
        cls.before=input_fingerprints(cls.snapshot.settings)

    def test_verified_demo_and_audit_alignment(self):
        s=self.snapshot
        self.assertEqual(s.raw.unit_id.nunique(),20)
        self.assertFalse(set(s.raw.unit_id)&set(s.manifest['training_engine_ids']))
        self.assertTrue((s.metrics.observations==3472).all())
        self.assertTrue((s.metrics.engines==20).all())
        self.assertAlmostEqual(s.metrics.set_index('model').loc['gradient_boosting','mae'],20.5357146059923)

    def test_all_saved_models_reproduce_replay_endpoints(self):
        s=self.snapshot
        for name in s.predictions:
            with self.subTest(model=name):
                predictor=load_predictor(s,name)
                first=fleet_predictions(s,name,predictor,100)
                second=fleet_predictions(s,name,load_predictor(s,name),100)
                np.testing.assert_array_equal(first.predicted_rul,second.predicted_rul)
                saved=s.predictions[name].query('cycle == 100').set_index('unit_id')
                np.testing.assert_allclose(first.predicted_rul,saved.loc[first.unit_id,'predicted_rul'],atol=1e-4,rtol=0)
                self.assertEqual(len(first),20)
        self.assertEqual(self.before,input_fingerprints(s.settings))

    def test_sequence_schema_short_history_and_nan(self):
        s=self.snapshot
        history=engine_history(s.raw,3,100)
        directory=s.settings.artifact_dir/'models/FD001/gradient_boosting'
        batch=prepare_inference(history,directory/'preprocessing/manifest.json')
        self.assertEqual(batch.x.shape,(1,20,75))
        for name in s.predictions:
            predictor=load_predictor(s,name)
            with self.subTest(model=name):
                with self.assertRaises(ValueError): predictor.predict(batch.x,s.manifest['feature_names'][::-1])
                with self.assertRaises(ValueError): predictor.predict(batch.x[:,:19],s.manifest['feature_names'])
                with self.assertRaises(ValueError): infer_history(s,name,predictor,history.iloc[:19])
        bad=history.copy()
        bad.loc[bad.index[0],s.manifest['selected_sensors'][0]]=np.nan
        with self.assertRaises(ValueError): infer_history(s,'gradient_boosting',load_predictor(s,'gradient_boosting'),bad)

    def test_stale_metrics_rejected(self):
        s=self.snapshot
        report={'tables':{'metrics':s.metrics.to_dict('records')}}
        report['tables']['metrics'][0]['mae']+=1
        with self.assertRaisesRegex(ArtifactError,'disagrees'):
            check_saved_metrics(s.predictions,report)

    def test_missing_model_and_bad_feature_metadata(self):
        s=self.snapshot
        with patch('aevora.dashboard.data.read_json',side_effect=ArtifactError('Missing local artifact')):
            with self.assertRaises(ArtifactError): load_predictor(s,'ridge')
        bad=deepcopy(s.manifest)
        bad['feature_names']=bad['feature_names'][::-1]
        with patch('aevora.dashboard.data.read_json',return_value=bad):
            with self.assertRaisesRegex(ArtifactError,'feature order'):
                load_predictor(s,'ridge')
        with self.assertRaises(ValueError): load_predictor(s,'unknown')

    def test_cache_inventory_does_not_reference_test_files(self):
        token=cache_token(CONFIG)
        self.assertFalse(any(Path(row[0]).name.startswith(('test_FD001','RUL_FD001')) for row in token))
