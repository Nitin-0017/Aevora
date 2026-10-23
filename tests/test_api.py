"""API contract/inference tests against existing local artifacts; no model training."""
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from aevora.api.app import create_app, CONFIG
from aevora.api.service import Service
from aevora.audit import input_fingerprints


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        if not (root/'artifacts/reports/FD001/performance_audit/audit.json').exists():
            raise unittest.SkipTest('Local audited bundle required; no training fallback')
        from aevora.data.loader import load_split
        def only_train(settings,split):
            assert split=='train','Official test access forbidden'
            return load_split(settings,split)
        with patch('aevora.training.load_split',side_effect=only_train),patch('aevora.dashboard.data.load_split',side_effect=only_train),patch('aevora.preprocessing.FeatureScaler.fit',side_effect=AssertionError('No refit')):
            cls.service=Service(CONFIG)
        cls.hashes=input_fingerprints(cls.service.snapshot.settings)

    def setUp(self):
        self.client=TestClient(create_app(self.service))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None,None,None)

    def test_health_engines_models(self):
        self.assertTrue(self.client.get('/api/health').json()['artifacts_ready'])
        self.assertEqual(len(self.client.get('/api/engines').json()['engines']),20)
        models=self.client.get('/api/models').json()['models']
        self.assertEqual(len(models),4)
        self.assertTrue(all(m['feature_count']==75 for m in models))

    def test_prediction_all_models_and_fleet(self):
        for model in ('baseline','ridge','gradient_boosting','lstm'):
            r=self.client.post('/api/predict',json={'engine_id':3,'model':model,'cycle':100})
            self.assertEqual(r.status_code,200,r.text)
            row=r.json()['predictions'][0]
            expected=self.service.snapshot.predictions[model].query('unit_id==3 and cycle==100').predicted_rul.iloc[0]
            self.assertAlmostEqual(row['raw_rul'],expected,places=4)
            self.assertEqual(row['latest_cycle'],100)
            self.assertIn('Not a guaranteed',row['warning'])
        fleet=self.client.post('/api/fleet-predictions',json={}).json()['predictions']
        self.assertEqual(len(fleet),20)
        self.assertEqual([r['raw_rul'] for r in fleet],sorted(r['raw_rul'] for r in fleet))
        self.assertEqual(self.hashes,input_fingerprints(self.service.snapshot.settings))

    def test_history_and_invalid_engine_sensor(self):
        r=self.client.get('/api/engines/3/history?cycle=100&sensor=sensor_2').json()
        self.assertEqual(len(r['readings']),100)
        for url in ['/api/engines/999/history','/api/engines/3/history?sensor=unknown']:
            self.assertEqual(self.client.get(url).status_code,422)

    def test_schema_invalid_requests(self):
        for body in ({},{'engine_id':True},{'engine_id':'3'},{'engine_id':3,'cycle':19},
                     {'engine_id':3,'model':'unknown'},{'engine_id':3,'review_threshold':100.,'monitor_threshold':30.},
                     {'engine_id':3,'unexpected':'field'}):
            r=self.client.post('/api/predict',json=body)
            self.assertEqual(r.status_code,422,body)
            self.assertEqual(r.json()['error']['code'],'invalid_request')

    def test_negative_and_nonfinite_outputs(self):
        predictor=self.service.predictor('baseline')
        with patch.object(predictor,'predict',return_value=np.array([-2.])):
            row=self.client.post('/api/predict',json={'engine_id':3,'model':'baseline'}).json()['predictions'][0]
            self.assertEqual(row['raw_rul'],-2)
            self.assertFalse(row['valid'])
            self.assertEqual(row['status'],'Unable to assess')
        with patch.object(predictor,'predict',return_value=np.array([np.nan])):
            r=self.client.post('/api/predict',json={'engine_id':3,'model':'baseline'})
            self.assertEqual(r.status_code,422)
            self.assertIn('Unable to assess',r.json()['error']['message'])

    def test_metrics_reliability_and_diagnostics(self):
        rows=self.client.get('/api/model-metrics').json()['metrics']
        self.assertEqual(rows,self.service.snapshot.metrics.to_dict('records'))
        r=self.client.get('/api/reliability')
        self.assertEqual(r.status_code,200,r.text)
        self.assertAlmostEqual(r.json()['middle_bias'],19.98545,places=4)
        self.assertAlmostEqual(r.json()['largest_overprediction'],108.46605,places=4)
        d=self.client.get('/api/diagnostics/gradient_boosting').json()
        self.assertEqual(len(d['points']),3472)
        self.assertEqual(sum(b['count'] for b in d['residual_bins']),3472)
        self.assertEqual(len(d['per_engine']),20)

    def test_missing_artifacts_and_model_failure(self):
        from aevora.dashboard.data import ArtifactError
        with patch.object(self.service,'check',side_effect=ArtifactError('missing')):
            self.assertEqual(self.client.get('/api/health').status_code,503)
            self.assertEqual(self.client.get('/api/models').json()['error']['code'],'artifacts_unavailable')
        with patch.object(self.service,'predictor',side_effect=ArtifactError('missing model')):
            self.assertEqual(self.client.post('/api/predict',json={'engine_id':3}).status_code,503)

    def test_cors_restricted(self):
        good=self.client.options('/api/predict',headers={'Origin':'http://127.0.0.1:5173','Access-Control-Request-Method':'POST'})
        self.assertEqual(good.headers['access-control-allow-origin'],'http://127.0.0.1:5173')
        bad=self.client.options('/api/predict',headers={'Origin':'https://unrelated.example','Access-Control-Request-Method':'POST'})
        self.assertNotIn('access-control-allow-origin',bad.headers)

    def test_release_prediction_and_sensor_parity_across_engines(self):
        from aevora.dashboard.data import engine_history, infer_history, load_predictor
        for engine in self.service.engines()[:3]:
            engine_id = engine['engine_id']
            history = engine_history(self.service.snapshot.raw, engine_id, 100)
            sensor = self.client.get(
                f'/api/engines/{engine_id}/history?cycle=100&sensor=sensor_2'
            ).json()['readings']
            self.assertEqual([r['cycle'] for r in sensor], history.cycle.tolist())
            self.assertEqual([r['value'] for r in sensor], history.sensor_2.tolist())
            for model in ('baseline', 'ridge', 'gradient_boosting', 'lstm'):
                with self.subTest(engine=engine_id, model=model):
                    response = self.client.post('/api/predict', json={
                        'engine_id': engine_id, 'model': model, 'cycle': 100,
                    })
                    self.assertEqual(response.status_code, 200, response.text)
                    row = response.json()['predictions'][0]
                    direct = infer_history(self.service.snapshot, model,
                        load_predictor(self.service.snapshot, model), history)
                    self.assertAlmostEqual(row['raw_rul'], direct.predicted_rul.iloc[0], places=5)
                    self.assertEqual(row['engine_id'], engine_id)
                    self.assertEqual(row['model'], model)
                    self.assertEqual(row['latest_cycle'], int(history.cycle.max()))
        self.assertEqual(self.hashes, input_fingerprints(self.service.snapshot.settings))
