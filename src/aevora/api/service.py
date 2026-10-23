"""Framework-independent API service; reuses existing inference and audit adapters."""
from datetime import datetime, timezone
from threading import RLock
import numpy as np
from ..dashboard.data import (load_snapshot, load_predictor, infer_history, fleet_predictions,
                              engine_history, cache_token, MODEL_NAMES, ArtifactError)
from ..dashboard.operator import operator_summary, GUIDANCE_LABEL

DISCLOSURE = 'NASA C-MAPSS FD001 · Demonstration data · Not live equipment'
WARNING = 'Illustrative decision support only. Not a guaranteed failure time or a validated safety status.'


class Service:
    def __init__(self, config):
        self.config = config
        self.snapshot = load_snapshot(config)
        self.token = cache_token(config)
        self.models = {}
        self.lock = RLock()

    def check(self):
        if cache_token(self.config) != self.token:
            raise ArtifactError('Artifacts changed. Restart the API with the matching audited bundle.')

    def predictor(self, model):
        if model not in self.models:
            self.models[model] = load_predictor(self.snapshot, model)
        return self.models[model]

    def engines(self):
        return [{'engine_id': int(unit), 'available_cycles': int(group.cycle.max())}
                for unit, group in self.snapshot.raw.groupby('unit_id')]

    def history(self, engine, cycle, sensor):
        s = self.snapshot
        if sensor not in s.manifest['selected_sensors']:
            raise ValueError('Select one of the available sensor IDs.')
        frame = engine_history(s.raw, engine, cycle)
        return {'engine_id': engine, 'sensor': sensor, 'units': 'Recorded dataset units',
                'readings': [{'cycle': int(row.cycle), 'value': float(getattr(row, sensor))}
                             for row in frame.itertuples()]}

    def record(self, row, model, high, elevated, timestamp):
        value = float(row.predicted_rul)
        if not np.isfinite(value):
            raise ValueError('Model returned a non-finite prediction.')
        interpretation = operator_summary(value, high, elevated)
        return {'engine_id': int(row.unit_id), 'model': model, 'raw_rul': value,
                'valid': value >= 0, 'status': interpretation['status'], 'level': interpretation['level'],
                'recommendation': interpretation['action'], 'reason': interpretation['reason'],
                'latest_cycle': int(row.cycle), 'available_cycles': int(row.cycle),
                'preprocessing_version': self.snapshot.manifest['version'], 'model_version': 1,
                'generated_at': timestamp, 'warning': WARNING, 'guidance': GUIDANCE_LABEL}

    def predict(self, request, fleet=False):
        with self.lock:
            self.check()
            s = self.snapshot
            predictor = self.predictor(request.model)
            if fleet:
                frame = fleet_predictions(s, request.model, predictor, request.cycle)
            else:
                history = engine_history(s.raw, request.engine_id, request.cycle)
                frame = infer_history(s, request.model, predictor, history)
            timestamp = datetime.now(timezone.utc).isoformat()
            records = [self.record(row, request.model, request.review_threshold, request.monitor_threshold, timestamp)
                       for row in frame.itertuples()]
            return {'predictions': records, 'generated_at': timestamp, 'disclosure': DISCLOSURE}

    def diagnostics(self, model):
        frame = self.snapshot.predictions[model].copy()
        frame['residual'] = frame.predicted_rul - frame.rul
        counts, bins = np.histogram(frame.residual, bins=30)
        return {'model': model, 'observations': len(frame),
                'points': frame[['unit_id','cycle','rul','predicted_rul']].to_dict('records'),
                'residual_bins': [{'cycles': float((bins[i]+bins[i+1])/2), 'count': int(count)}
                                  for i,count in enumerate(counts)],
                'per_engine': [{'engine_id': int(unit), 'mae': float(abs(g.residual).mean()), 'observations': len(g)}
                               for unit,g in frame.groupby('unit_id')]}

    def reliability(self):
        s = self.snapshot
        frame = s.predictions['gradient_boosting']
        residual = frame.predicted_rul - frame.rul
        middle = residual[(frame.rul > 30) & (frame.rul <= 100)]
        score = s.metrics.set_index('model').loc['gradient_boosting']
        # Compare recalculated displayed figures with the Phase 6 report, not README constants.
        tables = s.reliability['tables']
        pooled = next(r for r in tables['model_metrics'] if r['model']=='gradient_boosting')
        band = next(r for r in tables['grouped_errors'] if r['model']=='gradient_boosting' and r['group_type']=='rul' and r['band']=='middle_31_100')
        if not np.isclose(residual.mean(),pooled['mean_signed_error']) or not np.isclose(middle.mean(),band['mean_signed_error']):
            raise ArtifactError('Reliability report does not match the verified predictions.')
        return {'mean_bias': float(residual.mean()), 'middle_bias': float(middle.mean()),
                'largest_overprediction': float(residual.max()), 'p95_absolute_error': float(score.p95_absolute_error),
                'negative_predictions': [{'model': name, 'count': int((f.predicted_rul<0).sum())} for name,f in s.predictions.items()],
                'noise_probe': tables['stability'], 'perturbation': s.reliability['perturbation'],
                'scope': 'Saved 1% synthetic noise sensitivity; not calibrated uncertainty or failure probability.'}
