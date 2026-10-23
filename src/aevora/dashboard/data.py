"""Validated dashboard data access and inference; no Streamlit or training calls."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from ..audit import input_fingerprints
from ..classical import Predictor
from ..config import load_config
from ..data.loader import load_split
from ..evaluation import verified_predictions
from ..prepare import prepare_inference

MODEL_NAMES = {'gradient_boosting': 'Boosted trees', 'lstm': 'LSTM',
               'ridge': 'Ridge', 'baseline': 'Median baseline'}
DEMO_LABEL = 'NASA C-MAPSS FD001 · validation-engine replay · not live equipment'
VALIDATION_LABEL = 'Development validation · 3,472 observations / 20 engines · used for model selection'


class ArtifactError(ValueError):
    """Actionable artifact/provenance failure presented by the dashboard."""


@dataclass
class Snapshot:
    settings: object
    manifest: dict
    raw: pd.DataFrame
    predictions: dict
    metrics: pd.DataFrame
    reliability: dict
    metadata: dict


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise ArtifactError(f'Missing local artifact: {path}. Restore the matching Phase 3–6 artifact bundle; no automatic training is performed.')
    return json.loads(path.read_text())


def cache_token(config):
    """Invalidate UI caches when local inputs/config/report files change."""
    settings = load_config(config)
    paths = [Path(config), settings.train_path]
    for folder in ('models', 'prepared'):
        paths.extend(p for p in (settings.artifact_dir / folder / settings.dataset).rglob('*') if p.is_file())
    paths.extend(settings.artifact_dir / 'reports' / settings.dataset / folder / 'audit.json'
                 for folder in ('performance_audit', 'reliability'))
    return tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) if p.exists() else (str(p), None, None)
                 for p in sorted(paths))


def check_saved_metrics(frames, audit):
    """Reject stale or altered presentation metrics, using identical saved endpoints."""
    rows = audit['tables']['metrics']
    if len(rows) != 4 or {r['model'] for r in rows} != set(MODEL_NAMES):
        raise ArtifactError('Performance audit must contain exactly four supported models.')
    for row in rows:
        frame = frames[row['model']]
        residual = frame.predicted_rul.to_numpy() - frame.rul.to_numpy()
        absolute = np.abs(residual)
        expected = dict(observations=len(frame), engines=frame.unit_id.nunique(),
                        mae=float(absolute.mean()), rmse=float(np.sqrt(np.mean(residual**2))),
                        r2=r2_score(frame.rul, frame.predicted_rul), mean_signed_error=float(residual.mean()),
                        median_absolute_error=float(np.median(absolute)), max_absolute_error=float(absolute.max()),
                        p90_absolute_error=float(np.quantile(absolute,.9)), p95_absolute_error=float(np.quantile(absolute,.95)),
                        negative_pct=float(100*np.mean(frame.predicted_rul < 0)), nonfinite_pct=0.)
        expected.update({f'within_{t}_pct': float(100*np.mean(absolute <= t)) for t in (5,10,20,30)})
        for key, value in expected.items():
            if not np.isclose(row[key], value, rtol=1e-10, atol=1e-10):
                raise ArtifactError(f'Saved audit disagrees with predictions: {row["model"]} / {key}. Restore matching reports.')
    return pd.DataFrame(rows)


def load_snapshot(config):
    settings = load_config(config)
    try:
        before = input_fingerprints(settings)
        report_root = settings.artifact_dir / 'reports' / settings.dataset
        audit = read_json(report_root / 'performance_audit/audit.json')
        reliability = read_json(report_root / 'reliability/audit.json')
        for report in (audit, reliability):
            if report['input_sha256'] != before:
                raise ArtifactError('Reports do not match current model/preprocessing artifacts. Restore the audited bundle.')
        manifest, _, frames, metadata = verified_predictions(settings, include_lstm=True)
        metrics = check_saved_metrics(frames, audit)
        raw = load_split(settings, 'train')
        raw = raw[raw.unit_id.isin(manifest['validation_engine_ids'])].copy()
        if raw.empty or sorted(raw.unit_id.unique()) != manifest['validation_engine_ids']:
            raise ArtifactError('Validation demonstration engines are missing.')
        if before != input_fingerprints(settings):
            raise ArtifactError('Artifacts changed during verification; reload after files finish updating.')
        return Snapshot(settings, manifest, raw, frames, metrics, reliability, metadata)
    except ArtifactError:
        raise
    except (OSError, ValueError, KeyError, AssertionError, TypeError) as exc:
        raise ArtifactError(f'Cannot verify dashboard inputs: {exc}. Restore matching local Phase 3–6 data, models and reports.') from exc


def engine_history(raw, engine, cycle):
    if engine not in raw.unit_id.unique():
        raise ValueError('Select an available demonstration engine.')
    if isinstance(cycle, bool) or int(cycle) != cycle or cycle < 1:
        raise ValueError('Replay cycle must be a positive integer.')
    result = raw[(raw.unit_id == engine) & (raw.cycle <= cycle)].sort_values('cycle').copy()
    if result.empty:
        raise ValueError('No readings at this replay cycle.')
    return result


def risk_state(prediction, high=30., elevated=100.):
    """Illustrative inclusive boundaries, not failure probabilities. Never clip."""
    if not np.isfinite([high, elevated]).all() or not 0 <= high < elevated:
        raise ValueError('Thresholds must satisfy 0 ≤ high < elevated.')
    if prediction is None or not np.isfinite(prediction):
        return {'raw_prediction': prediction, 'display_prediction': None, 'status': 'Invalid / unavailable', 'level': 'invalid'}
    value = float(prediction)
    if value < 0:
        status, level = 'Invalid negative RUL — review model output', 'invalid'
    elif value <= high:
        status, level = 'High concern (illustrative)', 'high'
    elif value <= elevated:
        status, level = 'Elevated concern (illustrative)', 'elevated'
    else:
        status, level = 'Above demonstration thresholds — not a safety assurance', 'neutral'
    return {'raw_prediction': value, 'display_prediction': value, 'status': status, 'level': level}


def load_predictor(snapshot, model):
    if model not in MODEL_NAMES:
        raise ValueError('Unsupported saved model.')
    directory = snapshot.settings.artifact_dir / 'models' / snapshot.settings.dataset / model
    manifest = read_json(directory / 'preprocessing/manifest.json')
    if model == 'lstm':
        from ..lstm import LSTMPredictor
        predictor = LSTMPredictor.load(directory)
    else:
        predictor = Predictor.load(directory)
    if (manifest['version'] != 1 or tuple(manifest['feature_names']) != predictor.features
            or manifest['feature_names'] != snapshot.manifest['feature_names']
            or manifest['config']['sequence_window'] != predictor.length
            or predictor.length != snapshot.settings.sequence_window):
        raise ArtifactError('Model feature order or sequence dimensions differ from frozen preprocessing.')
    return predictor


def infer_history(snapshot, model, predictor, history):
    directory = snapshot.settings.artifact_dir / 'models' / snapshot.settings.dataset / model
    batch = prepare_inference(history, directory / 'preprocessing/manifest.json')
    predicted = predictor.predict(batch.x, snapshot.manifest['feature_names'])
    if predicted.shape != (len(batch.endpoints),) or not np.isfinite(predicted).all():
        raise ValueError('Model returned missing or nonfinite predictions; estimates cannot be displayed.')
    result = batch.endpoints.copy()
    result['predicted_rul'] = predicted
    return result


def fleet_predictions(snapshot, model, predictor, cycle):
    history = snapshot.raw[snapshot.raw.cycle <= cycle]
    return infer_history(snapshot, model, predictor, history).sort_values('predicted_rul').reset_index(drop=True)
