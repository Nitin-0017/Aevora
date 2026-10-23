"""Descriptive validation reliability metrics, never model fitting or final-test scoring."""

import numpy as np
import pandas as pd
from .training import metrics

METRIC_KEYS = ('mae', 'rmse', 'mean_signed_error', 'median_signed_error',
               'residual_std', 'p95_absolute_error', 'max_absolute_error',
               'underprediction_fraction', 'overprediction_fraction', 'exact_fraction',
               'negative_predictions', 'minimum_prediction', 'maximum_prediction')


def validate_predictions(frame):
    required = ['unit_id', 'cycle', 'rul', 'predicted_rul']
    if not frame.columns.is_unique or not set(required).issubset(frame.columns):
        raise ValueError('Predictions require unique engine/cycle/target/prediction columns')
    values = frame[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite prediction or target values')
    if (values[:, :2] <= 0).any() or (values[:, :2] != np.floor(values[:, :2])).any():
        raise ValueError('Invalid engine/cycle keys')
    if (values[:, 2] < 0).any() or frame.duplicated(['unit_id', 'cycle']).any():
        raise ValueError('Negative target or duplicate endpoint')


def describe_errors(frame):
    """Residual = prediction - actual; positive means remaining life is overstated."""
    validate_predictions(frame)
    count = len(frame)
    result = {'observations': count, 'engines': int(frame.unit_id.nunique())}
    if not count:
        return {**result, **dict.fromkeys(METRIC_KEYS)}
    actual, prediction = frame.rul.to_numpy(), frame.predicted_rul.to_numpy()
    residual = prediction - actual
    absolute = np.abs(residual)
    return {**result, **metrics(actual, prediction),
            'mean_signed_error': float(residual.mean()), 'median_signed_error': float(np.median(residual)),
            'residual_std': float(residual.std(ddof=0)) if count > 1 else None,
            'p95_absolute_error': float(np.quantile(absolute, .95)), 'max_absolute_error': float(absolute.max()),
            'underprediction_fraction': float(np.mean(residual < 0)),
            'overprediction_fraction': float(np.mean(residual > 0)), 'exact_fraction': float(np.mean(residual == 0)),
            'negative_predictions': int((prediction < 0).sum()),
            'minimum_prediction': float(prediction.min()), 'maximum_prediction': float(prediction.max())}


def engine_metrics(frames):
    rows = []
    for name, frame in frames.items():
        validate_predictions(frame)
        for unit, group in frame.groupby('unit_id', sort=True):
            rows.append({'model': name, 'unit_id': int(unit), **describe_errors(group)})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    benchmark = result[result.model == 'gradient_boosting'].set_index('unit_id')
    if benchmark.empty:
        raise ValueError('Boosted-tree benchmark is required')
    for measure in ('mae', 'rmse'):
        result[f'{measure}_minus_boosted'] = result[measure] - result.unit_id.map(benchmark[measure])
    if result[['mae_minus_boosted', 'rmse_minus_boosted']].isna().any().any():
        raise ValueError('Benchmark engine coverage differs')
    return result


def engine_variability(per_engine):
    rows = []
    for name, group in per_engine.groupby('model', sort=False):
        rows.append({'model': name, 'engines': len(group), 'macro_mae': float(group.mae.mean()),
                     'engine_mae_std': float(group.mae.std(ddof=0)) if len(group) > 1 else None,
                     'minimum_engine_mae': float(group.mae.min()), 'maximum_engine_mae': float(group.mae.max()),
                     'best_engine': int(group.sort_values(['mae', 'unit_id']).iloc[0].unit_id),
                     'worst_engine': int(group.sort_values(['mae', 'unit_id'], ascending=[False, True]).iloc[0].unit_id)})
    return pd.DataFrame(rows)
