"""A fixed synthetic sensor-noise probe, not calibrated predictive uncertainty."""

import numpy as np
import pandas as pd
from .classical import Predictor
from .data.loader import load_split
from .features import sensor_features
from .lstm import LSTMPredictor
from .preprocessing import FeatureScaler
from .sequences import make_windows
from .training import metrics


NOISE_FRACTION = .01


def perturb_sensors(history, sensors, training_scales, seed, fraction=NOISE_FRACTION):
    if not np.isfinite(fraction) or fraction < 0 or len(sensors) != len(training_scales):
        raise ValueError('Invalid noise specification')
    if not np.isfinite(training_scales).all() or min(training_scales) <= 0:
        raise ValueError('Noise scales must be positive training statistics')
    perturbed = history.sort_values(['unit_id', 'cycle'], kind='stable').reset_index(drop=True).copy()
    values = perturbed[sensors].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Sensor noise cannot repair missing/nonfinite input')
    noise = np.random.default_rng(seed).normal(size=values.shape) * np.asarray(training_scales) * fraction
    perturbed[sensors] = values + noise
    return perturbed


def stability_probe(settings, manifest, batch, frames):
    raw = load_split(settings, 'train')
    history = raw[raw.unit_id.isin(manifest['validation_engine_ids'])].copy()
    scaler = FeatureScaler.load(settings.artifact_dir / 'prepared' / settings.dataset / 'scaler.json')
    sensors, features = manifest['selected_sensors'], manifest['feature_names']
    scales = [scaler.scales[scaler.feature_names.index(sensor)] for sensor in sensors]
    perturbed = perturb_sensors(history, sensors, scales, settings.random_seed)
    engineered = sensor_features(perturbed, sensors, settings.rolling_window)
    windows = make_windows(scaler.transform(engineered), features, settings.sequence_window)
    if not windows.endpoints.equals(batch.endpoints):
        raise ValueError('Perturbation changed endpoint membership/order')
    summaries, by_engine, contracts = [], [], []
    for name, clean in frames.items():
        directory = settings.artifact_dir / 'models' / settings.dataset / name
        predictor = LSTMPredictor.load(directory) if name == 'lstm' else Predictor.load(directory)
        prediction = predictor.predict(windows.x, features)
        # Clean is already reproduced against this same saved predictor by verified_predictions.
        delta = prediction - clean.predicted_rul.to_numpy()
        score = metrics(batch.y, prediction)
        summaries.append({'model': name, 'observations': len(delta), 'noise_fraction': NOISE_FRACTION,
                          'seed': settings.random_seed, 'mean_absolute_prediction_change': float(np.abs(delta).mean()),
                          'p95_absolute_prediction_change': float(np.quantile(np.abs(delta), .95)),
                          'max_absolute_prediction_change': float(np.abs(delta).max()),
                          'fraction_change_over_5_cycles': float(np.mean(np.abs(delta) > 5)),
                          'perturbed_mae': score['mae'], 'perturbed_rmse': score['rmse'],
                          'mae_change': score['mae'] - metrics(batch.y, clean.predicted_rul.to_numpy())['mae']})
        for unit in sorted(batch.endpoints.unit_id.unique()):
            subset = delta[batch.endpoints.unit_id.to_numpy() == unit]
            by_engine.append({'model': name, 'unit_id': int(unit), 'observations': len(subset),
                              'mean_absolute_prediction_change': float(np.abs(subset).mean()),
                              'max_absolute_prediction_change': float(np.abs(subset).max())})
        x = batch.x[:1].copy()
        missing = x.copy(); missing[0, 0, 0] = np.nan
        for check, bad_x, bad_names in [('feature_order', x, features[::-1]),
                                        ('missing_values', missing, features),
                                        ('short_window', x[:, :-1], features)]:
            rejected = False
            try:
                predictor.predict(bad_x, bad_names)
            except ValueError:
                rejected = True
            contracts.append({'model': name, 'check': check, 'rejected': rejected})
            if not rejected:
                raise ValueError(f'{name} accepted invalid {check} input')
    return pd.DataFrame(summaries), pd.DataFrame(by_engine), pd.DataFrame(contracts)
