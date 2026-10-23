"""Evaluate the best saved LSTM on verified development-validation sequences."""

import argparse
import json
import os
import numpy as np
import pandas as pd
from .config import load_config
from .evaluation import align_predictions, error_tables, plot_errors
from .lstm import LSTMPredictor
from .training import digest, load_prepared, metrics


def validated_predictions(settings, manifest, batch):
    directory = settings.artifact_dir / 'models' / settings.dataset / 'lstm'
    report = json.loads((directory / 'metadata.json').read_text())
    source = settings.artifact_dir / 'prepared' / settings.dataset
    hashes = {p.name: digest(p) for p in sorted(source.iterdir()) if p.is_file()}
    if report['prepared_sha256'] != hashes or report['checkpoint_sha256'] != digest(directory / 'best.pt'):
        raise ValueError('LSTM provenance mismatch')
    reference = batch.endpoints.copy()
    reference['rul'] = batch.y
    frame = pd.read_csv(directory / 'validation_predictions.csv')
    align_predictions(reference, frame)
    predictor = LSTMPredictor.load(directory)
    prediction = predictor.predict(batch.x, manifest['feature_names'])
    if not np.allclose(prediction, frame.predicted_rul, rtol=1e-12, atol=1e-12):
        raise ValueError('LSTM saved predictions differ from checkpoint')
    score = metrics(batch.y, prediction)
    for key, value in score.items():
        if not np.isclose(value, report[key], rtol=1e-12, atol=1e-12):
            raise ValueError('LSTM metric mismatch')
    history = pd.read_csv(directory / 'history.csv')
    best = report['best_epoch']
    if (history.epoch.tolist() != list(range(1, report['epochs_completed'] + 1))
            or best not in history.epoch.values
            or not np.isfinite(history.to_numpy()).all()
            or not np.isclose(history.loc[history.epoch == best, 'validation_mse'].iloc[0], score['rmse'] ** 2)):
        raise ValueError('LSTM checkpoint/history mismatch')
    frame['predicted_rul'] = prediction
    return frame, report, history


def evaluate_lstm(settings):
    manifest, batches = load_prepared(settings)
    frame, report, history = validated_predictions(settings, manifest, batches['validation'])
    engines, bands = error_tables(frame)
    output = settings.artifact_dir / 'reports' / settings.dataset / 'lstm_validation'
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(engines).to_csv(output / 'errors_by_engine.csv', index=False)
    pd.DataFrame(bands).to_csv(output / 'errors_by_rul_band.csv', index=False)
    frame.to_csv(output / 'predictions.csv', index=False)
    history.to_csv(output / 'history.csv', index=False)
    result = {'scope': 'Development validation used for early stopping; not final test performance',
              'model': 'lstm', **metrics(frame.rul.to_numpy(), frame.predicted_rul.to_numpy()),
              'observations': len(frame), 'engines': int(frame.unit_id.nunique()),
              'engine_macro_mae': float(np.mean([e['mae'] for e in engines])),
              'negative_predictions': int((frame.predicted_rul < 0).sum()),
              'epochs_completed': report['epochs_completed'], 'best_epoch': report['best_epoch'],
              'early_stopped': report['early_stopped'], 'fit_seconds': report['fit_seconds'],
              'checkpoint_sha256': report['checkpoint_sha256'], 'rul_bands': bands}
    (output / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot_errors({'lstm': frame}, pd.DataFrame(engines).assign(model='lstm'), output, settings.artifact_dir)
    plot_history(history, report['best_epoch'], output, settings.artifact_dir)
    return result


def plot_history(history, best_epoch, output, artifact_dir):
    os.environ.setdefault('MPLCONFIGDIR', str(artifact_dir / '.matplotlib'))
    os.environ.setdefault('MPL_IGNORE_SYSTEM_FONTS', '1')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history.epoch, history.train_mse, marker='o', label='Training (online, dropout enabled)')
    ax.plot(history.epoch, history.validation_mse, marker='o', label='Validation (epoch end)')
    ax.axvline(best_epoch, linestyle='--', color='black', label=f'Restored epoch {best_epoch}')
    ax.set(xlabel='Epoch', ylabel='MSE (cycles squared)', title='LSTM learning curves · development split')
    ax.set_xticks(history.epoch)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate_lstm(load_config(args.config)), indent=2))


if __name__ == '__main__':
    main()
