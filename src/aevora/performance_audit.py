"""Extended descriptive audit of saved development predictions; never fits models."""
import argparse
import json
import os
import numpy as np
import pandas as pd
from .audit import input_fingerprints, write_tables
from .config import load_config
from .evaluation import verified_predictions, rul_band_masks, align_predictions

TOLERANCES = (5, 10, 20, 30)


def regression_summary(y, prediction):
    """Full-denominator percentages; invalid predictions invalidate error scores.

    Empty denominators and constant-target/singleton R² are undefined (None).
    Nonfinite predictions count as tolerance misses, never silently disappear.
    """
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or prediction.shape != y.shape or not np.isfinite(y).all():
        raise ValueError('Expected aligned vectors with finite ground truth')
    n = len(y)
    finite = np.isfinite(prediction)
    result = {'observations': n,
              'negative_pct': float(100 * np.count_nonzero(finite & (prediction < 0)) / n) if n else None,
              'nonfinite_pct': float(100 * np.count_nonzero(~finite) / n) if n else None}
    absolute = np.full(n, np.inf)
    absolute[finite] = np.abs(prediction[finite] - y[finite])
    for tolerance in TOLERANCES:
        result[f'within_{tolerance}_pct'] = float(100 * np.count_nonzero(absolute <= tolerance) / n) if n else None
    keys = ('mae', 'rmse', 'r2', 'median_absolute_error', 'mean_signed_error',
            'max_absolute_error', 'p90_absolute_error', 'p95_absolute_error')
    result.update(dict.fromkeys(keys))
    if n and finite.all():
        residual = prediction - y
        sst = float(np.sum((y - y.mean()) ** 2))
        result.update(mae=float(absolute.mean()), rmse=float(np.sqrt(np.mean(residual ** 2))),
                      r2=float(1 - np.sum(residual ** 2) / sst) if n > 1 and sst > 0 else None,
                      median_absolute_error=float(np.median(absolute)), mean_signed_error=float(residual.mean()),
                      max_absolute_error=float(absolute.max()),
                      p90_absolute_error=float(np.quantile(absolute, .90, method='linear')),
                      p95_absolute_error=float(np.quantile(absolute, .95, method='linear')))
    return result


def improvement(score, reference):
    """Percentage reduction relative to reference MAE; negative means worse."""
    return 100 * (reference - score) / reference if reference > 0 else None


def summarize_frames(frames):
    reference = next(iter(frames.values()))
    overall, engines, bands = [], [], []
    for model, frame in frames.items():
        align_predictions(reference, frame)
        def row(group):
            return {'model': model, 'engines': int(group.unit_id.nunique()),
                    **regression_summary(group.rul, group.predicted_rul)}
        overall.append(row(frame))
        engines.extend({'unit_id': int(unit), **row(group)} for unit, group in frame.groupby('unit_id'))
        bands.extend({'band': name, **row(frame[mask])} for name, mask in rul_band_masks(frame))
    baseline = next(row['mae'] for row in overall if row['model'] == 'baseline')
    for row in overall:
        competitor = min((other for other in overall if other['model'] != row['model']), key=lambda other: other['mae'])
        row.update(mae_improvement_vs_baseline_pct=improvement(row['mae'], baseline),
                   best_alternative=competitor['model'],
                   mae_improvement_vs_best_alternative_pct=improvement(row['mae'], competitor['mae']))
    return {'metrics': pd.DataFrame(overall), 'per_engine': pd.DataFrame(engines), 'rul_bands': pd.DataFrame(bands)}


def plot_performance(frames, tables, output, artifact_dir):
    os.environ.setdefault('MPLCONFIGDIR', str(artifact_dir / '.matplotlib'))
    os.environ.setdefault('MPL_IGNORE_SYSTEM_FONTS', '1')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reference = next(iter(frames.values()))
    caption = f'Development validation · n={len(reference):,} per model · {reference.unit_id.nunique()} engines'
    def save(fig, name):
        fig.suptitle(caption)
        fig.tight_layout()
        fig.savefig(output / f'{name}.png', dpi=150)
        plt.close(fig)
    values = np.concatenate([f[['rul', 'predicted_rul']].to_numpy().ravel() for f in frames.values()])
    limits = (min(0, values.min()) - 5, values.max() + 5)
    residuals = {name: f.predicted_rul - f.rul for name, f in frames.items()}
    bins = np.linspace(min(r.min() for r in residuals.values()), max(r.max() for r in residuals.values()), 51)
    for mode in ('actual_vs_predicted', 'residual_distribution', 'error_vs_actual_rul'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
        for ax, (name, frame) in zip(axes.ravel(), frames.items()):
            residual = residuals[name]
            ax.set_title(name)
            if mode == 'actual_vs_predicted':
                ax.scatter(frame.rul, frame.predicted_rul, s=5, alpha=.2)
                ax.plot(limits, limits, 'k--', label='Ideal prediction')
                ax.set(xlabel='Actual RUL (cycles)', ylabel='Predicted RUL (cycles)', xlim=limits, ylim=limits)
                ax.legend()
            elif mode == 'residual_distribution':
                ax.hist(residual, bins=bins)
                ax.axvline(0, color='black', linestyle='--')
                ax.set(xlabel='Prediction − actual RUL (cycles)', ylabel='Observations')
            else:
                ax.scatter(frame.rul, np.abs(residual), s=5, alpha=.2)
                ax.set(xlabel='Actual RUL (cycles)', ylabel='Absolute error (cycles)', ylim=(0, max(float(np.abs(r).max()) for r in residuals.values()) * 1.05 + 1))
        save(fig, mode)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, residual in residuals.items():
        ordered = np.sort(np.abs(residual))
        ax.step(ordered, 100 * np.arange(1, len(ordered) + 1) / len(ordered), where='post', label=name)
    ax.set(xlabel='Absolute error (cycles)', ylabel='Cumulative observations (%)', title='Absolute error distribution')
    ax.legend()
    save(fig, 'absolute_error_distribution')
    scores = tables['metrics'].set_index('model')
    fig, ax = plt.subplots(figsize=(9, 5))
    scores.mae.plot.bar(ax=ax, rot=0)
    ax.set(ylabel='Mean absolute error (cycles)', xlabel='Model')
    save(fig, 'mae_by_model')
    fig, ax = plt.subplots(figsize=(10, 5))
    tolerance = scores[[f'within_{t}_pct' for t in TOLERANCES]].T
    tolerance.index = [f'±{t}' for t in TOLERANCES]
    tolerance.plot.bar(ax=ax, rot=0)
    ax.set(xlabel='Inclusive error tolerance (cycles)', ylabel='Predictions within tolerance (%)', ylim=(0, 100))
    save(fig, 'error_tolerances')
    fig, ax = plt.subplots(figsize=(15, 6))
    tables['per_engine'].pivot(index='unit_id', columns='model', values='mae')[list(frames)].plot.bar(ax=ax)
    counts = reference.groupby('unit_id').size()
    ax.set_xticklabels([f'{unit} (n={count})' for unit, count in counts.items()], rotation=90)
    ax.set(xlabel='Validation engine and observation count per model', ylabel='Mean absolute error (cycles)')
    save(fig, 'per_engine_comparison')


def run_audit(settings):
    before = input_fingerprints(settings)
    manifest, batches, verified, _ = verified_predictions(settings, include_lstm=True)
    # Calculate from original CSV values after reproduction/provenance verification.
    frames = {name: pd.read_csv(settings.artifact_dir / 'models' / settings.dataset / name / 'validation_predictions.csv')
              for name in verified}
    for name in frames:
        align_predictions(verified[name], frames[name])
        np.testing.assert_allclose(frames[name].predicted_rul, verified[name].predicted_rul, rtol=1e-12, atol=1e-12)
    tables = summarize_frames(frames)
    output = settings.artifact_dir / 'reports' / settings.dataset / 'performance_audit'
    records = write_tables(tables, output)
    plot_performance(frames, tables, output, settings.artifact_dir)
    prior = settings.artifact_dir / 'reports' / settings.dataset / 'reliability' / 'audit.json'
    stability = None
    if prior.exists():
        previous = json.loads(prior.read_text())
        if previous['input_sha256'] == before:
            stability = {'source': str(prior.relative_to(settings.project_root)),
                         'status': 'Reused prior measurement; identical model/preparation hashes; not rerun',
                         'perturbation': previous['perturbation'], 'rows': previous['tables']['stability']}
    if before != input_fingerprints(settings):
        raise RuntimeError('Audit changed model or preparation inputs')
    result = {'scope': 'Development validation, selected models; no official test access or refitting',
              'input_sha256': before, 'inputs_unchanged': True,
              'training_shape': list(batches['train'].x.shape), 'validation_shape': list(batches['validation'].x.shape),
              'training_engine_ids': manifest['training_engine_ids'],
              'validation_engine_ids': manifest['validation_engine_ids'],
              'exact_endpoint_target_alignment': True, 'tables': records, 'prior_stability': stability}
    (output / 'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(run_audit(load_config(args.config))['tables']['metrics'], indent=2))


if __name__ == '__main__':
    main()
