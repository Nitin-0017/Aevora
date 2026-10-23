"""Descriptive audit plots; no significance or uncertainty claims."""

import os
import numpy as np
from .evaluation import plot_errors


def plot_audit(frames, tables, output, artifact_dir):
    plot_errors(frames, tables['per_engine'], output, artifact_dir)
    os.environ.setdefault('MPLCONFIGDIR', str(artifact_dir / '.matplotlib'))
    os.environ.setdefault('MPL_IGNORE_SYSTEM_FONTS', '1')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def panels():
        fig, grid = plt.subplots(1, len(frames), figsize=(4.6*len(frames), 4), squeeze=False)
        return fig, grid[0]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / name, dpi=150)
        plt.close(fig)

    fig, axes = panels()
    for ax, (name, frame) in zip(axes, frames.items()):
        ax.scatter(frame.rul, frame.predicted_rul - frame.rul, s=5, alpha=.2)
        ax.axhline(0, color='black', linestyle='--')
        ax.set(title=name, xlabel='Actual RUL (cycles)', ylabel='Prediction − actual (cycles)')
    save(fig, 'residuals_vs_rul.png')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, frame in frames.items():
        values = np.sort(np.abs(frame.predicted_rul - frame.rul))
        ax.step(values, np.arange(1, len(values)+1)/len(values), where='post', label=name)
    ax.set(xlabel='Absolute error (cycles)', ylabel='Fraction at or below error', title='Empirical absolute error distributions')
    ax.legend()
    save(fig, 'absolute_error_ecdf.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    groups = tables['grouped_errors'].query("group_type == 'rul'")
    order = ['near_failure_0_30', 'middle_31_100', 'early_over_100']
    for ax, measure in zip(axes, ['mae', 'mean_signed_error']):
        groups.pivot(index='band', columns='model', values=measure).reindex(order).plot.bar(ax=ax)
        ax.axhline(0, color='black', linewidth=.8)
        ax.set(xlabel='Actual RUL band', ylabel=f'{measure} (cycles)')
        ax.tick_params(axis='x', rotation=15)
    save(fig, 'errors_by_rul_band.png')

    fig, axes = panels()
    for ax, (name, frame) in zip(axes, frames.items()):
        top = tables['largest_errors'].query('model == @name').head(10).iloc[::-1]
        ax.barh([f"E{int(r.unit_id)} C{int(r.cycle)}" for r in top.itertuples()], top.residual)
        ax.axvline(0, color='black', linewidth=.8)
        ax.set(title=name, xlabel='Signed error (cycles)')
    save(fig, 'largest_errors.png')

    fig, axes = panels()
    for ax, name in zip(axes, frames):
        values = tables['per_engine'].query('model == @name')
        ax.scatter(values.last_observed_cycle, values.mae)
        for row in values.itertuples():
            ax.annotate(str(row.unit_id), (row.last_observed_cycle, row.mae), fontsize=7)
        ax.set(title=name, xlabel='Complete validation history length (retrospective)', ylabel='Engine MAE (cycles)')
    save(fig, 'error_vs_history_length.png')
