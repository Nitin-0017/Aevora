"""Read-only model audit; generated outputs live in a separate reliability report directory."""

import argparse
import json
import pandas as pd
from .config import load_config
from .evaluation import verified_predictions
from .reliability import build_tables
from .training import digest


def input_fingerprints(settings):
    result = {}
    for root in (settings.artifact_dir / 'models' / settings.dataset,
                 settings.artifact_dir / 'prepared' / settings.dataset):
        for path in sorted(root.rglob('*')):
            if path.is_file():
                result[str(path.relative_to(settings.artifact_dir))] = digest(path)
    return result


def write_tables(tables, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    # pandas converts missing/insufficient-group values to JSON null.
    return {name: json.loads(table.to_json(orient='records', double_precision=15)) for name, table in tables.items()}


def run_audit(settings):
    before = input_fingerprints(settings)
    manifest, batches, frames, metadata = verified_predictions(settings, include_lstm=True)
    tables = build_tables(frames)
    output = settings.artifact_dir / 'reports' / settings.dataset / 'reliability'
    records = write_tables(tables, output)
    alignment = frames['gradient_boosting'][['unit_id', 'cycle', 'rul']]
    alignment.to_csv(output / 'validation_endpoints.csv', index=False)
    if before != input_fingerprints(settings):
        raise RuntimeError('Audit inputs changed during execution')
    report = {'scope': 'Descriptive development-validation audit; no refitting or official test access',
              'residual_definition': 'prediction minus actual RUL; positive overstates remaining life',
              'validation_engine_ids': manifest['validation_engine_ids'],
              'training_engine_ids': manifest['training_engine_ids'],
              'source_sha256': manifest['source_sha256'],
              'validation_endpoints_sha256': digest(output / 'validation_endpoints.csv'),
              'input_sha256': before, 'tables': records}
    (output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    report = run_audit(load_config(args.config))
    print(json.dumps(report['tables']['model_metrics'], indent=2))


if __name__ == '__main__':
    main()
