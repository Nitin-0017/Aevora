# Phase 2 validation and exploration

Run from the repository root after the minimal setup in README:

```bash
PYTHONPATH=src .venv/bin/python -m aevora.data --config configs/default.toml --split train
PYTHONPATH=src .venv/bin/python -m aevora.eda --config configs/default.toml
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

The first command prints JSON. `--split test` is supported for trajectory-only
validation; neither command reads official RUL targets. Missing/invalid files
produce an actionable error and nonzero exit status. EDA validates input before
creating outputs. Existing outputs from earlier runs are not evidence of a later
failed run; check exit status and the report's source hash.

## Checks and constraints

Exactly 26 columns; explicit numeric parsing; missing/nonfinite values; positive
integer unit/cycle values; duplicate engine-cycle pairs; cycle order and gaps.
This FD001 raw-history contract requires cycles to start at 1 and progress by 1.
It is deliberately not a validator for future arbitrary inference windows.
No rows are dropped, sorted, imputed, or otherwise repaired by the loader.

`quality_report(frame)` returns diagnostics without modifying input. NaN counts
as both missing and nonfinite; the totals are not disjoint. Malformed strings
produce parse errors with line numbers. Valid IDs yield per-engine row counts
and cycle bounds. Invalid schemas/IDs cannot yield trustworthy engine summaries.

`load_rul_targets(path, test_frame)` is an explicit future evaluation API. It
checks finite nonnegative integer targets, one per complete test engine 1..N,
and maps them by ascending ID rather than file appearance. Phase 2 exercises this
only with temporary fixtures. No per-row training RUL is calculated.

Constant sensors have one unique recorded value. Near-constant sensors have a
modal exact-value frequency >=99%, excluding constants. This unit-independent
heuristic describes the recorded precision; it is not physical evidence and does
not remove features. Later feature-selection decisions must use training splits.

## Artifacts

Output path derives from configured `artifact_dir`: `reports/FD001/train/`.

- `quality_report.json`: checks, counts, cycle statistics, input SHA256, versions.
- `sensor_setting_summary.csv`: summary statistics for sensors and settings.
- `engine_cycle_counts.csv`: observed history length per engine.
- `per_engine_sensor_summary.csv`: sensor mean/std/min/max by engine.
- `engine_cycle_distribution.png`: distribution of history lengths.
- `sensor_distributions.png`: first three nonconstant sensors, pooled rows.
- `sensor_histories.png`: those sensors for the lowest three engine IDs.
- `sensor_variation_by_engine.png`: within-engine sensor variability distribution.
- `findings.md`: concise computed observations and limitations.

Plot selection is deterministic, not selected to support a degradation story.
Longer engine histories contribute more rows to pooled statistics. Interpret
those alongside engine-level summaries. Re-running replaces these named artifacts.
Training-set exploration is not independent validation and provides no model
performance estimate. Simulation findings do not establish industrial reliability.
