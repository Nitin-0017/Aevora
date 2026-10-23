# Preparing training data

## RUL targets

For complete training histories, remaining useful life is the last observed
cycle for that engine minus the current cycle. An engine ending at cycle 100
has RUL 10 at cycle 90 and RUL 0 at cycle 100. These are cycles, not days.

`training_rul(frame, run_to_failure=True, cap=None)` returns engine ID, cycle,
`rul_uncapped`, and `rul`, retaining input row order and index. An optional positive
integer `rul_cap` in the TOML config clips only `rul`; omitted means uncapped.
The function requires explicit confirmation of run-to-failure provenance. It
cannot determine from values whether a caller supplied a truncated history.
Only the official training split is used by the preparation workflow. No official
test data or target file is needed. Targets must never become input features.

## Engine split and fitted preprocessing

Split sorted unique engine IDs using NumPy's seeded generator. Validation gets
`ceil(engine_count * validation_fraction)` engines; both partitions must be nonempty.
All rows of an engine stay together. Seed, exact IDs, and source hash will be saved.

`select_sensors(training)` removes only exact constants in training engines.
Phase 2 whole-dataset EDA does not determine the selection. Near-constant sensor
removal is deliberately not enabled. Settings and identifiers are not features.

`FeatureScaler.fit` requires the declared training engine IDs and checks that
the supplied data matches them. It learns feature means and population standard
deviations from training rows only (longer engines contribute more rows).
A zero standard deviation uses scale 1. `transform` uses immutable fitted state;
it never refits. IDs/cycles stay unchanged as metadata, and feature order is explicit.
JSON save/load preserves columns, means, scales, training IDs, row count and version.
As with any API, the caller must supply an honest training-ID declaration.

Shared history validation rejects duplicate keys, gaps, invalid IDs and nonfinite
features. It sorts a copy by engine/cycle and permits partial histories starting
after cycle 1; callers align targets by keys rather than DataFrame row positions.

## Historical features

For each selected sensor, retain its current value and add a trailing mean,
population standard deviation (ddof=0), one-cycle change, and least-squares slope
against cycle position. `rolling_window` defaults to 5. The slope is in sensor
units per cycle. Consecutive cycles are required. Output is sorted by engine/cycle.

Early cycles use only available past/current readings (min_periods=1). The first
observation has zero standard deviation, change and slope. There is no backward
fill, centered window, future reading, or cross-engine context. At inference,
provide the full available history from cycle 1 so early-window conventions
match training. A changing feature is not proof of a physical failure mechanism.

## Twenty-cycle windows and inference

`sequence_window` now defaults to 20. Each window has 20 consecutive rows from
one engine, with stride 1. Its target is RUL at the last cycle, joined by
engine/cycle keys, never row index. Arrays have shape `[windows, 20, features]`
and float32 inputs; target arrays are separate. Endpoint IDs/cycles are metadata.
Short engines are skipped and listed in training reports. If all are short,
the result is an empty array with the correct three-dimensional shape.

`prepare_inference(history, manifest_path)` loads saved selection and scaler state,
builds the same features, and returns the latest window per engine without labels.
Supply only history available at prediction time, starting at cycle 1. It rejects
short histories rather than padding or pretending that missing context exists.
Future-looking endpoint targets are allowed as supervised labels, never features.
The fitted scaler is frozen across validation/inference; training-wide fitted
statistics are normal preprocessing, distinct from per-row future sensor leakage.

## Run and reproduce

From the repository root, using the existing NumPy/pandas environment:

```bash
PYTHONPATH=src .venv/bin/python -m aevora.prepare --config configs/default.toml --save-arrays
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Only `train_FD001.txt` is read. Output goes to configured
`artifact_dir/prepared/FD001/`: `manifest.json`, `scaler.json`, train/validation
label CSVs (capped and uncapped), endpoint CSVs, and optional compressed NPZ arrays.
Load NPZ with `allow_pickle=False`. Omit `--save-arrays` to compute checks and save
metadata only. Check `arrays_saved` in the latest manifest: existing arrays from
an older run are not removed and must not be used when it is false. Re-running
overwrites named outputs; use a different artifact directory for separate runs.

The manifest records source SHA256, exact engine IDs, selected sensors, ordered
feature names, all configuration values and runtime versions. Keep it with its
matching scaler and arrays; all are ignored by Git. Current fitting covers all
training rows, while windows omit the first 19 endpoints per engine. Later model
comparisons must use identical endpoints. No model performance is measured here.
