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
