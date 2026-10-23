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
