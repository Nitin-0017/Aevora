# AEVORA FD001 training exploration

Observed 20,631 rows, 26 columns, 100 engines.

History lengths: min 128, max 362, mean 206.31, median 199.00 cycles.

Missing: 0; nonfinite: 0; duplicate engine-cycle records: 0.

Exactly constant sensors: sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19.

Near-constant, excluding constant: none. Threshold: one exact recorded value accounts for at least 99% of training rows.

Plot selection is deterministic: first three nonconstant sensors and lowest three engine IDs. Pooled histograms give longer histories more weight. Per-engine summaries are also provided.

These are descriptive observations. Variation does not establish a physical cause or prove impending failure. Constant detection is not automatic feature selection. No training labels, features, models, test-target analysis, or performance estimates were produced.

Source SHA256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`

## Acquisition and execution

Downloaded the archive linked by NASA PCoE on 2026-10-23. Archive SHA256:
`c9c5dec12a945a82e8bb4446589d7fb3cc057b5e5d81fa1a12e25ee9912ad3b2`.
Extracted only train_FD001.txt and test_FD001.txt. RUL_FD001.txt remains
unextracted and unread inside the original archive at `../../work/cmapss.zip`.
The archive is local scratch material and is not required by the EDA command.

The local `.venv` reuses installed system packages and adds Matplotlib 3.11.2
and its plotting dependencies. No future ML stack was installed. The README
provides a fresh-environment recipe. Exact runtime versions are in the JSON report.

Real-data EDA generated four PNG plots, three CSV summaries, a JSON quality
report, and a Markdown findings report under `artifacts/reports/FD001/train/`.
Raw files and generated artifacts remain ignored. The summary here is a reviewed
documentation snapshot; rerun the EDA command to regenerate its source outputs.

No models, training labels, rolling features, or train/validation split were
implemented. Official target validation against real test engines is deferred
to final evaluation; the API is covered by fixture-based tests.
