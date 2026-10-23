# Phase 6 — Reliability audit and failure analysis

## Executive summary

Boosted trees retain the lowest pooled validation MAE/RMSE, but their average
conceals substantial overprediction in the middle RUL range and concentrated
large errors. LSTM has higher between-engine variability and positive near-failure
bias. Ridge produces physically invalid negative RUL predictions. None of these
models has been shown suitable for autonomous maintenance decisions.

This phase changed no benchmark model, scaler, split, prediction file or training
configuration. Input hashes before and after the audit match. No official test
trajectories or RUL targets were read. No benchmark model was retrained.

## Methodology and boundaries

Reused all four saved model bundles and validation predictions. The shared loader
reconstructed the Phase 3 windows from official **training** histories, verified
source/scaler/feature provenance, and reproduced predictions with the saved models.
All four use the same 3,472 endpoint targets from 20 engines; there are no subset
eligibility differences. Input is 20 × 75; classical models summarize this to 225
features. Training comprises 80 separate engines. RUL targets are uncapped.

Validation engines: 3, 5, 19, 22, 25, 26, 28, 34, 43, 51, 52, 53, 57, 60, 62,
72, 82, 93, 98, 100. Exact cycle IDs and targets are saved in
[validation_endpoints.csv](../artifacts/reports/FD001/reliability/validation_endpoints.csv).

**Residual = predicted RUL − actual RUL.** Positive residual overstates remaining
life and could delay maintenance; negative residual understates remaining life
and could prompt premature maintenance. These are potential consequences, not
measured operational outcomes. MAE weights absolute errors; RMSE emphasizes large
errors. Reported quantiles and population standard deviations are descriptive.
Empty groups have null metrics; single-row residual variability is null. No
confidence intervals, uncertainty calibration or statistical-significance claims
are made. Overlapping windows are correlated and long engines receive more weight.

## Model-level errors

| Model | MAE | RMSE | Mean signed | Median signed | P95 absolute | Worst absolute | Negative RUL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 46.41 | 54.85 | +6.12 | +9.00 | 92.00 | 154.00 | 0 |
| ridge | 27.48 | 33.60 | +11.73 | +13.80 | 63.04 | 96.26 | 207 |
| gradient_boosting | 20.54 | 28.72 | +9.83 | +5.45 | 60.34 | 108.47 | 10 |
| lstm | 23.11 | 32.26 | +12.08 | +7.54 | 70.44 | 110.51 | 0 |

All clean and perturbed predictions were finite. Ridge has 207 negative predictions
(minimum −81.69 cycles), boosted trees 10 (minimum −2.64); all occur in the
near-failure band. LSTM and baseline had none in this run. No clipping was applied.
RUL must be nonnegative, but there is no defensible universal upper limit inferred
from the largest training label. Finite/nonnegative output does not establish accuracy.

## Per-engine variation and benchmark differences

| Model | Equal-engine MAE | Engine MAE std | Best engine / MAE | Worst engine / MAE | Engines beating boosted MAE |
| --- | ---: | ---: | --- | --- | ---: |
| baseline | 45.44 | 6.06 | 93 / 39.56 | 5 / 65.98 | 1 |
| ridge | 28.19 | 8.16 | 34 / 17.27 | 28 / 48.45 | 1 |
| gradient_boosting | 21.23 | 9.47 | 34 / 9.77 | 28 / 44.73 | 0 |
| lstm | 24.21 | 14.67 | 51 / 10.15 | 57 / 62.99 | 9 |

Full counts, MAE/RMSE, signed means/medians, P95/worst errors, residual variability
and MAE/RMSE differences from boosted trees are in
[per_engine.csv](../artifacts/reports/FD001/reliability/per_engine.csv). Positive
`*_minus_boosted` means worse than boosted trees on that engine. LSTM beats boosted
trees on 9 engines but loses overall, illustrating why a single aggregate is incomplete.

The five largest boosted-tree engine MAEs are:

| Engine | Rows | MAE | RMSE | Mean signed | Median signed | Worst absolute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 28 | 146 | 44.73 | 53.40 | +43.19 | +44.34 | 96.12 |
| 57 | 118 | 38.41 | 46.42 | +37.81 | +43.49 | 83.17 |
| 43 | 188 | 35.28 | 46.48 | +33.94 | +28.74 | 108.47 |
| 98 | 137 | 30.98 | 35.17 | +30.47 | +35.54 | 60.05 |
| 93 | 136 | 29.31 | 39.00 | +28.63 | +22.44 | 102.11 |

These identify model-specific failures on this split, not inherently difficult
engines or physical fault causes. The best/worst engines differ between models.

## RUL-range bias and sample balance

Retain the pre-existing ranges: near failure [0,30], middle (30,100], early >100.
Counts are 620, 1,400 and 1,452 respectively; each band covers all 20 engines.
These are retrospective target-defined groups, unavailable as known labels online.

| Model | RUL band | Rows | Engines | MAE | RMSE | Mean signed | Under % | Over % | Residual std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | near_failure_0_30 | 620 | 20 | 80.00 | 80.50 | +80.00 | 0.0 | 100.0 | 8.94 |
| baseline | middle_31_100 | 1400 | 20 | 29.93 | 35.76 | +29.50 | 7.1 | 91.4 | 20.21 |
| baseline | early_over_100 | 1452 | 20 | 47.97 | 56.51 | -47.97 | 100.0 | 0.0 | 29.87 |
| ridge | near_failure_0_30 | 620 | 20 | 21.95 | 27.17 | -1.54 | 49.0 | 51.0 | 27.13 |
| ridge | middle_31_100 | 1400 | 20 | 33.88 | 39.46 | +32.10 | 8.9 | 91.1 | 22.96 |
| ridge | early_over_100 | 1452 | 20 | 23.67 | 29.70 | -2.24 | 50.0 | 50.0 | 29.62 |
| gradient_boosting | near_failure_0_30 | 620 | 20 | 4.64 | 6.07 | +1.18 | 45.5 | 54.5 | 5.96 |
| gradient_boosting | middle_31_100 | 1400 | 20 | 23.96 | 32.41 | +19.99 | 23.4 | 76.6 | 25.51 |
| gradient_boosting | early_over_100 | 1452 | 20 | 24.02 | 30.72 | +3.74 | 45.4 | 54.6 | 30.49 |
| lstm | near_failure_0_30 | 620 | 20 | 7.89 | 10.25 | +6.86 | 14.7 | 85.3 | 7.62 |
| lstm | middle_31_100 | 1400 | 20 | 27.41 | 38.10 | +23.17 | 24.6 | 75.4 | 30.25 |
| lstm | early_over_100 | 1452 | 20 | 25.47 | 32.30 | +3.62 | 45.1 | 54.9 | 32.10 |

The constant baseline overpredicts every near-failure row by an average 80 cycles.
Its modest pooled signed bias hides opposite biases in early and late life.
Boosted-tree middle-life mean overprediction is +19.99 cycles, with 76.6% of rows
overpredicted. LSTM near-failure mean error is +6.86 cycles and 85.3% of these
rows overpredict. Ridge middle-life MAE (33.88) is worse than baseline (29.93).

Residual dispersion changes with RUL: boosted-tree residual standard deviation
rises from 5.96 near failure to 30.49 in early life; LSTM from 7.62 to 32.10.
This is descriptive evidence of nonconstant error variance (heteroscedasticity),
not a statistical test. Uniform error bars would obscure this variation.

## Available history and operating conditions

Available-cycle bands are distinct from RUL bands. Actual evaluated cycles start
at 20 because a complete window is required. Counts: cycles 20–50: 620; 51–100:
1,000; >100: 1,852, each covering 20 engines.

| Model | MAE at cycles ≤50 | MAE at 51–100 | MAE at >100 |
| --- | ---: | ---: | ---: |
| baseline | 62.72 | 32.05 | 48.71 |
| ridge | 28.03 | 28.12 | 26.95 |
| gradient_boosting | 27.74 | 27.53 | 14.35 |
| lstm | 28.05 | 31.86 | 16.73 |

Boosted-tree errors are smaller at later available cycles, but age, true RUL and
engine composition are confounded; this does not prove that longer history causes
better predictions. Every prediction still uses a 20-cycle window. Complete
validation trajectories span 137–269 cycles; their total lengths are retrospective
outcomes, never model inputs. The history-length scatter labels all 20 engines.
FD001 does not support a broad operating-regime generalization claim here. This
audit measures engine clusters; it does not infer regime labels or physical causes.

## Extreme errors and diagnostics

| Model | Engine | Cycle | Actual RUL | Prediction | Signed error |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 5 | 20 | 249 | 95.00 | -154.00 |
| ridge | 22 | 21 | 181 | 84.74 | -96.26 |
| gradient_boosting | 43 | 122 | 85 | 193.47 | +108.47 |
| lstm | 57 | 84 | 53 | 163.51 | +110.51 |

Of boosted trees’ top 20 absolute errors, 11 occur on engine 43; 10 of LSTM’s top
20 occur on engine 57; 12 of Ridge’s top 20 occur on engine 28. Nearby cycles
overlap and are correlated; these are not counts of independent failure events.
Top-20 rankings use deterministic engine/cycle tie breaking and include signs.

![Residuals by actual RUL](../artifacts/reports/FD001/reliability/residuals_vs_rul.png)

![Largest signed errors](../artifacts/reports/FD001/reliability/largest_errors.png)

Additional plots in the same output directory: `actual_vs_predicted.png`,
`residual_distributions.png`, `absolute_error_ecdf.png`, `absolute_error_by_engine.png`,
`errors_by_rul_band.png`, and `error_vs_history_length.png`. Axis scales in the
residual and extreme-error panels are independent; compare numeric axes, not bar
lengths across panels. Tables retain every engine and the full selected subset.

## Synthetic stability probe

A single fixed probe adds independent zero-mean Gaussian noise to all 15 selected
raw sensors on validation histories only. Noise standard deviation is 1% of each
sensor’s saved training standard deviation; seed 42. Unselected channels, engine
IDs, cycles and targets are unchanged. Causal features and windows are regenerated
through the frozen scaler; exact endpoint membership is asserted. No parameter
or threshold was chosen by searching for favorable results. The >5-cycle change
threshold is a descriptive count, not an operational acceptance limit.

| Model | Mean absolute shift | P95 shift | Max shift | Shift >5 cycles % | Perturbed MAE | MAE change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.000 | 0.000 | 0.000 | 0.00 | 46.415 | +0.000 |
| ridge | 0.548 | 1.302 | 2.437 | 0.00 | 27.441 | -0.038 |
| gradient_boosting | 1.204 | 4.186 | 16.540 | 3.28 | 20.500 | -0.036 |
| lstm | 0.288 | 0.788 | 1.833 | 0.00 | 23.096 | -0.016 |

The boosted-tree maximum shift is 16.54 cycles despite a mean shift of 1.20;
3.28% of observations move by more than five cycles. This warrants further
sensor-sensitivity work, even though pooled MAE barely changes. The constant
baseline’s zero sensitivity is not evidence of useful reliability. Small MAE
decreases under one noise draw are not improvements or statistically significant
effects. Noise is synthetic, uncalibrated to hardware, and does not model drift,
dropouts, sensor correlations or real operational failures. These are not uncertainty
intervals and are not used to modify the benchmark models.

## Input contracts and operational risks

All four saved predictors rejected reordered feature names, a NaN input and a
short window (12 checks). Missing values are rejected, not silently imputed;
full raw-history inference also requires complete history from cycle 1. This
means a future interface must expose invalid-input failures rather than invent
predictions. Correctly named but semantically wrong sensor units cannot be caught
by ordering checks alone. No online missing-data recovery was added.

The same validation engines selected classical hyperparameters and the LSTM
checkpoint, so audit scores are optimistic development evidence. There are only
20 validation engines, one dataset and one noise seed. No group-bootstrap
intervals, calibration study, external dataset or final official test evaluation
was performed. Error magnitudes are operating cycles, not hours or days.

## Recommendations for the next approved phase

1. Keep boosted trees as the current development benchmark, with explicit warnings
   about middle-life overprediction, rare large shifts and invalid negative outputs.
2. Show per-engine and RUL-band diagnostics alongside aggregate metrics in any
   future dashboard; preserve data/schema validation and display errors clearly.
3. Define safety/decision costs and a separately evaluated handling policy for
   negative outputs before deploying; do not silently clip or revise current scores.
4. Freeze selection and define final-endpoint evaluation before opening official
   test targets. Calibration, realistic sensor-corruption studies and other datasets
   need separately approved protocols; never tune on final-test feedback.

## Reproduction and verification

From the repository root:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -W error -m aevora.audit --config configs/default.toml
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -W error -m unittest discover -s tests -v
```

Outputs: configured `artifact_dir/reports/FD001/reliability/`. `audit.json` records
all tables, the perturbation definition, exact engine IDs and input SHA256 hashes.
CSVs include model metrics, per-engine comparisons, variability, grouped errors,
largest errors, perturbation summaries/per-engine shifts, input-contract checks
and exact validation endpoints. Generated outputs remain ignored by Git.

The existing 55 tests passed before changes. The completed suite has 62 tests;
all pass with warnings treated as errors. Known-example tests verify signs,
MAE/RMSE, band boundaries, empty groups, singleton variability, finite values,
and ranking/alignment. A tiny synthetic end-to-end fixture forbids refitting
during the audit, verifies training-only reads, checks unchanged input hashes,
and reproduces JSON exactly. Fixture training does not touch benchmark models.
Full-data audit artifacts were also generated and visually inspected.

No final test evaluation or model tuning is authorized by this audit. Work stops
after Phase 6 pending approval.
