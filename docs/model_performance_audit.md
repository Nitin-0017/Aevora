# Comprehensive model performance audit

Audit date: 2026-10-24. Repository HEAD: `a93768c` (Phase 6). Scope: existing saved **development-validation** predictions, not an unseen-test evaluation. No benchmark training, model modification, test trajectory access, or official test-target access was performed. No commits or pushes were made.

## Findings

Boosted trees are the strongest overall development candidate: lowest MAE, RMSE, median absolute error, P90/P95 absolute error, highest R², and highest coverage at each requested tolerance. Their MAE is 55.76% below the median baseline and 11.15% below LSTM. This is descriptive superiority on this split, not evidence of statistical significance or production readiness.

Their typical absolute error is 13.33 cycles by the median and 20.54 by the mean. Only 42.08% of predictions fall within ±10 cycles; 62.24% fall within ±20. P95 error is 60.34 cycles and the worst error is 108.47. The mean signed error of +9.83 indicates overstatement of remaining life. These errors are material for maintenance decisions.

Ridge has the smallest observed maximum error (96.26 cycles), despite worse averages and 207 negative predictions. LSTM has no negative predictions, but larger pooled error and more variation across engines. The constant baseline has the smallest aggregate signed bias and perfect constancy under sensor perturbations; neither implies useful predictions. No model dominates every criterion.

## Evidence and protocol verification

The audit reconstructed saved training and validation tensors from the official **training trajectories only**, with the frozen scaler. It independently derived uncapped endpoint RUL as an engine's final observed training-dataset cycle minus the window endpoint cycle, checked saved labels and window arrays exactly, and verified source and preparation hashes.

All four saved model bundles were loaded for inference; reproduced predictions agreed with their saved CSVs at rtol/atol 1e-12. Metrics below were computed directly from those original CSV values, not copied from documentation. Engine/cycle/target keys and ordering matched exactly across all four models and the reconstructed reference. No subset filtering, clipping, or dropped invalid rows was applied.

Training tensor: **15,259 × 20 × 75**, from 80 engines. Validation tensor: **3,472 × 20 × 75**, from 20 separate engines. Classical models summarize the same windows into 225 features. All models evaluate all eligible chronological endpoints (stride 1), excluding each engine's first 19 cycles. There are no model-specific eligibility differences. Raw partitions contain 16,779 training rows and 3,852 validation rows.

The 75 ordered features comprise 15 training-selected sensors, each with its current value, trailing 5-cycle mean, population standard deviation, first difference, and trailing slope. Sensor selection and scaler fitting use training engines only; rolling calculations are causal and grouped by engine. Windows cannot cross engine boundaries. Full lifetime is used only to construct retrospective labels, never as an input feature. The code and saved provenance show no detected cross-engine or future-feature leakage; this is not proof against every possible source of dataset bias.

Validation engines: 3, 5, 19, 22, 25, 26, 28, 34, 43, 51, 52, 53, 57, 60, 62, 72, 82, 93, 98, 100. Seed: 42. RUL is uncapped.

SHA-256 fingerprints of all model and preparation files were identical before and after execution. The complete fingerprint inventory and split IDs are in `audit.json`.

## Metric definitions and edge cases

Let y be actual RUL, p the prediction, e = p − y, and n the number of evaluated observations. All error measures below are in cycles except R² and percentages.

- **MAE:** mean |e|. Average error magnitude, with each observation equally weighted.
- **RMSE:** sqrt(mean e²). Gives larger errors more weight.
- **R²:** 1 − sum(e²)/sum((y − mean(y))²). Dimensionless; negative means worse squared error than predicting this validation subset's mean. That reference is distinct from the fitted training-median baseline.
- **Median absolute error:** median |e|; a typical-error summary less sensitive to extremes.
- **Mean signed error / bias:** mean e. Positive means overprediction of remaining life, negative means underprediction; opposite errors can cancel. It is not a second independent metric.
- **Maximum absolute error:** max |e| on these observations, not a bound on future errors.
- **P90/P95 absolute error:** 90th/95th percentile of |e|, using NumPy linear interpolation. These are descriptive quantiles, not confidence or prediction intervals.
- **Within ±t cycles:** 100 × count(|e| ≤ t)/n, inclusive, for t = 5, 10, 20, 30. These are regression tolerance rates, not classification accuracy.
- **Negative percentage:** 100 × count(finite p < 0)/n. Nonfinite percentage is 100 × count(NaN or ±infinity)/n; invalid values are counted separately.
- **Observation and engine counts:** number of endpoints and unique engine IDs. Equal-engine MAE is the unweighted mean of the engine MAEs, unlike pooled observation-weighted MAE.

The metric helper returns null for empty denominators, and null R² for singleton or constant-target groups. Nonfinite predictions invalidate error scores rather than silently excluding rows; they count as tolerance misses. Invalid target values or vector dimensions are rejected. The production audit loader additionally fails closed if predictions are nonfinite or observations do not align. All actual audited predictions are finite, so none of these exceptions changes the results.

## Verified pooled metrics

Every row below uses n = 3,472 observations and 20 engines.

| model | mae | rmse | r2 | median_absolute_error | mean_signed_error |
| --- | --- | --- | --- | --- | --- |
| baseline | 46.41 | 54.85 | -0.01 | 44.00 | 6.12 |
| ridge | 27.48 | 33.60 | 0.62 | 24.07 | 11.73 |
| gradient_boosting | 20.54 | 28.72 | 0.72 | 13.33 | 9.83 |
| lstm | 23.11 | 32.26 | 0.65 | 14.46 | 12.08 |

R² with additional precision: baseline -0.012607, ridge 0.619996, gradient_boosting 0.722395, lstm 0.649741.

| model | p90_absolute_error | p95_absolute_error | max_absolute_error | negative_pct | nonfinite_pct |
| --- | --- | --- | --- | --- | --- |
| baseline | 86.00 | 92.00 | 154.00 | 0.00 | 0.00 |
| ridge | 54.80 | 63.04 | 96.26 | 5.96 | 0.00 |
| gradient_boosting | 49.08 | 60.34 | 108.47 | 0.29 | 0.00 |
| lstm | 57.30 | 70.44 | 110.51 | 0.00 | 0.00 |

| model | within_5_pct | within_10_pct | within_20_pct | within_30_pct |
| --- | --- | --- | --- | --- |
| baseline | 6.34 | 12.10 | 23.62 | 34.91 |
| ridge | 10.83 | 22.21 | 42.80 | 60.11 |
| gradient_boosting | 25.09 | 42.08 | 62.24 | 73.21 |
| lstm | 20.65 | 38.62 | 59.39 | 71.46 |

Negative counts: median baseline 0, Ridge 207, boosted trees 10, LSTM 0. All negative predictions occur in the 0–30 actual-RUL band. They remain in all metrics; no post-hoc clipping is used. Nonfinite count: 0 for every model.

## MAE comparisons

Percentage improvement = 100 × (reference MAE − model MAE)/reference MAE. Positive means lower error; negative means worse. The best alternative is the other model with the smallest pooled MAE; it excludes the model being compared. A zero reference MAE produces an undefined percentage, represented as null.

| model | mae_improvement_vs_baseline_pct | best_alternative | mae_improvement_vs_best_alternative_pct |
| --- | --- | --- | --- |
| baseline | 0.00 | gradient_boosting | -126.02 |
| ridge | 40.80 | gradient_boosting | -33.81 |
| gradient_boosting | 55.76 | lstm | 11.15 |
| lstm | 50.21 | gradient_boosting | -12.54 |

## Engine breakdown

Full metrics, tail errors, bias, tolerance percentages and counts for each engine/model are in `per_engine.csv`. The table below shows MAE in cycles; each engine uses identical endpoints across models.

| unit_id | observations | baseline | gradient_boosting | lstm | ridge |
| --- | --- | --- | --- | --- | --- |
| 3 | 160 | 41.50 | 20.00 | 31.98 | 33.36 |
| 5 | 250 | 65.98 | 16.42 | 28.82 | 29.91 |
| 19 | 139 | 39.61 | 12.82 | 15.35 | 25.85 |
| 22 | 183 | 45.84 | 16.85 | 17.38 | 24.73 |
| 25 | 211 | 53.22 | 20.61 | 14.65 | 20.72 |
| 26 | 180 | 45.17 | 18.25 | 13.19 | 23.73 |
| 28 | 146 | 39.97 | 44.73 | 43.14 | 48.45 |
| 34 | 176 | 44.32 | 9.77 | 10.74 | 17.27 |
| 43 | 188 | 47.01 | 35.28 | 43.72 | 29.34 |
| 51 | 194 | 48.51 | 10.98 | 10.15 | 26.84 |
| 52 | 194 | 48.51 | 18.95 | 12.55 | 21.19 |
| 53 | 176 | 44.32 | 15.37 | 10.41 | 19.58 |
| 57 | 118 | 40.79 | 38.41 | 62.99 | 41.81 |
| 60 | 153 | 40.61 | 19.94 | 24.38 | 32.80 |
| 62 | 161 | 41.65 | 10.11 | 12.67 | 19.18 |
| 72 | 194 | 48.51 | 18.27 | 34.39 | 24.22 |
| 82 | 195 | 48.77 | 23.24 | 13.25 | 30.70 |
| 93 | 136 | 39.56 | 29.31 | 44.20 | 38.40 |
| 98 | 137 | 39.57 | 30.98 | 28.72 | 36.19 |
| 100 | 181 | 45.39 | 14.27 | 11.59 | 19.51 |

| model | equal_engine_mae | engine_mae_population_std | best_engine_mae | worst_engine_mae |
| --- | --- | --- | --- | --- |
| baseline | 45.44 | 6.06 | 39.56 | 65.98 |
| gradient_boosting | 21.23 | 9.47 | 9.77 | 44.73 |
| lstm | 24.21 | 14.67 | 10.15 | 62.99 |
| ridge | 28.19 | 8.16 | 17.27 | 48.45 |

LSTM beats boosted-tree MAE on 9 of 20 engines, yet has worse pooled and equal-engine MAE. Its worst engine is 57 (MAE 62.99), compared with boosted trees' worst engine 28 (MAE 44.73). This shows why pooled averages alone are insufficient. Longer trajectories contribute more overlapping endpoints and therefore receive more pooled weight.

## RUL-range breakdown

Existing ranges are retained: near failure [0,30], middle (30,100], early >100 cycles. Every band contains all 20 validation engines. Bands are based on actual RUL and can be known retrospectively only. All remaining requested metrics by band are saved in `rul_bands.csv`.

| model | band | observations | mae | rmse | mean_signed_error | within_10_pct | within_20_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | near_failure_0_30 | 620 | 80.00 | 80.50 | 80.00 | 0.00 | 0.00 |
| baseline | middle_31_100 | 1400 | 29.93 | 35.76 | 29.50 | 22.86 | 37.14 |
| baseline | early_over_100 | 1452 | 47.97 | 56.51 | -47.97 | 6.89 | 20.66 |
| ridge | near_failure_0_30 | 620 | 21.95 | 27.17 | -1.54 | 28.06 | 53.23 |
| ridge | middle_31_100 | 1400 | 33.88 | 39.46 | 32.10 | 13.93 | 28.79 |
| ridge | early_over_100 | 1452 | 23.67 | 29.70 | -2.24 | 27.69 | 51.86 |
| gradient_boosting | near_failure_0_30 | 620 | 4.64 | 6.07 | 1.18 | 91.29 | 99.35 |
| gradient_boosting | middle_31_100 | 1400 | 23.96 | 32.41 | 19.99 | 33.00 | 57.14 |
| gradient_boosting | early_over_100 | 1452 | 24.02 | 30.72 | 3.74 | 29.82 | 51.31 |
| lstm | near_failure_0_30 | 620 | 7.89 | 10.25 | 6.86 | 74.68 | 93.87 |
| lstm | middle_31_100 | 1400 | 27.41 | 38.10 | 23.17 | 36.71 | 55.21 |
| lstm | early_over_100 | 1452 | 25.47 | 32.30 | 3.62 | 25.07 | 48.69 |

Boosted-tree near-failure MAE is 4.64 cycles, but middle-life MAE is 23.96 with mean signed error +19.99. LSTM near-failure bias is +6.86 and middle-life bias +23.17. Ridge underpredicts slightly on average near failure (−1.54) and early life (−2.24), while strongly overpredicting in middle life (+32.10). The baseline's +6.12 pooled bias hides +80.00 near failure and −47.97 in early life. A small pooled bias can therefore be misleading.

Ridge has slightly lower early-life MAE than boosted trees (23.67 vs 24.02), despite losing overall. Residual spread varies with RUL; a single uniform error allowance would conceal this variation.

## Extreme errors and stability

| model | engine | cycle | actual_rul | prediction | signed_error |
| --- | --- | --- | --- | --- | --- |
| baseline | 5 | 20 | 249.00 | 95.00 | -154.00 |
| ridge | 22 | 21 | 181.00 | 84.74 | -96.26 |
| gradient_boosting | 43 | 122 | 85.00 | 193.47 | 108.47 |
| lstm | 57 | 84 | 53.00 | 163.51 | 110.51 |

Positive errors can postpone maintenance and negative errors can prompt premature maintenance; these are plausible consequences, not measured operational outcomes. Worst errors are individual correlated endpoints, not independent failure-event rates.

The Phase 6 synthetic sensitivity results below were reused only after all current model and preparation hashes matched that audit. The perturbation experiment was **not rerun**. It added independent zero-mean Gaussian noise to selected raw sensors, at 1% of each sensor's frozen training standard deviation, seed 42, and regenerated causal features.

| model | observations | noise_fraction | seed | mean_absolute_prediction_change | p95_absolute_prediction_change | max_absolute_prediction_change | fraction_change_over_5_cycles | perturbed_mae | perturbed_rmse | mae_change |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3472 | 0.01 | 42 | 0.00 | 0.00 | 0.00 | 0.00 | 46.41 | 54.85 | 0.00 |
| ridge | 3472 | 0.01 | 42 | 0.55 | 1.30 | 2.44 | 0.00 | 27.44 | 33.54 | -0.04 |
| gradient_boosting | 3472 | 0.01 | 42 | 1.20 | 4.19 | 16.54 | 0.03 | 20.50 | 28.68 | -0.04 |
| lstm | 3472 | 0.01 | 42 | 0.29 | 0.79 | 1.83 | 0.00 | 23.10 | 32.23 | -0.02 |

Prediction shifts and errors in that table are in cycles; `fraction_change_over_5_cycles` is a fraction (boosted trees: 3.28%), not a percentage.

This measures sensitivity to one small artificial perturbation, not real sensor corruption, calibrated uncertainty, or operational reliability. Ridge and LSTM were less sensitive in prediction shifts than boosted trees; the constant baseline cannot respond to sensor changes. Slightly improved noisy MAE is not evidence that noise improves generalization. Between-engine variability above is a different type of instability.

## Generalization and production readiness

**Current results do not support a production-readiness claim.** They demonstrate useful predictive signal under one development protocol. Validation data selected Ridge regularization and boosted-tree leaf count using RMSE, and selected the LSTM checkpoint through early stopping. Repeated model comparison and reliability analysis further expose this same split to development decisions. Selection creates potential optimism; its magnitude cannot be quantified from these saved results alone. No unbiased generalization estimate or significance claim is available.

The 3,472 windows are highly correlated and come from only 20 held-out engines. They are not 3,472 independent experiments. This is one seed and one dataset subset (FD001), without a demonstrated range of field conditions. Complete held-out training trajectories at every eligible endpoint differ from an official truncated-trajectory evaluation at one endpoint per engine. Negative predictions, positive near-failure bias, large tail errors, and unequal engine performance remain unresolved.

A later untouched evaluation can use the existing frozen inference pipeline on one final available window per official test engine. Before opening labels, freeze the primary model (boosted trees is the present development candidate), any secondary comparisons, feature order/scaler, target definition, eligibility and short-history policy, metrics, and negative-output handling. Report eligibility counts. Then, **only with user approval**, read the official endpoint RUL targets once and evaluate without refitting or selecting models from the outcome. Do not compare that endpoint-only metric directly with today's all-window protocol as if the populations were identical. This audit has neither opened those files nor executed that evaluation; historical non-use is documented project provenance, not something file contents could prove.

Before strong field-generalization claims, additional evidence should include engine-grouped development resampling (nested selection where applicable), engine-level uncertainty estimates, an untouched external cohort, sensor missingness/drift/fault stress tests, and maintenance-cost-sensitive evaluation with domain experts. Calibrated prediction intervals require a separate calibration protocol. Deployment additionally needs monitoring and a human-reviewed decision policy. None of these experiments was performed in this audit.

## Outputs and reproduction

Run from the repository with its existing environment:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -W error -m aevora.performance_audit --config configs/default.toml
```

This verifies existing artifacts and regenerates this audit's CSV/JSON and plots; it does not train models or regenerate this narrative report. Outputs are under `artifacts/reports/FD001/performance_audit/` (ignored by Git, following existing conventions):

- `audit.json`: exact metrics, model/preparation hashes, tensor shapes, split IDs, alignment result and prior stability provenance.
- `metrics.csv`, `per_engine.csv`, `rul_bands.csv`: all extended metrics and group counts.
- Seven PNG diagnostics below, with cycle units and sample counts.

![actual vs predicted](../artifacts/reports/FD001/performance_audit/actual_vs_predicted.png)

![residual distribution](../artifacts/reports/FD001/performance_audit/residual_distribution.png)

![absolute error distribution](../artifacts/reports/FD001/performance_audit/absolute_error_distribution.png)

![mae by model](../artifacts/reports/FD001/performance_audit/mae_by_model.png)

![error tolerances](../artifacts/reports/FD001/performance_audit/error_tolerances.png)

![per engine comparison](../artifacts/reports/FD001/performance_audit/per_engine_comparison.png)

![error vs actual rul](../artifacts/reports/FD001/performance_audit/error_vs_actual_rul.png)


## Verification and exact source changes

Full suite: **66 tests passed**, with warnings treated as errors. Four new analytical tests cover known error scores and quantiles, inclusive tolerance boundaries, constant/singleton/empty targets, invalid values, exact alignment, group coverage and competitor-relative improvement calculations. Existing full-suite training checks use temporary synthetic fixtures; no saved benchmark was retrained.

New source/document files only:

- `src/aevora/performance_audit.py`
- `tests/test_performance_audit.py`
- `docs/model_performance_audit.md`

No existing tracked file, model checkpoint, preprocessing artifact, configuration, commit or remote was changed. The audit creates the eleven output files listed above (four CSV/JSON files and seven PNGs). Matplotlib may also maintain its ignored cache. All numbers are verified descriptive results except explicitly stated interpretations and proposed future evaluation; no future performance estimates are asserted.
