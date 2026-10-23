# Classical model training

Phase 4 uses the Phase 3 scaler and windows without refitting preprocessing.
The loader reconstructs every window from the official training file using the
saved scaler, and checks source hash, split, sensor selection, feature order,
labels and endpoint CSVs against the saved arrays. Official test data is not read.

Each 20 × 75 window becomes one 225-dimensional vector: the 75 last values,
then 75 historical means, then 75 population standard deviations. The target
is RUL at the final cycle. These statistics use only available history. They
reduce CPU cost compared with 1,500 flattened features but lose detailed temporal
ordering; existing trailing slopes and changes partly retain local dynamics.
No additional normalization or target clipping is applied. Negative predictions
are retained in evaluation, so metrics do not hide impossible estimates.

The baseline uses median **training window targets**, not validation targets or
all pre-window rows. Ridge searches alpha × {1, 10, 100}, where alpha comes from
`configs/default.toml` (default candidates 1, 10, 100); duplicates are removed.
SVD is deterministic and handles correlated summary features. The candidate
with lowest validation RMSE wins, with first-candidate tie breaking. This is a
small development search, not an independent test estimate or nested validation.

Run from the repository root:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m aevora.training --config configs/default.toml --model baseline
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m aevora.training --config configs/default.toml --model ridge
```

Models go under configured `artifact_dir/models/FD001/<model>/`. Each bundle
contains the estimator pickle, ordered feature schema, copied preprocessing,
metadata (candidate parameters, fit times, versions, source/artifact hashes),
and validation predictions with engine/cycle keys. Load only trusted local
pickles. Keep the bundle intact and use the recorded library versions.
`Predictor.load(bundle).predict(windows, ordered_feature_names)` enforces schema;
`predict_history(bundle, available_history)` reuses the saved preprocessing and
returns each engine's latest prediction. Paths inside bundles are portable.

HistGradientBoostingRegressor searches two leaf budgets: floor(configured leaves
/ 2), at least 2, and configured leaves (defaults 7 and 15). Other defaults are
200 iterations, learning rate 0.05, L2 regularization 1.0 and seed 42. The fixed
iteration budget is modest; early stopping is deliberately disabled because the
installed estimator's internal random validation split would mix overlapping
windows from the same engines. Only the existing engine-disjoint validation
partition selects the candidate, by the same RMSE rule as Ridge. No validation
rows are used to fit a tree or scaler. CPU thread limits are set before Python
starts in the documented commands; no GPU or API key is required.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m aevora.training --config configs/default.toml --model gradient_boosting
```

## Evaluation contract

`python -m aevora.evaluation --config configs/default.toml` verifies preparation,
requires identical preparation hashes for all models, checks saved target and
engine/cycle order, reloads each estimator and verifies its saved predictions and
metrics. It does not retrain. Reports go under configured
`artifact_dir/reports/FD001/validation/`: metrics JSON/CSV, per-engine MAE/RMSE,
RUL-band errors, actual-versus-predicted scatter, residual histograms and engine
MAE bars. Residual means prediction minus actual; positive is overprediction.

Bands are near failure 0–30 cycles, middle >30–100, and early >100. Labels are
integer cycles. Empty bands have null metrics, not fabricated zero error.
MAE averages absolute error; RMSE squares errors before averaging and therefore
penalizes large misses more strongly. They can rank models differently when one
has fewer typical errors but some much larger misses. Both have cycle units.

All 3,472 endpoints from 20 validation engines are evaluated with uncapped RUL.
Overlapping windows are correlated; these are not 3,472 independent engines.
Longer engines contribute more observations. Per-engine errors and macro MAE
(equal engine weights) complement pooled metrics. This protocol differs from
one final endpoint per truncated official test engine. Validation is reused for
tuning, so scores are optimistic development estimates; there is no confidence
interval or industrial reliability claim. No imputation, additional scaling or
calibration is fitted in this phase. Inference needs full available histories
from cycle 1 and at least 20 cycles per engine, as in Phase 3.

Training times measure estimator `.fit()` only, excluding verification, summary
construction, prediction and serialization. Search time sums all candidate fits;
the selected fit is retained without refitting. Commands overwrite each named
bundle/report; use a separate configured artifact directory to preserve another
experiment (prepare its Phase 3 artifacts first). Git ignores generated data,
models and reports; the small reviewed findings document records this run.
