# Phase 5 measured findings

One planned CPU LSTM run; no hyperparameter sweep or official test access.
All results below are development-validation scores, not final test performance.

## Data and model

- Training: 15,259 sequences × 20 timesteps × 75 ordered features, 80 engines.
- Validation: 3,472 sequences × 20 timesteps × 75 features, 20 disjoint engines.
- Labels: uncapped endpoint RUL. Frozen Phase 3 input scaling and engine split.
- Architecture: one unidirectional 32-unit LSTM, last output → dropout 0.1 → linear scalar.
- Parameters: 13,985. Runtime: Python 3.13.1, PyTorch 2.14.1.
- MPS was built but unavailable; used deterministic CPU operations with 2 threads.
- Seed 42; batch 64; AdamW learning rate 0.001, weight decay 0.0001; gradient clip 1.
- Training target mean 101.094633, standard deviation 66.093405, stored in checkpoint.
- Loss: standardized-target MSE; history losses converted to cycles squared.

## Training outcome

Completed 6 epochs out of 30; patience 5 triggered early stopping.
Restored epoch 1, selected by validation MSE, before saving and evaluation.
Measured training-plus-validation loop time: 5.219127 seconds; preparation and saving excluded.
Validation RMSE by epoch: 32.257, 33.074, 32.404, 36.332, 34.877, 35.710 cycles.
Training MSE fell from 1720.797 to 148.182 while validation error worsened:
evidence of overfitting on this development split. No extra run was used to chase
the boosted-tree score. Epoch 3 had lower MAE but higher RMSE than epoch 1;
selection followed the declared MSE criterion, rather than choosing metrics afterward.

## Aligned comparison

| Model | MAE (cycles) | RMSE (cycles) | Engine-macro MAE |
| --- | ---: | ---: | ---: |
| baseline | 46.414747 | 54.846329 | 45.439338 |
| ridge | 27.479194 | 33.598605 | 28.189250 |
| gradient_boosting | 20.535715 | 28.717121 | 21.228417 |
| lstm | 23.111861 | 32.256839 | 24.212676 |

All four models use exactly the same 3,472 engine/cycle targets. LSTM improves
on the median and Ridge here, but boosted trees remain best overall.
LSTM negative predictions: 0 in this validation run; outputs are not clipped.

| LSTM RUL band | Observations | MAE | RMSE |
| --- | ---: | ---: | ---: |
| near_failure_0_30 | 620 | 7.886613 | 10.251669 |
| middle_31_100 | 1400 | 27.405138 | 38.100274 |
| early_over_100 | 1452 | 25.473477 | 32.303501 |

The checkpoint was reloaded and reproduced all validation predictions exactly.
Validation was used for early stopping/model selection. Windows overlap, longer
histories receive more pooled weight, and only one split/seed was measured.
No confidence intervals or deployment-readiness claims follow from these results.
Classical fit times and LSTM training-plus-validation time have different scopes.

## Verification and artifacts

Initial suite: 45 passed. Final suite: 55 passed, 0 failures, 0 skips, with
warnings treated as errors. Tests cover dimensions, ordering, finite output,
fixed-seed training, engine overlap rejection, actual early-stop restoration,
checkpoint round trips, frozen preprocessing checksums, raw-prefix inference,
training-only data access, corruption rejection and aligned comparison metrics.
One synthetic-fixture feature-count assertion was corrected during development;
the real FD001 feature count remains 75.

- Model bundle: `artifacts/models/FD001/lstm/`.
- LSTM curves and diagnostics: `artifacts/reports/FD001/lstm_validation/`.
- Four-model comparison: `artifacts/reports/FD001/validation_with_lstm/`.
- Existing classical artifacts remain unchanged; generated outputs are Git-ignored.

Checkpoint SHA256: `9004cb04d7d2bf05c9bda9d8ed28696e0a9d3079ddd038d99e5a4881c4cb3b80`.
Training source SHA256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.

See [LSTM usage and evaluation contract](lstm.md) for commands and file descriptions.
Phase 6 has not started. Official test trajectories and RUL targets were not read.
