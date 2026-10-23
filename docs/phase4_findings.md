# Phase 4 measured findings

FD001; seed 42; uncapped RUL; frozen Phase 3 preprocessing. Training: 15,259
windows from 80 engines. Validation: 3,472 windows from 20 disjoint engines.
Input: last/mean/population-standard-deviation summaries, 225 dimensions.

| Model | MAE (cycles) | RMSE (cycles) | Selected fit (s) | All candidate fits (s) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 46.414747 | 54.846329 | 0.002049 | 0.002049 |
| ridge | 27.479194 | 33.598605 | 0.287863 | 0.638618 |
| gradient_boosting | 20.535715 | 28.717121 | 2.253095 | 4.012925 |

Baseline predicts the training-target median, 95 cycles. Ridge selected alpha 1
from {1, 10, 100}. Boosting selected 15 leaves from {7, 15}, with 200 iterations,
learning rate 0.05, L2 1.0, no internal early stopping. Selection used validation
RMSE. Times are measured local CPU estimator fits, not full workflow duration.

| Model | Near failure MAE (0–30) | Middle MAE (31–100) | Early MAE (>100) |
| --- | ---: | ---: | ---: |
| baseline | 80.0000 | 29.9286 | 47.9697 |
| ridge | 21.9545 | 33.8753 | 23.6712 |
| gradient_boosting | 4.6403 | 23.9604 | 24.0210 |

Band counts: 620 near-failure, 1,400 middle-life, 1,452 early-life observations.
All three bands include all 20 validation engines. Boosting has the lowest overall
MAE and RMSE and much lower near-failure error, but Ridge is slightly better on
early-life MAE and RMSE. Boosting is not uniformly superior across conditions.

Negative predictions were retained: baseline 0, Ridge 207, boosting 10. Such
estimates are physically invalid and are a known limitation, not silently clipped.
Engine-macro MAE: baseline 45.4393, Ridge 28.1893, boosting 21.2284 cycles.

Validation selected hyperparameters, so these are development scores, not final
test performance. Windows overlap; long histories receive more pooled weight.
No official test trajectories or official RUL targets were opened in this phase.
These results do not establish real-world maintenance savings or reliability.

Artifacts: `artifacts/models/FD001/{baseline,ridge,gradient_boosting}/` and
`artifacts/reports/FD001/validation/`. Each bundle contains estimator, ordered
schema, frozen preprocessing, candidate metadata and endpoint predictions.
Machine-readable reports contain full precision scores, per-engine/band errors
and input artifact hashes. Three diagnostic plots use measured predictions.

Verification: the initial 36 tests passed; the final suite passed 45 tests with
warnings treated as errors and no skips or failures. Tests cover summary order,
corrupt-target rejection, disjointness, all model reloads, deterministic fits,
bundle relocation, raw-prefix inference and identical evaluation alignment.

Training source SHA256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
Runtime: Python 3.13.1; NumPy 2.4.2; pandas 2.3.3; scikit-learn 1.6.1.
See [training contract](classical_models.md) for reproducible commands.
