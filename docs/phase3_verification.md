# Phase 3 verification

Source training file SHA256: `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8`.
Seed: 42; validation fraction: 0.2; RUL cap: None.

- Training: 80 engines, 16779 rows.
- Validation: 20 engines, 3852 rows.
- Selected sensors: 15; engineered features: 75.
- Training windows: [15259, 20, 75].
- Validation windows: [3472, 20, 75].
- Short engines skipped: {'train': [], 'validation': []}.

The data preparation command saved scaler/manifest JSON, separate capped and
uncapped label columns in two CSVs, two endpoint CSVs, and two compressed window
archives under ignored `artifacts/prepared/FD001/`.

Independent artifact checks confirmed labels match window endpoint keys, all
window inputs are finite, partitions are disjoint, and saved scaler means equal
means recomputed solely from the training partition. Unit tests also check that
changing validation values leaves saved scaler state identical and that changing
future sensor values leaves earlier features/windows unchanged. Saved inference
uses the same features and frozen scaler as training preparation.

These are preparation checks, not model-performance metrics. No estimator or
LSTM was trained; official test trajectories and targets were not read.
Scaling includes all training rows. Histories are overlapping within a partition;
future model comparisons must use the same engine split and prediction endpoints.
