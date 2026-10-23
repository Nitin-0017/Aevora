# Phase 5 LSTM

The CPU model consumes the existing scaled chronological `[batch, 20, 75]`
windows directly. It uses one unidirectional LSTM layer with 32 hidden units,
last-timestep output, dropout 0.1 and a linear scalar head. It resets recurrent
state for every window; it never carries state between engines or batches.

The central TOML controls hidden size, layer count, dropout, batch size (64),
AdamW learning rate (0.001), weight decay (0.0001), gradient norm clipping (1),
maximum epochs (30), patience (5), minimum validation-MSE improvement (0), CPU
thread count (2) and device (`cpu`). The project seed is 42. CPU deterministic
algorithms and a seeded batch-shuffle generator are enabled. Timestep order
inside each window is preserved even though training windows are shuffled.
Determinism is tested within the same runtime; different platforms/library
versions or prediction batch sizes can introduce floating-point differences.

Target mean and population standard deviation are computed **only from training
window targets** (scale floored at 1). Training minimizes standardized MSE;
the model converts its output back to cycles using saved buffers. This avoids
poor optimization from the large target magnitude. No additional input scaler
is fitted; frozen Phase 3 preprocessing is reused. Predictions are not clipped.

Early stopping monitors validation MSE in cycles squared, not training loss.
An independent snapshot is retained when loss improves by more than `min_delta`.
After five consecutive non-improvements, training stops; otherwise it ends at
30 epochs. In both cases the best snapshot is restored, then serialized. The
training loop and checkpoint round trip explicitly verify restoration. A tie is
not an improvement. Validation is used only for selection and early stopping.

Epoch history contains online training MSE (dropout enabled, weights changing
through the epoch) and end-of-epoch validation MSE/MAE/RMSE (evaluation mode).
These curves have different measurement conditions. Training duration includes
all epoch fits and validation passes, unlike classical estimator-only fit time.

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -W error -m aevora.lstm_training --config configs/default.toml
```

The entrypoint uses `load_prepared`, which reconstructs and validates the actual
saved windows, endpoint targets, engine partitions and scaler schema using only
`train_FD001.txt`. It never opens official test trajectories or RUL targets.
Training tensors are `(15259,20,75)` from 80 engines; validation tensors are
`(3472,20,75)` from 20 other engines. This is the same endpoint set used by all
three classical models. Their summary representation has 225 dimensions;
the LSTM uses the ordered sequence instead. Labels remain uncapped.

The ignored `artifact_dir/models/FD001/lstm/` bundle contains `best.pt` (weights
and schema, loaded with `weights_only=True`), `schema.json`, `metadata.json`,
`history.csv`, endpoint `validation_predictions.csv`, and a portable copy of
`preprocessing/manifest.json` and `scaler.json`. Metadata records configuration,
versions, shapes, seed, best epoch, duration, metrics and source/artifact hashes.
Preprocessing checksums prevent pairing the checkpoint with another scaler.
This is an inference checkpoint, not an optimizer-state training-resume bundle.
Use separate artifact directories for experiments; named outputs are overwritten.

```python
from aevora.lstm import LSTMPredictor, predict_history
predictor = LSTMPredictor.load(bundle_directory)
rul_cycles = predictor.predict(prepared_windows, ordered_feature_names)
latest_predictions = predict_history(bundle_directory, available_history)
```

Sequence prediction rejects wrong shape/order and nonfinite inputs/outputs.
Raw-history prediction reuses Phase 3 preprocessing, requires all available
history from cycle 1 and at least 20 cycles, and returns each engine's latest
endpoint. Feature order, length and frozen preprocessing remain coupled to the
checkpoint. Only load model artifacts from trusted sources.

## Validation diagnostics

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -W error -m aevora.lstm_evaluation --config configs/default.toml
```

Evaluation reloads the checkpoint, verifies its hash and preparation hashes,
checks the exact endpoint/target order, and reproduces saved predictions and
metrics. It writes to `artifact_dir/reports/FD001/lstm_validation/`: metrics JSON,
predictions, history, per-engine and RUL-band CSVs, scatter, residual histogram,
engine MAE bars and training curves. Bands match Phase 4: 0–30, >30–100 and >100
RUL cycles. Residual is prediction minus actual. No models are retrained here.
