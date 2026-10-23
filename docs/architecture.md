# Planned engineering contracts

These contracts guide the project. Phases 1–4 implement data preparation and
classical model development. Final test evaluation and LSTM remain future work.

## Data and labels

FD001 training engines run to failure: raw RUL is each engine's maximum observed
cycle minus its current cycle. Official test trajectories are truncated; their
provided targets describe the remaining life at the final observed cycle.
Do not derive test labels from the last observed cycle as if it were failure.
Validate schema, numeric values, unique engine/cycle pairs, chronological order
and expected cycle continuity before feature generation. Do not silently repair
ambiguous raw-data problems.

## Development split and causality

Persist disjoint training and validation engine IDs with a fixed seed. Rolling
features must group by engine and use only current/past observations. Sequence
windows never cross engine boundaries. Decide and document short-history behavior
before training. Feature selection, imputation and scaling are learned only on
training engines. Baseline constant is learned only from training labels.
Any RUL clipping policy must be declared and selected on development data.

## Fair comparisons

Compare models on identical engines, target definitions and prediction endpoints.
Do not compare an LSTM's complete-window subset to all rows for a tree model.
Report sample counts, row-level errors and engine-level aggregates. Validation
may include a predefined truncation protocol to better match test endpoints;
choose it before tuning, never using official test targets.

Freeze model selection and preprocessing before official test evaluation. Score
one prediction per test engine at its last observed cycle. Never train or tune on
official test targets; record any post-test changes as subsequent experiments.

## Artifacts and inference

Save configuration, engine split, feature schema/order, fitted preprocessing,
model weights, seed, software versions and evaluation reports together. Dashboard
and evaluation must call the same inference implementation. Return errors for
unsupported or invalid inputs rather than invented predictions.

## Reliability experiments

Apply noise/missingness to input histories, regenerate features through the same
pipeline, and compare against unmodified inputs. Define corruption strengths and
random seeds. Never introduce future observations through interpolation.
Experimental intervals require separate calibration engines and coverage/width
evaluation; do not label arbitrary bands as calibrated uncertainty.

## Scope and UI

CPU-first small experiments. Small PyTorch LSTM follows classical baselines.
Brand: dark navy/charcoal, restrained cyan, amber warnings, readable charts.
Streamlit will display real saved results with model/data provenance. No invented
metrics, fabricated screenshots, real-time claims from recorded playback, or
claims of production safety. Cycles are not automatically hours or days.
