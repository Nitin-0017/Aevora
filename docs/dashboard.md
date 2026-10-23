# Phase 7 — Interactive predictive maintenance dashboard

AEVORA — Intelligence Before Failure.

## Launch and requirements

From the repository directory, using the existing virtual environment:

```sh
cd /Users/nitinkumar/Documents/Codex/2026-10-23/he/outputs/aevora
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m streamlit run dashboard.py --server.address 127.0.0.1
```

Open http://127.0.0.1:8501. No API keys, network services, authentication or cloud setup are needed. Keep the server bound to localhost for this local demonstration. Run from the repository so Streamlit loads `.streamlit/config.toml`.

Tested imports in the existing `.venv/bin/python`: Streamlit 1.54.0 and Altair 6.0.0. The dependency source remains `pyproject.toml`; the dashboard requires Streamlit >=1.54 (the width API used by the UI) and explicitly declares Altair >=6. Plotly is declared in the earlier project dependencies but absent from the current environment; this dashboard does not import or require Plotly. A fresh environment can install the declared project with `python -m pip install -e '.[dev]'` when network access is available. Dependencies are version ranges, not a lockfile.

Required **trusted local** inputs, resolved through `configs/default.toml`:

- `data/raw/CMAPSSData/train_FD001.txt` only. Official test files are never loaded.
- `artifacts/prepared/FD001/`: manifest, frozen scaler, labels, endpoint tables and saved training/validation window arrays.
- `artifacts/models/FD001/{baseline,ridge,gradient_boosting,lstm}/`: existing complete model bundles, schemas, preprocessing copies, metadata, validation predictions and LSTM history/checkpoint.
- `artifacts/reports/FD001/performance_audit/audit.json` and `artifacts/reports/FD001/reliability/audit.json`.

These inputs exist locally and remain Git-ignored. A fresh Git clone alone cannot show predictions. Restore the matching local audited artifact bundle; do not create placeholder data or retrain merely to launch the UI. Missing/mismatched artifacts produce an actionable error. Pickled classical model bundles must come from a trusted source. The three pre-existing uncommitted performance-audit source/document files are preserved and are not imported by the dashboard; their saved audit output is required.

## Architecture and integrity

- `dashboard.py`: explicit launch entry point, also resolves the local `src` package without requiring an editable install.
- `src/aevora/dashboard/data.py`: data access, report validation, engine selection, risk interpretation and frozen inference adapters. Independent of Streamlit.
- `src/aevora/dashboard/app.py`: page composition, session controls and cache boundaries.
- `src/aevora/dashboard/views.py`: fleet and engine views with interactive Altair charts.
- `src/aevora/dashboard/reports.py`: RUL workflow, model comparison, reliability and data-information views.
- `.streamlit/config.toml`: navy theme, cyan accent and disabled usage telemetry.

On first use, the app checks both audit fingerprint inventories against current model/preparation files, reconstructs prepared windows and targets from training histories, reproduces saved predictions with saved models, and checks displayed audit metrics against those predictions. This does **not** fit preprocessing or models. All four bundles and both reports must match; the app fails closed rather than silently serving a partial or mixed-version fleet.

Model loaders validate schema v1 and feature ordering. Inference uses `prepare_inference` with each model's saved manifest/scaler and validates 20 × 75 sequences; classical models compute their existing 225-feature representation. Input must be finite, chronological, contiguous, complete from cycle 1 and at least 20 cycles long. The model's stored ordered feature list must match the saved preprocessing. Prediction vectors must match endpoint counts and be finite. No arbitrary upload or manual sensor-input form is offered.

Verified snapshots and model objects are cached as resources; fleet predictions are cached data. Cache keys include config, raw **training** file, model/preparation files and report file sizes/mtime signatures. Changing local inputs invalidates caches and triggers verification. “Run / refresh inference” clears fleet predictions and reruns inference with cached frozen models. No mutation of cached models or data is performed. For a trusted local app this avoids model loading on every widget change; it is not a defense against adversarial edits that deliberately preserve file metadata. Restart after restoring a bundle if in doubt.

## Views and interactions

1. **Fleet overview:** 20 available validation engines, finite-prediction count, negative-output count, a distribution and ascending-RUL table with textual statuses. The 80 training engines and official test engines are excluded.
2. **Engine explorer:** choose engine, sensor list and visible sensor-cycle range. Each sensor has a separate vertical scale and uses recorded values in dataset units. The prediction history is the actual saved validation series matched by engine/cycle; only endpoints up to the replay cutoff are shown.
3. **RUL prediction:** choose engine/model/replay cycle and refresh. Raw estimate, endpoint, available history, schema/preprocessing versions and feature provenance are shown. A successful computation does not imply a safe estimate.
4. **Model comparison:** saved audited MAE, RMSE, R², ±10/±20 coverage, signed bias, median/P95/maximum absolute errors; actual-versus-predicted with ideal line and residual histogram. Download verified metrics as CSV. All charts use the same 3,472 validation observations, regardless of fleet replay controls.
5. **Reliability insights:** boosted-tree middle-life bias (+19.99 cycles), overall bias (+9.83), largest overprediction (108.47), P95 absolute error (60.34), negative counts and the saved 1% Gaussian sensor-noise probe. It explicitly distinguishes sensitivity from uncertainty calibration.
6. **Model & data information:** FD001 context, training/validation/test distinctions, reserved official targets and limitations preventing production-readiness claims.

The default replay is through cycle 100. For any cutoff, inference uses the latest available endpoint at or before that cycle for each engine. Engines whose histories have ended retain that final endpoint; the table shows the actual cycle. Raising the cutoff to the maximum therefore displays complete run-to-failure endpoints, not a plausible live fleet. The UI states this limitation. Sensor chart range selection changes only visualization, never model preprocessing. No synthetic telemetry, future predictions, or interpolation is presented as real history.

## Illustrative risk bands

Sidebar controls are display-only; they never change model parameters or central experiment configuration. Defaults:

| Raw predicted RUL | Textual status |
|---|---|
| Nonfinite or missing | Invalid / unavailable; no numerical display |
| Below 0 | Invalid negative RUL — review model output; raw negative value remains visible |
| 0–30 inclusive | High concern (illustrative) |
| Above 30 through 100 inclusive | Elevated concern (illustrative) |
| Above 100 | Above demonstration thresholds — not a safety assurance |

Thresholds must be finite with `0 ≤ high < elevated`. No clipping or replacement occurs; numbers are formatted to two decimal places in the UI. High concern uses red, warnings/invalid outputs amber, neutral information cyan, always accompanied by text. These are not calibrated probabilities, validated industrial thresholds, or autonomous maintenance instructions. RUL is estimated operating cycles, not a guaranteed date or elapsed hours.

## Verification

Baseline: 66 tests passed with warnings as errors before dashboard edits. The full Phase 7 suite and smoke outcomes are recorded in the completion report below. Existing full-suite training tests use tiny temporary synthetic fixtures; no benchmark model was retrained for the dashboard.

New tests cover engine filtering, risk boundaries, negative/nonfinite/missing values, raw/display preservation, report consistency and tampering, sequence shape, feature ordering, short/NaN histories, missing bundles, reload reproducibility and reserved-test file access guards. Local artifact integration tests explicitly skip on source-only clones; skips must not be reported as successful model inference checks.

Streamlit `AppTest` executes application logic and interactions without a listening socket. It does not verify browser rendering, responsive layout, color contrast measurements or assistive-technology behavior. AppTest emits its standard `missing ScriptRunContext` **logging** warning in bare mode; this is not a Python warnings-as-errors test failure.

The normal Streamlit server launch was attempted in this sandbox and failed at localhost socket bind with `PermissionError: [Errno 1] Operation not permitted`. No browser-render or screenshot success is claimed. Run from normal macOS Terminal for browser verification:

1. Launch using the exact command above and open localhost:8501. Confirm dark navy styling, legible tables and the “not live equipment” label.
2. Confirm Fleet shows 20 engines and 20 finite estimates at cycle 100. Compare table ordering with the histogram and inspect endpoint cycles.
3. Switch among all four models and click refresh. At engine 3/cycle 100, observed estimates were baseline 95.00, Ridge 106.03, boosted trees 122.09 and LSTM 131.05 cycles.
4. Select another engine, change sensors and visible history range, then deselect all sensors to check the empty state. Change replay to 20 and verify eligible inference and aligned prediction history.
5. Exercise threshold boundaries and an invalid threshold order. At negative predictions, verify the raw negative number and invalid-model warning remain visible.
6. Open comparison and reliability pages; confirm the development-validation label, ideal reference line, residual axes, negative counts and sensitivity disclaimer. Confirm comparison metrics do not change with fleet cutoff.
7. Narrow the browser/mobile viewport and inspect chart/table scrolling, keyboard navigation and readable statuses. These browser-only checks remain manual.
8. Stop the app with Ctrl-C. Do not modify/delete model artifacts to test missing states; automated tests exercise that path safely.

## Git and scope

No datasets, fitted models, caches or generated reports are committed. The original three performance-audit files remain uncommitted and unchanged. Phase 7 commits contain only dashboard source, theme, entry point, tests, dependency declarations and documentation.

Initial fetch failed with `Could not resolve host: github.com`. Cached `origin/main` matched Phase 6 HEAD `a93768c`, but live remote state could not be verified. No retry loop or push was attempted. After reviewing the local work, a normal Terminal can fetch, verify compatible history and push normally; never force-push. Phase 8 has not started.

## Completion record

- Baseline: 66/66 tests passed.
- Final full suite: **77/77 tests passed**, zero failures/errors/skips, with Python warnings treated as errors (27.852 seconds in this environment).
- Six page flows, four model selections, refresh/cycle controls, sensor empty state, invalid thresholds and missing-artifact messaging passed Streamlit AppTest.
- An initial test-harness selector error was corrected; the final full run passed. Streamlit's module-level test temporary directory is explicitly cleaned up to avoid an interpreter-shutdown ResourceWarning.
- No browser render or visual accessibility test was possible because localhost socket binding is sandbox-restricted. No deployment was attempted.
- All 33 model/preprocessing hashes match the pre-existing audit. All three earlier uncommitted audit files retain their original hashes.
- No official test data/targets were accessed or benchmark models retrained.

Exact Phase 7 file inventory:

```text
.streamlit/config.toml
dashboard.py
src/aevora/dashboard/__init__.py
src/aevora/dashboard/data.py
src/aevora/dashboard/app.py
src/aevora/dashboard/views.py
src/aevora/dashboard/reports.py
tests/test_dashboard_data.py
tests/test_dashboard_integration.py
tests/test_dashboard_app.py
docs/dashboard.md
README.md
pyproject.toml
```
