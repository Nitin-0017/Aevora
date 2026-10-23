# Phase 8 — release readiness audit

Audit date: 2026-10-24. Starting source commit: `397fd68`. The checkout was clean
and synchronized with cached `origin/main`. Fresh remote verification failed with
`Could not resolve host: github.com`; cached tracking state is not fresh proof.

## Decision

Suitable for a documented local portfolio demonstration with the matching
artifact bundle. **Not production-ready. Live browser end-to-end verification
and clean-machine installation remain unverified.** No model parameters, saved
checkpoints, target definitions or reported performance were changed.

## Verified evidence

[Machine-readable audit evidence](phase8_evidence.json) records versions, all
four metric rows, reliability values and twelve actual API predictions. The audit
loaded existing models and recomputed metrics from saved predictions, checking
matching engine/cycle/target alignment through `verified_predictions` and
`check_saved_metrics`. No official test RUL target was opened.

For engines 3, 5 and 19 at replay cycle 100, all four API predictions matched
independently loaded inference calls to five decimal places. Examples:

| Engine | Boosted-tree prediction (cycles) |
|---|---:|
| 3 | 122.08812 |
| 5 | 167.17889 |
| 19 | 35.14468 |

Three sensor-2 API histories matched the actual source sensor values and cycles.
All 33 model/preprocessing artifact SHA256 hashes matched the prior audited
bundle. Frozen scaler, ordered 75-feature schema and 20-step windows are checked
on load; classical models summarize windows into 225 features. Training and
validation engines remain separate. Model loading does not trigger fitting.

API tests exercise missing artifacts, incompatible input, negative/nonfinite
outputs, all four models, verified metrics and restricted CORS. Errors return
structured messages, not client stack traces; server logs can contain diagnostic
paths. Trusted pickle artifacts are required. The service is local-only and has
no authentication or production serving configuration.

## Automated results

Baseline: 91 Python tests passed. Final: **92 passed, 0 failed, 0 skipped**
with warnings treated as errors (30.053 seconds), including 9 API tests and the
new cross-engine parity check. Streamlit emitted logging messages about its
absent runtime during headless tests; these were not Python warning failures. Existing ML tests can train tiny
synthetic fixtures in temporary directories; no benchmark model was retrained.

Frontend: 8 passed, 0 failed, 0 skipped. The interaction test verifies a fresh
backend request on Run prediction, engine selection, negative output and data
rendering. Tests use labelled fixtures; separate Python integration tests use
real saved models. TypeScript/production build and Prettier passed. No ESLint
configuration exists, so no ESLint result is claimed.

## Live execution and visual checks

Uvicorn completed artifact initialization, then failed binding `127.0.0.1:8000`
with `Errno 1: operation not permitted`. Vite failed binding `127.0.0.1:5173`
with `listen EPERM`. No live-server or browser success is claimed. TestClient
runs the API lifecycle in-process; it does not prove HTTP proxy/browser behavior.

Local logo and turbine assets exist and are included in `frontend/dist/assets/`.
The original logo was previously checked byte-for-byte against the supplied file.
The turbine is AI-generated and labelled accordingly; provenance is in
`frontend/public/assets/README.md`. No generated image is evidence of the UI.

Static inspection found labelled inputs, semantic buttons, aria-current navigation,
visible focus rules, textual status labels, reduced-motion rules and responsive
stacking/mobile navigation. Pixel-level contrast, keyboard traversal, chart sizing,
console errors and desktop/mobile visual fidelity still require a real browser.

## Manual verification checklist

Start the two servers using README commands in normal macOS Terminal, then:

1. Open `/api/health`; confirm `ready`. Open frontend port 5173 and browser devtools.
2. Select engine 3, boosted trees, replay cycle 100; Run prediction. Confirm
   approximately 122.08812 cycles, engine 3 and latest cycle 100.
3. Switch engine/model; verify card metadata and real sensor history update.
   Change sensor; verify axis field name, tooltip and recorded cycle range.
4. Visit all five pages; check development-validation/demo disclosures and metrics.
5. Stop backend and rerun prediction; confirm an error, not a stale reassuring card.
6. Test keyboard-only navigation, visible focus, mobile menu, 390px/768px/1440px
   widths, charts/tables and local image loading. Check console/network errors.
7. Capture genuine screenshots only after these checks; record results here.

## Artifact setup

The release intentionally excludes raw datasets, model binaries and generated
reports. There is currently no published downloadable checkpoint bundle. Restore
a trusted matching bundle from the project owner, retaining all relative paths:

- `data/raw/CMAPSSData/train_FD001.txt` (NASA acquisition: `dataset.md`).
- `artifacts/prepared/FD001/`: manifests, scaler, arrays and endpoints.
- `artifacts/models/FD001/{baseline,ridge,gradient_boosting,lstm}/`: estimator or
  checkpoint, schema, metadata, predictions and copied preprocessing.
- `artifacts/reports/FD001/{performance_audit,reliability}/audit.json` and related
  evaluation outputs. Reports must match bundle hashes.

Do not extract/read official test targets for this demonstration. If no bundle
is available, the following is an **explicit new experiment**, not automatic
startup or an exact reproduction guarantee. It was not executed in Phase 8:

```bash
PYTHONPATH=src .venv/bin/python -m aevora.prepare --config configs/default.toml --save-arrays
# Set thread limits for all following model/evaluation commands:
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
PYTHONPATH=src .venv/bin/python -m aevora.training --config configs/default.toml --model baseline
PYTHONPATH=src .venv/bin/python -m aevora.training --config configs/default.toml --model ridge
PYTHONPATH=src .venv/bin/python -m aevora.training --config configs/default.toml --model gradient_boosting
PYTHONPATH=src .venv/bin/python -m aevora.lstm_training --config configs/default.toml
PYTHONPATH=src .venv/bin/python -m aevora.evaluation --config configs/default.toml
PYTHONPATH=src .venv/bin/python -m aevora.lstm_evaluation --config configs/default.toml
PYTHONPATH=src .venv/bin/python -m aevora.audit --config configs/default.toml
PYTHONPATH=src .venv/bin/python -m aevora.performance_audit --config configs/default.toml
```

Back up existing bundles before regeneration. These commands overwrite named
artifacts. Model versions and floating-point behavior can change with libraries
or hardware; regenerate matching reports and report newly measured metrics.

## Environment and hygiene

Observed Python 3.13.1; Node 25.2.0; npm 11.6.2. Python requires >=3.11; installed
Vite 7.3.1 requires Node `^20.19.0 || >=22.12.0`. Key Python package versions are in
the evidence JSON. Python ranges are not a full dependency lock, and saved
scikit-learn estimators should use the recorded compatible version (1.6.1 here).
No fresh-environment installation or online dependency-vulnerability audit was
performed under restricted networking.

Tracked-file review found no dataset/model binary, virtual environment or cache
paths; no matches for common private-key/GitHub-token/OpenAI-key patterns; and no
`/Users/` paths in runtime source. This limited scan is not a security certification.
Largest tracked asset is the intentional 2.4 MB turbine PNG. `.gitignore` excludes
local datasets, artifacts, environments, secrets and frontend build/cache paths.
No useful artifacts were deleted. API origins are limited to localhost frontend
port 5173; the Vite development proxy points at backend port 8000.

## Release checklist

| Item | Verified status |
|---|---|
| Full Python suite | Passed: 92 tests, 0 failed, 0 skipped, warnings as errors |
| Frontend tests, build, TypeScript and formatting | Passed |
| Backend artifact initialization | Passed |
| Backend/frontend live servers | Blocked: sandbox socket permissions |
| Actual inference and source sensor parity | Passed in-process, 12 predictions / 3 histories |
| Saved metrics and artifact integrity | Passed, 33 unchanged hashes |
| Installation and launch documentation | Updated; fresh installation unverified |
| Required local image assets/build inclusion | Passed; visual browser loading unverified |
| Desktop/mobile/browser accessibility | Static review only; manual checks pending |
| Official test targets | Not accessed in this audit |
| Secrets and repository hygiene | Limited tracked-source checks passed |
| Final diff/staging review | Only README, release documentation/evidence and API test changes |
| Remote verification/publishing | Blocked: GitHub DNS |

Release remains conditional on the manual checks above. Passing tests does not
establish independent generalization performance or operational safety.
