# React + FastAPI dashboard

AEVORA now includes a React 19 + TypeScript/Vite frontend (`frontend/`) and a
FastAPI service (`src/aevora/api/`). The Streamlit prototype remains available.
This is a local NASA FD001 demonstration, not live equipment or a production
maintenance system. Development validation was used for model selection.

## Launch

From the repository directory, use the existing Python environment and audited
local artifact bundle. A fresh checkout alone does not contain datasets or models.
Install dependencies once (network access is required unless already cached):

```bash
.venv/bin/python -m pip install -e '.[api,dev]'
cd frontend
npm ci
```

In terminal 1, from the repository root:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m uvicorn aevora.api.app:app --host 127.0.0.1 --port 8000
```

In terminal 2:

```bash
cd frontend
npm run dev
```

Open **http://127.0.0.1:5173**. API health is at
http://127.0.0.1:8000/api/health and interactive API documentation at
http://127.0.0.1:8000/docs. Vite proxies `/api` to port 8000. CORS permits only
the localhost/127.0.0.1 frontend origins on port 5173. No credentials or API keys
are required. The supplied thread limits constrain numerical-library resource use.
Once dependencies and artifacts are installed, the application works offline.

If startup reports `operation not permitted` while binding a port in an isolated
runner, launch these commands from normal macOS Terminal. If port 5173 is already
occupied, stop its existing server first; Vite will not silently switch ports.

## Pages and behavior

- Fleet overview: actual eligible validation engines, replay cutoff and model
  selection, ordered predictions, valid/invalid counts, selected-engine card.
- Engine explorer: actual recorded sensor history through the selected cutoff,
  sensor and engine selectors, current estimate and preprocessing details.
- Model comparison: verified metrics for four models on the same 3,472 endpoints,
  actual/predicted scatter, residual histogram and per-engine MAE.
- Reliability: saved audit findings and explicit limitations of the noise probe.

Review/monitor thresholds default to 30/100 cycles and are illustrative, editable
bands, not validated safety limits. Negative predictions remain visible and are
marked unable to assess. Non-finite inference results fail the request rather than
being serialized as valid predictions. A failed fleet request displays an error;
it does not silently show partial or stale results. Optional hours conversion
requires a positive rate supplied by the user. No failure date is calculated.
Sensor identifiers are not assigned invented engineering meanings.

## API and artifact contract

GET endpoints: `/api/health`, `/api/engines`,
`/api/engines/{engine_id}/history?cycle=100&sensor=sensor_2`, `/api/models`,
`/api/model-metrics`, `/api/diagnostics/{model}`, `/api/reliability`.
POST endpoints: `/api/predict`, `/api/fleet-predictions`.
Example prediction body:

```json
{"engine_id":3,"model":"gradient_boosting","cycle":100,"review_threshold":30,"monitor_threshold":100}
```

Fleet requests omit `engine_id`. Supported model identifiers are `baseline`,
`ridge`, `gradient_boosting`, and `lstm`. Pydantic rejects extra fields, malformed
numbers, unknown models and unordered thresholds. Errors use
`{"error":{"code":"...","message":"..."}}` with 422 or 503 responses.
Health returns a readiness status and 503 when artifacts are unavailable.

The service uses `dashboard.data.load_snapshot`, `load_predictor`, `infer_history`
and `fleet_predictions`, with the existing frozen preprocessing and operator
interpretation functions. It verifies saved predictions and audit results before
serving them. Models load lazily and are cached; inference is synchronized.
Artifact changes invalidate the cache and require a restart. Required inputs are
those configured in `configs/default.toml`: training trajectories, prepared
FD001 manifest/scaler/sequences, saved classical/LSTM models, predictions and
Phase 6/performance-audit reports. Preserve the matching local bundle. Do not use
untrusted serialized model files. Missing or incompatible artifacts produce an
unavailable state, not substitute predictions.

The API does not fit preprocessing, retrain models, or read official test targets.
Model/preprocessing version fields describe the artifact schema, not a deployed
release hash. Timestamps are generated after inference. These are local
single-process development servers; authentication, hosting and production
hardening are outside this implementation.

## Verification

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -W error -m unittest discover -s tests -v
npm test --prefix frontend
npm run build --prefix frontend
npm run format --prefix frontend
```

The build includes strict TypeScript checking. Prettier checks formatting; ESLint
is not configured. API tests exercise the real saved bundle using TestClient,
verify all four models, schema/CORS/error cases and artifact integrity. They skip
bundle-dependent integration cases if artifacts are absent. Frontend tests use
clearly separated fixtures under `frontend/tests`, including DOM interaction,
loading/errors, raw negative output and optional conversion. Production code
fetches real API responses only.

For manual verification, open all four pages, switch engines/models/sensors,
change the replay cutoff, confirm cycles-only output without a rate, and check
errors after stopping the backend. Browser appearance must be reviewed locally
when the execution environment blocks server ports.

## Reference overview layout

The overview follows the supplied reference: engine/model controls and Run
prediction, a prominent status/RUL card, an illustrative turbine panel, actual
sensor history and engine metadata. Fleet counts and the priority table follow
these cards. Navigation now includes Data Information and a collapsible mobile
menu. Run prediction issues a fresh backend request; changing the selected engine
updates its card and sensor request. Rounded whole cycles and the raw output to
two decimals are shown separately. Negative outputs remain invalid.

The original silver logo is stored unchanged at
`frontend/public/assets/aevora-logo.jpg`. The AI-generated turbine illustration is
stored at `frontend/public/assets/turbofan.png` and explicitly labelled as an
illustration, not live equipment. Both are bundled locally for offline use. Image
provenance and the generation prompt are recorded in
`frontend/public/assets/README.md`. The screenshot's example numbers and
timestamps are never used as project data.
