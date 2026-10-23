# AEVORA — Predictive Maintenance Intelligence

**Intelligence Before Failure.**

AEVORA is a research prototype for estimating simulated turbofan engine Remaining
Useful Life (RUL), in operating cycles, from historical sensor measurements.
It will help explore inspection prioritization, not authorize maintenance actions.
Simulation results do not demonstrate factory savings or industrial reliability.

## Current status

Phases 1–3: foundation, ingestion/validation, training-only EDA, and leakage-safe
training-data preparation.
The NASA-linked FD001 training/test files are available locally. Official RUL
targets remain unextracted in the downloaded archive. No model has been trained.
The dashboard and all prediction functionality are planned, not implemented.

## Setup

Use Python 3.11 or newer. Initial foundation checks ran on Python 3.13.1.
From the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m aevora --config configs/default.toml
python -m pytest
```

`pyproject.toml` is the dependency source of truth. Installation requires network
access and compatible wheels. The full ML stack has not yet been installed or
verified together; version ranges are not a reproducible lockfile. After resolving
the environment, record exact versions with `python -m pip freeze` alongside each
experiment. CPU is the initial device; no API key or paid service is needed.

To check the foundation without installing dependencies:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m aevora --config configs/default.toml
```

## Planned architecture

```text
NASA C-MAPSS FD001 files
    -> validated loader and engine-based development split
    -> causal rolling features / historical sequence windows
    -> training-only preprocessing
    -> mean baseline, Ridge, gradient boosting, small LSTM
    -> shared inference interface and saved artifacts
    -> development evaluation / final official test evaluation
    -> robustness studies and Streamlit dashboard
```

The configuration defines seed, paths, validation fraction, sequence length, and
initial model settings. These are starting choices, not tuned results. Dataset
paths resolve against the configuration's project root, not the shell directory.
The configuration checker never opens dataset files, including official targets.

## Repository layout

```text
configs/default.toml      Central configuration
src/aevora/              Package and configuration-check CLI
tests/                   Foundation tests (unittest and pytest compatible)
docs/architecture.md     Evaluation and implementation contracts
docs/roadmap.md          Phase gates and proposed commit boundaries
data/                    Future local datasets; ignored by Git
artifacts/               Future models/reports; ignored by Git
```

Data and EDA artifact directories are populated locally and ignored by Git.
Future selected evaluation summaries may be committed under documentation after
review. Raw data, fitted artifacts, environments and secrets stay out of Git.

## Development protocol

Split by engine; never randomly split overlapping windows across train/validation.
Use only present/past readings for predictions. Learn preprocessing only from the
training partition. Select models and hyperparameters using development data.
Keep official test targets sealed until final evaluation, with frozen settings.

See [architecture](docs/architecture.md) and [roadmap](docs/roadmap.md).
Each phase ends with tests, a diff review, and user approval before the next phase.

## GitHub

The repository uses `main` and has an existing `origin` remote. Phase work is
committed locally; pushing requires explicit approval. Never commit datasets or keys.

## Phase 2: validation and EDA

See [dataset acquisition](docs/dataset.md), [EDA usage](docs/eda.md), and
[observed findings](docs/phase2_findings.md). Minimal Phase 2 setup (no PyTorch):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy>=1.26,<3' 'pandas>=2.2,<3' 'matplotlib>=3.8,<4'
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m aevora.data --config configs/default.toml --split train
PYTHONPATH=src python -m aevora.eda --config configs/default.toml
```

Reports go to `<artifact_dir>/reports/FD001/train/`; change `artifact_dir` in the
existing TOML config to relocate them. All EDA statistics use training data only.
Loader tests use small artificial temporary fixtures; no real data is required
for tests. The plotting integration test skips explicitly if Matplotlib is absent.

## Phase 3: prepare model inputs

```bash
PYTHONPATH=src .venv/bin/python -m aevora.prepare --config configs/default.toml --save-arrays
```

This reads only official training histories, splits by engine, creates RUL labels,
selects nonconstant sensors on training engines, engineers historical features,
fits training-only scaling, and builds 20-cycle windows. No predictive model is
trained. Saved preprocessing is reused by `prepare_inference` without refitting.

See [preparation rules](docs/preparation.md) for formulas, early-history behavior,
artifact usage and limitations, and [Phase 3 verification](docs/phase3_verification.md)
for the measured preparation counts. Generated arrays remain local and ignored.
