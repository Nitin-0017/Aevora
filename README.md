# AEVORA — Predictive Maintenance Intelligence

**Intelligence Before Failure.**

AEVORA is a research prototype for estimating simulated turbofan engine Remaining
Useful Life (RUL), in operating cycles, from historical sensor measurements.
It will help explore inspection prioritization, not authorize maintenance actions.
Simulation results do not demonstrate factory savings or industrial reliability.

## Current status

Phase 1 foundation only: package, configuration, setup instructions and tests.
No dataset has been downloaded, no model trained, and no metrics measured.
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

Data and artifact directories are intentionally not populated in Phase 1.
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

A local repository is initialized on `main`; no remote or commits are configured
as part of this foundation preparation. Review the three proposed commit groups
in the roadmap before committing. To publish later, create an empty GitHub
repository, add its URL with `git remote add origin <repository-url>`, and push
reviewed commits with `git push -u origin main`. Never commit dataset files or keys.
