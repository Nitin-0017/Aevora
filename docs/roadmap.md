# Phase gates and commit plan

Target about 30 substantive commits; never pad history. Each phase begins with
status inspection and ends with relevant checks, diff review, reporting and user
approval. Phases 1 and 2 are implemented; subsequent work requires approval.

| Phase | Scope | Target commits |
| --- | --- | --- |
| 1 | Foundation and configuration | 3 |
| 2 | Dataset loading, validation, EDA | 4 |
| 3 | Labels, preprocessing, causal features | 4 |
| 4 | Classical models and evaluation | 4 |
| 5 | LSTM and fair comparison | 3 |
| 6 | Frozen final evaluation, robustness, model card | 5 |
| 7 | Dashboard with shared real inference | 5 |
| 8 | Integration checks, documentation, release | 2 |

## Prepared Phase 1 commit boundaries

No commits are made automatically in this preparation step. Inspect each staged
diff before committing; stage only the listed files.

1. `chore: initialize AEVORA package and environment setup`
   - `.gitignore`, `pyproject.toml`, `README.md`, `src/aevora/__init__.py`
   - Check: parse `pyproject.toml` with `tomllib`; import `aevora` using
     `PYTHONPATH=src`; review setup and ignore patterns.
2. `feat: add validated central experiment configuration`
   - `configs/default.toml`, `src/aevora/config.py`, `src/aevora/__main__.py`
   - Check: `PYTHONPATH=src python3 -m aevora --config configs/default.toml`.
3. `test: verify foundation and document evaluation contracts`
   - `tests/test_foundation.py`, `docs/architecture.md`, `docs/roadmap.md`
   - Check: `PYTHONPATH=src python3 -m unittest discover -s tests -v`;
     `git diff --check` and inspect all added files before committing.

## Phase 2 implementation

Acquisition documentation, strict loader and quality reports, fixture tests, and
training-only EDA are implemented. See `dataset.md`, `eda.md`, and
`phase2_findings.md`. Four substantive Phase 2 commits capture this work.

## Phase 3 gate

Wait for explicit approval before adding labels, splits, preprocessing, or
features. Official test targets remain sealed until final evaluation.
