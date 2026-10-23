# Phase gates and commit plan

Target about 30 substantive commits; never pad history. Each phase begins with
status inspection and ends with relevant checks, diff review, reporting and user
approval. Only Phase 1 is implemented here; the remaining work is planned.

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

## Phase 2 gate

Wait for explicit user approval. Then verify dataset provenance and access,
implement the loader and schema validation, and perform training-data EDA. Keep
official test targets unopened until the final evaluation phase. Do not fabricate
data if download or access fails.
