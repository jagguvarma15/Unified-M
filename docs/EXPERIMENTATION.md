# Experimentation plan (replacing notebooks)

The previous **Jupyter notebook** (`notebooks/experimental_analysis.ipynb`) has been removed. Here is the replacement approach.

## Why move away from notebooks

- Hard to diff and review; outputs can get committed.
- Not easy to run in CI or from a single command.
- Dependency on a kernel and environment that can drift.

## Replacement strategy

1. **Runnable scripts** (`scripts/`)
   - Plain Python scripts for synthetic data, transforms, and evaluation.
   - Run with `PYTHONPATH=src python scripts/<script>.py`.
   - Can be wired into CI or Make targets.

2. **Documentation** (`docs/`)
   - **Data sources:** table of required/optional inputs and key fields (see `DATA_AND_TRANSFORMS.md` or add a short section there).
   - **Transforms:** adstock and saturation formulas, when to use which (in docs, not in a notebook).
   - **Evaluation:** metrics definitions and interpretation (in docs or in `models/evaluation.py` docstrings).

3. **Tests** (`tests/`)
   - `test_transforms.py` and `test_optimization.py` already cover core logic.
   - Add or extend tests for evaluation and calibration so “experimental” behavior is regression-tested.

4. **CLI** (`cli demo`)
   - Synthetic data generation already lives in `cli demo`; scripts or docs can point users there instead of duplicating logic in a notebook.

## Next steps

- Add `scripts/demo_transforms.py` (and optionally `demo_evaluation.py`) when you want runnable examples.
- Flesh out `docs/DATA_AND_TRANSFORMS.md` with the data-sources table and transform notes from the old notebook.
- Keep all executable experimentation in `scripts/` and `tests/`, and all narrative in `docs/`.
