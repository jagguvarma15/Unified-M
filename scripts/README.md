# Scripts — Runnable demos (replacement for notebooks)

This directory holds **runnable Python scripts** for data, transforms, and evaluation. They replace the former Jupyter notebook with something that is:

- **Reproducible** — Run from the repo root; no kernel or notebook server.
- **CI-friendly** — Scripts can be executed in GitHub Actions or locally.
- **Versioned** — Plain Python; diffs and reviews are straightforward.
- **Documented** — See `docs/DATA_AND_TRANSFORMS.md` for data sources and transform formulas.

## Plan

| Purpose | Script | Notes |
|--------|--------|--------|
| **Synthetic data** | Use `PYTHONPATH=src python -m cli demo` | Already in CLI; writes to `data/`. |
| **Transforms demo** | `demo_transforms.py` | Adstock & saturation examples, optional plots. |
| **Evaluation demo** | `demo_evaluation.py` | Metrics (MAPE, R², RMSE) and residual diagnostics. |
| **Data sources** | — | Described in `docs/DATA_AND_TRANSFORMS.md`. |

## Settings

- **PYTHONPATH:** Must include `src` so that `transforms`, `models`, etc. resolve. From repo root:
  - `PYTHONPATH=src python scripts/<script>.py`
  - Or use the Makefile: `make demo-transforms`, `make demo-evaluation`, or `make scripts SCRIPT=scripts/your_script.py`
- **Config:** Scripts use the same config as the CLI. Optional: ensure `config.yaml` exists in the project root (see `config.yaml` and `docs/EXPERIMENTATION.md`).
- **Data:** For scripts that need data, generate it first with `make dev` (or `python -m cli demo`); outputs go to `data/` and `runs/` per `config.yaml`.
- **Optional env:** See `docs/ENV_VARS.md` for server and connector variables.

## Running scripts

From the repo root:

```bash
PYTHONPATH=src python scripts/demo_transforms.py
PYTHONPATH=src python scripts/demo_evaluation.py
# Or via Makefile:
make demo-transforms
make scripts SCRIPT=scripts/your_script.py
```

## Docs

- **Data sources & transforms:** `docs/DATA_AND_TRANSFORMS.md` (data table, adstock/saturation formulas).
- **Architecture:** `docs/ARCHITECTURE.md`.
- **Implementation:** `docs/IMPLEMENTATION.md`.
