# Unified-M Architecture

This document reflects the current implementation in this repository.

## System Overview

Unified-M uses a batch-first architecture:

- High-latency pipeline computes features, model outputs, reconciliation, and optimization.
- Low-latency serving reads versioned run artifacts and returns precomputed responses.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HIGH-LATENCY PIPELINE                           │
│                                                                         │
│  Connect → Quality Gates → Transform → Train → Reconcile → Optimize    │
│                                                                         │
│  Outputs: runs/<run_id>/{manifest.json, *.parquet, *.json, model/}     │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOW-LATENCY SERVING                             │
│                                                                         │
│     FastAPI (`src/server/app.py`)  ↔  React + Vite (`ui/src`)          │
│                                                                         │
│  API returns precomputed artifacts; no model fitting on hot path.       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Backend Components

### CLI and config

- `src/cli.py`: user-facing commands (`run`, `serve`, `ingest`, `transform`, etc.).
- `src/config.py`: typed config and filesystem path management.

### Pipeline execution

- `src/pipeline/runner.py`: end-to-end orchestration.
- `src/pipeline/modes.py`: run mode resolution (local-dev, weekly-prod, backfill, etc.).

### Data connectivity

- `src/connectors/local.py`: local file loading and auto-detection.
- `src/connectors/database.py`: DB connectors (Postgres, MySQL, SQL Server, SQLite).
- `src/connectors/cloud.py`: cloud storage connectors.
- `src/connectors/ad_platforms/`: platform-specific ad connectors.

### Model layer

- `src/core/base_model.py`: backend model interface.
- `src/models/registry.py`: model backend discovery/registration.
- `src/models/builtin.py`: default Ridge + adstock + saturation backend.
- Optional adapters: `src/models/pymc_adapter.py`, `src/models/meridian_adapter.py`, `src/models/numpyro_adapter.py`.

### Measurement logic

- `src/transforms/`: feature engineering, adstock, saturation.
- `src/reconciliation/`: fusion of MMM, tests, and attribution.
- `src/optimization/`: budget allocation and scenarios.
- `src/quality/`: data quality gates and PII scanning.

### Serving layer

- `src/server/app.py`: FastAPI endpoints.
- `src/server/cache.py`: Redis/Rust/Python cache backends.
- `src/server/jobs.py`: async pipeline job state.
- `src/server/auth.py`: bearer-token middleware and route protection.

### Artifact and contracts

- `src/core/artifacts.py`: versioned run storage under `runs/`.
- `src/core/contracts.py`: typed contracts for manifests and domain payloads.

## Frontend Components

- `ui/src/main.tsx`: app bootstrap and providers.
- `ui/src/App.tsx`: route wiring.
- `ui/src/pages/`: dashboards and analytical views.
- `ui/src/lib/api.ts`: typed API client wrappers.
- `ui/src/lib/queries.ts`: React Query bindings.

## Data and Artifact Flow

### 1. Pipeline run

1. Input data is loaded from configured files/connectors.
2. Quality checks run before training.
3. MMM-ready features are created.
4. Model backend is fitted.
5. Reconciliation combines signals into unified channel estimates.
6. Optimization computes recommended allocations.
7. Outputs are written into a new `runs/<run_id>/` directory.

### 2. Serving

1. API endpoints read latest run artifacts from `runs/latest` (or newest run fallback).
2. Responses are normalized to stable API contracts.
3. UI fetches API data via React Query and renders pages.

## Current Directory Map

```
src/
  cli.py
  config.py
  core/
  connectors/
  transforms/
  models/
  reconciliation/
  optimization/
  quality/
  pipeline/
  server/
ui/
  src/
tests/
runs/
docs/
```

## Notes on Stability

- Runtime behavior should be inferred from code in `src/` and `ui/src/`.
- This file is checked in CI by a docs-sync script to prevent stale architecture terms.
