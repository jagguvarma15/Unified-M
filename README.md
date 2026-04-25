# Unified-M


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-blue.svg)](https://www.typescriptlang.org/)
[![SolidJS](https://img.shields.io/badge/SolidJS-1.9-blue.svg)](https://www.solidjs.com/)
[![Bun](https://img.shields.io/badge/Bun-1.3-black.svg)](https://bun.sh/)
[![uv](https://img.shields.io/badge/uv-managed-ffc021.svg)](https://docs.astral.sh/uv/)

> **Status: Pre-stable.** Core pipeline, API, and dashboard are functional. Enterprise connectors and new pages are landing. Expect breaking changes before v1.0.

**Unified Marketing Measurement Platform** — an end-to-end framework that fuses Marketing Mix Modeling (MMM), incrementality tests, and attribution data into a single source of truth for channel-level lift with calibrated uncertainty. Produces stable budget recommendations, scenario simulations, and real-time what-if analysis through a modern enterprise dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HIGH-LATENCY PIPELINE                           │
│  (Scheduled / On-demand)                                                │
│                                                                         │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  ┌───────┐  ┌──────────────┐│
│  │ Ingest  │→ │ Validate │→ │ Transform │→ │ Train │→ │  Reconcile   ││
│  └─────────┘  └──────────┘  └───────────┘  └───────┘  └──────────────┘│
│       ↓                                                      ↓         │
│  Raw Data                                              ┌──────────┐    │
│  (Parquet)                                             │ Optimize │    │
│                                                        └──────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                           Precomputed Outputs
                           (Parquet + JSON)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         LOW-LATENCY SERVING                             │
│                                                                         │
│  ┌──────────────────┐              ┌──────────────────────────────────┐ │
│  │    FastAPI        │    ←────→   │     SolidJS + Vite Dashboard     │ │
│  │  (REST API)       │             │   (TypeScript, Recharts, TW)     │ │
│  │  + Redis cache    │             │   + ⌘K palette, date picker      │ │
│  └──────────────────┘              └──────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- **Python 3.10+** with [uv](https://docs.astral.sh/uv/) (recommended) or pip
- **[Bun](https://bun.sh)** for the frontend (Node.js also works via `npm`)

### Installation

```bash
git clone https://github.com/jagguvarma15/Unified-M.git
cd Unified-M

# Backend
uv sync                    # install from lockfile (reproducible)

# Frontend
cd ui && bun install       # install JS dependencies
```

<details>
<summary>Pip fallback (no uv)</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```
</details>

### Run the Demo

```bash
# Generate synthetic data + train the model
uv run python -m cli demo

# Start the API server
uv run python -m cli serve --port 8000

# In another terminal: start the UI
cd ui && bun dev
```

Open [http://localhost:5173](http://localhost:5173). The dashboard auto-populates once a pipeline run completes.

**Makefile shortcuts:** `make dev` (demo) · `make serve` (API) · `make ui` (frontend) · `make start` (both in background) · `make stop`

## Dashboard & UI

The dashboard is a SolidJS single-page application with React islands for Recharts. Key features:

| Feature | Description |
|---------|-------------|
| **KPI Cards** | Sparklines, trend arrows, % change vs prior period |
| **Chart Export** | CSV and PNG export on every chart panel |
| **Channel Drill-down** | Click any channel → slide-over detail panel with spend, ROAS, lift, budget |
| **Date Range Picker** | Global filter: Last 30d / 90d / 1Y / Custom range |
| **Command Palette** | `⌘K` to jump to any page — keyboard-first navigation |
| **Collapsible Sidebar** | Full or icon-only (48px) mode, persisted |
| **Sortable Tables** | Column sorting, sticky headers, row selection |
| **Toast Notifications** | Undo support, progress bar, auto-dismiss |
| **Animated Empty States** | Guided step CTAs for onboarding |
| **Density Toggle** | Compact / Default / Comfortable spacing |

### Pages

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | KPI cards, model fit, contribution share, waterfall, ROAS, allocation, efficiency map, residuals, timeline |
| Data | `/data` | Upload status, file management |
| Connections | `/datapoint` | Enterprise data source wizard (see [Connectors](#data-source-connectors)) |
| Runs | `/runs` | Pipeline run history, comparison |
| Contributions | `/contributions` | Channel contribution decomposition |
| Response Curves | `/curves` | Saturation & diminishing returns |
| ROAS Analysis | `/roas` | Per-channel return on ad spend |
| Channel Insights | `/channel-insights` | Marginal ROI, headroom, saturation status |
| Diagnostics | `/diagnostics` | Model fit, residuals, metrics |
| Budget Optimizer | `/optimization` | Constrained optimal allocation |
| Scenario Planner | `/scenarios` | What-if scenario comparison |
| **Budget Simulator** | `/budget-simulator` | Real-time slider-based what-if with live KPIs |
| Spend Pacing | `/spend-pacing` | Budget pace tracking |
| Calibration | `/calibration` | Experiment calibration |
| Stability | `/stability` | Parameter drift monitoring |
| Data Quality | `/data-quality` | Quality gate results |
| **Alerts Center** | `/alerts` | Configurable threshold alerts per channel |
| **Attribution Explorer** | `/attribution` | Touchpoint flow, model comparison (MMM vs heuristic) |
| Executive Summary | `/report` | Auto-generated report |
| **Report Builder** | `/report-builder` | Drag-and-drop canvas, export to markdown |
| Settings | `/settings` | Configuration |

## Data Source Connectors

The Connections page provides an enterprise-grade wizard for adding data sources across four priority tiers:

| Tier | Category | Sources |
|------|----------|---------|
| **1** | Paid Media APIs | Google Ads, Meta Ads, TikTok, LinkedIn, Pinterest, Snapchat, X/Twitter, Apple Search Ads |
| **2** | Analytics & Revenue | Google Analytics 4, Adobe Analytics, Shopify, Salesforce CRM |
| **3** | Warehouses & Databases | BigQuery, Snowflake, Redshift, Databricks, DuckDB, PostgreSQL, MySQL, S3, Azure Blob |
| **4** | External Signals | Holidays/Events, Weather (NOAA/Open-Meteo), FRED Economic Data |

Each connector includes:
- **Step-by-step wizard** — choose category → select provider → configure credentials
- **Test Connection** — verify connectivity before saving
- **Column Mapping Wizard** — map source columns to MMM schema (`date`, `channel_spend`, `target_kpi`, etc.)
- **File Upload** — CSV, Parquet, Excel direct upload as fallback

## Project Structure

```
unified-m/
├── src/
│   ├── core/             # Contracts, artifacts, base model, exceptions
│   ├── connectors/       # Data loaders & enterprise connectors
│   │   ├── ad_platforms/  # Google Ads, Meta, TikTok, Amazon
│   │   └── external/      # FRED, holidays, weather
│   ├── models/           # MMM backends (builtin Ridge, PyMC, Meridian)
│   ├── pipeline/         # End-to-end runner, modes
│   ├── transforms/       # Adstock, saturation, features, Rust accel
│   ├── reconciliation/   # Fusion engine (MMM + tests + attribution)
│   ├── optimization/     # Budget allocator & scenarios
│   ├── quality/          # Data quality gates, PII scanner
│   ├── experiments/      # Experiment framework
│   ├── orchestration/    # Prefect & Dagster adapters
│   ├── server/           # FastAPI app, auth, cache, jobs, schemas
│   ├── config.py         # Configuration management
│   └── cli.py            # CLI commands
├── ui/
│   └── src/
│       ├── pages/        # 20 pages (Dashboard, Attribution, Alerts, etc.)
│       ├── components/   # MetricCard, ChartCard, Table, CommandPalette, etc.
│       └── lib/          # API client, dateRange, toast, icons, queries
├── rust/                 # Optional Rust acceleration (adstock, saturation)
├── tests/                # Backend test suite (6 test modules)
├── scripts/              # OpenAPI export, docs sync check
├── docs/                 # Architecture, data schemas, env vars, etc.
├── data/                 # Lakehouse zones (raw → bronze → silver → gold)
├── .github/              # CI workflows, Dependabot, issue/PR templates
├── Dockerfile            # Multi-stage (Python API + Nginx UI)
├── docker-compose.yml    # Full local stack (API + UI + Redis + Dagster)
├── pyproject.toml        # Python dependencies (uv managed)
├── config.yaml           # Default configuration
└── Makefile              # Developer workflow targets
```

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **MMM Backends** | Built-in (Ridge), PyMC-Marketing, Google Meridian |
| **Optimization** | scipy.optimize (SLSQP) |
| **Data** | Parquet, DuckDB, Pandas, PyArrow |
| **Validation** | Pydantic v2 |
| **Frontend** | SolidJS 1.9, TypeScript 5.6, Vite 7 |
| **Charts** | Recharts (via React island bridge) |
| **Styling** | Tailwind CSS 3.4 |
| **Icons** | Lucide (wrapped for SolidJS) |
| **State** | TanStack Solid Query |
| **Runtime** | Bun 1.3 |
| **Cache** | Redis (optional, falls back to in-memory) |
| **Containers** | Docker multi-stage, docker-compose |
| **CI** | GitHub Actions (lint, test, build, bundle budget, docs sync) |
| **Rust Accel** | Optional PyO3 bindings for adstock/saturation |


Concurrency is enabled — pushing again to the same PR cancels the previous run.

## Development

```bash
# Install everything (backend + frontend)
make install-all

# Run tests
make test                  # or: uv run pytest tests/ -v --cov=src

# Lint & type-check
make lint                  # or: uv run ruff check src/
make typecheck             # or: uv run mypy src/

# All checks at once
make check

# Start dev servers (background, with logs)
make start                 # API on :8000, UI on :5173
make stop                  # clean shutdown

# Docker
make docker-up             # API + UI + Redis
make docker-down
```

## Configuration

Copy `.env.example` to `.env` for API keys and secrets. Edit `config.yaml` for model and pipeline settings:

```yaml
project_name: "My MMM Project"
environment: "production"

storage:
  raw_path: "data/raw"
  gold_path: "data/gold"

model:
  backend: "builtin"       # builtin | pymc | meridian
  adstock_max_lag: 8
  n_samples: 1000

reconciliation:
  mmm_weight: 0.5
  incrementality_weight: 0.3
  attribution_weight: 0.2

optimization:
  method: "SLSQP"
  min_channel_budget_pct: 0.0
  max_channel_budget_pct: 1.0
```

See [docs/ENV_VARS.md](docs/ENV_VARS.md) for all environment variables and [docs/DATA_SCHEMAS.md](docs/DATA_SCHEMAS.md) for input data formats.

## Data Schemas

### Media Spend
```
date | channel | spend | impressions | clicks
```

### Outcomes
```
date | revenue | conversions
```

### Incrementality Tests
```
test_id | channel | start_date | end_date | lift_estimate | lift_ci_lower | lift_ci_upper | test_type
```

## API

FastAPI serves all precomputed outputs at `http://localhost:8000`. Key endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + latest run info |
| GET | `/api/v1/runs` | Pipeline run history |
| GET | `/api/v1/contributions` | Channel contribution data |
| GET | `/api/v1/roas` | ROAS by channel |
| GET | `/api/v1/diagnostics` | Model fit metrics + chart data |
| GET | `/api/v1/optimization` | Current vs optimal allocation |
| GET | `/api/v1/waterfall` | Response decomposition |
| GET | `/api/v1/response-curves` | Saturation curves per channel |
| GET | `/api/v1/channel-insights` | Marginal ROI, headroom |
| GET | `/api/v1/calibration` | Experiment calibration results |
| GET | `/api/v1/stability` | Parameter drift alerts |
| GET | `/api/v1/data-quality` | Quality gate results |
| POST | `/api/v1/pipeline/run` | Trigger async pipeline run |
| GET/POST | `/api/v1/connectors` | CRUD for saved data connections |

Full OpenAPI docs available at `/docs` when the server is running.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! This project is pre-stable, so please open an issue before large PRs to align on direction. See `.github/PULL_REQUEST_TEMPLATE.md` for PR guidelines.
