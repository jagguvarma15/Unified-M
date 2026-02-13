# Unified-M Makefile
# ===================
# Developer workflow targets for all run modes.

PYTHON   := PYTHONPATH=src python
SHELL    := /bin/bash
.DEFAULT_GOAL := help

# ── Run Modes ────────────────────────────────────────────────

.PHONY: dev
dev: ## Local dev run (fast Ridge model, sample data)
	$(PYTHON) -m cli demo

.PHONY: run
run: ## Full pipeline run with builtin model
	$(PYTHON) -m cli run --model builtin --target revenue

.PHONY: weekly-prod
weekly-prod: ## Weekly production run (full Bayesian model)
	$(PYTHON) -m cli run --model pymc --mode weekly-prod

.PHONY: backfill
backfill: ## Backfill run for a date range (usage: make backfill START=2024-01-01 END=2024-12-31)
	$(PYTHON) -m cli run --mode backfill --start $(START) --end $(END)

.PHONY: what-if
what-if: ## What-if scenario (no retraining, usage: make what-if BUDGET=120000)
	$(PYTHON) -m cli scenario --budget $(BUDGET)

.PHONY: calibrate
calibrate: ## Experiment calibration run (usage: make calibrate TEST_FILE=data/raw/test.parquet)
	$(PYTHON) -m cli calibrate --test-file $(TEST_FILE)

# ── Serving ──────────────────────────────────────────────────

.PHONY: serve
serve: ## Start FastAPI server
	$(PYTHON) -m cli serve --port 8000

.PHONY: ui
ui: ## Start React dev server
	cd ui && bun dev

.PHONY: serve-all
serve-all: ## Start API + UI in background
	$(PYTHON) -m cli serve --port 8000 &
	cd ui && bun dev

# ── Docker ───────────────────────────────────────────────────

.PHONY: docker-up
docker-up: ## Start Docker serving stack (api + ui + redis)
	docker compose up api ui redis -d

.PHONY: docker-down
docker-down: ## Stop Docker stack
	docker compose down

.PHONY: docker-batch
docker-batch: ## Run batch pipeline in Docker
	docker compose run --rm pipeline

.PHONY: docker-build
docker-build: ## Build Docker images
	docker compose build

# ── Scripts (demos; replace notebooks) ────────────────────────

.PHONY: scripts
scripts: ## Run script with PYTHONPATH set (usage: make scripts SCRIPT=scripts/demo_transforms.py)
	$(PYTHON) $(SCRIPT)

.PHONY: demo-transforms
demo-transforms: ## Demo adstock & saturation transforms (requires scripts/demo_transforms.py)
	@test -f scripts/demo_transforms.py && $(PYTHON) scripts/demo_transforms.py || (echo "Add scripts/demo_transforms.py first. See scripts/README.md"; exit 1)

.PHONY: demo-evaluation
demo-evaluation: ## Demo evaluation metrics (requires scripts/demo_evaluation.py)
	@test -f scripts/demo_evaluation.py && $(PYTHON) scripts/demo_evaluation.py || (echo "Add scripts/demo_evaluation.py first. See scripts/README.md"; exit 1)

# ── Data & Quality ───────────────────────────────────────────

.PHONY: ingest
ingest: ## Ingest raw data sources
	$(PYTHON) -m cli ingest --source data/raw --output data/bronze

.PHONY: validate
validate: ## Run data quality gates
	$(PYTHON) -m cli validate --input data/bronze

.PHONY: transform
transform: ## Transform bronze -> silver -> gold
	$(PYTHON) -m cli transform --input data/bronze --output data/gold

# ── Testing & Linting ────────────────────────────────────────

.PHONY: test
test: ## Run test suite
	pytest tests/ -v --cov=src

.PHONY: lint
lint: ## Run linter
	ruff check src/

.PHONY: typecheck
typecheck: ## Run type checker
	mypy src/

.PHONY: check
check: lint typecheck test ## Run all checks

# ── Utilities ────────────────────────────────────────────────

.PHONY: install
install: ## Install Python dependencies
	pip install -e ".[dev]"

.PHONY: install-ui
install-ui: ## Install UI dependencies
	cd ui && bun install

.PHONY: install-all
install-all: install install-ui ## Install everything

.PHONY: clean
clean: ## Remove generated artifacts
	rm -rf runs/ data/bronze data/silver data/gold data/processed
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
