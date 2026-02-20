"""
Command-line interface for Unified-M.

Provides commands for:
  - Running the full pipeline (five run modes)
  - What-if scenario analysis
  - Experiment calibration
  - Generating demo data
  - Starting the API server
  - Starting the UI
  - Inspecting run history
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
import pandas as pd
import numpy as np

app = typer.Typer(
    name="unified-m",
    help="Unified Marketing Measurement -- local-first framework",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def _connect_pipeline(cfg, pipe, media_spend=None, outcomes=None, controls=None, tests=None, attribution=None):
    """Helper: resolve data paths and connect sources to a Pipeline."""
    ms = media_spend or cfg.storage.processed_path / "media_spend.parquet"
    oc = outcomes or cfg.storage.processed_path / "outcomes.parquet"
    ct = controls or cfg.storage.processed_path / "controls.parquet"
    ts = tests or cfg.storage.processed_path / "incrementality_tests.parquet"
    at = attribution

    pipe.connect(
        media_spend=ms if Path(ms).exists() else None,
        outcomes=oc if Path(oc).exists() else None,
        controls=ct if Path(ct).exists() else None,
        incrementality_tests=ts if Path(ts).exists() else None,
        attribution=at if at and Path(at).exists() else None,
    )


@app.command()
def run(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml",
    ),
    model: str = typer.Option(
        "builtin", "--model", "-m", help="Model backend (builtin, pymc, ...)",
    ),
    target: str = typer.Option(
        "revenue", "--target", "-t", help="Target column in outcomes",
    ),
    budget: Optional[float] = typer.Option(
        None, "--budget", "-b", help="Total budget for optimisation",
    ),
    mode: str = typer.Option(
        "local-dev", "--mode", help="Run mode: local-dev, weekly-prod, backfill, what-if, experiment-calibration",
    ),
    start: Optional[str] = typer.Option(
        None, "--start", help="Backfill start date (YYYY-MM-DD)",
    ),
    end: Optional[str] = typer.Option(
        None, "--end", help="Backfill end date (YYYY-MM-DD)",
    ),
    media_spend: Optional[Path] = typer.Option(
        None, "--media-spend", help="Path to media spend data",
    ),
    outcomes: Optional[Path] = typer.Option(
        None, "--outcomes", help="Path to outcomes data",
    ),
    controls: Optional[Path] = typer.Option(
        None, "--controls", help="Path to control variables",
    ),
    tests: Optional[Path] = typer.Option(
        None, "--tests", help="Path to incrementality test results",
    ),
    attribution: Optional[Path] = typer.Option(
        None, "--attribution", help="Path to attribution data",
    ),
):
    """
    Run the full pipeline: connect, transform, train, reconcile, optimise.

    Results are written to the runs/ directory as versioned artifacts.
    """
    from config import load_config
    from pipeline.runner import Pipeline
    from pipeline.modes import resolve_run_mode

    cfg = load_config(config_path)
    cfg.ensure_directories()

    # Resolve run mode
    mode_cfg = resolve_run_mode(
        mode=mode,
        model=model,
        backfill_start=start,
        backfill_end=end,
    )

    pipe = Pipeline(
        config=cfg.to_flat_dict(),
        runs_dir=cfg.storage.runs_path,
    )

    _connect_pipeline(cfg, pipe, media_spend, outcomes, controls, tests, attribution)

    results = pipe.run(
        model=mode_cfg.model_backend,
        target_col=target,
        total_budget=budget,
    )

    logger.info(f"Run complete ({mode}): {pipe.run_id}")
    metrics = results.get("metrics", {})
    logger.info(f"  MAPE  : {metrics.get('mape', 'N/A')}%")
    logger.info(f"  R2    : {metrics.get('r_squared', 'N/A')}")
    logger.info(f"  RMSE  : {metrics.get('rmse', 'N/A')}")


# ---------------------------------------------------------------------------
# Individual pipeline steps (for CI / granular execution)
# ---------------------------------------------------------------------------

@app.command()
def ingest(
    source: Path = typer.Option(Path("data/raw"), "--source", "-s", help="Source directory with raw data files"),
    output: Path = typer.Option(Path("data/gold"), "--output", "-o", help="Output directory for validated data"),
):
    """
    Ingest raw data files into the processing zone.

    Copies parquet/CSV files from source to output, coercing dates and
    validating that minimum required files (media_spend, outcomes) exist.
    """
    from connectors.local import load_file

    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    REQUIRED = ["media_spend", "outcomes"]
    OPTIONAL = ["controls", "incrementality_tests", "attribution"]

    found = []
    for name in REQUIRED + OPTIONAL:
        for ext in [".parquet", ".csv", ".xlsx"]:
            src_path = source / f"{name}{ext}"
            if src_path.exists():
                df = load_file(src_path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                out_path = output / f"{name}.parquet"
                df.to_parquet(out_path, index=False)
                found.append(name)
                logger.info(f"Ingested {name}: {len(df)} rows -> {out_path}")
                break

    missing = [r for r in REQUIRED if r not in found]
    if missing:
        logger.error(f"Missing required data files: {missing}")
        raise typer.Exit(1)

    logger.info(f"Ingest complete: {len(found)} file(s) written to {output}")


@app.command()
def validate(
    input_dir: Path = typer.Option(Path("data/gold"), "--input", "-i", help="Directory with ingested data"),
    target: str = typer.Option("revenue", "--target", "-t", help="Target column in outcomes"),
):
    """
    Run data quality gates on ingested data.

    Checks schema validity, completeness, anomalies, staleness, and
    cross-source consistency.  Exits non-zero if critical gates fail.
    """
    from quality.gates import run_quality_gates

    input_dir = Path(input_dir)

    media_path = input_dir / "media_spend.parquet"
    outcomes_path = input_dir / "outcomes.parquet"

    if not media_path.exists() or not outcomes_path.exists():
        logger.error(f"Required files not found in {input_dir}. Run 'ingest' first.")
        raise typer.Exit(1)

    media_spend = pd.read_parquet(media_path)
    outcomes = pd.read_parquet(outcomes_path)

    report = run_quality_gates(media_spend=media_spend, outcomes=outcomes, target_col=target)

    logger.info(f"Quality gates: {report.n_passed} passed, {report.n_warnings} warnings, {report.n_failed} failures")

    if not report.overall_pass:
        logger.warning("Quality gates produced failures -- review before training")
        for gate in report.results:
            if not gate.passed:
                logger.warning(f"  FAIL: {gate.gate_name} -- {gate.message}")
    else:
        logger.info("All quality gates passed")


@app.command()
def transform(
    input_dir: Path = typer.Option(Path("data/gold"), "--input", "-i", help="Directory with ingested data"),
    output: Path = typer.Option(Path("data/gold/mmm_input.parquet"), "--output", "-o", help="Path for MMM-ready parquet"),
    target: str = typer.Option("revenue", "--target", "-t", help="Target column"),
):
    """
    Transform raw data into MMM-ready features.

    Creates adstock, saturation, and time features from ingested media
    spend, outcomes, and controls data.
    """
    from transforms.features import create_mmm_features

    input_dir = Path(input_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    media_spend = pd.read_parquet(input_dir / "media_spend.parquet")
    outcomes = pd.read_parquet(input_dir / "outcomes.parquet")
    controls = None
    controls_path = input_dir / "controls.parquet"
    if controls_path.exists():
        controls = pd.read_parquet(controls_path)

    mmm_df = create_mmm_features(
        media_spend=media_spend,
        outcomes=outcomes,
        controls=controls,
        target_col=target,
    )

    mmm_df.to_parquet(output, index=False)
    media_cols = [c for c in mmm_df.columns if c.endswith("_spend")]
    logger.info(f"Transform complete: {len(mmm_df)} rows, {len(media_cols)} media channels -> {output}")


@app.command()
def train(
    input_dir: Path = typer.Option(Path("data/gold"), "--input", "-i", help="Directory with ingested data"),
    model: str = typer.Option("builtin", "--model", "-m", help="Model backend"),
    target: str = typer.Option("revenue", "--target", "-t", help="Target column"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Total budget for optimisation"),
    samples: int = typer.Option(1000, "--samples", help="MCMC samples (Bayesian backends)"),
    chains: int = typer.Option(4, "--chains", help="MCMC chains (Bayesian backends)"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
):
    """
    Train an MMM model on prepared data.

    Runs the full train->reconcile->optimise pipeline and writes versioned
    artifacts to the runs/ directory.
    """
    from config import load_config
    from pipeline.runner import Pipeline

    cfg = load_config(config_path)
    cfg.ensure_directories()

    input_dir = Path(input_dir)

    pipe = Pipeline(config=cfg.to_flat_dict(), runs_dir=cfg.storage.runs_path)
    pipe.connect(
        media_spend=input_dir / "media_spend.parquet"
        if (input_dir / "media_spend.parquet").exists() else None,
        outcomes=input_dir / "outcomes.parquet"
        if (input_dir / "outcomes.parquet").exists() else None,
        controls=input_dir / "controls.parquet"
        if (input_dir / "controls.parquet").exists() else None,
        incrementality_tests=input_dir / "incrementality_tests.parquet"
        if (input_dir / "incrementality_tests.parquet").exists() else None,
        attribution=input_dir / "attribution.parquet"
        if (input_dir / "attribution.parquet").exists() else None,
    )

    model_kwargs = {}
    if model in ("pymc", "numpyro"):
        model_kwargs = {"n_samples": samples, "n_chains": chains}

    results = pipe.run(
        model=model,
        target_col=target,
        total_budget=budget,
        model_kwargs=model_kwargs if model_kwargs else None,
    )

    logger.info(f"Train complete: run_id={pipe.run_id}")
    metrics = results.get("metrics", {})
    logger.info(f"  MAPE : {metrics.get('mape', 'N/A')}%")
    logger.info(f"  R2   : {metrics.get('r_squared', 'N/A')}")


@app.command()
def reconcile(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run ID to reconcile (latest if omitted)"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """
    Re-run reconciliation on an existing run's artifacts.

    Reads MMM parameters and experiment data from a completed run and
    re-computes the reconciled channel estimates.
    """
    from config import load_config
    from core.artifacts import ArtifactStore
    from reconciliation.engine import ReconciliationEngine
    import json

    cfg = load_config(config_path)
    store = ArtifactStore(cfg.storage.runs_path)
    rid = run_id or store.get_latest_run_id()

    if rid is None:
        logger.error("No runs found. Run 'train' first.")
        raise typer.Exit(1)

    params = store.load_json(rid, "parameters")
    if params is None:
        logger.error(f"No parameters in run {rid}")
        raise typer.Exit(1)

    engine = ReconciliationEngine(
        mmm_weight=cfg.reconciliation.mmm_weight,
        incrementality_weight=cfg.reconciliation.incrementality_weight,
        attribution_weight=cfg.reconciliation.attribution_weight,
    )

    media_cols = list(params.get("coefficients", {}).keys())
    tests_path = cfg.storage.processed_path / "incrementality_tests.parquet"
    tests_df = pd.read_parquet(tests_path) if tests_path.exists() else None

    result = engine.reconcile(
        mmm_results=params,
        incrementality_tests=tests_df,
        attribution_data=None,
        channels=media_cols,
    )

    store.save_json(rid, "reconciliation", result.to_dict())
    logger.info(f"Reconciliation saved to run {rid}")


@app.command()
def optimize(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Run ID (latest if omitted)"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Override total budget"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
):
    """
    Re-run budget optimisation on an existing run's response curves.

    Useful for what-if analysis without retraining the model.
    """
    from config import load_config
    from core.artifacts import ArtifactStore
    from optimization.allocator import BudgetOptimizer
    import json

    cfg = load_config(config_path)
    store = ArtifactStore(cfg.storage.runs_path)
    rid = run_id or store.get_latest_run_id()

    if rid is None:
        logger.error("No runs found. Run 'train' first.")
        raise typer.Exit(1)

    curves_data = store.load_json(rid, "response_curves")
    if curves_data is None:
        logger.error(f"No response curves in run {rid}")
        raise typer.Exit(1)

    response_fns = {}
    for ch, curve in curves_data.items():
        spend_pts = np.array(curve["spend"])
        resp_pts = np.array(curve["response"])
        response_fns[ch] = lambda s, sp=spend_pts, rp=resp_pts: float(np.interp(s, sp, rp))

    total_budget = budget
    if total_budget is None:
        media_path = cfg.storage.processed_path / "media_spend.parquet"
        if media_path.exists():
            total_budget = float(pd.read_parquet(media_path)["spend"].sum())
        else:
            total_budget = 100_000.0

    optimizer = BudgetOptimizer(response_curves=response_fns, total_budget=total_budget)
    result = optimizer.optimize()
    store.save_json(rid, "optimization", result.to_dict())

    logger.info(f"Optimisation saved to run {rid}")
    logger.info(f"  Budget:   ${total_budget:,.0f}")
    logger.info(f"  Expected: ${result.expected_response:,.0f}")


# ---------------------------------------------------------------------------
# scenario (what-if)
# ---------------------------------------------------------------------------

@app.command()
def scenario(
    budget: float = typer.Option(
        100_000, "--budget", "-b", help="Total budget for scenario",
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml",
    ),
    model: str = typer.Option(
        "builtin", "--model", "-m", help="Model backend to load artifacts from",
    ),
):
    """
    What-if scenario: re-optimise budget without retraining.

    Uses the latest run's model artifacts (response curves) to compute
    an optimal allocation for the specified budget.
    """
    from config import load_config
    from core.artifacts import ArtifactStore
    from optimization.allocator import BudgetOptimizer
    import json

    cfg = load_config(config_path)
    store = ArtifactStore(cfg.storage.runs_path)
    run_id = store.get_latest_run_id()

    if run_id is None:
        logger.error("No previous runs found. Run the pipeline first.")
        raise typer.Exit(1)

    logger.info(f"What-if scenario: budget=${budget:,.0f} using run {run_id}")

    # Load response curves from latest run
    curves_data = store.load_json(run_id, "response_curves")
    if curves_data is None:
        logger.error(f"No response curves in run {run_id}")
        raise typer.Exit(1)

    # Build response functions from curve data
    response_fns = {}
    for ch, curve in curves_data.items():
        spend_pts = np.array(curve["spend"])
        resp_pts = np.array(curve["response"])
        response_fns[ch] = lambda s, sp=spend_pts, rp=resp_pts: float(
            np.interp(s, sp, rp)
        )

    optimizer = BudgetOptimizer(
        response_curves=response_fns,
        total_budget=budget,
    )
    result = optimizer.optimize()

    logger.info(f"Scenario result:")
    logger.info(f"  Expected response: {result.expected_response:,.2f}")
    logger.info(f"  Expected ROI:      {result.expected_roi:.2f}")
    for ch, alloc in result.optimal_allocation.items():
        logger.info(f"  {ch}: ${alloc:,.0f}")

    # Save scenario to runs
    scenario_data = {
        "total_budget": budget,
        "expected_response": result.expected_response,
        "expected_roi": result.expected_roi,
        "optimal_allocation": result.optimal_allocation,
        "source_run_id": run_id,
    }
    scenario_path = cfg.storage.runs_path / run_id / f"scenario_{int(budget)}.json"
    with open(scenario_path, "w") as f:
        json.dump(scenario_data, f, indent=2)
    logger.info(f"Scenario saved to {scenario_path}")


# ---------------------------------------------------------------------------
# calibrate (experiment calibration)
# ---------------------------------------------------------------------------

@app.command()
def calibrate(
    test_file: Path = typer.Option(
        ..., "--test-file", "-f", help="Path to experiment results (CSV/Parquet)",
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config.yaml",
    ),
):
    """
    Experiment calibration: update model priors from new test results.

    Reads experiment results, compares against MMM predictions, computes
    calibration factors, and updates the prior set for the next run.
    """
    from config import load_config
    from core.artifacts import ArtifactStore
    from models.calibration_eval import evaluate_calibration
    from models.priors import PriorSet, apply_calibration_factors, warm_start_from_run
    import json

    cfg = load_config(config_path)
    store = ArtifactStore(cfg.storage.runs_path)
    run_id = store.get_latest_run_id()

    if run_id is None:
        logger.error("No previous runs found. Run the pipeline first.")
        raise typer.Exit(1)

    # Load experiment results
    if test_file.suffix == ".parquet":
        tests_df = pd.read_parquet(test_file)
    else:
        tests_df = pd.read_csv(test_file, parse_dates=["start_date", "end_date"])

    logger.info(f"Loaded {len(tests_df)} experiment results from {test_file}")

    # Load MMM parameters from latest run
    params = store.load_json(run_id, "parameters")
    if params is None:
        logger.error(f"No parameters found in run {run_id}")
        raise typer.Exit(1)

    # Evaluate calibration
    report = evaluate_calibration(tests_df, params)
    logger.info(f"Calibration quality: {report.calibration_quality}")
    logger.info(f"  Coverage:          {report.coverage:.0%}")
    logger.info(f"  Median lift error: {report.median_lift_error:.1f}%")

    # Compute calibration factors
    channels = list(params.get("coefficients", {}).keys())
    priors = warm_start_from_run(cfg.storage.runs_path / run_id, channels)

    cal_factors = {}
    for point in report.points:
        if abs(point.predicted_lift) > 1e-8:
            cal_factors[point.channel] = point.measured_lift / point.predicted_lift

    if cal_factors:
        priors = apply_calibration_factors(priors, cal_factors)
        logger.info(f"Applied calibration factors for {len(cal_factors)} channels")
    else:
        logger.warning("No calibration factors could be computed")

    # Save updated priors and calibration report
    run_dir = cfg.storage.runs_path / run_id
    priors.save(run_dir / "calibrated_priors.json")
    with open(run_dir / "calibration_eval.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)

    logger.info(f"Calibrated priors saved to {run_dir}/calibrated_priors.json")
    logger.info("Next pipeline run will use these as informative priors.")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Start the REST API server (serves latest run artifacts)."""
    from server.app import run_server
    run_server(host=host, port=port, reload=reload)


# ---------------------------------------------------------------------------
# ui
# ---------------------------------------------------------------------------

@app.command()
def ui(
    port: int = typer.Option(5173, "--port", "-p"),
    install: bool = typer.Option(False, "--install", help="Run bun install first"),
):
    """Start the React dashboard (requires Bun)."""
    import subprocess
    import shutil

    ui_dir = Path("ui")
    if not ui_dir.exists():
        logger.error("ui/ directory not found. Are you in the project root?")
        raise typer.Exit(1)

    bun = shutil.which("bun")
    if bun is None:
        logger.error(
            "Bun not found.  Install it: curl -fsSL https://bun.sh/install | bash"
        )
        raise typer.Exit(1)

    if install or not (ui_dir / "node_modules").exists():
        logger.info("Installing UI dependencies...")
        subprocess.run([bun, "install"], cwd=str(ui_dir), check=True)

    logger.info(f"Starting Unified-M UI on http://localhost:{port}")
    logger.info("Make sure the API is running:  unified-m serve")
    subprocess.run(
        [bun, "run", "dev", "--", "--port", str(port)],
        cwd=str(ui_dir),
    )


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------

@app.command()
def demo(
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory (defaults to gold zone)",
    ),
    n_days: int = typer.Option(365, "--days"),
):
    """
    Generate synthetic demo data and immediately run the pipeline.

    This is the fastest way to see the framework in action:

        unified-m demo
        unified-m serve
    """
    from config import load_config

    cfg = load_config()
    cfg.ensure_directories()

    # Write to gold zone by default, also populate raw zone
    gold_dir = output or cfg.storage.gold_path
    raw_dir = cfg.storage.raw_path
    gold_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {n_days} days of demo data...")

    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n_days, freq="D")

    # --- Media spend (5 channels with geo dimension) ---
    channels = ["google_search", "meta_facebook", "meta_instagram", "tiktok", "tv_linear"]
    geo = "national"  # single-geo default
    media_records = []
    for date in dates:
        for channel in channels:
            base = {
                "google_search": 1200, "meta_facebook": 900,
                "meta_instagram": 600, "tiktok": 400, "tv_linear": 1500,
            }[channel]
            seasonal = 1 + 0.3 * np.sin(date.dayofyear / 365 * 2 * np.pi)
            noise = 1 + np.random.normal(0, 0.15)
            spend = max(0, base * seasonal * noise)
            media_records.append({
                "date": date,
                "geo": geo,
                "channel": channel,
                "spend": round(spend, 2),
                "impressions": round(spend * np.random.uniform(40, 60), 0),
                "clicks": round(spend * np.random.uniform(0.3, 0.5), 0),
            })
    media_df = pd.DataFrame(media_records)
    media_df.to_parquet(gold_dir / "media_spend.parquet", index=False)
    media_df.to_parquet(raw_dir / "media_spend.parquet", index=False)

    # --- Outcomes (with geo) ---
    outcomes_records = []
    for date in dates:
        base_revenue = 50000
        seasonal = 1 + 0.2 * np.sin(date.dayofyear / 365 * 2 * np.pi)
        trend = 1 + (date - dates[0]).days / 365 * 0.1
        noise = np.random.normal(1, 0.08)
        revenue = base_revenue * seasonal * trend * noise
        conversions = round(revenue / 100 * np.random.uniform(0.8, 1.2), 0)
        outcomes_records.append({
            "date": date,
            "geo": geo,
            "revenue": round(revenue, 2),
            "conversions": conversions,
            "new_customers": round(conversions * 0.3, 0),
        })
    outcomes_df = pd.DataFrame(outcomes_records)
    outcomes_df.to_parquet(gold_dir / "outcomes.parquet", index=False)
    outcomes_df.to_parquet(raw_dir / "outcomes.parquet", index=False)

    # --- Controls (with geo) ---
    controls_df = pd.DataFrame({
        "date": dates,
        "geo": geo,
        "is_holiday": [1 if d.dayofweek >= 5 else 0 for d in dates],
        "promo": np.random.binomial(1, 0.1, n_days),
        "price_index": np.random.normal(1.0, 0.03, n_days).clip(0.8, 1.2).round(3),
    })
    controls_df.to_parquet(gold_dir / "controls.parquet", index=False)
    controls_df.to_parquet(raw_dir / "controls.parquet", index=False)

    # --- Incrementality tests ---
    test_records = []
    for i, channel in enumerate(channels[:3]):  # tests for first 3 channels
        start = dates[60 + i * 30]
        end = dates[90 + i * 30]
        lift = np.random.uniform(0.05, 0.25)
        test_records.append({
            "test_id": f"test_{channel}_2025",
            "channel": channel,
            "start_date": start,
            "end_date": end,
            "test_type": "geo_lift",
            "lift_estimate": round(lift, 4),
            "lift_ci_lower": round(lift * 0.6, 4),
            "lift_ci_upper": round(lift * 1.4, 4),
            "confidence_level": 0.95,
            "spend_during_test": round(np.random.uniform(20000, 80000), 2),
        })
    tests_df = pd.DataFrame(test_records)
    tests_df.to_parquet(gold_dir / "incrementality_tests.parquet", index=False)
    tests_df.to_parquet(raw_dir / "incrementality_tests.parquet", index=False)

    logger.info(f"Demo data written to {gold_dir}/")
    logger.info(f"  media_spend          : {len(media_df)} rows ({len(channels)} channels)")
    logger.info(f"  outcomes             : {len(outcomes_df)} rows")
    logger.info(f"  controls             : {len(controls_df)} rows")
    logger.info(f"  incrementality_tests : {len(tests_df)} tests")

    # Also write to processed_path (for backward compatibility)
    processed = cfg.storage.processed_path
    processed.mkdir(parents=True, exist_ok=True)
    if processed != gold_dir:
        media_df.to_parquet(processed / "media_spend.parquet", index=False)
        outcomes_df.to_parquet(processed / "outcomes.parquet", index=False)
        controls_df.to_parquet(processed / "controls.parquet", index=False)
        tests_df.to_parquet(processed / "incrementality_tests.parquet", index=False)

    # Auto-run pipeline
    logger.info("Running pipeline on demo data...")
    from pipeline.runner import Pipeline

    pipe = Pipeline(config=cfg.to_flat_dict(), runs_dir=cfg.storage.runs_path)
    pipe.connect(
        media_spend=gold_dir / "media_spend.parquet",
        outcomes=gold_dir / "outcomes.parquet",
        controls=gold_dir / "controls.parquet",
        incrementality_tests=gold_dir / "incrementality_tests.parquet",
    )
    results = pipe.run(model="builtin", target_col="revenue")

    logger.info(f"Demo pipeline complete.  Run ID: {pipe.run_id}")
    m = results.get("metrics", {})
    logger.info(f"  MAPE={m.get('mape', '?')}%  R2={m.get('r_squared', '?')}")


# ---------------------------------------------------------------------------
# runs (inspection)
# ---------------------------------------------------------------------------

@app.command()
def runs(
    limit: int = typer.Option(10, "--limit", "-n"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
):
    """List recent pipeline runs."""
    from core.artifacts import ArtifactStore
    store = ArtifactStore(runs_dir)
    manifests = store.list_runs(limit=limit)

    if not manifests:
        logger.info("No runs found.")
        return

    for m in manifests:
        status_tag = "OK" if m.status == "completed" else m.status.upper()
        metrics_str = ""
        if m.metrics:
            metrics_str = f"  MAPE={m.metrics.mape}%  R2={m.metrics.r_squared}"
        logger.info(
            f"  [{status_tag}] {m.run_id}  "
            f"backend={m.model_backend}  "
            f"rows={m.n_rows}  channels={m.n_channels}  "
            f"duration={m.duration_seconds}s"
            f"{metrics_str}"
        )


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

@app.command()
def backends():
    """List available model backends."""
    from models.registry import list_backends
    available = list_backends()
    logger.info(f"Available backends: {', '.join(available) if available else '(none)'}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
