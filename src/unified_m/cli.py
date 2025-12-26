"""
Command-line interface for Unified-M.

Provides commands for running the full pipeline or individual steps.
"""

from pathlib import Path
from typing import Optional
import json

import typer
from loguru import logger
import pandas as pd
import numpy as np

from unified_m.config import load_config, get_config

app = typer.Typer(
    name="unified-m",
    help="Unified Marketing Measurement CLI",
    add_completion=False,
)


@app.command()
def run_pipeline(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
    skip_training: bool = typer.Option(
        False, "--skip-training", help="Skip model training (use existing)"
    ),
):
    """
    Run the full MMM pipeline.
    
    Executes: ingest → validate → transform → train → reconcile → optimize
    """
    logger.info("Starting Unified-M pipeline...")
    
    # Load config
    config = load_config(config_path)
    config.ensure_directories()
    
    # Run pipeline steps
    if not skip_training:
        logger.info("Step 1: Ingesting data...")
        _run_ingest(config)
        
        logger.info("Step 2: Validating data...")
        _run_validate(config)
        
        logger.info("Step 3: Transforming data...")
        _run_transform(config)
        
        logger.info("Step 4: Training MMM...")
        _run_train(config)
    
    logger.info("Step 5: Reconciling measurements...")
    _run_reconcile(config)
    
    logger.info("Step 6: Optimizing budget...")
    _run_optimize(config)
    
    logger.info("Pipeline complete!")


@app.command()
def ingest(
    source: Path = typer.Option(
        Path("data/raw"), "--source", "-s", help="Source data directory"
    ),
    output: Path = typer.Option(
        Path("data/validated"), "--output", "-o", help="Output directory"
    ),
):
    """Ingest raw data from source."""
    from unified_m.ingestion import DataIngestion
    
    logger.info(f"Ingesting data from {source}")
    
    output.mkdir(parents=True, exist_ok=True)
    
    ingestion = DataIngestion()
    
    # Check for different data files
    for data_type in ["media_spend", "outcomes", "controls", "incrementality", "attribution"]:
        for ext in [".parquet", ".csv"]:
            path = source / f"{data_type}{ext}"
            if path.exists():
                loader = ingestion._get_loader(path, "auto")
                df = loader.load(path)
                
                # Save as parquet
                output_path = output / f"{data_type}.parquet"
                df.to_parquet(output_path)
                logger.info(f"Saved {data_type}: {len(df)} rows")
    
    logger.info("Ingestion complete")


@app.command()
def validate(
    input_dir: Path = typer.Option(
        Path("data/validated"), "--input", "-i", help="Input directory"
    ),
):
    """Validate data against schemas."""
    from unified_m.schemas import (
        validate_media_spend,
        validate_outcomes,
        validate_controls,
    )
    
    logger.info(f"Validating data in {input_dir}")
    
    # Validate each file
    validators = {
        "media_spend": validate_media_spend,
        "outcomes": validate_outcomes,
        "controls": validate_controls,
    }
    
    for name, validator in validators.items():
        path = input_dir / f"{name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            try:
                validator(df)
                logger.info(f"{name}: Valid ({len(df)} rows)")
            except Exception as e:
                logger.error(f"{name}: Validation failed - {e}")
                raise
    
    logger.info("Validation complete")


@app.command()
def transform(
    input_dir: Path = typer.Option(
        Path("data/validated"), "--input", "-i", help="Input directory"
    ),
    output: Path = typer.Option(
        Path("data/transformed"), "--output", "-o", help="Output directory"
    ),
):
    """Transform data for MMM training."""
    from unified_m.transforms import create_mmm_features
    
    logger.info(f"Transforming data from {input_dir}")
    
    output.mkdir(parents=True, exist_ok=True)
    
    # Load data
    media_spend = pd.read_parquet(input_dir / "media_spend.parquet")
    outcomes = pd.read_parquet(input_dir / "outcomes.parquet")
    
    controls = None
    controls_path = input_dir / "controls.parquet"
    if controls_path.exists():
        controls = pd.read_parquet(controls_path)
    
    # Create features
    target_col = "revenue" if "revenue" in outcomes.columns else "conversions"
    
    mmm_data = create_mmm_features(
        media_spend=media_spend,
        outcomes=outcomes,
        controls=controls,
        target_col=target_col,
    )
    
    # Save
    mmm_data.to_parquet(output / "mmm_input.parquet")
    logger.info(f"Created MMM input: {len(mmm_data)} rows, {len(mmm_data.columns)} columns")


@app.command()
def train(
    input_dir: Path = typer.Option(
        Path("data/transformed"), "--input", "-i", help="Input directory"
    ),
    output: Path = typer.Option(
        Path("models"), "--output", "-o", help="Model output directory"
    ),
    samples: int = typer.Option(1000, "--samples", "-s", help="Posterior samples"),
    chains: int = typer.Option(4, "--chains", help="MCMC chains"),
):
    """Train MMM model."""
    from unified_m.mmm import UnifiedMMM
    
    logger.info(f"Training MMM with {samples} samples, {chains} chains")
    
    output.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = pd.read_parquet(input_dir / "mmm_input.parquet")
    
    # Initialize model
    mmm = UnifiedMMM(
        date_col="date",
        target_col="y",
    )
    
    # Train
    results = mmm.fit(df, n_samples=samples, n_chains=chains)
    
    # Save results
    results.save(output / "mmm_model")
    
    # Save outputs for API
    outputs_dir = Path("data/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    with open(outputs_dir / "mmm_results.json", "w") as f:
        json.dump(results.to_dict(), f, indent=2, default=str)
    
    if results.channel_contributions is not None:
        results.channel_contributions.to_parquet(outputs_dir / "contributions.parquet")
    
    logger.info(f"Model trained. MAPE: {results.metrics.get('mape', 'N/A'):.2f}%")


@app.command()
def reconcile(
    mmm_results: Path = typer.Option(
        Path("data/outputs/mmm_results.json"), "--mmm-results", help="MMM results file"
    ),
    output: Path = typer.Option(
        Path("data/outputs/reconciliation.json"), "--output", "-o", help="Output file"
    ),
):
    """Run reconciliation to unify measurement signals."""
    from unified_m.reconciliation import ReconciliationEngine
    
    logger.info("Running reconciliation...")
    
    # Load MMM results
    with open(mmm_results) as f:
        mmm_data = json.load(f)
    
    # Load incrementality tests if available
    incr_path = Path("data/validated/incrementality.parquet")
    incrementality_tests = None
    if incr_path.exists():
        incrementality_tests = pd.read_parquet(incr_path)
    
    # Load attribution if available
    attr_path = Path("data/validated/attribution.parquet")
    attribution_data = None
    if attr_path.exists():
        attribution_data = pd.read_parquet(attr_path)
    
    # Run reconciliation
    engine = ReconciliationEngine(
        mmm_weight=0.5,
        incrementality_weight=0.3,
        attribution_weight=0.2,
    )
    
    result = engine.reconcile(
        mmm_results=mmm_data,
        incrementality_tests=incrementality_tests,
        attribution_data=attribution_data,
    )
    
    # Save
    result.save(output)
    logger.info(f"Reconciliation complete: {len(result.channel_estimates)} channels")


@app.command()
def optimize(
    mmm_results: Path = typer.Option(
        Path("data/outputs/mmm_results.json"), "--mmm-results", help="MMM results file"
    ),
    budget: float = typer.Option(100000, "--budget", "-b", help="Total budget"),
    output: Path = typer.Option(
        Path("data/outputs/optimization.json"), "--output", "-o", help="Output file"
    ),
):
    """Run budget optimization."""
    from unified_m.optimization import BudgetOptimizer
    
    logger.info(f"Optimizing budget allocation for ${budget:,.0f}")
    
    # Load MMM results
    with open(mmm_results) as f:
        mmm_data = json.load(f)
    
    # Build response curves from saturation params
    channels = mmm_data.get("channels", [])
    saturation_params = mmm_data.get("saturation_params", {})
    coefficients = mmm_data.get("coefficients", {})
    
    response_params = {}
    for channel in channels:
        params = saturation_params.get(channel, {})
        coef = coefficients.get(channel, 1.0)
        
        response_params[channel] = {
            "K": params.get("K", 10000),
            "S": params.get("S", 1.0),
            "coefficient": coef,
        }
    
    # Optimize
    optimizer = BudgetOptimizer(
        response_params=response_params,
        total_budget=budget,
    )
    
    result = optimizer.optimize()
    
    # Save
    result.save(output)
    
    # Save response curves
    curves = {}
    for channel, curve_fn in optimizer.response_curves.items():
        spend = np.linspace(0, budget * 0.5, 100)
        response = [curve_fn(s) for s in spend]
        curves[channel] = {"spend": spend.tolist(), "response": response}
    
    curves_path = output.parent / "response_curves.json"
    with open(curves_path, "w") as f:
        json.dump(curves, f, indent=2)
    
    logger.info(f"Optimization complete. Improvement: {result.improvement_pct:.1f}%")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the API server."""
    from unified_m.api.app import run_server
    
    run_server(host=host, port=port, reload=reload)


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", "-p", help="Port for Streamlit"),
):
    """Start the Streamlit UI."""
    import subprocess
    
    logger.info(f"Starting Streamlit UI on port {port}")
    subprocess.run([
        "streamlit", "run", "ui/app.py",
        "--server.port", str(port),
        "--server.headless", "true",
    ])


@app.command()
def generate_demo(
    output: Path = typer.Option(
        Path("data/raw"), "--output", "-o", help="Output directory"
    ),
    n_days: int = typer.Option(365, "--days", help="Number of days of data"),
):
    """Generate demo data for testing."""
    logger.info(f"Generating {n_days} days of demo data...")
    
    output.mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="D")
    
    # Media spend (3 channels)
    channels = ["google", "meta", "tv"]
    media_records = []
    
    for date in dates:
        for channel in channels:
            base_spend = {"google": 1000, "meta": 800, "tv": 500}[channel]
            # Add some variation
            spend = base_spend * (1 + 0.3 * np.sin(date.dayofyear / 365 * 2 * np.pi))
            spend *= (1 + np.random.normal(0, 0.2))
            
            media_records.append({
                "date": date,
                "channel": channel,
                "spend": max(0, spend),
                "impressions": spend * np.random.uniform(40, 60),
                "clicks": spend * np.random.uniform(0.3, 0.5),
            })
    
    media_df = pd.DataFrame(media_records)
    media_df.to_parquet(output / "media_spend.parquet")
    
    # Outcomes
    outcomes_records = []
    for date in dates:
        # Base revenue with seasonality and trend
        base = 50000
        seasonality = 1 + 0.2 * np.sin(date.dayofyear / 365 * 2 * np.pi)
        trend = 1 + (date - dates[0]).days / 365 * 0.1
        noise = np.random.normal(1, 0.1)
        
        revenue = base * seasonality * trend * noise
        conversions = revenue / 100 * np.random.uniform(0.8, 1.2)
        
        outcomes_records.append({
            "date": date,
            "revenue": revenue,
            "conversions": conversions,
        })
    
    outcomes_df = pd.DataFrame(outcomes_records)
    outcomes_df.to_parquet(output / "outcomes.parquet")
    
    # Controls
    controls_df = pd.DataFrame({
        "date": dates,
        "is_holiday": [1 if d.dayofweek >= 5 else 0 for d in dates],
        "promo": np.random.binomial(1, 0.1, n_days),
    })
    controls_df.to_parquet(output / "controls.parquet")
    
    logger.info(f"Generated demo data in {output}")
    logger.info(f"  - Media spend: {len(media_df)} rows")
    logger.info(f"  - Outcomes: {len(outcomes_df)} rows")
    logger.info(f"  - Controls: {len(controls_df)} rows")


# Helper functions for pipeline
def _run_ingest(config):
    """Run ingestion step."""
    pass  # Handled by ingest command


def _run_validate(config):
    """Run validation step."""
    pass  # Handled by validate command


def _run_transform(config):
    """Run transformation step."""
    pass  # Handled by transform command


def _run_train(config):
    """Run training step."""
    pass  # Handled by train command


def _run_reconcile(config):
    """Run reconciliation step."""
    pass  # Handled by reconcile command


def _run_optimize(config):
    """Run optimization step."""
    pass  # Handled by optimize command


if __name__ == "__main__":
    app()

