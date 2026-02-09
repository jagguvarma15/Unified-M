"""
Command-line interface for Unified-M.

Provides commands for:
  - Running the full pipeline
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

    cfg = load_config(config_path)
    cfg.ensure_directories()

    # Resolve data paths
    ms = media_spend or cfg.storage.processed_path / "media_spend.parquet"
    oc = outcomes or cfg.storage.processed_path / "outcomes.parquet"
    ct = controls or cfg.storage.processed_path / "controls.parquet"
    ts = tests or cfg.storage.processed_path / "incrementality_tests.parquet"
    at = attribution

    pipe = Pipeline(
        config=cfg.to_flat_dict(),
        runs_dir=cfg.storage.runs_path,
    )

    pipe.connect(
        media_spend=ms if Path(ms).exists() else None,
        outcomes=oc if Path(oc).exists() else None,
        controls=ct if Path(ct).exists() else None,
        incrementality_tests=ts if Path(ts).exists() else None,
        attribution=at if at and Path(at).exists() else None,
    )

    results = pipe.run(
        model=model,
        target_col=target,
        total_budget=budget,
    )

    logger.info(f"Run complete: {pipe.run_id}")
    metrics = results.get("metrics", {})
    logger.info(f"  MAPE  : {metrics.get('mape', 'N/A')}%")
    logger.info(f"  R2    : {metrics.get('r_squared', 'N/A')}")
    logger.info(f"  RMSE  : {metrics.get('rmse', 'N/A')}")


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
    output: Path = typer.Option(
        Path("data/processed"), "--output", "-o",
    ),
    n_days: int = typer.Option(365, "--days"),
):
    """
    Generate synthetic demo data and immediately run the pipeline.

    This is the fastest way to see the framework in action:

        unified-m demo
        unified-m serve
    """
    logger.info(f"Generating {n_days} days of demo data...")
    output.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n_days, freq="D")

    # --- Media spend (3 channels) ---
    channels = ["google", "meta", "tv"]
    media_records = []
    for date in dates:
        for channel in channels:
            base = {"google": 1000, "meta": 800, "tv": 500}[channel]
            seasonal = 1 + 0.3 * np.sin(date.dayofyear / 365 * 2 * np.pi)
            noise = 1 + np.random.normal(0, 0.15)
            spend = max(0, base * seasonal * noise)
            media_records.append({
                "date": date,
                "channel": channel,
                "spend": round(spend, 2),
                "impressions": round(spend * np.random.uniform(40, 60), 0),
                "clicks": round(spend * np.random.uniform(0.3, 0.5), 0),
            })
    media_df = pd.DataFrame(media_records)
    media_df.to_parquet(output / "media_spend.parquet", index=False)

    # --- Outcomes ---
    outcomes_records = []
    for date in dates:
        base_revenue = 50000
        seasonal = 1 + 0.2 * np.sin(date.dayofyear / 365 * 2 * np.pi)
        trend = 1 + (date - dates[0]).days / 365 * 0.1
        noise = np.random.normal(1, 0.08)
        revenue = base_revenue * seasonal * trend * noise
        outcomes_records.append({
            "date": date,
            "revenue": round(revenue, 2),
            "conversions": round(revenue / 100 * np.random.uniform(0.8, 1.2), 0),
        })
    outcomes_df = pd.DataFrame(outcomes_records)
    outcomes_df.to_parquet(output / "outcomes.parquet", index=False)

    # --- Controls ---
    controls_df = pd.DataFrame({
        "date": dates,
        "is_holiday": [1 if d.dayofweek >= 5 else 0 for d in dates],
        "promo": np.random.binomial(1, 0.1, n_days),
    })
    controls_df.to_parquet(output / "controls.parquet", index=False)

    # --- Incrementality tests ---
    test_records = []
    for i, channel in enumerate(channels):
        start = dates[60 + i * 30]
        end = dates[90 + i * 30]
        lift = np.random.uniform(0.05, 0.25)
        test_records.append({
            "test_id": f"test_{channel}_2024",
            "channel": channel,
            "start_date": start,
            "end_date": end,
            "test_type": "geo_lift",
            "lift_estimate": round(lift, 4),
            "lift_ci_lower": round(lift * 0.6, 4),
            "lift_ci_upper": round(lift * 1.4, 4),
            "confidence_level": 0.95,
        })
    tests_df = pd.DataFrame(test_records)
    tests_df.to_parquet(output / "incrementality_tests.parquet", index=False)

    logger.info(f"Demo data written to {output}/")
    logger.info(f"  media_spend          : {len(media_df)} rows")
    logger.info(f"  outcomes             : {len(outcomes_df)} rows")
    logger.info(f"  controls             : {len(controls_df)} rows")
    logger.info(f"  incrementality_tests : {len(tests_df)} tests")

    # Auto-run pipeline
    logger.info("Running pipeline on demo data...")
    from config import load_config
    from pipeline.runner import Pipeline

    cfg = load_config()
    cfg.ensure_directories()

    pipe = Pipeline(config=cfg.to_flat_dict(), runs_dir=cfg.storage.runs_path)
    pipe.connect(
        media_spend=output / "media_spend.parquet",
        outcomes=output / "outcomes.parquet",
        controls=output / "controls.parquet",
        incrementality_tests=output / "incrementality_tests.parquet",
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
