"""API contract tests for parameter payload normalization."""

from __future__ import annotations

import pandas as pd
import pytest

from core.artifacts import ArtifactStore
from server.app import create_app
from transforms.saturation import hill_saturation


@pytest.fixture
def client_with_run(tmp_path):
    runs_dir = tmp_path / "runs"
    store = ArtifactStore(runs_dir)
    run_id = store.create_run(config_snapshot={})
    app = create_app(runs_dir=runs_dir)
    return app, store, run_id


def _route_endpoint(app, path: str, method: str = "GET"):
    method = method.upper()
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route.endpoint
    raise AssertionError(f"Route {method} {path} not found")


def test_parameters_endpoint_normalizes_legacy_keys(client_with_run):
    app, store, run_id = client_with_run
    store.save_json(
        run_id,
        "parameters",
        {
            "coefficients": {"google_spend": 1.2},
            "adstock_params": {"google_spend": {"alpha": 0.7, "l_max": 8}},
            "saturation_params": {"google_spend": {"K": 1000.0, "S": 1.5}},
        },
    )

    endpoint = _route_endpoint(app, "/api/v1/parameters")
    body = endpoint()

    # Canonical keys for UI contracts.
    assert body["adstock"]["google_spend"]["decay"] == pytest.approx(0.7)
    assert body["adstock"]["google_spend"]["max_lag"] == 8
    assert body["saturation"]["google_spend"]["K"] == pytest.approx(1000.0)
    assert body["saturation"]["google_spend"]["S"] == pytest.approx(1.5)

    # Legacy keys remain for backward compatibility.
    assert "adstock_params" in body
    assert "saturation_params" in body


def test_channel_insights_uses_legacy_saturation_params(client_with_run):
    app, store, run_id = client_with_run
    current_spend = 1000.0
    coef = 2.0
    k_val = 2000.0
    s_val = 2.0

    store.save_json(
        run_id,
        "parameters",
        {
            "coefficients": {"google_spend": coef},
            "adstock_params": {"google_spend": {"alpha": 0.5, "l_max": 8}},
            "saturation_params": {"google_spend": {"K": k_val, "S": s_val}},
        },
    )
    store.save_json(
        run_id,
        "optimization",
        {
            "current_allocation": {"google_spend": current_spend},
            "optimal_allocation": {"google_spend": 1100.0},
        },
    )

    endpoint = _route_endpoint(app, "/api/v1/channel-insights")
    body = endpoint()
    assert body["channels"], "expected at least one channel insight"

    insight = next(ch for ch in body["channels"] if ch["channel"] == "google_spend")
    delta = max(current_spend * 0.01, 1.0)
    expected_marginal_roi = (
        float(hill_saturation([current_spend + delta], K=k_val, S=s_val)[0]) * coef
        - float(hill_saturation([current_spend], K=k_val, S=s_val)[0]) * coef
    ) / delta

    assert insight["marginal_roi"] == pytest.approx(expected_marginal_roi, abs=1e-6)


def test_reconciliation_endpoint_normalizes_ci_aliases(client_with_run):
    app, store, run_id = client_with_run
    store.save_json(
        run_id,
        "reconciliation",
        {
            "channel_estimates": {
                "google_spend": {
                    "lift_estimate": 10.0,
                    "lift_ci_lower": 8.0,
                    "lift_ci_upper": 12.0,
                    "confidence_score": 0.9,
                }
            },
            "total_incremental_value": 10.0,
        },
    )

    endpoint = _route_endpoint(app, "/api/v1/reconciliation")
    body = endpoint()
    est = body["channel_estimates"]["google_spend"]
    assert est["ci_lower"] == pytest.approx(8.0)
    assert est["ci_upper"] == pytest.approx(12.0)
    assert est["lift_ci_lower"] == pytest.approx(8.0)
    assert est["lift_ci_upper"] == pytest.approx(12.0)


def test_optimization_endpoint_normalizes_legacy_keys(client_with_run):
    app, store, run_id = client_with_run
    store.save_json(
        run_id,
        "optimization",
        {
            "channel_allocations": {"google_spend": 700.0, "meta_spend": 300.0},
            "current_allocations": {"google_spend": 600.0, "meta_spend": 400.0},
            "optimized_response": 123.4,
            "baseline_response": 120.0,
        },
    )

    endpoint = _route_endpoint(app, "/api/v1/optimization")
    body = endpoint()
    assert body["optimal_allocation"] == {"google_spend": 700.0, "meta_spend": 300.0}
    assert body["current_allocation"] == {"google_spend": 600.0, "meta_spend": 400.0}
    assert body["expected_response"] == pytest.approx(123.4)
    assert body["current_response"] == pytest.approx(120.0)
    assert body["total_budget"] == pytest.approx(1000.0)


def test_roas_uses_normalized_current_allocation_keys(client_with_run):
    app, store, run_id = client_with_run
    store.save_dataframe(
        run_id,
        "contributions",
        # Channel column key is *_spend while allocation key is raw.
        pd.DataFrame(
            [
                {"date": "2026-01-01", "google_spend": 40.0, "actual": 100.0, "predicted": 95.0},
                {"date": "2026-01-02", "google_spend": 60.0, "actual": 110.0, "predicted": 100.0},
            ]
        ),
    )
    store.save_json(
        run_id,
        "optimization",
        {"current_allocations": {"google": 50.0}},
    )
    store.save_json(
        run_id,
        "parameters",
        {"coefficients": {"google_spend": 2.0}},
    )

    endpoint = _route_endpoint(app, "/api/v1/roas")
    body = endpoint()
    google_row = next(ch for ch in body["channels"] if ch["channel"] == "google_spend")
    assert google_row["total_spend"] == pytest.approx(50.0)
    assert google_row["roas"] == pytest.approx(2.0)
