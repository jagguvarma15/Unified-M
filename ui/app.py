"""
Streamlit UI for Unified-M.

A modern, interactive dashboard for exploring MMM results,
reconciliation outputs, and optimization recommendations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import json
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Unified-M | Marketing Measurement",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern look
st.markdown("""
<style>
    /* Main theme */
    :root {
        --primary-color: #6366f1;
        --background-dark: #0f172a;
        --card-bg: #1e293b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #f59e0b;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #475569;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-delta-positive {
        color: #10b981;
    }
    
    .metric-delta-negative {
        color: #ef4444;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #6366f1;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1e293b;
    }
    
    /* Plotly chart backgrounds */
    .js-plotly-plot {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# =============================================================================
# Data Loading Functions
# =============================================================================

@st.cache_data(ttl=60)
def load_from_api(endpoint: str) -> dict | None:
    """Load data from the API."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except requests.exceptions.RequestException:
        return None


@st.cache_data(ttl=60)
def load_from_file(path: str) -> dict | None:
    """Load data directly from file."""
    file_path = Path(path)
    if not file_path.exists():
        return None
    
    if file_path.suffix == ".json":
        with open(file_path) as f:
            return json.load(f)
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
        return {"data": df.to_dict(orient="records")}
    return None


def load_contributions():
    """Load contribution data."""
    # Try API first
    data = load_from_api("/api/v1/contributions")
    if data:
        return data
    
    # Fallback to file
    return load_from_file("data/outputs/contributions.parquet")


def load_reconciliation():
    """Load reconciliation data."""
    data = load_from_api("/api/v1/reconciliation")
    if data:
        return data
    return load_from_file("data/outputs/reconciliation.json")


def load_optimization():
    """Load optimization data."""
    data = load_from_api("/api/v1/optimization")
    if data:
        return data
    return load_from_file("data/outputs/optimization.json")


def load_response_curves():
    """Load response curves."""
    data = load_from_api("/api/v1/response-curves")
    if data:
        return data
    return load_from_file("data/outputs/response_curves.json")


# =============================================================================
# Visualization Components
# =============================================================================

def create_contribution_chart(data: dict) -> go.Figure:
    """Create contribution decomposition waterfall chart."""
    if "summary" not in data:
        return None
    
    summary = data["summary"]
    channels = summary.get("channels", {})
    
    # Prepare data for waterfall
    labels = ["Baseline"] + list(channels.keys()) + ["Total"]
    values = [summary.get("baseline_contribution", 0)]
    values += [ch.get("total", 0) for ch in channels.values()]
    values.append(summary.get("total_contribution", 0))
    
    measures = ["absolute"] + ["relative"] * len(channels) + ["total"]
    
    fig = go.Figure(go.Waterfall(
        name="Contribution",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        connector={"line": {"color": "#475569"}},
        increasing={"marker": {"color": "#10b981"}},
        decreasing={"marker": {"color": "#ef4444"}},
        totals={"marker": {"color": "#6366f1"}},
        textposition="outside",
        text=[f"${v:,.0f}" for v in values],
    ))
    
    fig.update_layout(
        title="Channel Contribution Decomposition",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
        showlegend=False,
        height=400,
    )
    
    return fig


def create_contribution_share_chart(data: dict) -> go.Figure:
    """Create contribution share pie chart."""
    if "summary" not in data:
        return None
    
    channels = data["summary"].get("channels", {})
    
    labels = list(channels.keys())
    values = [ch.get("total", 0) for ch in channels.values()]
    
    # Color palette
    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors[:len(labels)],
        textinfo="label+percent",
        textposition="outside",
    )])
    
    fig.update_layout(
        title="Channel Contribution Share",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    
    return fig


def create_contribution_timeline(data: dict) -> go.Figure:
    """Create contribution timeline chart."""
    if "data" not in data:
        return None
    
    df = pd.DataFrame(data["data"])
    
    if "date" not in df.columns:
        return None
    
    df["date"] = pd.to_datetime(df["date"])
    
    # Get contribution columns
    contrib_cols = [c for c in df.columns if c.endswith("_contribution")]
    
    fig = go.Figure()
    
    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
    
    for i, col in enumerate(contrib_cols):
        channel = col.replace("_contribution", "")
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df[col],
            name=channel,
            mode="lines",
            stackgroup="one",
            line=dict(width=0),
            fillcolor=colors[i % len(colors)],
        ))
    
    fig.update_layout(
        title="Contributions Over Time",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    
    return fig


def create_reconciliation_chart(data: dict) -> go.Figure:
    """Create reconciliation confidence chart."""
    if "channel_estimates" not in data:
        return None
    
    estimates = data["channel_estimates"]
    
    channels = []
    lifts = []
    ci_lower = []
    ci_upper = []
    confidence = []
    
    for channel, est in estimates.items():
        channels.append(channel)
        lifts.append(est.get("lift_estimate", 0))
        ci_lower.append(est.get("lift_ci_lower", 0))
        ci_upper.append(est.get("lift_ci_upper", 0))
        confidence.append(est.get("confidence_score", 0))
    
    fig = go.Figure()
    
    # Error bars for CI
    fig.add_trace(go.Bar(
        x=channels,
        y=lifts,
        error_y=dict(
            type="data",
            symmetric=False,
            array=[u - l for u, l in zip(ci_upper, lifts)],
            arrayminus=[l - lo for l, lo in zip(lifts, ci_lower)],
            color="#94a3b8",
        ),
        marker_color=["#6366f1" if c > 0.7 else "#f59e0b" if c > 0.4 else "#ef4444" for c in confidence],
        text=[f"Conf: {c:.0%}" for c in confidence],
        textposition="outside",
    ))
    
    fig.update_layout(
        title="Reconciled Channel Lift Estimates with Uncertainty",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
        height=400,
        yaxis_title="Incremental Lift",
        xaxis_title="Channel",
    )
    
    return fig


def create_optimization_chart(data: dict) -> go.Figure:
    """Create optimization comparison chart."""
    if "optimal_allocation" not in data:
        return None
    
    optimal = data.get("optimal_allocation", {})
    current = data.get("current_allocation", {})
    
    channels = list(optimal.keys())
    optimal_values = [optimal.get(ch, 0) for ch in channels]
    current_values = [current.get(ch, 0) for ch in channels]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Current",
        x=channels,
        y=current_values,
        marker_color="#475569",
    ))
    
    fig.add_trace(go.Bar(
        name="Recommended",
        x=channels,
        y=optimal_values,
        marker_color="#6366f1",
    ))
    
    fig.update_layout(
        title="Current vs. Recommended Budget Allocation",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
        height=400,
        barmode="group",
        yaxis_title="Budget ($)",
        xaxis_title="Channel",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    
    return fig


def create_response_curves_chart(data: dict) -> go.Figure:
    """Create response curves chart."""
    fig = go.Figure()
    
    colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899"]
    
    for i, (channel, curve_data) in enumerate(data.items()):
        if isinstance(curve_data, list):
            spend = [p.get("spend", 0) for p in curve_data]
            response = [p.get("response", 0) for p in curve_data]
        else:
            spend = curve_data.get("spend", [])
            response = curve_data.get("response", [])
        
        fig.add_trace(go.Scatter(
            x=spend,
            y=response,
            name=channel,
            mode="lines",
            line=dict(color=colors[i % len(colors)], width=2),
        ))
    
    fig.update_layout(
        title="Channel Response Curves (Saturation)",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
        height=400,
        xaxis_title="Spend ($)",
        yaxis_title="Response",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    
    return fig


# =============================================================================
# Main Application
# =============================================================================

def main():
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50?text=Unified-M", width=150)
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Dashboard", "Contributions", "Reconciliation", "Optimization", "Response Curves"],
            label_visibility="collapsed",
        )
        
        st.markdown("---")
        
        # Data refresh
        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #64748b; font-size: 0.75rem;'>
            Unified-M v0.1.0<br>
            Last updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    if page == "Dashboard":
        render_dashboard()
    elif page == "Contributions":
        render_contributions()
    elif page == "Reconciliation":
        render_reconciliation()
    elif page == "Optimization":
        render_optimization()
    elif page == "Response Curves":
        render_response_curves()


def render_dashboard():
    """Render main dashboard."""
    st.title("Unified Marketing Measurement")
    st.markdown("*Strategic insights from MMM, Attribution, and Incrementality*")
    
    # Load data
    contributions = load_contributions()
    reconciliation = load_reconciliation()
    optimization = load_optimization()
    
    # Check if data is available
    has_data = any([contributions, reconciliation, optimization])
    
    if not has_data:
        st.warning("""
        **No data available yet.**
        
        Run the training pipeline to generate results:
        ```bash
        python -m cli run-pipeline
        ```
        
        Or use the demo data:
        ```bash
        python -m cli generate-demo
        ```
        """)
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_contribution = 0
        if contributions and "summary" in contributions:
            total_contribution = contributions["summary"].get("total_contribution", 0)
        st.metric(
            label="Total Attribution",
            value=f"${total_contribution:,.0f}",
            delta="From MMM",
        )
    
    with col2:
        total_incremental = 0
        if reconciliation:
            total_incremental = reconciliation.get("total_incremental_value", 0)
        st.metric(
            label="Reconciled Lift",
            value=f"${total_incremental:,.0f}",
            delta="Calibrated",
        )
    
    with col3:
        improvement = 0
        if optimization:
            improvement = optimization.get("improvement_pct", 0)
        st.metric(
            label="Optimization Potential",
            value=f"{improvement:,.1f}%",
            delta="Expected uplift",
        )
    
    with col4:
        n_channels = 0
        if contributions and "summary" in contributions:
            n_channels = len(contributions["summary"].get("channels", {}))
        st.metric(
            label="Active Channels",
            value=f"{n_channels}",
            delta="Measured",
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if contributions:
            fig = create_contribution_share_chart(contributions)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if reconciliation:
            fig = create_reconciliation_chart(reconciliation)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    # Timeline
    if contributions:
        fig = create_contribution_timeline(contributions)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def render_contributions():
    """Render contributions page."""
    st.title("Channel Contributions")
    st.markdown("*Decomposition of total response by marketing channel*")
    
    data = load_contributions()
    
    if not data:
        st.warning("No contribution data available. Run the training pipeline first.")
        return
    
    # Summary metrics
    if "summary" in data:
        summary = data["summary"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Total Contribution",
                f"${summary.get('total_contribution', 0):,.0f}",
            )
        
        with col2:
            st.metric(
                "Baseline",
                f"${summary.get('baseline_contribution', 0):,.0f}",
            )
    
    st.markdown("---")
    
    # Waterfall chart
    fig = create_contribution_chart(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Share chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_contribution_share_chart(data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_contribution_timeline(data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("### Detailed Data")
    if "data" in data:
        df = pd.DataFrame(data["data"])
        st.dataframe(df, use_container_width=True)


def render_reconciliation():
    """Render reconciliation page."""
    st.title("Reconciled Estimates")
    st.markdown("*Unified channel lift combining MMM, tests, and attribution*")
    
    data = load_reconciliation()
    
    if not data:
        st.warning("No reconciliation data available. Run the pipeline first.")
        return
    
    # Method info
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Reconciliation Method",
            data.get("method", "Unknown").replace("_", " ").title(),
        )
    
    with col2:
        st.metric(
            "Total Incremental Value",
            f"${data.get('total_incremental_value', 0):,.0f}",
        )
    
    st.markdown("---")
    
    # Chart
    fig = create_reconciliation_chart(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel details
    st.markdown("### Channel Details")
    
    if "channel_estimates" in data:
        estimates = data["channel_estimates"]
        
        for channel, est in estimates.items():
            with st.expander(f"{channel.replace('_', ' ').title()}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Lift Estimate", f"${est.get('lift_estimate', 0):,.2f}")
                
                with col2:
                    conf = est.get("confidence_score", 0)
                    st.metric(
                        "Confidence",
                        f"{conf:.0%}",
                        delta="High" if conf > 0.7 else "Medium" if conf > 0.4 else "Low",
                    )
                
                with col3:
                    st.metric(
                        "95% CI",
                        f"[{est.get('lift_ci_lower', 0):,.0f}, {est.get('lift_ci_upper', 0):,.0f}]",
                    )


def render_optimization():
    """Render optimization page."""
    st.title("Budget Optimization")
    st.markdown("*Recommended budget allocation to maximize ROI*")
    
    data = load_optimization()
    
    if not data:
        st.warning("No optimization data available. Run the pipeline first.")
        return
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Budget",
            f"${data.get('total_budget', 0):,.0f}",
        )
    
    with col2:
        st.metric(
            "Expected Response",
            f"${data.get('expected_response', 0):,.0f}",
        )
    
    with col3:
        improvement = data.get("improvement_pct", 0)
        st.metric(
            "Improvement vs Current",
            f"{improvement:+.1f}%",
            delta="Potential uplift",
        )
    
    st.markdown("---")
    
    # Comparison chart
    fig = create_optimization_chart(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations table
    st.markdown("### Recommendations")
    
    optimal = data.get("optimal_allocation", {})
    current = data.get("current_allocation", {})
    
    recommendations = []
    for channel in optimal.keys():
        opt = optimal.get(channel, 0)
        cur = current.get(channel, 0)
        change = opt - cur
        change_pct = (change / cur * 100) if cur > 0 else 0
        
        recommendations.append({
            "Channel": channel.replace("_", " ").title(),
            "Current ($)": f"${cur:,.0f}",
            "Recommended ($)": f"${opt:,.0f}",
            "Change ($)": f"${change:+,.0f}",
            "Change (%)": f"{change_pct:+.1f}%",
            "Action": "Increase" if change > 0 else "Decrease" if change < 0 else "Maintain",
        })
    
    df = pd.DataFrame(recommendations)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_response_curves():
    """Render response curves page."""
    st.title("Response Curves")
    st.markdown("*Channel saturation curves showing diminishing returns*")
    
    data = load_response_curves()
    
    if not data:
        st.warning("No response curve data available. Run the pipeline first.")
        return
    
    # Main chart
    fig = create_response_curves_chart(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual channel analysis
    st.markdown("### Channel Analysis")
    
    selected_channel = st.selectbox(
        "Select Channel",
        options=list(data.keys()),
    )
    
    if selected_channel:
        curve_data = data[selected_channel]
        
        if isinstance(curve_data, list):
            df = pd.DataFrame(curve_data)
        else:
            df = pd.DataFrame({
                "spend": curve_data.get("spend", []),
                "response": curve_data.get("response", []),
            })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Find saturation point (80% of max response)
            max_response = df["response"].max()
            saturation_idx = (df["response"] >= 0.8 * max_response).idxmax()
            saturation_spend = df.loc[saturation_idx, "spend"]
            
            st.metric(
                "Saturation Point (80%)",
                f"${saturation_spend:,.0f}",
                help="Spend level at 80% of maximum response",
            )
        
        with col2:
            # Compute marginal ROI at midpoint
            mid_idx = len(df) // 2
            if mid_idx < len(df) - 1:
                marginal = (
                    df.loc[mid_idx + 1, "response"] - df.loc[mid_idx, "response"]
                ) / (
                    df.loc[mid_idx + 1, "spend"] - df.loc[mid_idx, "spend"]
                )
                st.metric(
                    "Marginal ROI (at median)",
                    f"{marginal:.4f}",
                    help="Additional response per additional dollar",
                )


if __name__ == "__main__":
    main()

