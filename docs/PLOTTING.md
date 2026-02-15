# Plotting standard for Unified-M

Unified-M uses a **React + FastAPI** stack (not Streamlit). This document defines the plotting standard for MMM/UMM: which libraries to use, where, and how to get high-value, high-definition charts and exports.

---

## Current state

| Layer | Library | Use |
|-------|---------|-----|
| **Frontend (React)** | Recharts | All interactive dashboards: diagnostics, response curves, contributions, ROAS, calibration, stability, scenario planner, spend pacing. |
| **Backend (Python)** | None (JSON only) | API returns downsampled `chart` data; no server-side image generation yet. |

---

## Recommended stack (by use-case)

### 1. Frontend: Recharts today → ECharts when you need more

- **Recharts** (current): Good for consistent React charts, small multiples, tooltips. Keep it as the default for dashboards.
- **Apache ECharts** (optional upgrade): Use when you need **higher performance** (Canvas or SVG), more chart types, or production polish. Apache 2.0, works well in React ([echarts.apache.org](https://echarts.apache.org/)).

**Best for:** Contribution-over-time, response curves, marginal ROAS, calibration vs lift, stability/whipsaw, scenario comparisons.

### 2. Backend exports: Plotly + Matplotlib

- **Plotly (Python)**  
  - Interactive dashboards (hover, toggles, facets).  
  - Export **PNG/JPG/WebP + SVG/PDF** for high-definition reports.  
  - Best for: contribution-over-time with uncertainty bands, response curves, scenario comparisons.

- **Matplotlib**  
  - Publication-grade static figures, full control over typography and layout.  
  - Export **image or vector (PDF/SVG)**.  
  - Best for: final ROAS tables/plots, regression diagnostics, residual plots, static report exports.

Add these as **optional** dependencies when you implement “Export report” or batch PDF/SVG generation (see `requirements.txt`).

### 3. Big data (future): HoloViz / Datashader

- **Datashader + HoloViews** when you hit performance limits (e.g. millions of points).  
- Best for: massive scatter (impressions/clicks), dense time-series, geo heatmaps.

### 4. Declarative specs (optional): Vega-Lite / Altair

- **Vega-Lite** (JSON specs) for consistent, repeatable charts and templates.  
- Best for: standardized dashboards, interactive filtering, small-multiple comparisons.

---

## MMM/UMM chart list and suggested library

| Chart | Frontend (React) | Backend export |
|-------|------------------|----------------|
| Contribution waterfall | Recharts (or ECharts) | Plotly → SVG/PDF |
| Response curves (spend vs response) | Recharts | Plotly → SVG/PDF |
| Marginal ROAS by channel | Recharts | Plotly or Matplotlib |
| Calibration vs lift / error by channel | Recharts | Plotly or Matplotlib |
| Stability / whipsaw | Recharts | Plotly |
| Scenario comparison | Recharts | Plotly |
| Diagnostics (actual vs predicted, residuals) | Recharts | Matplotlib for paper-grade |
| ROAS tables/radar | Recharts | Matplotlib |

---

## Theme and consistency

- Use a **single palette** across frontend and backend (e.g. Tailwind/Recharts palette mirrored in Plotly/Matplotlib).
- Keep **axis labels, units, and legends** consistent (e.g. “Spend”, “Response”, “Channel”).
- For reports: **SVG or PDF** for vector quality; **PNG/WebP** when file size matters.

---

## Adoption path

1. **Short term:** Keep Recharts for all React dashboards; ensure existing charts match the list above (contribution waterfall, response curves, marginal ROAS, calibration, stability, scenario comparison, diagnostics).
2. **When adding “Export report”:** Add optional `plotly` and `matplotlib` to the backend; implement endpoints or CLI that generate SVG/PDF (and optionally PNG) using the same data as the API.
3. **If frontend performance becomes an issue:** Evaluate ECharts for heavy pages (e.g. many series or large datasets).
4. **If you hit big-data limits:** Consider Datashader/HoloViews for server-side rasterization of very large scatters or time-series.

---

## References

- [Plotly Python – static image export](https://plotly.com/python/static-image-export/)
- [Matplotlib savefig (PDF/SVG)](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html)
- [Vega-Lite](https://vega.github.io/vega-lite/)
- [HoloViews – large data (Datashader)](https://holoviews.org/user_guide/Large_Data.html)
- [Apache ECharts](https://echarts.apache.org/)
- [Datashader – performance](https://datashader.org/user_guide/Performance.html)
