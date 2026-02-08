# Unified-M Implementation Guide

This document provides detailed implementation information for the Unified Marketing Measurement platform.

## Mathematical Foundations

### 1. Adstock Transformations

Adstock models the **carryover effect** of advertising - today's ad spend affects not just today's outcomes, but also future outcomes with decaying impact.

#### Geometric Adstock (Most Common)

**Formula:**
```
adstock[t] = x[t] + α × adstock[t-1]
```

Equivalently, using convolution:
```
adstock[t] = Σ(k=0 to L) w[k] × x[t-k]
where w[k] = α^k / Σα^k (normalized weights)
```

**Parameters:**
- `α` (alpha): Decay rate, range [0, 1]
  - α = 0: No carryover (immediate effect only)
  - α = 0.5: Half-life of ~1 period
  - α = 0.9: Long carryover (TV, brand campaigns)
- `L` (l_max): Maximum lag to consider

**Interpretation:**
- **Half-life**: `t_half = ln(0.5) / ln(α)`
  - If α = 0.7, half-life ≈ 1.9 periods
  - If α = 0.9, half-life ≈ 6.6 periods

**Code:**
```python
def geometric_adstock(x, alpha, l_max=8, normalize=True):
    """Apply geometric adstock transformation."""
    weights = np.array([alpha ** i for i in range(l_max)])
    if normalize:
        weights = weights / weights.sum()
    return np.convolve(x, weights)[:len(x)]
```

#### Weibull Adstock (More Flexible)

**Formula:**
```
w[k] = (k/λ)^(c-1) × exp(-(k/λ)^c) / λ  (PDF form)
```

**Parameters:**
- `c` (shape): Controls decay shape
  - c < 1: Rapid initial decay
  - c = 1: Exponential (same as geometric)
  - c > 1: Delayed peak then decay
- `λ` (scale): Controls time scale

**Use Case:** When effect peaks after a delay (e.g., consideration period)

#### Delayed Adstock

**Formula:**
```
w[k] = (k+1)/(θ+1)           for k < θ (building up)
w[k] = α^(k-θ)               for k ≥ θ (decaying)
```

**Parameters:**
- `θ` (theta): Delay before peak effect
- `α`: Decay rate after peak

---

### 2. Saturation Transformations

Saturation models **diminishing returns** - doubling spend doesn't double response.

#### Hill Function (Most Common)

**Formula:**
```
y = x^S / (K^S + x^S)
```

**Parameters:**
- `K`: Half-saturation point (EC50)
  - Spend level at which response = 50% of maximum
  - Higher K = more spend needed to saturate
- `S`: Hill coefficient (steepness)
  - S = 1: Standard hyperbolic (Michaelis-Menten)
  - S > 1: Steeper S-curve
  - S < 1: Flatter curve

**Properties:**
- Output range: [0, 1]
- At x = 0: y = 0
- At x = K: y = 0.5
- As x → ∞: y → 1

**Marginal Response (Derivative):**
```
dy/dx = S × K^S × x^(S-1) / (K^S + x^S)^2
```

This tells you the incremental response per additional dollar at any spend level.

**Code:**
```python
def hill_saturation(x, K, S):
    """Apply Hill function saturation."""
    x = np.maximum(x, 0)
    x_s = np.power(x, S)
    K_s = np.power(K, S)
    return x_s / (K_s + x_s)
```

#### Logistic Saturation

**Formula:**
```
y = L / (1 + exp(-k × (x - x0)))
```

**Parameters:**
- `L`: Maximum value (asymptote)
- `k`: Steepness
- `x0`: Midpoint (inflection point)

**Note:** Unlike Hill, this doesn't start at 0 when x=0.

---

### 3. Marketing Mix Model (MMM)

#### Model Specification

**Full Model:**
```
y[t] = intercept + media_effect[t] + control_effect[t] + trend[t] + seasonality[t] + ε[t]

where:
  media_effect[t] = Σ(β_i × saturate(adstock(x_i[t])))
  seasonality[t] = Σ(a_k × sin(2πk×t/365) + b_k × cos(2πk×t/365))
  ε[t] ~ Normal(0, σ²)
```

#### Bayesian Formulation

**Priors:**
```
intercept ~ Normal(μ_y, σ_y)           # Centered on mean outcome
β_i ~ HalfNormal(σ_β)                  # Positive channel effects
α_i ~ Beta(2, 2)                       # Adstock decay [0, 1]
K_i ~ LogNormal(μ_K, σ_K)              # Half-saturation points
S_i ~ Gamma(2, 0.5)                    # Hill coefficients
σ ~ HalfNormal(σ_σ)                    # Noise scale
```

**Likelihood:**
```
y ~ Normal(μ, σ)
where μ = model prediction
```

#### Contribution Decomposition

Channel contributions are computed by:
1. Run model prediction with all channels
2. For each channel, compute its isolated effect:
   ```
   contribution_i[t] = β_i × saturate(adstock(x_i[t]))
   ```
3. Baseline = total prediction - Σ(contributions)

---

### 4. Reconciliation Methods

#### Weighted Average Fusion

**Formula:**
```
lift_final = (w_mmm × lift_mmm + w_incr × lift_incr + w_attr × lift_attr) / (w_mmm + w_incr + w_attr)
```

Where weights are normalized and can be adjusted based on:
- Data recency
- Statistical significance
- Business trust

**Confidence Intervals:**
```
CI = weighted average of individual CIs
     (conservative approach)
```

#### Bayesian Fusion

**Prior from MMM:**
```
θ_prior ~ Normal(μ_mmm, σ_mmm)
```

**Likelihood from Tests:**
```
test_result | θ ~ Normal(θ, σ_test)
```

**Posterior (conjugate update):**
```
σ²_post = 1 / (1/σ²_mmm + 1/σ²_test)
μ_post = σ²_post × (μ_mmm/σ²_mmm + test_result/σ²_test)
```

This appropriately weights more precise estimates higher.

---

### 5. Budget Optimization

#### Problem Formulation

**Objective:**
```
maximize: Σ response_i(spend_i)
```

**Subject to:**
```
Σ spend_i = total_budget              (budget constraint)
min_i ≤ spend_i ≤ max_i              (channel bounds)
```

#### Response Functions

From MMM saturation parameters:
```
response_i(spend) = coef_i × hill(spend; K_i, S_i)
                  = coef_i × spend^S_i / (K_i^S_i + spend^S_i)
```

#### Solution Method

Uses **Sequential Least Squares Programming (SLSQP)**:
1. Start with equal allocation
2. Compute gradient of total response
3. Move in direction that improves objective while respecting constraints
4. Repeat until convergence

**Optimality Condition:**
At optimal allocation, marginal ROI is equal across channels:
```
∂response_i/∂spend_i = ∂response_j/∂spend_j  for all i, j
```

---

## Evaluation Metrics

### Model Fit Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAPE** | `mean(|y - ŷ| / y) × 100` | % error, scale-independent |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Same units as y |
| **R²** | `1 - SS_res/SS_tot` | Variance explained (0-1) |
| **MAE** | `mean(|y - ŷ|)` | Average absolute error |

**Target Values for MMM:**
- MAPE < 15%: Good
- R² > 0.8: Good
- Residuals should be uncorrelated (Durbin-Watson ≈ 2)

### Posterior Diagnostics (Bayesian)

| Metric | Target | Meaning |
|--------|--------|---------|
| **R-hat** | < 1.01 | Chain convergence |
| **ESS** | > 400 | Effective sample size |
| **Divergences** | 0 | No sampling issues |

### Business Metrics

| Metric | Formula | Use |
|--------|---------|-----|
| **ROI** | `contribution / spend` | Channel efficiency |
| **Marginal ROI** | `d(response)/d(spend)` | Optimization signal |
| **Contribution Share** | `contribution_i / Σcontributions` | Budget allocation |

---

## Code Examples

### Complete Training Pipeline

```python
import pandas as pd
from transforms import apply_adstock, apply_saturation, create_mmm_features
from mmm import UnifiedMMM
from mmm.evaluation import evaluate_model, cross_validate

# 1. Load and prepare data
media_spend = pd.read_parquet("data/validated/media_spend.parquet")
outcomes = pd.read_parquet("data/validated/outcomes.parquet")
controls = pd.read_parquet("data/validated/controls.parquet")

# 2. Create features
mmm_data = create_mmm_features(
    media_spend=media_spend,
    outcomes=outcomes,
    controls=controls,
    target_col="revenue",
)

# 3. Train model
mmm = UnifiedMMM(
    date_col="date",
    target_col="y",
    adstock_max_lag=8,
    yearly_seasonality=2,
)

results = mmm.fit(
    mmm_data,
    n_samples=2000,
    n_chains=4,
    target_accept=0.9,
)

# 4. Evaluate
metrics = evaluate_model(
    y_true=mmm_data["y"].values,
    y_pred=mmm.predict(mmm_data),
)
print(f"MAPE: {metrics['mape']:.2f}%")
print(f"R²: {metrics['r2']:.3f}")

# 5. Cross-validate
cv_results = cross_validate(
    model_fn=lambda df: UnifiedMMM().fit(df),
    df=mmm_data,
    n_splits=5,
)
print(f"CV MAPE: {np.mean(cv_results['mape']):.2f}% ± {np.std(cv_results['mape']):.2f}%")
```

### Reconciliation Example

```python
from reconciliation import ReconciliationEngine

# Initialize engine with weights
engine = ReconciliationEngine(
    mmm_weight=0.5,           # Trust MMM for trend
    incrementality_weight=0.3, # Trust tests for calibration
    attribution_weight=0.2,    # Use for tactical insights
    fusion_method="bayesian",  # or "weighted_average"
)

# Load data sources
mmm_results = results.to_dict()
incrementality_tests = pd.read_parquet("data/validated/incrementality.parquet")
attribution_data = pd.read_parquet("data/validated/attribution.parquet")

# Run reconciliation
reconciled = engine.reconcile(
    mmm_results=mmm_results,
    incrementality_tests=incrementality_tests,
    attribution_data=attribution_data,
)

# View results
for channel, estimate in reconciled.channel_estimates.items():
    print(f"{channel}:")
    print(f"  Lift: {estimate.lift_estimate:.2f}")
    print(f"  95% CI: [{estimate.lift_ci_lower:.2f}, {estimate.lift_ci_upper:.2f}]")
    print(f"  Confidence: {estimate.confidence_score:.0%}")
```

### Optimization Example

```python
from optimization import BudgetOptimizer

# Build optimizer from MMM results
optimizer = BudgetOptimizer(
    response_params={
        "google": {"K": 10000, "S": 1.2, "coefficient": 2.5},
        "meta": {"K": 8000, "S": 1.0, "coefficient": 1.8},
        "tv": {"K": 50000, "S": 0.8, "coefficient": 0.5},
    },
    total_budget=100000,
    min_budget_pct=0.05,   # Min 5% per channel
    max_budget_pct=0.50,   # Max 50% per channel
)

# Set current allocation for comparison
optimizer.set_current_allocation({
    "google": 30000,
    "meta": 40000,
    "tv": 30000,
})

# Optimize
result = optimizer.optimize()

print(f"Expected improvement: {result.improvement_pct:.1f}%")
print(f"\nOptimal allocation:")
for channel, spend in result.optimal_allocation.items():
    current = result.current_allocation.get(channel, 0)
    change = (spend - current) / current * 100 if current > 0 else 0
    print(f"  {channel}: ${spend:,.0f} ({change:+.1f}%)")
```

---

## Best Practices

### Data Quality

1. **Completeness**: No missing dates in time series
2. **Consistency**: Same channel naming across sources
3. **Recency**: Use most recent data for training
4. **Granularity**: Daily data preferred for MMM

### Model Training

1. **Sufficient history**: 2+ years for seasonality
2. **Variation**: Ensure spend varies (flight patterns help)
3. **Priors**: Use domain knowledge for informative priors
4. **Diagnostics**: Check convergence before trusting results

### Reconciliation

1. **Test recency**: Weight recent tests higher
2. **Statistical significance**: Flag non-significant results
3. **Channel coverage**: Note channels without test data
4. **Documentation**: Record fusion rationale

### Optimization

1. **Constraints**: Set realistic min/max bounds
2. **Scenarios**: Run multiple budget scenarios
3. **Sensitivity**: Test robustness to parameter uncertainty
4. **Business rules**: Incorporate non-quantifiable factors
