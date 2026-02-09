# Unified-M Core (Rust)

High-performance Rust library for Unified-M's computationally intensive operations.

## What's Here

This Rust library provides **10-100x faster** implementations of:

1. **Adstock Transformations** (`src/adstock.rs`)
   - Geometric adstock (exponential decay)
   - Weibull adstock (flexible decay shape)
   - Called millions of times in optimization loops

2. **Saturation Functions** (`src/saturation.rs`)
   - Hill saturation
   - Logistic saturation
   - Used for response curve calculations

3. **Budget Optimization** (`src/optimization.rs`)
   - Constrained optimization (much faster than scipy.optimize)
   - Gradient-based method with projection

## Building

```bash
cd rust
cargo build --release
```

## Python Integration

The Rust code is called from Python via PyO3 bindings. To use:

```python
import unified_m_core

# Fast adstock transformation
adstocked = unified_m_core.geometric_adstock_rust(
    spend_array, alpha=0.7, l_max=8, normalize=True
)

# Fast budget optimization
optimal = unified_m_core.optimize_budget_rust(
    response_params=[("google", (1000.0, 1.0, 1.0)), ...],
    total_budget=50000.0,
    min_budget_pct=0.0,
    max_budget_pct=1.0,
    channel_constraints=[],
)
```

## Performance

- **Adstock**: 10-50x faster than NumPy convolution
- **Optimization**: 10-100x faster than scipy.optimize.minimize
- **Memory**: Lower memory footprint, better cache locality

## When to Use

- Large datasets (>100K rows)
- Many optimization iterations
- Production workloads requiring speed
- Development/prototyping (Python is faster to iterate) - use Python instead
