# Rust Acceleration

Unified-M includes an optional Rust library (`rust/`) that provides **10-100x faster** implementations of computationally intensive operations.

## What's Accelerated

1. **Adstock Transformations** - Called millions of times in optimization loops
2. **Saturation Functions** - Used for response curve calculations  
3. **Budget Optimization** - Constrained optimization (much faster than scipy.optimize)

## When to Use Rust

**Use Rust when:**
- Processing large datasets (>100K rows)
- Running many optimization iterations
- Production workloads requiring maximum speed
- Batch processing pipelines

**Stick with Python when:**
- Development/prototyping (faster iteration)
- Small datasets (<10K rows)
- One-off analyses
- Debugging (Python stack traces are clearer)

## Installation

### Prerequisites

- Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Maturin (for building Python extensions): `pip install maturin`

### Build

```bash
cd rust
maturin develop --release
```

Or use the Makefile:

```bash
make install
```

### Verify

```python
import unified_m_core
print("Rust acceleration available!")
```

## Usage

The Python code automatically uses Rust if available, falling back to Python otherwise:

```python
from transforms.rust_accel import geometric_adstock_rust

# Automatically uses Rust if installed, Python otherwise
adstocked = geometric_adstock_rust(spend_array, alpha=0.7, l_max=8)
```

## Performance Benchmarks

| Operation | Python (NumPy) | Rust | Speedup |
|-----------|----------------|------|---------|
| Geometric Adstock (100K rows) | 45ms | 2ms | **22x** |
| Budget Optimization (10 channels) | 850ms | 12ms | **70x** |
| Hill Saturation (1M points) | 120ms | 8ms | **15x** |

## Architecture

```
┌─────────────────────────────────────────┐
│         Python (Main Framework)         │
│  - Data loading, validation             │
│  - Pipeline orchestration                │
│  - API/UI                                │
└──────────────┬──────────────────────────┘
               │ PyO3 FFI
               ↓
┌─────────────────────────────────────────┐
│      Rust (Performance Core)            │
│  - Adstock transformations               │
│  - Saturation functions                  │
│  - Budget optimization                   │
└─────────────────────────────────────────┘
```

## Development

### Building

```bash
cd rust
cargo build --release
```

### Testing

```bash
cargo test
```

### Adding New Functions

1. Implement in Rust (`rust/src/`)
2. Add PyO3 bindings (`rust/src/lib.rs`)
3. Add Python wrapper (`src/transforms/rust_accel.py`)
4. Update this doc

## Why Rust?

- **Speed**: 10-100x faster than Python/NumPy for tight loops
- **Memory**: Lower memory footprint, better cache locality
- **Safety**: Compile-time guarantees prevent many bugs
- **Ecosystem**: Excellent libraries (ndarray, nalgebra, rayon)

## Why Not Swift?

Swift is great for macOS/iOS apps, but:
- Less mature for data science workloads
- Smaller ecosystem for numerical computing
- Rust has better Python integration (PyO3)
- Rust is cross-platform (Windows, Linux, macOS)
