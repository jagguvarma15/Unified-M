//! Unified-M Core: High-performance Rust library for marketing measurement.
//!
//! This module provides fast implementations of:
//! - Adstock transformations (geometric, Weibull, delayed)
//! - Saturation functions (Hill, logistic, Michaelis-Menten)
//! - Budget optimization (constrained optimization)
//!
//! Called from Python via PyO3 bindings.

use ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

mod adstock;
mod cache;
mod saturation;
mod optimization;

use adstock::*;
use cache::PyLruCache;
use saturation::*;
use optimization::*;

/// Python module definition
#[pymodule]
fn unified_m_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyLruCache>()?;
    m.add_function(wrap_pyfunction!(geometric_adstock_rust, m)?)?;
    m.add_function(wrap_pyfunction!(weibull_adstock_rust, m)?)?;
    m.add_function(wrap_pyfunction!(hill_saturation_rust, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_budget_rust, m)?)?;
    Ok(())
}

/// Geometric adstock transformation (10-50x faster than NumPy).
#[pyfunction]
#[pyo3(signature = (x, alpha, l_max = 8, normalize = true))]
fn geometric_adstock_rust(
    py: Python,
    x: PyReadonlyArray1<f64>,
    alpha: f64,
    l_max: usize,
    normalize: bool,
) -> Py<PyArray1<f64>> {
    let x_array = x.as_array();
    let result = geometric_adstock(&x_array, alpha, l_max, normalize);
    result.into_pyarray(py).to_owned()
}

/// Weibull adstock transformation.
#[pyfunction]
#[pyo3(signature = (x, shape, scale, l_max = 8))]
fn weibull_adstock_rust(
    py: Python,
    x: PyReadonlyArray1<f64>,
    shape: f64,
    scale: f64,
    l_max: usize,
) -> Py<PyArray1<f64>> {
    let x_array = x.as_array();
    let result = weibull_adstock(&x_array, shape, scale, l_max);
    result.into_pyarray(py).to_owned()
}

/// Hill saturation function.
#[pyfunction]
#[pyo3(signature = (x, k, s, coef = 1.0))]
fn hill_saturation_rust(
    py: Python,
    x: PyReadonlyArray1<f64>,
    k: f64,
    s: f64,
    coef: f64,
) -> Py<PyArray1<f64>> {
    let x_array = x.as_array();
    let result = hill_saturation(&x_array, k, s, coef);
    result.into_pyarray(py).to_owned()
}

/// Budget optimization (much faster than scipy.optimize).
#[pyfunction]
fn optimize_budget_rust(
    py: Python,
    response_params: Vec<(String, (f64, f64, f64))>, // (channel, (K, S, coef))
    total_budget: f64,
    min_budget_pct: f64,
    max_budget_pct: f64,
    channel_constraints: Vec<(String, (f64, f64))>, // (channel, (min, max))
) -> PyResult<Py<PyDict>> {
    let result = optimize_budget(
        &response_params,
        total_budget,
        min_budget_pct,
        max_budget_pct,
        &channel_constraints,
    )?;
    
    let dict = PyDict::new(py);
    for (channel, spend) in result {
        dict.set_item(channel, spend)?;
    }
    Ok(dict.into())
}
