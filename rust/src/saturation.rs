//! Saturation functions - diminishing returns in media response.

use ndarray::{Array1, ArrayView1};

/// Hill saturation function.
/// 
/// Formula: coef * x^S / (K^S + x^S)
/// 
/// This models diminishing returns where:
/// - K is the half-saturation point
/// - S controls the steepness (S=1 is Michaelis-Menten)
pub fn hill_saturation(
    x: &ArrayView1<f64>,
    k: f64,
    s: f64,
    coef: f64,
) -> Array1<f64> {
    x.map(|&val| {
        let x_val = val.max(0.0);
        let k_pow_s = k.powf(s);
        let x_pow_s = x_val.powf(s);
        coef * x_pow_s / (k_pow_s + x_pow_s)
    })
}

/// Logistic saturation.
pub fn logistic_saturation(
    x: &ArrayView1<f64>,
    lam: f64,
    coef: f64,
) -> Array1<f64> {
    x.map(|&val| {
        let x_val = val.max(0.0);
        coef * (1.0 - (-lam * x_val).exp())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_hill_saturation() {
        let x = array![0.0, 100.0, 1000.0, 10000.0];
        let result = hill_saturation(&x.view(), 1000.0, 1.0, 1.0);
        
        // Should increase but with diminishing returns
        assert!(result[1] < result[2]);
        assert!(result[2] < result[3]);
        // But growth should slow
        let growth_1_2 = result[2] - result[1];
        let growth_2_3 = result[3] - result[2];
        assert!(growth_2_3 < growth_1_2);
    }
}
