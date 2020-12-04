use crate::{
    math::*,
    regressor::{
        config::Config,
        regressor::Regressor,
    },
};

/// Conveniently define regularization term that is added to the MSE formula.
///
/// # None
/// MSE is unchanged: `1/|y| Σ (y - y·)^2`.
///
/// # L1
/// Term added: `-Σ |w|`.
///
/// # L2
/// Term added: `-Σ w^2`.
#[derive(Copy, Clone, Debug)]
pub enum Penalty {
    L1,
    L2,
    None,
}

impl Penalty {
    pub fn compute(&self, alpha: f64, weight: f64) -> f64 {
        match self {
            Penalty::L1 => alpha * if weight > 0f64 { 1f64 } else { -1f64 },
            Penalty::L2 => 2f64 * alpha * weight,
            Penalty::None => 0f64,
        }
    }
}

/// Assess the best value of alpha coefficient in L2 regularization.
///
/// # Arguments
///
/// * `X`: Observation matrix.
/// * `y`: Target vector.
/// * `k`: The parameter in k-fold algorithm.
/// * `grid`: Vector of possible values of alpha to use in the assessment.
///
/// # Examples
/// ```rust
/// let X = Matrix::read("./data/X.csv")?;
/// let y = Vector::read("./data/y.csv")?;
/// let grid = (0..250)
///     .map(|p| 1e-4 / p as f64)
///     .collect::<Vector>();
///
/// println!("Optimal alpha: {}", assess_alpha(&X, &y, 5, &grid));
/// ```
pub fn assess_alpha(X: &Matrix, y: &Vector, k: usize, grid: &Vec<f64>) -> f64 {
    let n = X.rows();
    let mut alpha = 0f64;
    let mut best_error = f64::MAX;

    let (XT, X0) = X.slice(n / k);
    let (yT, y0) = (y[0..n / k - 1].to_vec(), y[n / k..n - 1].to_vec());

    for (i, &p) in grid.iter().enumerate() {
        let model = Config::default()
            .iterations(1000)
            .penalty(Penalty::L2)
            .alpha(p)
            .to_SGD()
            .fit(XT.clone(), yT.clone());
        let error = model.mse(&X0, &y0);
        if error < best_error {
            best_error = error;
            alpha = p;
        }

        println!("-- Iteration {} out of {}", i, grid.len());
    }

    alpha
}
