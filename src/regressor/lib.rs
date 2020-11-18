use crate::math::{Matrix, Vector, dot, shuffle};
use crate::regressor::sgd::SGD;

macro_rules! builder_field {
    ($field:ident, $field_type:ty) => {
        pub fn $field(mut self, $field: $field_type) -> Self {
            self.$field = $field;
            self
        }
    };
}

#[derive(Copy, Clone, Debug)]
pub enum Penalty {
    L1,
    L2,
    None,
}

/// Base linear regressor interface.
/// Inspired by `Pipeline` class from `scikit-learn` library.
///
/// Only `fit` and `weights` methods are needed to be implemented.
/// Other methods have standard implementation and are auto-inherited.
pub trait Regressor {
    /// Assess the best weights of linear regression on the provided dataset.
    fn fit(self, X: Matrix, y: Vector) -> Self;

    /// Return weights of a fitted regressor.
    /// This method should be implemented to automatically inherit `predict` method.
    fn weights(&self) -> &Vector;

    /// Predict using the linear model.
    ///
    /// # Arguments
    ///
    /// * `X`: Matrix with the same number of features as of fitted matrix.
    ///
    /// # Return
    /// Vector of predicted values of target variable.
    fn predict(&self, X: &Matrix) -> Vector {
        let mut predictions = Vector::new();
        for i in 0..X.rows() {
            let prediction = self.weights()[0] + dot(&self.weights()[1..], &X[i]);
            predictions.push(prediction);
        }

        predictions
    }

    /// Assess the efficiency of a model with R-squared score.
    fn score(&self, X: &Matrix, y: &Vector) -> f64 {
        let predictions = &self.predict(X);
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let ssres = predictions.iter().zip(y.iter())
            .map(|(yh, y)| f64::powi(y - yh, 2))
            .sum::<f64>();
        let sstot = y.iter()
            .map(|y| f64::powi(y - mean, 2))
            .sum::<f64>();

        1f64 - (ssres / sstot)
    }

    /// Calculate mean squared deviation of an estimator.
    fn mse(&self, X: &Matrix, y: &Vector) -> f64 {
        let predictions = &self.predict(X);
        let error = predictions.iter().zip(y.iter())
            .map(|(yh, y)| f64::powi(y - yh, 2))
            .sum::<f64>();

        error / predictions.len() as f64
    }
}

pub fn assess_alpha(X: &Matrix, y: &Vector, k: usize, grid: &Vec<f64>) -> f64 {
    let n = X.rows();
    let mut alpha = 0f64;
    let mut best_error = f64::MAX;

    let (XT, yT) = (X.slice(0, n / k - 1), y[0..n / k - 1].to_vec());
    let (X0, y0) = (X.slice(n / k, n - 1), y[n / k..n - 1].to_vec());

    let mut i: usize = 0;
    for &p in grid.iter() {
        println!("Iteration: {} out of {}", i, grid.len());

        let model = SGD::default()
            .iterations(20000)
            .alpha(p)
            .fit(XT.clone(), yT.clone());
        let error = model.mse(&X0, &y0);
        if error < best_error {
            best_error = error;
            alpha = p;
        }

        i += 1;
    }

    alpha
}