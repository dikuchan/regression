use crate::math::{dot, Matrix, Vector};

/// Base linear regressor interface.
/// Inspired by `Pipeline` class from `scikit-learn` library.
///
/// Only `fit` and `weights` methods are needed to be implemented.
/// Other methods have standard implementation and are auto-inherited.
pub trait Regressor {
    /// Assess the best weights of linear regression on the provided dataset.
    fn fit(self, X: Matrix, y: Vector) -> Self;

    /// Return the weights of a fitted regressor.
    /// This method should be implemented in order to automatically inherit `predict` method.
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
        let ssres = predictions.iter().zip(y.iter())
            .map(|(yh, y)| (y - yh).powi(2))
            .sum::<f64>();
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        let sstot = y.iter()
            .map(|y| (y - mean).powi(2))
            .sum::<f64>();

        1f64 - (ssres / sstot)
    }

    /// Calculate mean squared deviation of an estimator.
    fn mse(&self, X: &Matrix, y: &Vector) -> f64 {
        let predictions = &self.predict(X);
        let error = predictions.iter().zip(y.iter())
            .map(|(yh, y)| (y - yh).powi(2))
            .sum::<f64>();

        error / predictions.len() as f64
    }
}
