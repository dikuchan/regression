use crate::math::{
    Matrix, Vector,
};

/// Base linear regressor interface.
/// Inspired by `Pipeline` class from `scikit-learn` library.
pub trait Regressor {
    fn fit(self, X: Matrix, y: Vector) -> Self;

    fn predict(&self, X: &Matrix) -> Vector;

    /// Assess model efficiency with coefficient of determination.
    fn score(&self, X: &Matrix, y: &Vector) -> f64 {
        let predictions = &self.predict(X);
        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let ssres = predictions.iter().zip(y.iter())
            .map(|(yh, y)| (y - yh) * (y - yh))
            .sum::<f64>();
        let sstot = y.iter()
            .map(|y| (y - mean) * (y - mean))
            .sum::<f64>();

        1f64 - (ssres / sstot)
    }
}

/// Calculate Mean Squared Error using trained linear regressor.
pub fn mse<T: Regressor>(R: &T, X: &Matrix, y: &Vector) -> f64 {
    let predictions = &R.predict(X);
    let error = predictions.iter().zip(y.iter())
        .map(|(y, yh)| (y - yh) * (y - yh))
        .sum::<f64>();

    error / predictions.len() as f64
}
