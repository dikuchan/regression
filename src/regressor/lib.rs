use crate::math::{
    Matrix, Vector,
    dot,
};

macro_rules! builder_field {
    ($field:ident, $field_type:ty) => {
        pub fn $field(mut self, $field: $field_type) -> Self {
            self.$field = $field;
            self
        }
    };
}

/// Base linear regressor interface.
/// Inspired by `Pipeline` class from `scikit-learn` library.
pub trait Regressor {
    fn fit(self, X: Matrix, y: Vector) -> Self;

    fn weights(&self) -> (f64, &Vector);

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
            let (intercept, weights) = &self.weights();
            let prediction = intercept + dot(weights, &X[i]);
            predictions.push(prediction);
        }

        predictions
    }

    /// Assess model efficiency with coefficient of determination.
    fn score(&self, X: &Matrix, y: &Vector) -> f64 {
        let predictions = &self.predict(X);
        let mean = y.iter().sum::<f64>() / y.len() as f64;
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
