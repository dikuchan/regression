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

    /// Assess model efficiency with coefficient of determination.
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
}
