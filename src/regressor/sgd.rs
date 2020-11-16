use crate::{
    regressor::lib::*,
    math::*,
};

use std::{
    default::Default
};

/// Main parameters are the same as parameters of the `SGDRegressor` in `scikit-learn` library.
///
/// Note: Regularisation is not implemented for this regressor.
#[derive(Clone, Debug)]
pub struct SGD {
    /// Size of mini-batches used in the gradient computation.
    batch_size: usize,
    /// The maximum number of passes over the training data.
    iterations: usize,
    /// The stopping criterion.
    /// Training will stop when `error > best_error - tolerance`.
    tolerance: f64,
    /// Whether or not the training data should be shuffled after each epoch.
    shuffle: bool,
    /// Should an important information be printed.
    verbose: bool,
    /// Number of iterations with no improvement to wait before early stopping.
    stumble: usize,
    /// Initial learning rate.
    eta: f64,
    /// Whether to use early stopping to terminate training when validation score is not improving.
    stopping: bool,
    weights: Vector,
    intercept: f64,
}

impl SGD {
    builder_field!(batch_size, usize);
    builder_field!(iterations, usize);
    builder_field!(tolerance, f64);
    builder_field!(shuffle, bool);
    builder_field!(verbose, bool);
    builder_field!(stumble, usize);
    builder_field!(eta, f64);
    builder_field!(stopping, bool);
}

impl Default for SGD {
    fn default() -> Self {
        Self {
            batch_size: 8,
            iterations: 1000,
            tolerance: 1e-3,
            shuffle: true,
            verbose: false,
            stumble: 5,
            eta: 1e-2,
            stopping: false,
            weights: Vec::new(),
            intercept: 0f64,
        }
    }
}

impl Regressor for SGD {
    /// Fit linear model with Stochastic Gradient Descent.
    ///
    /// # Arguments
    ///
    /// * `X`: Train matrix filled with `N` observations and `P` features. Former forms rows, latter columns.
    /// * `y`: Target vector of matrix `X`. One column with precisely `N` rows.
    ///
    /// # Examples
    /// ```rust
    /// let X = Matrix::read("./train_X.csv");
    /// let y = Vector::read("./train_y.csv");
    /// let gcd = GCD::default()
    ///     .batch_size(12)
    ///     .verbose(true)
    ///     .fit(X, y);
    ///
    /// let y = Vector::read("./test_y.csv");
    /// let prediction = gcd.predict(y);
    /// ```
    fn fit(mut self, mut X: Matrix, mut y: Vector) -> Self {
        // Squared loss is convex function, so start with zeroes.
        self.weights = vec![0f64; X.cols()] as Vector;
        self.intercept = 0f64;

        // If batches are too large, shrink them.
        if self.batch_size >= X.rows() {
            self.batch_size = if self.batch_size > 4 { X.rows() / 4 } else { 1 };
            if self.verbose { println!("Changed batch size to {}", self.batch_size); }
        }

        let mut best_error = f64::MAX;
        let mut stumble = 0usize;
        let mut best_weights = Vector::new();

        for e in 1..self.iterations {
            if self.verbose {
                if e % 100 == 0 { println!("Processed epoch #{}", e); }
            }
            // It is essential to reshuffle data.
            // Randomly permute all rows.
            if self.shuffle {
                shuffle(&mut X, &mut y);
            }
            // Inverse scaling for learning rate.
            // Default method in `sklearn`.
            let scaling = 1f64 / f64::powf(e as f64, 0.25);
            let mut new_weights = self.weights.clone();
            // Linear regression function is `w0 + w1 x1 + ... + wp xp = y`.
            // Precompute part of gradient that does not depend on `x`.
            let mut G = Vector::with_capacity(self.batch_size);
            for i in 0..self.batch_size {
                G.push(self.intercept + dot(&self.weights, &X[i]) - y[i]);
            };
            // For each weight `wi` find gradient using batch of observations.
            for j in 0..self.weights.len() {
                let mut dw = 0f64;
                for i in 0..self.batch_size {
                    dw += X[[i, j]] * G[i];
                }
                // Adjust weights.
                new_weights[j] -= scaling * self.eta / self.batch_size as f64 * dw;
            }
            // Separately process intercept, or `w0`.
            let mut di = 0f64;
            for i in 0..self.batch_size {
                di += self.intercept + dot(&self.weights, &X[i]) - y[i];
            }
            self.intercept -= scaling * self.eta / self.batch_size as f64 * di;
            self.weights = new_weights;

            if self.stopping {
                // If result is not improving, stop procedure early.
                // IMHO, MSE is more complex procedure than SGD.
                // TODO: Validation fraction.
                let error = mse(&self, &X, &y);
                if error > best_error - self.tolerance { stumble += 1; } else { stumble = 0; }
                if error < best_error { best_weights = self.weights.clone(); }
                best_error = if error < best_error { error } else { best_error };

                if stumble >= self.stumble {
                    // Return to weights with minimal MSE.
                    self.weights = best_weights;
                    if self.verbose { println!("Had to stop, no improvement for {} steps", self.stumble); }
                    return self;
                }
            }
        }

        self
    }

    /// Returns parameters of the trained model.
    ///
    /// If regressor was not yet trained via `fit` method, returns garbage.
    fn weights(&self) -> (f64, &Vector) {
        (self.intercept, &self.weights)
    }
}
