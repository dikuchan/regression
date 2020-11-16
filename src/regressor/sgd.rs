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
    batch: usize,
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
    // The proportion of training data to set aside as validation set for early stopping.
    fraction: f64,
    weights: Vector,
}

impl SGD {
    builder_field!(batch, usize);
    builder_field!(iterations, usize);
    builder_field!(tolerance, f64);
    builder_field!(shuffle, bool);
    builder_field!(verbose, bool);
    builder_field!(stumble, usize);
    builder_field!(eta, f64);
    builder_field!(stopping, bool);
    builder_field!(fraction, f64);
}

impl Default for SGD {
    fn default() -> Self {
        Self {
            batch: 8,
            iterations: 1000,
            tolerance: 1e-3,
            shuffle: true,
            verbose: false,
            stumble: 5,
            eta: 1e-2,
            stopping: false,
            fraction: 0.1,
            weights: Vec::new(),
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
    /// let prediction = gcd.predict(&y);
    /// ```
    fn fit(mut self, mut X: Matrix, mut y: Vector) -> Self {
        // Squared loss is convex function, so start with zeroes.
        self.weights = vec![0f64; X.cols() + 1] as Vector;

        // If batches are too large, shrink them.
        if self.batch >= X.rows() {
            self.batch = if X.rows() > 4 { X.rows() / 4 } else { 1 };
            if self.verbose { println!("Changed batch size to {}", self.batch); }
        }

        let fraction = self.fraction * (X.rows() as f64);
        let mut best_error = f64::MAX;
        let mut stumble = 0usize;
        let mut best_weights = Vector::new();

        for e in 1..self.iterations {
            if self.verbose {
                if e % 100 == 0 { println!("Processed epoch #{}", e); }
            }
            // It is essential to reshuffle data.
            // Randomly permute all rows.
            if self.shuffle { shuffle(&mut X, &mut y); }
            // Linear regression function is `w0 + w1 x1 + ... + wp xp = y`.
            // Precompute part of gradient that does not depend on `x`.
            let delta: Vector = (0..self.batch)
                .map(|i| self.weights[0] + dot(&self.weights[1..], &X[i]) - y[i])
                .collect();
            // For each weight `wi` find derivative using batch of observations.
            for j in 0..self.weights.len() {
                let mut derivative = 0f64;
                for i in 0..self.batch {
                    derivative += if j == 0 { delta[i] } else { X[[i, j - 1]] * delta[i] }
                }
                // Adjust weights.
                derivative /= self.batch as f64;
                // Inverse scaling of learning rate.
                // Default method in `sklearn`.
                let eta = self.eta / f64::powf(e as f64, 0.25);
                self.weights[j] -= eta * derivative;
            }

            // If result is not improving, stop procedure early.
            if self.stopping {
                // Compute MSE on part of dataset.
                let mut error = 0f64;
                for i in 0..fraction as usize {
                    let prediction = self.weights[0] + dot(&self.weights[1..], &X[i]);
                    error += f64::powi(prediction - y[i], 2);
                }
                error /= fraction;
                if error > best_error - self.tolerance { stumble += 1; } else { stumble = 0; }
                // Remember best weights.
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
    fn weights(&self) -> &Vector {
        &self.weights
    }
}
