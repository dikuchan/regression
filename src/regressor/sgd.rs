use crate::{
    regressor::lib::*,
    math::*,
};

use std::{
    default::Default
};

/// Main parameters are the same as parameters of the `SGDRegressor` in `scikit-learn` library.
#[derive(Clone, Debug)]
pub struct SGD {
    /// Size of a mini-batch used in the gradient computation.
    batch: usize,
    /// The maximum number of passes over the training data.
    iterations: usize,
    /// Constant that multiplies the regularization term.
    alpha: f64,
    /// The penalty (aka regularization term) to be used.
    penalty: Penalty,
    /// The stopping criterion.
    /// Training will stop when `error > best_error - tolerance`.
    tolerance: f64,
    /// Whether or not the training data should be shuffled after each epoch.
    shuffle: bool,
    /// Should an important information be printed.
    verbose: bool,
    /// Number of iterations with no improvement to wait before early stopping.
    stumble: usize,
    /// The initial learning rate.
    eta: f64,
    /// Whether to use early stopping to terminate training when validation score is not improving.
    stopping: bool,
    /// The proportion of training data to set aside as validation set for early stopping.
    fraction: f64,
    weights: Vector,
}

impl SGD {
    builder_field!(batch, usize);
    builder_field!(iterations, usize);
    builder_field!(alpha, f64);
    builder_field!(penalty, Penalty);
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
            alpha: 1e-4,
            penalty: Penalty::L2,
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
    /// Fit a linear model with Stochastic Gradient Descent.
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
                if e % 250 == 0 {
                    println!("Processed epoch #{}", e);
                    println!("Weights: {:?}", self.weights);
                }
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
                    derivative += (if j == 0 { 1f64 } else { X[[i, j - 1]] }) * delta[i];
                }
                // Adjust weights.
                derivative /= self.batch as f64;
                derivative += match self.penalty {
                    Penalty::L1 => self.alpha * if self.weights[j] > 0f64 { 1f64 } else { -1f64 },
                    Penalty::L2 => 2f64 * self.alpha * self.weights[j],
                    Penalty::None => 0f64,
                };
                // Inverse scaling of learning rate.
                // Default method in `sklearn`.
                let eta = self.eta / f64::powf(e as f64, 0.25);
                self.weights[j] -= eta * derivative;
            }

            // If result is not improving, stop procedure early.
            if self.stopping {
                // Compute MSE on a part of the dataset.
                let mut error = 0f64;
                for i in 0..fraction as usize {
                    let prediction = self.weights[0] + dot(&self.weights[1..], &X[i]);
                    error += f64::powi(prediction - y[i], 2);
                }
                error /= fraction;
                if error > best_error - self.tolerance { stumble += 1; } else { stumble = 0; }
                // Remember the best weights.
                if error < best_error { best_weights = self.weights.clone(); }
                best_error = if error < best_error { error } else { best_error };

                if stumble >= self.stumble {
                    // Return to the weights with minimal MSE.
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
