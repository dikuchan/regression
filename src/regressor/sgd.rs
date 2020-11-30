use crate::{
    regressor::lib::*,
    math::*,
};

use std::default::Default;

/// Main parameters are the same as parameters of the `SGDRegressor` in `scikit-learn` library.
#[derive(Clone, Debug)]
pub struct SGD {
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
    intercept: f64,
    weights: Vector,
}

impl SGD {
    builder_field!(iterations, usize);
    builder_field!(alpha, f64);
    builder_field!(penalty, Penalty);
    builder_field!(tolerance, f64);
    builder_field!(shuffle, bool);
    builder_field!(verbose, bool);
    builder_field!(stumble, usize);
    builder_field!(eta, f64);
}

impl Default for SGD {
    fn default() -> Self {
        Self {
            iterations: 1000,
            alpha: 1e-4,
            penalty: Penalty::L2,
            tolerance: 1e-3,
            shuffle: true,
            verbose: false,
            stumble: 6,
            eta: 1e-2,
            weights: Vec::new(),
            intercept: 0f64,
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
        self.weights = vec![0f64; X.cols()];

        // Scaling factor for eta.
        let mut t = 0usize;
        // Number of times loss didn't improve.
        let mut stumble = 0usize;
        let mut best_loss = f64::MAX;

        for e in 0..self.iterations {
            let mut loss = 0f64;
            // It is essential to reshuffle data.
            // Randomly permute all rows.
            if self.shuffle { shuffle(&mut X, &mut y); }

            for i in 0..X.rows() {
                t += 1;
                // Scale learning rate.
                // Default method in `sklearn`.
                let eta = self.eta / (t as f64).powf(0.25);
                // Precompute the part of derivative that doesn't depend on `X`.
                let delta = self.intercept + dot(&self.weights, &X[i]) - y[i];
                // Separately compute change of intercept.
                {
                    let penalty = self.penalty.compute(self.alpha, self.intercept);
                    let derivative = delta + penalty;
                    self.intercept -= eta * derivative;
                }
                for j in 0..X.cols() {
                    let penalty = self.penalty.compute(self.alpha, self.weights[j]);
                    let derivative = delta * X[[i, j]] + penalty;
                    self.weights[j] -= eta * derivative;
                }
                loss += delta.powi(2) / 2f64;
            }

            // Compute average loss.
            loss = loss / X.rows() as f64;
            if loss > best_loss - self.tolerance { stumble += 1; } else { stumble = 0; }
            if loss < best_loss { best_loss = loss; }

            if self.verbose {
                println!("-- Epoch {}, Norm: {}, Bias: {}, T: {}, Average loss: {:.06}",
                         e, norm(&self.weights), self.intercept, t, loss);
            }

            if stumble > self.stumble {
                if self.verbose { println!("Convergence after {} epochs", e); }
                return self;
            }
        }

        self
    }

    /// Return weights of the trained model.
    ///
    /// If regressor was not yet trained via `fit` method, return garbage.
    fn weights(&self) -> &Vector {
        &self.weights
    }

    /// Return bias of the trained model.
    ///
    /// If regressor was not yet trained via `fit` method, return garbage.
    fn intercept(&self) -> f64 {
        self.intercept
    }
}
