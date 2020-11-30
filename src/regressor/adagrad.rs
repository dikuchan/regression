use crate::{
    regressor::lib::*,
    math::*,
};

use std::default::Default;

/// For the complete documentation on regressors, see `SGD`.
#[derive(Clone, Debug)]
pub struct AdaGrad {
    iterations: usize,
    verbose: bool,
    stumble: usize,
    tolerance: f64,
    shuffle: bool,
    eta: f64,
    // A small value to avoid division by zero.
    epsilon: f64,
    intercept: f64,
    weights: Vector,
}

impl AdaGrad {
    builder_field!(iterations, usize);
    builder_field!(verbose, bool);
    builder_field!(stumble, usize);
    builder_field!(tolerance, f64);
    builder_field!(shuffle, bool);
    builder_field!(eta, f64);
    builder_field!(epsilon, f64);
}

impl Default for AdaGrad {
    fn default() -> Self {
        Self {
            iterations: 1000,
            verbose: false,
            stumble: 6,
            tolerance: 1e-3,
            shuffle: true,
            eta: 1e-2,
            epsilon: 1e-8,
            intercept: 0f64,
            weights: Vec::new(),
        }
    }
}

impl Regressor for AdaGrad {
    /// Fit a linear model with Adaptive Gradient Descent.
    ///
    /// # Arguments
    ///
    /// * `X`: Train matrix filled with `N` observations and `P` features.
    /// * `y`: Target vector of matrix `X`. One column with precisely `N` rows.
    fn fit(mut self, mut X: Matrix, mut y: Vector) -> Self {
        self.weights = vec![0f64; X.cols()];
        // For each weight store variance to adjust them.
        let mut G = vec![0f64; X.cols() + 1];

        let mut t = 0usize;
        let mut stumble = 0usize;
        let mut best_loss = f64::MAX;

        for e in 0..self.iterations {
            let mut loss = 0f64;
            if self.shuffle { shuffle(&mut X, &mut y) };
            for i in 0..X.rows() {
                t += 1;
                let delta = self.intercept + dot(&self.weights, &X[i]) - y[i];
                {
                    G[0] += delta.powi(2);
                    let eta = self.eta / (G[0] + self.epsilon).sqrt();
                    self.intercept -= eta * delta;
                }
                for j in 0..X.cols() {
                    let derivative = delta * X[[i, j]];
                    // Accumulate past gradient variance.
                    G[j + 1] += derivative.powi(2);
                    // Scale eta value.
                    let eta = self.eta / (G[j + 1] + self.epsilon).sqrt();
                    // Adjust weights.
                    self.weights[j] -= eta * derivative;
                }
            }

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

    fn weights(&self) -> &Vector {
        &self.weights
    }

    fn intercept(&self) -> f64 {
        self.intercept
    }
}
