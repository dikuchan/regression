use crate::{
    regressor::lib::*,
    math::*,
};

use std::default::Default;

/// For the complete documentation on regressors, see `SGD`.
#[derive(Clone, Debug)]
pub struct Adam {
    iterations: usize,
    verbose: bool,
    tolerance: f64,
    shuffle: bool,
    stumble: usize,
    eta: f64,
    // The conservation factor for gradient.
    beta_m: f64,
    // The conservation factor for eta.
    beta_v: f64,
    // A small value to avoid division by zero.
    epsilon: f64,
    weights: Vector,
}

impl Adam {
    builder_field!(iterations, usize);
    builder_field!(verbose, bool);
    builder_field!(tolerance, f64);
    builder_field!(stumble, usize);
    builder_field!(eta, f64);
    builder_field!(beta_m, f64);
    builder_field!(beta_v, f64);
    builder_field!(epsilon, f64);
}

impl Default for Adam {
    fn default() -> Self {
        Self {
            iterations: 1000,
            verbose: false,
            tolerance: 1e-3,
            shuffle: true,
            stumble: 6,
            eta: 1e-2,
            beta_m: 0.9,
            beta_v: 0.9,
            epsilon: 1e-8,
            weights: Vec::new(),
        }
    }
}

impl Regressor for Adam {
    /// Fit a linear model with Adaptive Moment Estimation.
    ///
    /// # Arguments
    ///
    /// * `X`: Train matrix filled with `N` observations and `P` features.
    /// * `y`: Target vector of matrix `X`. One column with precisely `N` rows.
    fn fit(mut self, mut X: Matrix, mut y: Vector) -> Self {
        self.weights = vec![0f64; 1 + X.cols()];
        // Precompute rate of changes of weights.
        let mut M = vec![0f64; 1 + X.cols()];
        let mut V = vec![0f64; 1 + X.cols()];

        let mut t = 0usize;
        let mut stumble = 0usize;
        let mut best_loss = f64::MAX;
        let mut best_weights = vec![0f64; 1 + X.cols()];

        for e in 1..self.iterations {
            if self.shuffle { shuffle(&mut X, &mut y); }

            let mut loss = 0f64;
            for i in 0..X.rows() {
                t += 1;
                let delta = self.weights[0] + dot(&self.weights[1..], &X[i]) - y[i];
                for j in 0..1 + X.cols() {
                    let derivative = delta * if j == 0 { 1f64 } else { X[[i, j - 1]] };

                    M[j] = self.beta_m * M[j] + (1f64 - self.beta_m) * derivative;
                    V[j] = self.beta_v * V[j] + (1f64 - self.beta_v) * derivative.powi(2);
                    let m = M[j] / (1f64 - self.beta_m.powi(e as i32));
                    let v = V[j] / (1f64 - self.beta_v.powi(e as i32));
                    let eta = self.eta / (v + self.epsilon).sqrt();

                    self.weights[j] -= eta * m;
                }
                loss += delta.powi(2) / 2f64;
            }

            loss = loss / X.rows() as f64;
            if loss > best_loss - self.tolerance { stumble += 1; } else { stumble = 0; }
            if loss < best_loss {
                best_weights = self.weights.clone();
                best_loss = loss;
            }

            if self.verbose {
                println!("-- Epoch {}, Norm: {}, Bias: {}, T: {}, Average loss: {:.06}",
                         e, norm(&self.weights[1..]), self.weights[0], t, loss);
            }

            if stumble > self.stumble {
                self.weights = best_weights;
                if self.verbose { println!("Convergence after {} epochs", e); }
                return self;
            }
        }

        self
    }

    fn weights(&self) -> &Vector {
        &self.weights
    }
}
