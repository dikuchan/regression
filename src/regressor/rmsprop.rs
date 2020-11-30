use crate::{
    regressor::lib::*,
    math::*,
};

use std::default::Default;

/// For the complete documentation on regressors, see `SGD`.
#[derive(Clone, Debug)]
pub struct RMSProp {
    iterations: usize,
    verbose: bool,
    tolerance: f64,
    shuffle: bool,
    stumble: usize,
    eta: f64,
    // The conservation factor.
    gamma: f64,
    // A small value to avoid division by zero.
    epsilon: f64,
    intercept: f64,
    weights: Vector,
}

impl RMSProp {
    builder_field!(iterations, usize);
    builder_field!(verbose, bool);
    builder_field!(tolerance, f64);
    builder_field!(shuffle, bool);
    builder_field!(stumble, usize);
    builder_field!(eta, f64);
    builder_field!(gamma, f64);
    builder_field!(epsilon, f64);

    fn update(&self, i: usize, d: f64, E: &mut Vector) -> f64 {
        E[i] = self.gamma * E[i] + (1f64 - self.gamma) * d.powi(2);
        self.eta / (E[i] + self.epsilon).sqrt()
    }
}

impl Default for RMSProp {
    fn default() -> Self {
        Self {
            iterations: 1000,
            verbose: false,
            tolerance: 1e-3,
            shuffle: true,
            stumble: 6,
            eta: 1e-2,
            gamma: 0.9,
            epsilon: 1e-8,
            intercept: 0f64,
            weights: Vec::new(),
        }
    }
}

impl Regressor for RMSProp {
    /// Fit a linear model with AdaGrad with root mean square propagation.
    ///
    /// # Arguments
    ///
    /// * `X`: Train matrix filled with `N` observations and `P` features.
    /// * `y`: Target vector of matrix `X`. One column with precisely `N` rows.
    fn fit(mut self, mut X: Matrix, mut y: Vector) -> Self {
        self.weights = vec![0f64; X.cols()];
        // For each weight store values to further adjust them.
        let mut E = vec![0f64; 1 + X.cols()];

        let mut t = 0usize;
        let mut stumble = 0usize;
        let mut best_loss = f64::MAX;

        for e in 0..self.iterations {
            let mut loss = 0f64;
            if self.shuffle { shuffle(&mut X, &mut y); }
            for i in 0..X.rows() {
                t += 1;
                let delta = self.intercept + dot(&self.weights, &X[i]) - y[i];
                {
                    let eta = self.update(0, delta, &mut E);
                    self.intercept -= eta * delta;
                }
                for j in 0..X.cols() {
                    let derivative = delta * X[[i, j]];
                    let eta = self.update(j + 1, derivative, &mut E);
                    self.weights[j] -= eta * derivative;
                }
                loss += delta.powi(2) / 2f64;
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
