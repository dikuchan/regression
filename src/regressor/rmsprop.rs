use crate::{
    regressor::lib::*,
    math::*,
};

use std::default::Default;

/// For the complete documentation on regressors, see `SGD`.
#[derive(Clone, Debug)]
pub struct RMSProp {
    /// Size of a mini-batch used in the gradient computation.
    batch: usize,
    /// The maximum number of passes over the training data.
    iterations: usize,
    /// Should an important information be printed.
    verbose: bool,
    /// The initial learning rate.
    eta: f64,
    // The conservation factor.
    gamma: f64,
    // A small value to avoid division by zero.
    epsilon: f64,
    weights: Vector,
}

impl RMSProp {
    builder_field!(batch, usize);
    builder_field!(iterations, usize);
    builder_field!(verbose, bool);
    builder_field!(eta, f64);
    builder_field!(gamma, f64);
    builder_field!(epsilon, f64);
}

impl Default for RMSProp {
    fn default() -> Self {
        Self {
            batch: 8,
            iterations: 1000,
            verbose: false,
            eta: 1e-2,
            gamma: 0.9,
            epsilon: 1e-8,
            weights: Vec::new(),
        }
    }
}

impl Regressor for RMSProp {
    /// Fit a linear model with AdaGrad with root mean square propagation.
    /// Note that RMSProp does not implement early stopping.
    ///
    /// # Arguments
    ///
    /// * `X`: Train matrix filled with `N` observations and `P` features.
    /// * `y`: Target vector of matrix `X`. One column with precisely `N` rows.
    ///
    /// # Examples
    /// ```rust
    /// let X = Matrix::read("./train_X.csv");
    /// let y = Vector::read("./train_y.csv");
    /// let rmsp = RMSProp::default()
    ///     .iterations(10000)
    ///     .verbose(true)
    ///     .fit(X, y);
    ///
    /// let y = Vector::read("./test_y.csv");
    /// let prediction = rmsp.predict(&y);
    /// ```
    fn fit(mut self, mut X: Matrix, mut y: Vector) -> Self {
        // Squared loss is convex function, so start with zeroes.
        self.weights = vec![0f64; 1 + X.cols()] as Vector;

        let mut EG = vec![0f64; 1 + X.cols()] as Vector;

        // If batches are too large, shrink them.
        if self.batch >= X.rows() {
            self.batch = if X.rows() > 4 { X.rows() / 4 } else { 1 };
            if self.verbose { println!("Changed batch size to {}", self.batch); }
        }

        for e in 1..self.iterations {
            if self.verbose {
                if e % 250 == 0 {
                    println!("Processed epoch #{}", e);
                    println!("Weights: {:?}", self.weights);
                }
            }
            // Randomly permute all rows.
            shuffle(&mut X, &mut y);
            // Linear regression function is `w0 + w1 x1 + ... + wp xp = y`.
            // Precompute part of gradient that does not depend on `x`.
            let delta: Vector = (0..self.batch)
                .map(|i| self.weights[0] + dot(&self.weights[1..], &X[i]) - y[i])
                .collect();
            // For each weight `wi` find derivative using batch of observations.
            for j in 0..self.weights.len() {
                let mut derivative = 0f64;
                // Derivative of intercept doesn't have `x` component.
                for i in 0..self.batch {
                    derivative += (if j == 0 { 1f64 } else { X[[i, j - 1]] }) * delta[i];
                }
                derivative /= self.batch as f64;
                // Calculate new eta values.
                EG[j] = self.gamma * EG[j] + (1f64 - self.gamma) * derivative.powi(2);
                // Scale eta value.
                let eta = self.eta / f64::sqrt(EG[j] + self.epsilon);
                // Adjust weights.
                self.weights[j] -= eta * derivative;
            }
        }

        self
    }

    fn weights(&self) -> &Vector {
        &self.weights
    }
}
