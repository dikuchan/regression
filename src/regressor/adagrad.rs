use crate::{
    regressor::lib::*,
    math::*,
};

use std::{
    default::Default
};

/// For complete documentation on regressors, see `SGD`.
#[derive(Clone, Debug)]
pub struct AdaGrad {
    /// Size of mini-batches used in the gradient computation.
    batch: usize,
    /// The maximum number of passes over the training data.
    iterations: usize,
    /// Whether or not the training data should be shuffled after each epoch.
    shuffle: bool,
    /// Should an important information be printed.
    verbose: bool,
    /// Initial learning rate.
    eta: f64,
    // Small value to avoid division by zero.
    epsilon: f64,
    weights: Vector,
}

impl AdaGrad {
    builder_field!(batch, usize);
    builder_field!(iterations, usize);
    builder_field!(shuffle, bool);
    builder_field!(verbose, bool);
    builder_field!(eta, f64);
    builder_field!(epsilon, f64);
}

impl Default for AdaGrad {
    fn default() -> Self {
        Self {
            batch: 8,
            iterations: 1000,
            shuffle: true,
            verbose: false,
            eta: 1e-2,
            epsilon: 1e-8,
            weights: Vec::new(),
        }
    }
}

impl Regressor for AdaGrad {
    /// Fit linear model with Adaptive Gradient Descent.
    /// Note that AdaGrad does not implement early stopping.
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
    /// let ag = AdaGrad::default()
    ///     .iterations(120000)
    ///     .verbose(true)
    ///     .fit(X, y);
    ///
    /// let y = Vector::read("./test_y.csv");
    /// let prediction = gcd.predict(&y);
    /// ```
    fn fit(mut self, mut X: Matrix, mut y: Vector) -> Self {
        // Squared loss is convex function, so start with zeroes.
        self.weights = vec![0f64; 1 + X.cols()] as Vector;
        // For each weight, store variance to adjust its weight.
        let mut G = vec![0f64; 1 + X.cols()] as Vector;

        // If batches are too large, shrink them.
        if self.batch >= X.rows() {
            self.batch = if X.rows() > 4 { X.rows() / 4 } else { 1 };
            if self.verbose { println!("Changed batch size to {}", self.batch); }
        }

        for _ in 1..self.iterations {
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
                // Derivative of intercept doesn't have component of X.
                for i in 0..self.batch {
                    derivative += if j == 0 { 1f64 } else { X[[i, j - 1]] } * delta[i];
                }
                derivative /= self.batch as f64;
                // Calculate new eta values.
                // Accumulate past gradient variance.
                G[j] += f64::powi(derivative, 2);
                // Scale eta value.
                let eta = self.eta / f64::sqrt(G[j] + self.epsilon);
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
