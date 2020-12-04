use crate::{
    math::*,
    regressor::{
        config::Config,
        regressor::Regressor,
    },
};

/// For the complete documentation on regressors, see `SGD`.
#[derive(Clone, Debug)]
pub struct AdaGrad {
    config: Config,
    weights: Vector,
}

impl AdaGrad {
    pub(crate) fn new(config: Config) -> Self {
        AdaGrad { config, weights: Vector::new() }
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
        self.weights = vec![0f64; 1 + X.cols()];
        // For each weight store variance to adjust them.
        let mut G = vec![0f64; 1 + X.cols()];

        let mut t = 0usize;
        let mut stumble = 0usize;
        let mut best_loss = f64::MAX;
        let mut best_weights = vec![0f64; 1 + X.cols()];

        for e in 0..self.config.iterations {
            if self.config.shuffle { shuffle(&mut X, &mut y) };

            let mut loss = 0f64;
            for i in 0..X.rows() {
                t += 1;
                let delta = self.weights[0] + dot(&self.weights[1..], &X[i]) - y[i];
                for j in 0..1 + X.cols() {
                    let derivative = delta * if j == 0 { 1f64 } else { X[[i, j - 1]] };
                    // Accumulate past gradient variance.
                    G[j] += derivative.powi(2);
                    // Scale eta.
                    let eta = self.config.eta / (G[j] + self.config.epsilon).sqrt();
                    // Adjust weights.
                    self.weights[j] -= eta * derivative;
                }
            }

            loss = loss / X.rows() as f64;
            if loss > best_loss - self.config.tolerance { stumble += 1; } else { stumble = 0; }
            if loss < best_loss {
                best_weights = self.weights.clone();
                best_loss = loss;
            }

            if self.config.verbose {
                println!("-- Epoch {}, Norm: {}, Bias: {}, T: {}, Average loss: {:.06}",
                         e, norm(&self.weights[1..]), self.weights[0], t, loss);
            }

            if stumble > self.config.stumble {
                self.weights = best_weights;
                if self.config.verbose { println!("Convergence after {} epochs", e); }
                return self;
            }
        }

        self
    }

    fn weights(&self) -> &Vector {
        &self.weights
    }
}
