use crate::{
    math::*,
    regressor::{
        config::Config,
        regressor::Regressor,
    },
};

regressor!(Adam);

impl Regressor for Adam {
    /// Fit a linear model with Adaptive Moment Estimation.
    ///
    /// # Arguments
    ///
    /// * `X`: Train matrix filled with `N` observations and `P` features.
    /// * `y`: Target vector of matrix `X`. One column with precisely `N` rows.
    fn fit(mut self, mut X: Matrix, mut y: Vector) -> Self {
        self.weights = vec![0f64; 1 + X.cols()];
        self.best_weights = vec![0f64; 1 + X.cols()];
        // Precompute rate of changes of weights.
        let mut M = vec![0f64; 1 + X.cols()];
        let mut V = vec![0f64; 1 + X.cols()];

        let mut t = 0usize;
        let mut stumble = 0usize;

        for e in 1..self.config.iterations {
            if self.config.shuffle { shuffle(&mut X, &mut y); }

            let mut loss = 0f64;
            for i in 0..X.rows() {
                t += 1;
                let delta = self.weights[0] + dot(&self.weights[1..], &X[i]) - y[i];
                for j in 0..1 + X.cols() {
                    let derivative = delta * if j == 0 { 1f64 } else { X[[i, j - 1]] };

                    M[j] = self.config.beta_m * M[j] + (1f64 - self.config.beta_m) * derivative;
                    V[j] = self.config.beta_v * V[j] + (1f64 - self.config.beta_v) * derivative.powi(2);
                    let m = M[j] / (1f64 - self.config.beta_m.powi(e as i32));
                    let v = V[j] / (1f64 - self.config.beta_v.powi(e as i32));
                    let eta = self.config.eta / (v + self.config.epsilon).sqrt();

                    self.weights[j] -= eta * m;
                }
                loss += delta.powi(2) / 2f64;
            }

            loss = loss / X.rows() as f64;
            if loss > self.best_loss - self.config.tolerance { stumble += 1; } else { stumble = 0; }
            if loss < self.best_loss {
                self.best_weights = self.weights.clone();
                self.best_loss = loss;
            }

            if self.config.verbose {
                println!("-- Epoch {}, Norm: {}, Bias: {}, T: {}, Average loss: {:.06}",
                         e, norm(&self.weights[1..]), self.weights[0], t, loss);
            }

            if stumble > self.config.stumble {
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
