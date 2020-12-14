use crate::{
    math::*,
    regressor::{
        config::Config,
        regressor::Regressor,
    },
};

#[derive(Clone, Debug)]
pub struct RMSProp {
    config: Config,
    weights: Vector,
    best_loss: f64,
    best_weights: Vector,
}

impl RMSProp {
    pub(crate) fn new(config: Config) -> Self {
        RMSProp {
            config,
            weights: Vector::new(),
            best_loss: f64::MAX,
            best_weights: Vector::new(),
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
        self.weights = vec![0f64; 1 + X.cols()];
        self.best_weights = vec![0f64; 1 + X.cols()];
        // For each weight store values to further adjust them.
        let mut E = vec![0f64; 1 + X.cols()];

        let mut t = 0usize;
        let mut stumble = 0usize;

        for e in 0..self.config.iterations {
            if self.config.shuffle { shuffle(&mut X, &mut y); }

            let mut loss = 0f64;
            for i in 0..X.rows() {
                t += 1;
                let delta = self.weights[0] + dot(&self.weights[1..], &X[i]) - y[i];
                for j in 0..1 + X.cols() {
                    let derivative = delta * if j == 0 { 1f64 } else { X[[i, j - 1]] };

                    E[j] = self.config.gamma * E[j] + (1f64 - self.config.gamma) * derivative.powi(2);
                    let eta = self.config.eta / (E[j] + self.config.epsilon).sqrt();

                    self.weights[j] -= eta * derivative;
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
