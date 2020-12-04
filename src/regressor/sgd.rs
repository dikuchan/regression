use crate::{
    math::*,
    regressor::{
        config::Config,
        regressor::Regressor,
    },
};

regressor!(SGD);

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
        self.weights = vec![0f64; 1 + X.cols()];
        self.best_weights = vec![0f64; 1 + X.cols()];

        // Scaling factor for eta.
        let mut t = 0usize;
        // Number of times loss didn't improve.
        let mut stumble = 0usize;

        for e in 0..self.config.iterations {
            // It is essential to reshuffle data.
            // Randomly permute all rows.
            if self.config.shuffle { shuffle(&mut X, &mut y); }

            let mut loss = 0f64;
            for i in 0..X.rows() {
                t += 1;
                // Scale learning rate.
                // Default method in `sklearn`.
                let eta = self.config.eta / (t as f64).powf(0.25);
                // Precompute the part of derivative that doesn't depend on `X`.
                let delta = self.weights[0] + dot(&self.weights[1..], &X[i]) - y[i];
                for j in 0..1 + X.cols() {
                    let penalty = self.config.penalty.compute(self.config.alpha, self.weights[j]);
                    let derivative = delta * if j == 0 { 1f64 } else { X[[i, j - 1]] } + penalty;
                    self.weights[j] -= eta * derivative;
                }
                loss += delta.powi(2) / 2f64;
            }

            // Compute average loss.
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

    /// Return weights of the trained model.
    ///
    /// If regressor was not yet trained via `fit` method, return garbage.
    fn weights(&self) -> &Vector {
        &self.weights
    }
}
