use crate::{
    regressor::{
        sgd::SGD,
        adagrad::AdaGrad,
        rmsprop::RMSProp,
        adam::Adam,
        utils::*,
    }
};

#[derive(Copy, Clone, Debug)]
pub struct Config {
    /// The maximum number of passes over the training data.
    pub iterations: usize,
    /// Constant that multiplies the regularization term.
    pub alpha: f64,
    /// The penalty (aka regularization term) to be used.
    pub penalty: Penalty,
    /// The stopping criterion.
    /// Training will stop when `error > best_error - tolerance`.
    pub tolerance: f64,
    /// Whether or not the training data should be shuffled after each epoch.
    pub shuffle: bool,
    /// Should an important information be printed.
    pub verbose: bool,
    /// Number of iterations with no improvement to wait before early stopping.
    pub stumble: usize,
    // The conservation factor.
    pub gamma: f64,
    // The conservation factor for gradient.
    pub beta_m: f64,
    // The conservation factor for eta.
    pub beta_v: f64,
    /// The initial learning rate.
    pub eta: f64,
    // A small value to avoid division by zero.
    pub epsilon: f64,
}

impl Config {
    builder_field!(iterations, usize);
    builder_field!(alpha, f64);
    builder_field!(penalty, Penalty);
    builder_field!(tolerance, f64);
    builder_field!(shuffle, bool);
    builder_field!(verbose, bool);
    builder_field!(stumble, usize);
    builder_field!(gamma, f64);
    builder_field!(beta_m, f64);
    builder_field!(beta_v, f64);
    builder_field!(eta, f64);
    builder_field!(epsilon, f64);

    /// Fit a linear model with Stochastic Gradient Descent.
    pub fn to_SGD(self) -> SGD { SGD::new(self) }

    /// Fit a linear model with Adaptive Gradient Descent.
    pub fn to_AdaGrad(self) -> AdaGrad { AdaGrad::new(self) }

    /// Fit a linear model with AdaGrad with root mean square propagation.
    pub fn to_RMSProp(self) -> RMSProp { RMSProp::new(self) }

    /// Fit a linear model with Adaptive Moment Estimation.
    pub fn to_Adam(self) -> Adam { Adam::new(self) }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            iterations: 1000,
            alpha: 1e-4,
            penalty: Penalty::L2,
            tolerance: 1e-3,
            shuffle: true,
            verbose: false,
            stumble: 6,
            gamma: 0.9,
            beta_m: 0.9,
            beta_v: 0.9,
            eta: 1e-2,
            epsilon: 1e-8,
        }
    }
}
