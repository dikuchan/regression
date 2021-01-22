use crate::{
    math::*,
    regressor::{
        config::Config,
        regressor::Regressor,
    },
};

macro_rules! builder_field {
    ($field:ident, $field_type:ty) => {
        pub fn $field(mut self, $field: $field_type) -> Self {
            self.$field = $field;
            self
        }
    };
}

/// Conveniently define regularization term that is added to the MSE formula.
#[derive(Copy, Clone, Debug)]
pub enum Penalty {
    L1,
    L2,
    None,
}

impl Penalty {
    pub fn compute(&self, alpha: f64, weight: f64) -> f64 {
        match self {
            Penalty::L1 => alpha * if weight > 0f64 { 1f64 } else { -1f64 },
            Penalty::L2 => 2f64 * alpha * weight,
            Penalty::None => 0f64,
        }
    }
}

/// K-Folds cross-validator.
///
/// Provides train and test indices to split data in train and test sets.
/// Split dataset into `k` consecutive folds (without shuffling by default).
///
/// Each fold is then used once as a validation while the `k-1` remaining folds form the training set.
///
/// # Examples
/// ```rust
/// let X = Matrix::read("./data/X.csv")?;
///
/// let folds = KFold::new(5, X, y);
/// for (train, test) in folds {
///     let (X, y) = train;
///     let (X, y) = test;
/// }
/// ```
#[derive(Clone)]
pub struct KFold<'a> {
    pub n: usize,
    batch: usize,
    rows: usize,
    X: &'a Matrix,
    y: &'a Vector,
}

impl<'a> KFold<'a> {
    pub fn new(n: usize, X: &'a Matrix, y: &'a Vector) -> Self {
        Self { n, batch: 1, rows: X.rows(), X, y }
    }
}

impl<'a> Iterator for KFold<'a> {
    type Item = ((Matrix, Vector), (Matrix, Vector));

    fn next(&mut self) -> Option<Self::Item> {
        if self.batch > self.n { return None; }

        let l = self.rows * (self.batch - 1) / self.n;
        let r = self.batch * self.rows / self.n;

        let test: Vec<usize> = (l..r).collect();
        let mut train: Vec<usize> = (0..l).collect();
        train.extend(r..self.rows);

        self.batch += 1;

        Some((
            slice(self.X, self.y, &train),
            slice(self.X, self.y, &test)
        ))
    }
}

/// Assess the best value of alpha coefficient in L2 regularization.
///
/// # Arguments
///
/// * `X`: Observation matrix.
/// * `y`: Target vector.
/// * `k`: Number of folds in cross-validation.
/// * `grid`: Vector of possible alpha values to use in the assessment.
///
/// # Examples
/// ```rust
/// let X = Matrix::read("./data/X.csv")?;
/// let y = Vector::read("./data/y.csv")?;
/// let grid = (0..250)
///     .map(|p| 1e-4 / p as f64)
///     .collect::<Vector>();
///
/// println!("Optimal alpha: {}", assess_alpha(&X, &y, 5, &grid, Penalty::L2));
/// ```
pub fn assess_alpha(X: &Matrix, y: &Vector, k: usize, grid: &Vec<f64>, penalty: Penalty) -> f64 {
    let mut alpha = 0f64;
    let mut best_error = f64::MAX;
    let mut best_iteration = 0usize;

    for (i, &p) in grid.iter().enumerate() {
        let mut error = 0f64;
        let folds = KFold::new(k, X, y);

        for (j, (train, test)) in folds.enumerate() {
            let (X, y) = train;
            let model = Config::default()
                .penalty(penalty)
                .alpha(p)
                .to_SGD()
                .fit(X, y);
            let (X, y) = test;
            error += model.mse(&X, &y);

            println!("-- Fold {}", j + 1);
        }
        error = error / k as f64;
        if error < best_error {
            best_error = error;
            best_iteration = i;
            alpha = p;
        }

        println!("-- Iteration {}, Error: {}, Alpha: {}", i + 1, error, p);
        println!("-- Best error: {} on iteration {}, Alpha: {}", best_error, best_iteration, alpha);
    }

    alpha
}
