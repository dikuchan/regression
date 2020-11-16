use crate::{
    math::*,
    regressor::{lib::*, sgd::*},
};

#[test]
fn test_matrix_generation() {
    let f = |x: f64| -> f64 { 2f64 * x + 10f64 };
    let (X, y) = generate(250, 1, (0f64, 20f64), f);

    let sgd = SGD::default()
        .fit(X.clone(), y.clone());

    assert!(sgd.score(&X, &y) > 0.6);
}
