use crate::{
    math::*,
    regressor::{
        lib::*,
        sgd::*,
        adagrad::*,
    },
};

#[test]
fn test_sgd() {
    let f = |x: f64| -> f64 { 2f64 * x + 10f64 };
    let (X, y) = generate(1000, 1, (0f64, 20f64), f);

    let sgd = SGD::default()
        .fit(X.clone(), y.clone());

    assert!(sgd.score(&X, &y) > 0.6);
}

#[test]
fn test_adagrad() {
    let f = |x: f64| -> f64 { 4f64 * x + 10f64 };
    let (X, y) = generate(1000, 1, (0f64, 40f64), f);

    let ag = AdaGrad::default()
        .iterations(50000)
        .fit(X.clone(), y.clone());

    assert!(ag.score(&X, &y) > 0.6);
}