use crate::{
    math::*,
    regressor::{
        lib::Regressor,
        sgd::SGD,
        adagrad::AdaGrad,
        rmsprop::RMSProp,
        adam::Adam,
    },
};

#[test]
fn test_sgd() {
    let f = |x: f64| { 10f64 + 2f64 * x };
    let (X, y) = generate(1000, (0f64, 20f64), f);

    let sgd = SGD::default()
        .fit(X.clone(), y.clone());

    assert!(sgd.score(&X, &y) > 0.6);
}

#[test]
fn test_adagrad() {
    let f = |x: f64| { 10f64 + 4f64 * x };
    let (X, y) = generate(1000, (0f64, 40f64), f);

    let ag = AdaGrad::default()
        .fit(X.clone(), y.clone());

    assert!(ag.score(&X, &y) > 0.6);
}

#[test]
fn test_rmsprop() {
    let f = |x: f64| { -10f64 + 3f64 * x };
    let (X, y) = generate(1000, (30f64, 60f64), f);

    let rmsp = RMSProp::default()
        .iterations(10000)
        .fit(X.clone(), y.clone());

    assert!(rmsp.score(&X, &y) > 0.6);
}

#[test]
fn test_adam() {
    let f = |x: f64| { -2f64 + 10f64 * x };
    let (X, y) = generate(1000, (2f64, 16f64), f);

    let adam = Adam::default()
        .iterations(10000)
        .fit(X.clone(), y.clone());

    assert!(adam.score(&X, &y) > 0.6);
}