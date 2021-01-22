#![allow(non_snake_case)]

use crate::{
    math::{FromCSV, Matrix, Vector},
    regressor::{
        config::Config,
        regressor::Regressor,
        utils::{assess_alpha as assess, Penalty},
    },
};

pub mod regressor;
pub mod math;

#[no_mangle]
pub extern "C" fn fit(buffer: *mut f64, n: usize, method: u64, iterations: u64, alpha: f64,
                      penalty: u64, tolerance: f64, shuffle: u64, verbose: u64, stumble: u64, eta: f64) {
    let X = Matrix::read("./data/train/X.csv").unwrap();
    let y = Vector::read("./data/train/y.csv").unwrap();

    let config = Config::default()
        .iterations(iterations as usize)
        .alpha(alpha)
        .penalty(match penalty {
            0 => Penalty::None,
            1 => Penalty::L1,
            2 => Penalty::L2,
            _ => unimplemented!()
        })
        .tolerance(tolerance)
        .shuffle(shuffle != 0)
        .verbose(verbose != 0)
        .stumble(stumble as usize)
        .eta(eta);
    let regressor: Box<dyn Regressor> = match method {
        1 => Box::new(config.to_SGD().fit(X, y)),
        2 => Box::new(config.to_AdaGrad().fit(X, y)),
        3 => Box::new(config.to_RMSProp().fit(X, y)),
        4 => Box::new(config.to_Adam().fit(X, y)),
        _ => unimplemented!()
    };

    let X = Matrix::read("./data/test/X.csv").unwrap();
    let y = Vector::read("./data/test/y.csv").unwrap();

    println!("MSE: {:.05}", regressor.mse(&X, &y));
    println!("R2 Score: {:.05}", regressor.score(&X, &y));
    println!("Weights: {:.05?}", regressor.weights());

    unsafe {
        std::slice::from_raw_parts_mut(buffer, n)
            .copy_from_slice(&regressor.weights())
    }
}

#[no_mangle]
pub extern "C" fn assess_alpha(k: u64, left: f64, right: f64, size: u64, penalty: u64) -> f64 {
    let X = Matrix::read("./data/train/X.csv").unwrap();
    let y = Vector::read("./data/train/y.csv").unwrap();
    let grid = (0..size)
        .map(|p| left + p as f64 * (right - left) / size as f64)
        .collect::<Vector>();
    let penalty = match penalty {
        0 => Penalty::None,
        1 => Penalty::L1,
        2 => Penalty::L2,
        _ => unimplemented!()
    };

    assess(&X, &y, k as usize, &grid, penalty)
}
