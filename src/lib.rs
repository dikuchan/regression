#![allow(non_snake_case)]

use std::{
    ffi::{CStr, OsStr},
    os::{
        raw::c_char,
        unix::ffi::OsStrExt,
    },
    path::Path,
};

use crate::{
    math::{FromCSV, Matrix, Vector},
    regressor::{
        config::Config,
        regressor::Regressor,
        utils::Penalty,
    },
};

pub mod regressor;
pub mod math;

#[no_mangle]
pub extern "C" fn fit(method: u64, X: *const c_char, y: *const c_char, iterations: u64,
                      alpha: f64, penalty: u64, tolerance: f64, shuffle: u64,
                      verbose: u64, stumble: u64, eta: f64, score: u64)
                      -> *const f64 {
    // I'm genuinely sorry for the mess.
    let X: &Path = unsafe { OsStr::from_bytes(CStr::from_ptr(X).to_bytes()).as_ref() };
    let y: &Path = unsafe { OsStr::from_bytes(CStr::from_ptr(y).to_bytes()).as_ref() };

    let X = Matrix::read(X).unwrap();
    let y = Vector::read(y).unwrap();

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

    if score != 0 {
        let X = Matrix::read("./data/test/X.csv").unwrap();
        let y = Vector::read("./data/test/y.csv").unwrap();

        println!("MSE: {:.05}", regressor.mse(&X, &y));
        println!("R2 Score: {:.05}", regressor.score(&X, &y));
        println!("Weights: {:.05?}", regressor.weights());
    }

    regressor.weights().as_ptr()
}
