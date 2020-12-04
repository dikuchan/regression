#![allow(non_snake_case)]

use std::error::Error;

use crate::{
    math::{FromCSV, Matrix, Vector},
    regressor::{
        config::Config,
        regressor::Regressor,
    },
};

pub mod regressor;
pub mod math;

#[cfg(test)]
pub mod test;

fn main() -> Result<(), Box<dyn Error>> {
    let X = Matrix::read("./data/train/X.csv")?;
    let y = Vector::read("./data/train/y.csv")?;

    let regressor = Config::default()
        .alpha(2e-6)
        .eta(1e-2)
        .iterations(1000)
        .stumble(12)
        .tolerance(1e-3)
        .verbose(true)
        .to_SGD()
        .fit(X, y);

    let X = Matrix::read("./data/test/X.csv")?;
    let y = Vector::read("./data/test/y.csv")?;

    println!("MSE: {:.05}", regressor.mse(&X, &y));
    println!("R2 Score: {:.05}", regressor.score(&X, &y));

    Ok(())
}
