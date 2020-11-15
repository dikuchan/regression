#![allow(non_snake_case)]

use crate::{
    math::{FromCSV, Matrix, Vector},
    regressor::{
        lib::Regressor,
        sgd::SGD,
    },
};
use std::error::Error;
use crate::regressor::lib::mse;

pub mod regressor;
pub mod math;

fn main() -> Result<(), Box<dyn Error>> {
    let X = Matrix::read("./data/train/X.csv")?;
    let y = Vector::read("./data/train/y.csv")?;

    let sgd = SGD::default()
        .iterations(25000)
        .fit(X, y);

    let X = Matrix::read("./data/test/X.csv")?;
    let y = Vector::read("./data/test/y.csv")?;

    let error = mse(&sgd, &X, &y);

    println!("Error: {}", error);

    Ok(())
}
