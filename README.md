# Regression

Various optimization methods for linear regression implemented from scratch.

### About

Course assignment @ St. Petersburg State University.

Construct multidimensional linear regression models using the dataset of airline delays by the [US Bureau of 
Transportation Statistics](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp).

### Methods

- [x] Mini-batch stochastic gradient descent
- [x] AdaGrad
- [x] RMSProp
- [x] Adam

With regularization:

- [x] Lasso
- [x] Ridge

### TODO

Implement drop feature for the one-hot encoding procedure.
Multi-collinearity causes excessively high intercept values.
See more [here](https://geoffruddock.com/one-hot-encoding-plus-linear-regression-equals-multi-collinearity/).

### Usage

`Makefile` is presented for convenience.

Python script `preprocess.py` is used for transforming data. 
Use `--help` to see more.

Source code of all implemented optimization methods is placed in `src/regressor`.

### Notes

Regularization was added to the classical SGD method in `src/regressor/sgd.rs` as an optional parameter.

L2-regularization was not implemented the for adaptive gradient methods due to 
[its inefficiency](https://stackoverflow.com/questions/42415319/should-i-avoid-to-use-l2-regularization-in-conjuntion-with-rmsprop).