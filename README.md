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

- [ ] Lasso
- [ ] Ridge

### Usage

`Makefile` is presented for convenience.

Python script `preprocess.py` is used for transforming data. 
Use `--help` to see more.

Source code of all implemented optimization methods is placed in `src/regressor`.
