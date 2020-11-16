# Regression

Various linear regression methods implemented from scratch.

### About

_Intelligent Systems_ course assignment @ St. Petersburg State University.

Construct the multidimensional linear regression model using dataset of airline delays by the [US Bureau of 
Transportation Statistics](https://www.transtats.bts.gov/OT_Delay/OT_DelayCause1.asp).

### Methods

Without `L1`, `L2`-regularisation:

- [x] Mini-batch stochastic gradient descent
- [x] AdaGrad
- [ ] RMSProp
- [ ] Adam

With regularisation:

- [ ] Lasso
- [ ] Ridge

### Usage

Python script is used for preprocessing data and transforming it into well-formed matrices. 
Launch `preprocess.py` with `--help` to learn about its usage.

Due to the fact that doing maths in Python without NumPy is insane, Rust was used for that purpose.
All methods are placed in `regressor` crate.
