from cffi import FFI


class Regressor:
    def __init__(self, method: str = 'sgd', iterations: int = 1000, alpha: float = 1e-4, penalty: str = 'l2',
                 tolerance: float = 1e-3, shuffle: bool = True, verbose: bool = True, stumble: int = 12,
                 eta: float = 1e-2):
        if method == 'sgd':
            self.method = 1
        elif method == 'adagrad':
            self.method = 2
        elif method == 'rmsprop':
            self.method = 3
        elif method == 'adam':
            self.method = 4
        else:
            self.method = 1
        self.iterations = iterations
        self.alpha = alpha
        if not penalty:
            self.penalty = 0
        elif penalty == 'l1':
            self.penalty = 1
        else:
            self.penalty = 2
        self.tolerance = tolerance
        self.shuffle = int(shuffle)
        self.verbose = int(verbose)
        self.stumble = stumble
        self.eta = eta

    def fit(self, X: str = './data/train/X.csv', y: str = './data/train/y.csv'):
        ffi = FFI()
        ffi.cdef("""
            double * fit(uint64_t method, uint64_t iterations, double alpha, uint64_t penalty, 
                         double tolerance, uint64_t shuffle, uint64_t verbose, uint64_t stumble, 
                         double eta);
        """)
        C = ffi.dlopen('target/release/libregression.so')

        regressor = C.fit(self.method, self.iterations, self.alpha, self.penalty, self.tolerance, self.shuffle,
                          self.verbose, self.stumble, self.eta)


if __name__ == '__main__':
    regression = Regressor(method='sgd', alpha=2e-6)
    regression.fit()
