import argparse
import csv
from collections import defaultdict
from math import sqrt

from cffi import FFI


class Regressor:
    def __init__(self, method: str = 'sgd', iterations: int = 1000, alpha: float = 1e-4,
                 penalty: str = 'l2',
                 tolerance: float = 1e-3, shuffle: bool = True, verbose: bool = True,
                 stumble: int = 12,
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

    def fit(self, n: int) -> [float]:
        """
        Fit regressor.
        :param n: Number of features.
        """
        ffi = FFI()
        ffi.cdef("""
            void fit(double * buffer, uint64_t n, uint64_t method, uint64_t iterations, double alpha, 
                     uint64_t penalty, double tolerance, uint64_t shuffle, uint64_t verbose, uint64_t stumble, double eta);
        """)
        C = ffi.dlopen('target/release/libregression.so')
        buffer = ffi.new(f'double [{n}]')
        C.fit(buffer, n, self.method, self.iterations, self.alpha, self.penalty,
              self.tolerance, self.shuffle, self.verbose, self.stumble, self.eta)
        weights = ffi.unpack(buffer, n)

        return weights


def assess_alpha(k: int, left: float, right: float, size: int, penalty: str) -> float:
    ffi = FFI()
    ffi.cdef("""
            double assess_alpha(uint64_t k, double left, double right, uint64_t size, uint64_t penalty);
    """)
    C = ffi.dlopen('target/release/libregression.so')
    if not penalty:
        penalty = 0
    elif penalty == 'l1':
        penalty = 1
    else:
        penalty = 2
    alpha = C.assess_alpha(k, left, right, size, penalty)

    return alpha


def onehot(X):
    features = [f for f in X.keys()
                if isinstance(X[f][-1], str)]
    replacement = {}
    for f in features:
        column = X[f]
        uniques = list(set(column))
        for u in uniques:
            replacement[f + str(u)] = [1. if v == u else 0. for v in column]
        X.pop(f)
    X.update(replacement)


def drop(X):
    for f in list(X.keys()):
        if isinstance(X[f][-1], str):
            X.pop(f)
        elif X[f].count(X[f][-1]) == len(X[f]):
            X.pop(f)
        else:
            column = [float(x) for x in X[f]]
            X[f] = column


def __standard(X):
    for f, column in X.items():
        mu = sum(column) / len(column)
        sigma = sqrt(sum([(v - mu) ** 2 for v in column]) / len(column))
        X[f] = [0. if sigma == 0. else (v - mu) / sigma for v in column]


def __minmax(X):
    for f, column in X.items():
        x, y = min(column), max(column)
        X[f] = [0. if x == y else (v - x) / (y - x) for v in column]


def scale(X, method='standard'):
    if method == 'standard':
        return __standard(X)
    elif method == 'minmax':
        return __minmax(X)
    else:
        raise ValueError('Unknown scaling method')


def split(X, k):
    length = max([len(v) for v in X.values()])
    __train, __test = {}, {}
    for f, column in X.items():
        __train[f], __test[f] = [], []
        for i in range(0, int(k * length)):
            __test[f].append(column[i])
        for i in range(int(k * length), length):
            __train[f].append(column[i])

    return __train, __test


def read_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        result = defaultdict(list)
        # Put value to corresponding feature.
        for row in reader:
            for k, v in row.items():
                try:
                    result[k].append(float(v))
                except ValueError:
                    result[k].append(v)
        result = dict(result)
        # Comma may occur on end of line.
        if '' in result:
            result.pop('', None)

        return result


def write_csv(filepath, X):
    with open(filepath, 'w+', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        length = max([len(v) for v in X.values()])
        for i in range(0, length):
            row = [X[v][i] for v in X.keys()]
            writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess data with airline delays.')

    parser.add_argument('filename', metavar='FILE', nargs='?', default='data.csv',
                        help='CSV file with data')
    parser.add_argument('--target', nargs='?', required=True, help='Target variable to predict')
    parser.add_argument('--drop', nargs='+', help='Columns to drop from data')
    parser.add_argument('--scale', nargs='?', choices=['standard', 'minmax'], default='minmax',
                        help='Method of scaling of numerical data')
    parser.add_argument('--split', nargs='?', type=float, default=0.2,
                        help='Test to train split in %%')
    parser.add_argument('--ohe', action='store_true',
                        help='Apply one-hot encoding to string columns, drop otherwise')
    parser.add_argument('--outliers', nargs='?', type=int, default=3,
                        help='Remove observations further than n standard derivations')
    parser.add_argument('--method', choices=['sgd', 'adagrad', 'rmsprop', 'adam'], default='sgd',
                        help='Optimization method for regression')

    args = parser.parse_args()

    data = read_csv(args.filename)
    if args.drop:
        for feature in args.drop:
            data.pop(feature)
    if args.ohe:
        onehot(data)
    else:
        drop(data)
    scale(data, args.scale)
    train, test = split(data, args.split)

    # Write split data.
    write_csv('./data/train/y.csv', {'y': train[args.target]})
    train.pop(args.target)
    write_csv('./data/train/X.csv', train)

    write_csv('./data/test/y.csv', {'y': test[args.target]})
    test.pop(args.target)
    write_csv('./data/test/X.csv', test)

    # Find optimum.
    regression = Regressor(method='sgd', alpha=2e-6)
    regression.fit(n=15)
