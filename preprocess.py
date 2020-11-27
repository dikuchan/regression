#!/usr/bin/env python

import argparse
import csv

from collections import defaultdict
from math import sqrt
from progress.bar import Bar

bar = Bar('Processing data...', max=8)


def ohe(data):
    features = [f for f in data.keys()
                if isinstance(data[f][-1], str)]
    replacement = {}
    for f in features:
        column = data[f]
        uniques = list(set(column))
        for u in uniques:
            replacement[f + str(u)] = [1. if v == u else 0. for v in column]
        data.pop(f)
    data.update(replacement)
    bar.next()


def drop(data):
    for f in list(data.keys()):
        if isinstance(data[f][-1], str):
            data.pop(f)
        elif data[f].count(data[f][-1]) == len(data[f]):
            data.pop(f)
        else:
            column = [float(x) for x in data[f]]
            data[f] = column
    bar.next()


def __standard(data):
    for f, column in data.items():
        mu = sum(column) / len(column)
        sigma = sqrt(sum([(v - mu) ** 2 for v in column]) / len(column))
        data[f] = [0. if sigma == 0. else (v - mu) / sigma for v in column]
    bar.next()


def __minmax(data):
    for f, column in data.items():
        x, y = min(column), max(column)
        data[f] = [0. if x == y else (v - x) / (y - x) for v in column]
    bar.next()


def scale(data, method='standard'):
    if method == 'standard':
        return __standard(data)
    elif method == 'minmax':
        return __minmax(data)
    else:
        raise ValueError('Unknown scaling method')


def split(data, k):
    length = max([len(v) for v in data.values()])
    train, test = {}, {}
    for f, column in data.items():
        train[f], test[f] = [], []
        for i in range(0, int(k * length)):
            test[f].append(column[i])
        for i in range(int(k * length), length):
            train[f].append(column[i])
    bar.next()

    return train, test


def read_csv(filepath):
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        result = defaultdict(list)
        # Put value to corresponding feature
        for row in reader:
            for k, v in row.items():
                try:
                    result[k].append(float(v))
                except ValueError:
                    result[k].append(v)
        result = dict(result)
        # Comma may occur on end of line
        if '' in result:
            result.pop('', None)
        bar.next()

        return result


def write_csv(filepath, data):
    with open(filepath, 'w+', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        length = max([len(v) for v in data.values()])
        for i in range(0, length):
            row = [data[v][i] for v in data.keys()]
            writer.writerow(row)
        bar.next()


def main():
    parser = argparse.ArgumentParser(description='Preprocess data with airline delays.')
    parser.add_argument('filename', metavar='FILE', nargs='?', default='data.csv',
                        help='CSV file with data')
    parser.add_argument('--target', nargs='?', required=True,
                        help='Target variable to predict')
    parser.add_argument('--drop', nargs='+',
                        help='Columns to drop from data')
    parser.add_argument('--scale', nargs='?', choices=['standard', 'minmax'], default='minmax',
                        help='Method of scaling of numerical data')
    parser.add_argument('--split', nargs='?', type=float, default=0.2,
                        help='Test to train split in %%')
    parser.add_argument('--ohe', action='store_true',
                        help='Apply one-hot encoding to string columns, drop otherwise')
    parser.add_argument('--outliers', nargs='?', type=int, default=3,
                        help='Remove observations further than n standard derivations')
    args = parser.parse_args()

    data = read_csv(args.filename)
    if args.drop:
        for feature in args.drop:
            data.pop(feature)
    if args.ohe:
        ohe(data)
    else:
        drop(data)
    scale(data, args.scale)
    train, test = split(data, args.split)

    write_csv('./data/train/y.csv', {'y': train[args.target]})
    train.pop(args.target)
    write_csv('./data/train/X.csv', train)

    write_csv('./data/test/y.csv', {'y': test[args.target]})
    test.pop(args.target)
    write_csv('./data/test/X.csv', test)


if __name__ == '__main__':
    main()
    bar.finish()
