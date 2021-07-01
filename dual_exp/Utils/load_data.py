import numpy as np
import csv
import h5py


def load_bibtex(path='Bibtex/Bibtex_data.txt'):

    """
        Load Dataset Bibtex
    """

    with open(path, 'r') as file:
        data = file.read()
    data = data.split('\n')
    n, p, d = list(map(int, data[0].split(' ')))
    X = np.zeros((n, p))
    Y = np.zeros((n, d))
    for i in range(n):
        z = data[i + 1].split(' ')
        y_idx = list(map(int, z[0].split(',')))
        Y[i, y_idx] = 1
        x_idx = z[1:]
        x_idx = list(map(lambda s: int(s.split(':')[0]), x_idx))
        X[i, x_idx] = 1

    return X, Y, n, p, d


def load_delicious(path='Delicious/Delicious_data.txt'):
    """
        Load Dataset Delicious
    """

    with open(path, 'r') as file:
        data = file.read()
    data = data.split('\n')
    n, p, d = list(map(int, data[0].split(' ')))
    X = np.zeros((n, p))
    Y = np.zeros((n, d))
    for i in range(n):
        z = data[i + 1].split(' ')
        if z[0] != '':
            y_idx = list(map(int, z[0].split(',')))
            Y[i, y_idx] = 1
        x_idx = z[1:]
        x_idx = list(map(lambda s: int(s.split(':')[0]), x_idx))
        X[i, x_idx] = 1

    return X, Y, n, p, d


def load_yeast(path='YEAST_Elisseeff_Weston_2002.csv'):
    """
        Load Dataset YEAST
    """

    p = 103
    d = 14
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        data = list(reader)

    n = len(data)

    X = np.zeros((n, p))
    Y = np.zeros((n, d))
    for i in range(n):
        X[i] = np.array(data[i][:p])
        Y[i] = np.array([True if data[i][p + j] == 'TRUE' else False
                         for j in range(d)])

    return X, Y, n, p, d


def load_scene(path='Data/Scene/scene_train'):

    if path[-1] == 'n':
        n, p, d = 1211, 294, 6
    else:
        n, p, d = 1196, 294, 6

    with open(path, 'r') as file:
        data = file.read()

    data = data.split('\n')

    X = np.zeros((n, p))
    Y = np.zeros((n, d))

    for i in range(n):
        z = data[i].split(' ')
        y_idx = list(map(int, z[0].split(',')))
        Y[i, y_idx] = 1
        x = z[1:]
        x = list(map(lambda s: float(s.split(':')[1]), x))
        X[i, :] = x

    return X, Y, n, p, d


def load_usps(path='Data/usps.h5'):

    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X_tr = train.get('data')[:]
        y_tr = train.get('target')[:]
        test = hf.get('test')
        X_te = test.get('data')[:]
        y_te = test.get('target')[:]

    return X_tr, y_tr, X_te, y_te
