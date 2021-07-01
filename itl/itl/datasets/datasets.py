from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.datasets.mldata import fetch_mldata
from sklearn.datasets.kddcup99 import fetch_kddcup99
from sklearn.datasets.covtype import fetch_covtype
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle as sh
from pandas import read_csv

from .fetch_ml_mieux import fetch_spambase, fetch_annthyroid, fetch_arrhythmia
from .fetch_ml_mieux import fetch_pendigits, fetch_pima, fetch_wilt
from .fetch_ml_mieux import fetch_internet_ads, fetch_adult
from ..utils import data_array, Problem
from ..preprocessing import IdentityScaler

from operalib.datasets import toy_data_quantile


__all__ = ['import_regression_data', 'NAMES',
           'NAMES_REGRESSION', 'NAMES_ANOMALY',
           'toy_data_quantile', 'anomaly_rate', 'import_one_class_data']

NAMES_REGRESSION = ["CobarOre", "engel", "birthwt", "crabs",
                    "GAGurine", "geyser", "gilgais", "topo", "mcycle",
                    "cpus", "BostonHousing", "caution", "ftcollinssnow",
                    "highway", "heights", "sniffer", "snowgeese", "ufc",
                    "BigMac2003", "UN3"]

NAMES_ANOMALY = ['http', 'smtp', 'shuttle', 'forestcover', 'ionosphere',
                 'spambase', 'annthyroid', 'arrhythmia', 'pendigits',
                 'pima', 'wilt', 'adult']

NAMES = NAMES_REGRESSION + NAMES_ANOMALY


def anomaly_rate(y):
    return (y > 0).sum() / y.size


def split_standardized_generator(X, y, n_splits=10,
                                 test_size='default', train_size=None,
                                 fillna=True, scaler_x=True, scaler_y=True,
                                 problem=Problem.Regression, random_state=0):
    rs = ShuffleSplit(n_splits, test_size, train_size, random_state)
    for train_idx, test_idx in rs.split(X):
        imp = Imputer() if fillna else None # Deal with NaN values
        scaler_inputs = IdentityScaler() if scaler_x else None
        if scaler_y:
            if problem == Problem.Classification:
                scaler_targets = LabelBinarizer(pos_label=1, neg_label=-1)
            elif problem == Problem.Regression:
                scaler_targets = StandardScaler()
            else:
                raise BaseException('Invalid Problem')
        else:
            scaler_targets = None
        X_train = scaler_inputs.fit_transform(
            imp.fit_transform(data_array(X)[train_idx, :]))
        if test_idx.shape[0] > 0:
            X_test = scaler_inputs.transform(
                imp.transform(data_array(X)[test_idx, :]))
        else:
            X_test = None
        y_train = scaler_targets.fit_transform(
            imp.fit_transform(data_array(y)[train_idx, :]))
        if test_idx.shape[0] > 0:
            y_test = scaler_targets.transform(
                imp.transform(data_array(y)[test_idx, :]))
        else:
            y_test = None
        yield (X_train, X_test, y_train, y_test,
               (scaler_inputs, scaler_targets, imp))

def import_one_class_data(name, n_splits=1, test_size=0, train_size=None,
                          fillna=True, scaler_x=True, scaler_y=True,
                          anomaly_max=0.1, percent10_kdd=False,
                          continuous=True, random_state=0):
    '''
    Parameters
    ----------
    name : string, dataset to return
        - datasets:
            'http', 'smtp', 'shuttle', 'forestcover',
            'ionosphere', 'spambase', 'annthyroid', 'arrhythmia',
            'pendigits', 'pima', 'wilt', 'adult'
    anomaly_max : float in (0, 1), default=0.1
        max proportion of anomalies.
    percent10_kdd : bool, default=False
        Whether to load only 10 percent of the kdd data.
    scale : bool, default=True
        Whether to scale dataset.
    shuffle : bool, default=True
        Whether to shuffle dataset.
    continuous: bool, default=True
        Whether to remove discontinuous attributes.
    '''

    print('loading data {}'.format(name))

    if name.casefold() == 'adult':
        dataset = fetch_adult(shuffle=False)
        X = dataset.data
        y = dataset.target
        # anormal data are those with label >50K:
        y = np.all((y != b' <=50K', y != b' <=50K.'), axis=0).astype(int)

    elif name.casefold() == 'wilt':
        dataset = fetch_wilt(shuffle=False)
        X = dataset.data
        y = dataset.target
        y = (y == b'w').astype(int)

    elif name.casefold() == 'pima':
        dataset = fetch_pima(shuffle=False)
        X = dataset.data
        y = dataset.target

    elif name.casefold() == 'pendigits':
        dataset = fetch_pendigits(shuffle=False)
        X = dataset.data
        y = dataset.target
        y = (y == 4).astype(int)
        # anomalies = class 4

    elif name.casefold() == 'arrhythmia':
        dataset = fetch_arrhythmia(shuffle=False)
        X = dataset.data
        y = dataset.target
        # rm 5 features containing some '?' (XXX to be mentionned in paper)
        X = np.delete(X, [10, 11, 12, 13, 14], axis=1)
        # rm non-continuous features:
        if continuous is True:
            l = []
            for j in range(X.shape[1]):
                if len(set(X[:, j])) < 10:
                    l += [j]
            X = np.delete(X, l, axis=1)
        y = (y != 1).astype(int)
        # normal data are then those of class 1

    elif name.casefold() == 'annthyroid':
        dataset = fetch_annthyroid(shuffle=False)
        X = dataset.data
        y = dataset.target
        # rm 1-15 features taking only 2 values:
        if continuous is True:
            X = np.delete(X, range(1, 16), axis=1)
        y = (y != 3).astype(int)
        # normal data are then those of class 3

    elif name.casefold() == 'spambase':
        dataset = fetch_spambase(shuffle=False)
        X = dataset.data
        y = dataset.target

    elif name.casefold() == 'ionosphere':
        dataset = fetch_mldata('ionosphere')
        X = dataset.data
        y = dataset.target
        # rm first two features which are not continuous (take only 2 values):
        if continuous is True:
            X = np.delete(X, [0, 1], axis=1)
        y = (y != 1).astype(int)

    elif name in ['http', 'smtp', 'sa', 'sf']:
        dataset = fetch_kddcup99(subset=name, shuffle=False,
                                 percent10=percent10_kdd)
        X = dataset.data
        y = dataset.target

    elif name.casefold() == 'shuttle':
        dataset = fetch_mldata('shuttle')
        X = dataset.data
        y = dataset.target
        # we remove data with label 4
        # normal data are then those of class 1
        s = (y != 4)
        X = X[s, :]
        y = y[s]
        y = (y != 1).astype(int)

    elif name.casefold() == 'forestcover':
        dataset = fetch_covtype(shuffle=False)
        X = dataset.data
        y = dataset.target
        # normal data are those with attribute 2
        # abnormal those with attribute 4
        s = (y == 2) + (y == 4)
        X = X[s, :]
        y = y[s]
        # rm discontinnuous features:
        if continuous is True:
            l = []
            for j in range(X.shape[1]):
                if len(set(X[:, j])) < 10:
                    l += [j]
            X = np.delete(X, l, axis=1)
        # X = np.delete(X, [28, 50], axis=1)
        y = (y != 2).astype(int)
    else:
        raise ValueError('Unknown dataset')

    print('vectorizing data')

    X = data_array(X)

    if name.casefold() == 'sf':
        lb = LabelBinarizer()
        lb.fit(X[:, 1])
        x1 = lb.transform(X[:, 1])
        X = np.c_[X[:, :1], x1, X[:, 2:]]
        y = (y != 'normal.').astype(np.float)

    elif name.casefold() == 'sa':
        lb = LabelBinarizer()
        lb.fit(X[:, 1])
        x1 = lb.transform(X[:, 1])
        lb.fit(X[:, 2])
        x2 = lb.transform(X[:, 2])
        lb.fit(X[:, 3])
        x3 = lb.transform(X[:, 3])
        X = np.c_[X[:, :1], x1, x2, x3, X[:, 4:]]
        y = (y != 'normal.').astype(np.float)

    elif name.casefold() == 'http' or name.casefold() == 'smtp':
        y = (y != b'normal.') # 1 -> anomaly, 0 -> normal
    y = data_array(y)

    # take max 10 % of abnormal data:
    if anomaly_max is not None:
        index_normal = np.isclose(y.ravel(), 0)
        index_abnormal = np.isclose(y.ravel(), 1)
        if index_abnormal.sum() > anomaly_max * index_normal.sum():
            X_normal = X[index_normal, :]
            X_abnormal = X[index_abnormal, :]
            n_anomalies = X_abnormal.shape[0]
            n_anomalies_max = int(np.floor(0.1 * index_normal.sum()))
            r = sh(np.arange(n_anomalies))[:n_anomalies_max]
            X = np.r_[X_normal, X_abnormal[r, :]]
            y = np.array([0] * X_normal.shape[0] + [1] * n_anomalies_max)

    dite = split_standardized_generator(X, y, n_splits, test_size, train_size,
                                        fillna, scaler_x, scaler_y,
                                        Problem.Classification, random_state)
    return dite

def import_regression_data(dataset):
    """ Import datasets to numpy arrays. Inputs missing values and standard
    scales inputs and outputs. Datasets are:
        - CobarOre
        - engel
        - birthwt
        - crabs
        - GAGurine
        - geyser
        - gilgais
        - topo
        - mcycle
        - cpus
        - BostonHousing
        - caution
        - ftcollinssnow
        - highway
        - heights
        - sniffer
        - snowgeese
        - ufc
        - BigMac2003
        - UN3
        See global variable NAMES for a full list of names."""
    dirname = os.path.dirname(os.path.abspath(__file__)) + '/'
    delimiter = ','
    imp = Imputer()  # Deal with NaN values
    scaler_inputs = StandardScaler()  # Standardize the inputs
    scaler_targets = StandardScaler()  # Standardize the outputs

    if dataset == 'CobarOre':
        dirname += 'quantreg/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[['x', 'y']].values
        targets = data['z'].values
    elif dataset == 'engel':
        dirname += 'quantreg/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[['income']].values
        targets = data['foodexp'].values
    elif dataset == 'birthwt':
        dirname += 'MASS/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["age", "lwt", "race", "smoke", "ptl", "ht",
                       "ui"]].values
        targets = data['bwt'].values
    elif dataset == 'crabs':
        dirname += 'MASS/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["FL", "RW", "CL", "BD"]].values
        targets = data['CW'].values
    elif dataset == 'GAGurine':
        dirname += 'MASS/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["Age"]].values
        targets = data['GAG'].values
    elif dataset == 'geyser':
        dirname += 'MASS/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["duration"]].values
        targets = data['waiting'].values
    elif dataset == 'gilgais':
        dirname += 'MASS/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["pH00", "pH30", "pH80", "e00", "e30", "c00", "c30",
                       "c80"]].values
        targets = data['e80'].values
    elif dataset == 'topo':
        dirname += 'MASS/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["x", "y"]].values
        targets = data['z'].values
    elif dataset == 'mcycle':
        dirname += 'MASS/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["times"]].values
        targets = data["accel"].values
    elif dataset == 'cpus':
        dirname += 'MASS/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["syct", "mmin", "mmax", "cach", "chmin", "chmax",
                       "perf"]].values
        targets = data["estperf"].values
    elif dataset == 'BostonHousing':
        dirname += 'mlbench/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["crim", "zn", "indus", "chas", "nox", "rm", "age",
                       "dis", "rad", "tax", "ptratio", "b", "lstat"]].values
        targets = data["medv"].values
    elif dataset == 'caution':
        dirname += 'alr3/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["x1", "x2"]].values
        targets = data["y"].values
    elif dataset == 'ftcollinssnow':
        dirname += 'alr3/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["Early"]].values
        targets = data["Late"].values
    elif dataset == 'highway':
        dirname += 'alr3/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["ADT", "Trks", "Lane", "Acpt", "Sigs", "Itg", "Slim",
                       "Len", "Lwid", "Shld", "Hwy"]].values
        targets = data["Rate"].values
    elif dataset == 'heights':
        dirname += 'alr3/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["Mheight"]].values
        targets = data["Dheight"].values
    elif dataset == 'sniffer':
        dirname += 'alr3/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["TankTemp", "GasTemp", "TankPres", "GasPres"]].values
        targets = data["Y"].values
    elif dataset == 'snowgeese':
        dirname += 'alr3/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["obs1", "obs2"]].values
        targets = data["photo"].values
    elif dataset == 'ufc':
        dirname += 'alr3/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["Plot", "Tree", "Dbh"]].values
        targets = data["Height"].values
    elif dataset == 'BigMac2003':
        dirname += 'alr3/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["Bread", "Rice", "FoodIndex", "Bus", "Apt", "TeachGI",
                       "TeachNI", "TaxRate", "TeachHours"]].values
        targets = data["BigMac"].values
    elif dataset == 'UN3':
        dirname += 'alr3/'
        filename = dirname + dataset + '.csv'
        data = read_csv(filename, delimiter=delimiter, index_col=0)
        inputs = data[["ModernC", "Change", "PPgdp", "Frate", "Pop",
                       "Fertility"]].values
        targets = data["Purban"].values
    else:
        raise ValueError('Unknown dataset')
    inputs = np.array(inputs, dtype=np.float, ndmin=2)
    targets = np.array(targets, dtype=np.float, ndmin=2).T
    inputs = imp.fit_transform(inputs)
    inputs = scaler_inputs.fit_transform(inputs)
    targets = scaler_targets.fit_transform(targets)
    return inputs, targets, (scaler_inputs, scaler_targets, imp)
