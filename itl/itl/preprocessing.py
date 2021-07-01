from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.base import TransformerMixin
from scipy.stats.mstats import mquantiles
from scipy.spatial.distance import pdist

from .utils import data_array

__all__ = ['IdentityScaler', 'pdist_quantile']


class IdentityScaler(TransformerMixin):
    """ IdentityScaler. Does nothing."""

    def __init__(self):
        pass

    def fit(self, X):
        """ Does nothing. """
        return self

    def transform(self, X):
        """ Does nothing. """
        return X

    def inverse_transform(self, X):
        """ Does nothing. """
        return X


def pdist_quantile(X, distance='sqeuclidean', quantile=.5, gamma=False):
    if callable(X):
        X_ = X()[0]
    else:
        X_ = data_array(X)
    sigma = mquantiles(pdist(X_, distance), quantile)
    if not gamma:
        return np.sqrt(sigma)
    else:
        return .5 / sigma
