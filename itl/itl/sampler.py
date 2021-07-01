from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from scipy.integrate import quad
from sobol_seq import i4_sobol_generate
from scipy.stats import uniform, rv_continuous
from ghalton import Halton as halton_generate

from .utils import *

__all__ = ['Sampler', 'Sobol', 'Dirac', 'GaussLegendre',
           'SobolUniform', 'SobolUniform_0p1', 'SobolUniform_m1p1', 'Halton',
           'HaltonUniform', 'HaltonUniform_0p1', 'HaltonUniform_m1p1',
           'GaussLegendreUniform', 'GaussLegendreUniform_0p1',
           'GaussLegendreUniform_m1p1', 'GaussChebyshev',
           'GaussChebyshevUniform', 'GaussChebyshevUniform_0p1',
           'GaussChebyshevUniform_m1p1', 'Random', 'RandomUniform',
           'RandomUniform_0p1', 'RandomUniform_m1p1', 'RVContinuous']


class RVContinuous(rv_continuous):

    def construct(self, f=lambda x: x):
        self.f = f
        self.alpha = quad(self.f, self.a, self.b)[0]
        return self

    def _pdf(self, x):
        return self.f(x) / self.alpha


class Sampler(object):

    def __init__(self, n_samples):
        self.n_samples = n_samples

    def get_support(self):
        if hasattr(self, 'distribution'):
            return (self.distribution.ppf(self.distribution.a),
                    self.distribution.ppf(self.distribution.b))
        elif hasattr(self, 'loc'):
            return (np.min(self.loc), np.max(self.loc))
        else:
            raise BaseException('No suppport found.')

    def _package(self, anchors, weights):
        anchors = data_array(anchors)
        weights = data_array(weights)
        return anchors, weights


class GaussLegendre(Sampler):

    def __init__(self, n_samples=100, distribution=uniform(0, 1)):
        super(GaussLegendre, self).__init__(n_samples)
        self.distribution = distribution

    def __call__(self):
        support = self.get_support()
        if np.abs(support[0]) < np.inf:
            anchors, weights = np.polynomial.legendre.leggauss(self.n_samples)
            anchors = (.5 * (anchors + 1) * (support[1] - support[0]) +
                       support[0])
            weights = 2. / (support[1] - support[0]) * weights
        else:
            raise NotImplementedError('Invalid support {} for '
                                      'Gauss-Legendre '
                                      'Quadrature.'.format(support))
        return self._package(anchors, weights * self.distribution.pdf(anchors))


class GaussLegendreUniform(GaussLegendre):

    def __init__(self, n_samples=100, support=[0, 1]):
        super(GaussLegendreUniform, self).__init__(n_samples,
                                                   uniform(support[0],
                                                           support[1] -
                                                           support[0]))
        self.support = support


class GaussLegendreUniform_0p1(GaussLegendreUniform):

    def __init__(self, n_samples=100):
        super(GaussLegendreUniform_0p1, self).__init__(n_samples, [0, 1])


class GaussLegendreUniform_m1p1(GaussLegendreUniform):

    def __init__(self, n_samples=100):
        super(GaussLegendreUniform_m1p1, self).__init__(n_samples, [-1, 1])


class GaussChebyshev(Sampler):

    def __init__(self, n_samples=100, distribution=uniform(0, 1)):
        super(GaussChebyshev, self).__init__(n_samples)
        self.distribution = distribution

    def __call__(self):
        support = self.get_support()
        if np.abs(support[0]) < np.inf:
            anchors, weights = np.polynomial.chebyshev.chebgauss(
                self.n_samples)
            anchors = (.5 * (anchors + 1) * (support[1] - support[0]) +
                       support[0])
            weights = 2. / (support[1] - support[0]) * weights
        else:
            raise NotImplementedError('Invalid support {} for '
                                      'Gauss-Legendre '
                                      'Quadrature.'.format(support))
        return self._package(anchors, weights * self.distribution.pdf(anchors))


class GaussChebyshevUniform(GaussChebyshev):

    def __init__(self, n_samples=100, support=[0, 1]):
        super(GaussChebyshevUniform, self).__init__(n_samples,
                                                   uniform(support[0],
                                                           support[1] -
                                                           support[0]))
        self.support = support


class GaussChebyshevUniform_0p1(GaussChebyshevUniform):

    def __init__(self, n_samples=100):
        super(GaussChebyshevUniform_0p1, self).__init__(n_samples, [0, 1])


class GaussChebyshevUniform_m1p1(GaussChebyshevUniform):

    def __init__(self, n_samples=100):
        super(GaussChebyshevUniform_m1p1, self).__init__(n_samples, [-1, 1])


class Dirac(Sampler):

    def __init__(self, loc, weights=None):
        super(Dirac, self).__init__(np.array(loc).reshape(-1, 1).size)
        self.loc = loc
        self.weights = weights

    def __call__(self):
        return self._package(self.loc,
                             default(self.weights, 1. / self.n_samples))


class Sobol(Sampler):

    def __init__(self, n_samples=100, distribution=uniform(0, 1)):
        super(Sobol, self).__init__(n_samples)
        self.distribution = distribution

    def __call__(self):
        uniform_pr = np.asarray(
            i4_sobol_generate(1, self.n_samples)).reshape(-1, 1)
        anchors = self.distribution.ppf(uniform_pr)
        return self._package(anchors, 1. / self.n_samples)

class SobolUniform(Sobol):

    def __init__(self, n_samples=100, support=[0, 1]):
        super(SobolUniform, self).__init__(n_samples,
                                           uniform(support[0],
                                                   support[1] - support[0]))
        self.support = support


class SobolUniform_0p1(SobolUniform):

    def __init__(self, n_samples=100):
        super(SobolUniform_0p1, self).__init__(n_samples, [0, 1])


class SobolUniform_m1p1(SobolUniform):

    def __init__(self, n_samples=100):
        super(SobolUniform_m1p1, self).__init__(n_samples, [-1, 1])


class Halton(Sampler):

    def __init__(self, n_samples=100, distribution=uniform(0, 1)):
        super(Halton, self).__init__(n_samples)
        self.distribution = distribution

    def __call__(self):
        sequencer = halton_generate(1)
        uniform_pr = np.asarray(sequencer.get(self.n_samples)).reshape(-1, 1)
        anchors = self.distribution.ppf(uniform_pr)
        return self._package(anchors, 1. / self.n_samples)


class HaltonUniform(Halton):

    def __init__(self, n_samples=100, support=[0, 1]):
        super(HaltonUniform, self).__init__(n_samples,
                                            uniform(support[0],
                                                    support[1] - support[0]))
        self.support = support


class HaltonUniform_0p1(HaltonUniform):

    def __init__(self, n_samples=100):
        super(HaltonUniform_0p1, self).__init__(n_samples, [0, 1])


class HaltonUniform_m1p1(HaltonUniform):

    def __init__(self, n_samples=100):
        super(HaltonUniform_m1p1, self).__init__(n_samples, [-1, 1])


class Random(Sampler):

    def __init__(self, n_samples=100, distribution=uniform(0, 1)):
        super(Random, self).__init__(n_samples)
        self.distribution = distribution

    def __call__(self):
        return self._package(self.distribution.rvs(self.n_samples),
                             1. / self.n_samples)


class RandomUniform(Halton):

    def __init__(self, n_samples=100, support=[0, 1]):
        super(RandomUniform, self).__init__(n_samples,
                                            uniform(support[0],
                                                    support[1] - support[0]))
        self.support = support


class RandomUniform_0p1(HaltonUniform):

    def __init__(self, n_samples=100):
        super(RandomUniform_0p1, self).__init__(n_samples, [0, 1])


class RandomUniform_m1p1(HaltonUniform):

    def __init__(self, n_samples=100):
        super(RandomUniform_m1p1, self).__init__(n_samples, [-1, 1])
