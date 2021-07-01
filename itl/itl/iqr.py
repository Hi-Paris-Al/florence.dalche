from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import numpy as np

from . import cost
from . import penalty

from .kernel import *
from .model import *
from .solver import *
from .sampler import *
from .risk import *
from .utils import *
from .scoped_lru_cache import *
from .summary import *
from .estimator import *


__all__ = ['QuantileReg']


class QuantileReg(Estimator):

    def __init__(self, model=ITLModel(Gaussian(), Gaussian(), Gaussian()),
                 lbda={}, sampler=GaussLegendreUniform_0p1(),
                 cost=cost.Pinball(), penalty=penalty.Hinge(),
                 device='cpu', solver='NQN-L-BFGS-B',
                 solver_param={}, config_param={}, summary={},
                 debug={}):
        super(QuantileReg, self).__init__(model, lbda, sampler, cost, penalty,
                                          device,
                                          solver, solver_param, config_param,
                                          summary, debug)

    def _construct(self, X, y, weights):
        with tf.variable_scope('risk'):
            self.risk_ = IQR(X, y, weights,
                             self.sampler, self.cost, self.model, self.penalty,
                             self.lbda).construct()()
        return self

    def fit(self, X, y, weights=None, scope='fit'):
        with tf.variable_scope(scope):
            if self.device == 'cpu':
                device = '/device:CPU:0'
            elif self.device == 'gpu':
                device = '/device:GPU:0'
            else:
                device = self.device
            with tf.device(device):
                res = self._fit_bfgs(X, y, weights)
        return self

    def predict(self, X, T, scope='predict'):
        return self.decision_function(X, T, scope)

    def pinball_score(self, X, T, y):
        """ Score using the pinball loss."""
        return ploss(self.predict(X, T), y, T)

    def crossing_score(self, X, T, y):
        """ Score using the pinball loss."""
        return closs(self.predict(X, T), y, T)

    def score(self, X, T, y):
        """ Score using the pinball loss."""
        return self.pinball_score(X, T, y)


class IQR(Risk):

    def __init__(self, X, y, weights_X=None,
                 sampler=GaussLegendreUniform_0p1(),
                 cost=cost.Pinball(), model=KernelModel(),
                 non_crossing=penalty.SquaredHinge(),
                 lbda={}):
        super(IQR, self).__init__(X)
        self.y = y
        self.weights_X = weights_X
        self.sampler = sampler
        self.cost = cost
        self.model = model
        self.non_crossing = non_crossing
        self.lbda = lbda

    def construct(self, weights_X=None):
        super(IQR, self).construct(self.sampler(), weights_X)
        if not isinstance(self.cost, (cost.HuberPinball,
                                      cost.Pinball,
                                      cost.SquaredPinball)):
            raise BaseException('Invalid loss for Quantile Regression.')
        if (self.sampler.get_support()[0] < 0 or
           self.sampler.get_support()[1] > 1):
            raise BaseException('Support of hyperparameter distribution must '
                                'be a subset of [0, 1] for Quantile '
                                'Regression')
        self.model.construct(self.X, self.get_anchors_t(), init='random')
        self.cost.construct(tilt=self.get_anchors_t())
        self.non_crossing.construct(self.model, margin=0)
        lbda = {
            'rkhs': 1.,
            'p_rkhs': None,
            'crossing': None,
            'scale': None
        }
        lbda.update(self.lbda)
        self.lambda_ = {}
        self.lambda_['rkhs'] = default(lbda['rkhs'], 1.)
        self.lambda_['rkhs'] = self.lambda_['rkhs'] / (2 * self.X.shape[0])
        self.lambda_['p_rkhs'] = default(lbda['p_rkhs'], 0.)
        self.lambda_['p_rkhs'] = self.lambda_['p_rkhs'] / (2 * self.X.shape[0])
        self.lambda_['crossing'] = default(lbda['crossing'], 0.)
        self.lambda_['crossing'] = self.lambda_['crossing'] / self.X.shape[0]
        self.lambda_['scale'] = default(lbda['scale'], 0.)
        return self

    def get_lambda(self, name=None):
        if not hasattr(self, 'lambda_'):
            raise BaseException('Risk must be constructed first')
        if name is None:
            return [self.get_lambda('rkhs'),
                    self.get_lambda('crossing')]
        else:
            return self.lambda_[name]

    def __call__(self):
        with tf.variable_scope('predictions'):
            pred = self.model()
        with tf.variable_scope('model_regularization'):
            lambda_m = self.get_lambda('rkhs')
            squared_norm = self.model.squared_norm()
            squared_norm = (lambda_m * squared_norm
                            if lambda_m > 0 else 0)
            lambda_p = self.get_lambda('p_rkhs')
            partial_squared_norm = self.model.squared_norm(comp_p=[0])
            partial_squared_norm = (lambda_p * partial_squared_norm
                                    if lambda_p > 0 else 0)
        with tf.variable_scope('pinball_cost'):
            cost = self.cost(self.y, pred)
        with tf.variable_scope('crossing_penalty'):
            lambda_nc = self.get_lambda('crossing')
            cross_penalty = (lambda_nc * self.non_crossing()
                             if lambda_nc > 0 else 0)
        with tf.variable_scope('sparse_scales'):
            sparse_scales = 0.
            lbda_s = self.get_lambda('scale')
            if lbda_s > 0:
                for v in tf.get_collection('scales'):
                    sparse_scales = sparse_scales + v
        with tf.variable_scope('cost_penalty_aggregation'):
            risk = (self._fast_reduce(cost + cross_penalty) +
                    squared_norm + sparse_scales + partial_squared_norm)
        with tf.variable_scope('summary'):
            tf.summary.scalar('risk', risk)
            tf.summary.scalar('cost', self._fast_reduce(cost))
            if lambda_nc > 0:
                tf.summary.scalar('cross', self._fast_reduce(cross_penalty))
            tf.summary.scalar('scales', sparse_scales)
        return risk
