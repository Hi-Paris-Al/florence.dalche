from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import auc

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


__all__ = ['DensityEst']


class DensityEst(Estimator):

    def __init__(self, model=ITLModel(Gaussian(), Gaussian(), Gaussian()),
                 lbda={}, sampler=GaussLegendreUniform_0p1(),
                 cost=cost.Hinge(),
                 device='cpu', solver='NQN-L-BFGS-B',
                 solver_param={}, config_param={}, summary={}, debug={}):
        super(DensityEst, self).__init__(model, lbda, sampler, cost, None,
                                         device,
                                         solver, solver_param, config_param,
                                         summary, debug)

    def _construct(self, X, y=None, weights=None):
        with tf.variable_scope('risk'):
            self.risk_ = IOCSVM(X, weights,
                                self.sampler, self.cost, self.model,
                                self.lbda).construct()()
        return self

    def fit(self, X, y=None, weights=None, scope='fit'):
        with tf.variable_scope(scope):
            if self.device == 'cpu':
                device = '/device:CPU:0'
            elif self.device == 'gpu':
                device = '/device:GPU:0'
            else:
                device = self.device
            with tf.device(device):
                res = self._fit_bfgs(X, None, weights)
        return self

    #def quantile(self, X):
        #T = tf.placeholder(tf.float32, shape=(1, 1))
        #pred = self._build_predictor(X, T)
        #config = tf.ConfigProto(allow_soft_placement=True)
        #config.gpu_options.allow_growth = True
        #if self.device == 'cpu':
            #device = '/cpu:0'
        #elif self.device == 'gpu':
            #device = '/gpu:0'
        #else:
            #raise('Unknown device')
        #with tf.device(device):
            #with tf.Session(config=config) as sess:
                #sess.run(tf.global_variables_initializer())
                #def f(nu, idx):
                    #return sess.run(pred,
                                    #feed_dict={T: np.array(nu, ndmin=2)})[idx,
                                                                          #0]
                #res = np.array([brentq(f, 0, 1, idx)
                                #for idx in range(X.shape[0])])
        #tf.reset_default_graph()
        #return res

    def decision_function(self, X, tau, scope=''):
        with tf.variable_scope(scope):
            if self.device == 'cpu':
                device = '/device:XLA_CPU:0'
            elif self.device == 'gpu':
                device = '/device:XLA_GPU:0'
            else:
                device = self.device
            with tf.device(device):
                Summary(self.sess_, self.summary_).construct('test')
                pred = self.sess_.run(self.model.model_left(
                    data_array(X), data_array(tau)) -
                    self.model.model_right(data_array(X), data_array(tau)))
        return pred

    def predict(self, X, T, scope='predict'):
        return np.sign(self.decision_function(X, T, scope))

    def score(self, X, y, n_points=100, scope='score'):
        return np.mean(self.predict(X) == y)

    def ROC(self, X, y, n_points=100, plot=False):
        self.transformer_ = LabelBinarizer(pos_label=1, neg_label=-1)
        yy = self.transformer_.fit_transform(y)
        nu = np.linspace(0, 1, n_points)
        pred = -self.decision_function(X, nu)
        p = np.sum(yy > 0)
        n = np.sum(yy < 0)
        tp = np.sum(np.logical_and((pred > 0), (yy > 0)), axis=0)
        fp = np.sum(np.logical_and((pred > 0), (yy < 0)), axis=0)
        tpr = tp / p
        fpr = fp / n
        idxs = np.argsort(fpr)
        fpr = np.hstack(([0], fpr[idxs], [1]))
        tpr = np.hstack(([0], tpr[idxs], [1]))
        auc_roc = auc(fpr, tpr)
        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(fpr, tpr)
            plt.title('Receiver Operating Characteristic Curve. '
                      'AUC: {}'.format(auc_roc))
            plt.show()
        return {'FPR': fpr, 'TPR': tpr, 'AUC': auc_roc, 'nu': nu[idxs]}


    def PR(self, X, y, n_points=100, plot=False):
        self.transformer_ = LabelBinarizer(pos_label=1, neg_label=-1)
        yy = self.transformer_.fit_transform(y)
        nu = np.linspace(0, 1, n_points)
        pred = -self.decision_function(X, nu)
        tp = np.sum(np.logical_and((pred > 0), (yy > 0)), axis=0)
        fp = np.sum(np.logical_and((pred > 0), (yy < 0)), axis=0)
        p = np.sum(yy > 0)
        precision = tp / (tp + fp)
        recall = tp / p
        idxs = np.argsort(recall)
        recall = np.hstack(([0], recall[idxs], [1]))
        precision = np.hstack(([1], precision[idxs], [0]))
        auc_roc = auc(recall, precision)
        if plot:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(recall, precision)
            plt.title('Precision-Recall Curve. '
                      'AUC: {}'.format(auc_roc))
            plt.show()
        return {'recall': recall, 'precision': precision,
                'AUC': auc_roc, 'nu': nu[idxs]}


class IOCSVM(Risk):

    def __init__(self, X, weights_X=None,
                 sampler=GaussLegendreUniform_0p1(),
                 cost=cost.Hinge(), model=ITLModel(),
                 lbda={}):
        super(IOCSVM, self).__init__(X)
        self.weights_X = weights_X
        self.sampler = sampler
        self.cost = cost
        self.model = model
        self.lbda = lbda

    def construct(self, weights_X=None):
        super(IOCSVM, self).construct(self.sampler(), weights_X)
        if not isinstance(self.cost, (cost.HuberHinge,
                                      cost.Hinge,
                                      cost.SquaredHinge)):
            raise BaseException('Invalid loss for Minimum-Volume Set '
                                'Estimation')
        if (self.sampler.get_support()[0] < 0 or
           self.sampler.get_support()[1] > 1):
            raise BaseException('Support of hyperparameter distribution must '
                                'be a subset of [0, 1] for Density Estimation')
        self.model.construct(self.X, self.get_anchors_t(), init='random')
        self.cost.construct(margin=self.model.model_right())
        lbda = {
            'rkhs': 1e-5,
            'scale': None
        }
        lbda.update(self.lbda)
        self.lambda_ = {}
        self.lambda_['rkhs'] = default(lbda['rkhs'], 1.)
        self.lambda_['rkhs'] = (self.lambda_['rkhs'] /
                                (2 * self.get_anchors_t().shape[0]))
        self.lambda_['scale'] = default(lbda['scale'], 0.)
        return self

    def get_lambda(self, name=None):
        if not hasattr(self, 'lambda_'):
            raise BaseException('Risk must be constructed first')
        if name is None:
            return self.lambda_.values()
        else:
            return self.lambda_[name]

    def __call__(self):
        with tf.variable_scope('predictions'):
            pred = self.model.model_left()
        with tf.variable_scope('margin_regularization'):
            lambda_m = self.get_lambda('rkhs')
            squared_norm = (lambda_m * self.model.model_right.squared_norm()
                            if lambda_m > 0 else 0.)
        with tf.variable_scope('hinge_cost'):
            cost = self.cost(1, pred) / tf.reshape(self.get_anchors_t(),
                                                   [1, -1])
        with tf.variable_scope('pointwise_theta_regularization'):
            partial_pred = self.model.model_left(comp_p=[1])
            tau_reg = tf.reduce_sum(partial_pred * pred, [0], keepdims=True)
        with tf.variable_scope('margin_shift'):
            margin_shift = self.model.model_right()
        with tf.variable_scope('sparse_scales'):
            sparse_scales = 0.
            lbda_s = self.get_lambda('scale')
            if lbda_s > 0:
                for v in tf.get_collection('scales'):
                    sparse_scales = sparse_scales + v
                sparse_scales = lbda_s * sparse_scales
        with tf.variable_scope('cost_penalty_aggregation'):
            risk = (self._fast_reduce(cost) +
                    self._fast_reduce((tau_reg / 2 - margin_shift) /
                                       self.get_weights_X()) +
                    squared_norm + sparse_scales)
            risk = tf.reshape(risk, [])
        with tf.variable_scope('summary'):
            tf.summary.scalar('risk', risk)
            tf.summary.tensor_summary('cost_test', cost)
            tf.summary.scalar('cost', self._fast_reduce(cost))
            tf.summary.scalar('tau_reg', self._fast_reduce(tau_reg))
            tf.summary.scalar('margin_shift', self._fast_reduce(margin_shift))
            tf.summary.scalar('scales', sparse_scales)
        return risk

