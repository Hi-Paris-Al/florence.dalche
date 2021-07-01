from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from math import sqrt, isclose

from .kernel import *
from .kernel import _KroneckerLinop
from .utils import default, tf_type, variable_summaries, get_env_precision
from .scoped_lru_cache import scope


__all__ = ['KernelModel', 'KernelDerivativeModel', 'ITLModel']


class Model(object):

    def __init__(self):
        pass

    def __call__(self, X):
        raise NotImplementedError('Abstract class \'Model\' '
                                  'cannot be instanciated')

    def construct(self, *args, **kwargs):
        raise NotImplementedError('Abstract class \'Model\' '
                                  'cannot be instanciated')

    @property
    def squared_norm(self):
        raise NotImplementedError('Abstract class \'Model\' '
                                  'cannot be instanciated')


class KernelModel(Model):

    def __init__(self, kernel=Decomposable(Gaussian(), Gaussian()),
                 trainable=True, random_state=0):
        self.kernel = kernel
        self.random_state = random_state
        self.trainable = trainable

    @scope('KernelModel_predict')
    def __call__(self, *args, **kwargs):
        args += (None, ) * (self.kernel.n_args() - len(args))
        return self.kernel(*args, **kwargs) @ self.coef

    @scope('KernelModel_add')
    def __add__(self, model):
        return KernelModelSum(self, model)

    @scope('KernelModel_radd')
    def __radd__(self, model):
        return KernelModelSum(model, self)

    @scope('KernelModel_Jac')
    def Jac(self, *args, **kwargs):
        args += (None, ) * (self.kernel.n_args() - len(args))
        return self.kernel.Jac(*args, side='l', **kwargs) @ self.coef

    @scope('KernelModel_Construct')
    def construct(self, *args, **kwargs):
        try:
            precision = os.environ['ITL_PRECISION']
        except KeyError:
            precision = 'fp32'
        self.kernel.set_anchors(*args)
        N = self.kernel.n_basis()
        try:
            init = kwargs['init']
        except KeyError:
            init = None
        if init is None or init == 'zeros':
            init_vect = tf.zeros(shape=[N, 1], dtype=tf_type(precision))
        elif init == 'random':
            init_vect = tf.random_normal(stddev=1. / np.sqrt(N), shape=[N, 1],
                                         dtype=tf_type(precision))
        else:
            init_vect = init
        self.coef = tf.Variable(init_vect,
                                trainable=self.trainable,
                                name='model_param')
        variable_summaries('param', self.coef)
        return self

    @scope('KernelModel_norm')
    def squared_norm(self, comp_p=[]):
        pred = self()
        partial_pred = self(comp_p=comp_p)
        val = tf.reshape(tf.reshape(partial_pred, [1, -1]) @
                         tf.reshape(pred, [-1, 1]), [])
        tf.summary.scalar('norm', val)
        return val


class KernelDerivativeModel(KernelModel):

    def __init__(self, kernel=Decomposable(Gaussian(), Gaussian()),
                 trainable=True, random_state=0):
        self.kernel = kernel
        self.random_state = random_state
        self.trainable = trainable

    @scope('KernelDerivativeModel_predict')
    def __call__(self, *args):
        return (self._predict(*args) +
                self._predict_Jac(*args, side='r', comp_d=[1]))

    @scope('KernelDerivativeModel_Jac')
    def Jac(self, *args, **kwargs):
        if kwargs['comp_d'] != [1]:
            raise ValueError('Unsupported Jacobian')
        return (self._predict_Jac(*args, side='l', comp_d=[1]) +
                self._predict_Hess(*args, comp_d=[1]))

    @scope('KernelDerivativeModel_predict_impl')
    def _predict(self, *args, **kwargs):
        args += (None, ) * (self.kernel.n_args() - len(args))
        return (self.kernel(*args, **kwargs) @
                self.coef)

    @scope('KernelDerivativeModel_predict_jac_impl')
    def _predict_Jac(self, *args, **kwargs):
        if kwargs['comp_d'] != [1]:
            raise ValueError('Unsupported Jacobian')
        try:
            comp_p = kwargs['comp_p']
        except KeyError:
            comp_p = [0, 1]
        args += (None, ) * (self.kernel.n_args() - len(args))
        if kwargs['side'] == 'r':
            return (self.kernel.Jac(*args, side='r',
                                    comp_p=comp_p,
                                    comp_d=[1]) @ self.coef_derivative)
        elif kwargs['side'] == 'l':
            return (self.kernel.Jac(*args, side='l',
                                    comp_p=comp_p,
                                    comp_d=[1]) @ self.coef)
        else:
            raise ValueError('side must be \'l\' or \'r\'')

    @scope('KernelDerivativeModel_predict_hess_impl')
    def _predict_Hess(self, *args, **kwargs):
        if kwargs['comp_d'] != [1]:
            raise ValueError('Unsupported Hessian')
        else:
            comp_d = kwargs['comp_d']
        args += (None, ) * (self.kernel.n_args() - len(args))
        return (self.kernel.Hess(*args, **kwargs) @ self.coef_derivative)

    @scope('KernelDerivativeModel_Construct')
    def construct(self, *args, **kwargs):
        precision = tf_type(get_env_precision())
        self.kernel.set_anchors(*args)
        N = self.kernel.n_basis()
        N_derivative = (self.kernel.n_basis_Jac(comp_d=[1]))
        try:
            init = kwargs['init']
        except KeyError:
            init = None
        if init is None or init == 'zeros':
            init_vect = tf.zeros(shape=[N, 1],
                                 dtype=precision)
            init_vect_derivative = tf.zeros(shape=[N_derivative, 1],
                                            dtype=precision)
        elif init == 'random':
            stdtot = 1. / (N + N_derivative)
            init_vect = tf.random_normal(stddev=stdtot,
                                         shape=[N, 1],
                                         dtype=precision)
            init_vect_derivative = tf.random_normal(stddev=stdtot,
                                                    shape=[N_derivative, 1],
                                                    dtype=precision)
        else:
            init_vect = init[0]
            init_vect_derivative = init[1]
        self.coef = tf.Variable(init_vect,
                                trainable=self.trainable,
                                name='model_param')
        self.coef_derivative = tf.Variable(init_vect_derivative,
                                           trainable=self.trainable,
                                           name='model_derivative_param')
        variable_summaries('param', self.coef)
        variable_summaries('param_derivative', self.coef_derivative)
        return self

    @scope('KernelDerivativeModel_norm')
    def squared_norm(self, comp_p=[]):
        pred = self._predict()
        partial_pred = self._predict(comp_p=comp_p)
        val_p = tf.reshape(tf.reshape(partial_pred, [1, -1]) @
                           tf.reshape(pred, [-1, 1]), [])
        tf.summary.scalar('norm_pred', val_p)

        hess = self._predict_Hess(comp_d=[1])
        partial_hess = self._predict_Hess(comp_p=comp_p, comp_d=[1])
        val_h = tf.reshape(tf.reshape(partial_hess, [1, -1]) @
                           tf.reshape(hess, [-1, 1]), [])
        tf.summary.scalar('norm_hess', val_h)

        jac_l = self._predict_Jac(side='l', comp_d=[1])
        partial_jac_l = self._predict_Jac(side='l',
                                          comp_p=comp_p, comp_d=[1])
        val_j_l = tf.reshape(tf.reshape(partial_jac_l, [1, -1]) @
                             tf.reshape(jac_l, [-1, 1]), [])
        tf.summary.scalar('norm_jac_l', val_j_l)

        jac_r = self._predict_Jac(side='r', comp_d=[1])
        partial_jac_r = self._predict_Jac(side='r',
                                          comp_p=comp_p, comp_d=[1])
        val_j_r = tf.reshape(tf.reshape(partial_jac_r, [1, -1]) @
                             tf.reshape(jac_r, [-1, 1]), [])
        tf.summary.scalar('norm_jac_r', val_j_r)

        val = val_p + val_h + val_j_l + val_j_r

        tf.summary.scalar('norm_squared', val)
        return val


def ITLModel(kernel_X=Gaussian(), kernel_t=Gaussian(), kernel_b=None,
             derivative=False, trainable=True, scale_bias=1,
             random_state=0):
    model = KernelDerivativeModel if derivative else KernelModel
    kernel_body = Decomposable(kernel_X, kernel_t)
    body = model(kernel_body, trainable, random_state)
    if np.isclose(scale_bias, 0) or (kernel_b is None):
        return body
    kernel_bias = Decomposable(Constant(np.sqrt(scale_bias)), kernel_b)
    bias = model(kernel_bias, trainable, random_state)
    return body + bias


class KernelModelSum(KernelModel):

    def __init__(self, model_left, model_right):
        self.model_left = model_left
        self.model_right = model_right

    @scope('KernelModelSum_predict')
    def __call__(self, *args, **kwargs):
        return (self.model_left(*args, **kwargs) +
                self.model_right(*args, **kwargs))

    @scope('KernelModelSum_Jac')
    def Jac(self, *args, **kwargs):
        return (self.model_left.Jac(*args, **kwargs) +
                self.model_right.Jac(*args, **kwargs))

    @scope('KernelModelSum_construct')
    def construct(self, *args, **kwargs):
        self.model_left.construct(*args, **kwargs)
        self.model_right.construct(*args, **kwargs)

    @scope('KernelModelSum_norm')
    def squared_norm(self, comp_p=[]):
        return (self.model_left.squared_norm(comp_p=comp_p) +
                self.model_right.squared_norm(comp_p=comp_p))

