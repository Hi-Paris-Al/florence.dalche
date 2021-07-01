from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.linalg as linalg

from distutils.version import LooseVersion

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

from sklearn.metrics.pairwise import additive_chi2_kernel
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances

from .utils import (default, get_type, variable_summaries, tf_type,
                    get_env_precision)
from .scoped_lru_cache import scope


__all__ = ['Kernel',
           'Gaussian', 'Laplacian', 'Constant', 'Linear', 'Polynomial',
           'Impulse', 'ExponentiatedChi2', 'Even', 'Projection', 'Map',
           'Periodic', 'Decomposable', 'Intersection']


@scope('pairwise_euclidean')
def _fast_pairwise_euclidean(input1, input2, use_scikit=True):
    if input1 is input2:
        return _fast_pairwise_euclidean_gram(input1, use_scikit)
    if (use_scikit and
       isinstance(input1, np.ndarray) and
       isinstance(input2, np.ndarray)):
        return tf.constant(euclidean_distances(input1, input2, squared=True),
                           dtype=tf_type(get_env_precision()))
    square1 = tf.reshape(tf.reduce_sum(input1 * input1, 1,
                                       name="reduce_input1"), [-1, 1])
    square2 = tf.reshape(tf.reduce_sum(input2 * input2, 1,
                                       name="reduce_input2"), [1, -1])
    return tf.maximum(square1 -
                      2 * tf.matmul(input1, input2, transpose_b=True,
                                    name="inner_product") +
                      square2, 0, name="positivity_constraint")


@scope('pairwise_euclidean_gram')
def _fast_pairwise_euclidean_gram(input1, use_scikit=True):
    if (use_scikit and isinstance(input1, np.ndarray)):
        return tf.constant(euclidean_distances(input1, squared=True),
                           dtype=tf_type(get_env_precision()))
    square1 = tf.reshape(tf.reduce_sum(input1 * input1, 1,
                                       name="reduce_input1"), [-1, 1])
    return tf.maximum(square1 -
                      2 * tf.matmul(input1, input1, transpose_b=True,
                                    name="inner_product") +
                      tf.transpose(square1), 0,
                      name="positivity_constraint")


@scope('pairwise_manhattan')
def _pairwise_manhattan(input1, input2, use_scikit=True):
    if input1 is input2:
        return _pairwise_manhattan_gram(input1, use_scikit)
    if (use_scikit and
       isinstance(input1, np.ndarray) and
       isinstance(input2, np.ndarray)):
        return tf.constant(manhattan_distances(input1, input2),
                           dtype=get_type(input1))
    return tf.reduce_sum(tf.abs(tf.expand_dims(input1, 1) -
                                tf.expand_dims(input2, 0)), 2)


@scope('pairwise_manhattan_gram')
def _pairwise_manhattan_gram(input1, use_scikit=True):
    if (use_scikit and isinstance(input1, np.ndarray)):
        return tf.constant(manhattan_distances(input1), dtype=get_type(input1))
    return tf.reduce_sum(tf.abs(tf.expand_dims(input1, 1) -
                                tf.expand_dims(input1, 0)), 2)


@scope('pairwise_equal')
def _pairwise_equal(input1, input2, eps, use_scikit=False):
    if input1 is input2:
        return _pairwise_equal_gram(input1, use_scikit)
    return tf.cast(_fast_pairwise_euclidean(input1, input2) < eps,
                   get_type(input1))


@scope('pairwise_equal_gram')
def _pairwise_equal_gram(input1, use_scikit=False):
    return tf.eye(input1.shape[0], dtype=get_type(input1))


@scope('pairwise_minimum')
def _pairwise_min(input1, input2, use_scikit=False):
    test = tf.reduce_sum(tf.minimum(tf.expand_dims(input1, 1),
                                    tf.expand_dims(input2, 0)), 2)
    return test

@scope('pairwise_minimum_gram')
def _pairwise_min_gram(input1, use_scikit=False):
    return tf.reduce_sum(tf.minimum(tf.expand_dims(input1, 1,
                                    tf.expand_dims(input1, 0))), 2)


@scope('pairwise_chi2')
def _pairwise_chi2(input1, input2, eps, use_scikit=True):
    if input1 is input2:
        return _pairwise_chi2_gram(input1, eps, use_scikit)
    if (use_scikit and
       isinstance(input1, np.ndarray) and
       isinstance(input2, np.ndarray)):
        return tf.constant(-additive_chi2_kernel(input1, input2))
    diff = tf.expand_dims(input1, 1) - tf.expand_dims(input2, 0)
    summ = tf.expand_dims(input1, 1) + tf.expand_dims(input2, 0)
    return tf.reduce_sum(tf.where(tf.abs(summ) < eps,
                         tf.zeros(shape=summ.shape), diff * diff / summ),
                         axis=2)


@scope('pairwise_chi2_gram')
def _pairwise_chi2_gram(input1, eps, use_scikit=True):
    if (use_scikit and isinstance(input1, np.ndarray)):
        return tf.constant(-additive_chi2_kernel(input1))
    diff = tf.expand_dims(input1, 1) - tf.expand_dims(input1, 0)
    summ = tf.expand_dims(input1, 1) + tf.expand_dims(input1, 0)
    return tf.reduce_sum(tf.where(tf.abs(summ) < eps,
                         tf.zeros(shape=summ.shape), diff * diff / summ),
                         axis=2)


class Kernel(object):

    def __init__(self):
        pass

    def __call__(self, X):
        raise NotImplementedError('Abstract class \'Kernel\' '
                                  'cannot be instanciated')

    def __add__(self, kernel):
        return KernelSum(self, kernel)

    def __radd__(self, kernel):
        return KernelSum(kernel, self)

    def __mul__(self, kernel):
        return KernelProduct(self, kernel)

    def __rmul__(self, kernel):
        return KernelProduct(kernel, self)

    def Jac(self, X=None):
        raise NotImplementedError('Abstract class \'Kernel\' '
                                  'cannot be instanciated')

    def n_basis(self, **kwargs):
        return int(self.get_anchors(**kwargs).shape[0])

    def n_basis_Jac(self, **kwargs):
        return self.n_basis(**kwargs) * self.n_dims(**kwargs)

    def n_args(self):
        return int(1)

    def n_dims(self, **kwargs):
        return self.get_anchors(**kwargs).shape[1]

    def get_anchors(self, **kwargs):
        if not hasattr(self, 'anchors_'):
            raise BaseException('No anchor(s) set.')
        try:
            if self.n_args() > 1:
                return self.anchors_[kwargs['idx']]
            else:
                return self.anchors_
        except KeyError:
                return self.anchors_

    def get_param(self, param_name):
        if not hasattr(self, 'params_'):
            raise BaseException('No parameter(s) set.')
        return self.params_[param_name]

    def set_param(self, param_name, val):
        if not hasattr(self, 'params_'):
            raise BaseException('No parameter(s) set.')
        self.params_[param_name] = val
        return self

    def Gram(self, suffix=None):
        if suffix is None:
            with tf.variable_scope('Gram'):
                return self.__call__()
        else:
            with tf.variable_scope('Gram_' + suffix):
                return self.__call__()


class Even(Kernel):

    def __init__(self, kernel):
        self.kernel = kernel

    @scope('kernel_symmetry')
    def __call__(self, X):
        X_ = default(X, self.get_anchors())
        return self.kernel(X_) + self.kernel(-X_)

    @scope('kernel_symmetry_Jacobian')
    def Jac(self, X):
        X_ = default(X, self.get_anchors())
        kmat = self.kernel.Jac(X) - self.kernel.Jac(-X_)
        variable_summaries('kernel_matrix', kmat)
        tf.summary.image('kernel_matrix',
                         tf.reshape(kmat,
                                    [1, kmat.shape[0], kmat.shape[1], 1]))
        return kmat

    def set_anchors(self, anchors):
        self.kernel.set_anchors(anchors)
        self.anchors_ = anchors
        return self

    def n_args(self):
        return int(self.kernel.n_args)


class Map(Kernel):

    def __init__(self, kernel, fun, fun_grad=None):
        self.kernel = kernel
        self.fun = fun
        self.fun_grad = fun_grad

    @scope('kernel_map')
    def __call__(self, X):
        X_ = default(X, self.get_anchors())
        kmat = self.kernel(self.fun(X_))
        variable_summaries('kernel_matrix', kmat)
        tf.summary.image('kernel_matrix',
                         tf.reshape(kmat,
                                    [1, kmat.shape[0], kmat.shape[1], 1]))
        return kmat

    @scope('kernel_map_Jacobian')
    def Jac(self, X):
        X_ = default(X, self.get_anchors())
        if self.fun_grad is None:
            raise NotImplementedError('Jacbobian for projections is not '
                                      'implemented because argument '
                                      '\'fun_grad\' was set to \'None\'')
        Jac_3d = self.kernel.Jac(X).reshape([X_.shape[0], X_.shape[1],
                                            self.get_anchors().shape[0]])
        return self.fun_grad(Jac_3d).reshape(
            X_.shape[0] * X_.shape[1], self.get_anchors().shape[0])

    def set_anchors(self, anchors):
        self.kernel.set_anchors(self.fun(anchors))
        self.anchors_ = anchors
        return self


class Projection(Map):

    def __init__(self, kernel, dimensions):

        def gather(X):
            return tf.reshape(tf.gather(X, dimensions, axis=1),
                              [X.reshape[0], -1])

        super(Projection, self).__init__(kernel, gather)
        self.dimensions = dimensions


class Periodic(Kernel):

    def __init__(self, kernel, period=None, period_trainable=False):
        self.kernel = kernel
        self.period = period
        self.period_trainable = period_trainable

    @scope('kernel_periodic')
    def __call__(self, X):
        kmat = self.kernel_periodic(X)
        variable_summaries('kernel_matrix', kmat)
        tf.summary.image('kernel_matrix',
                         tf.reshape(kmat,
                                    [1, kmat.shape[0], kmat.shape[1], 1]))
        return kmat

    @scope('kernel_periodic_Jacobian')
    def Jac(self, X):
        return self.kernel_periodic.Jac(X)

    def set_anchors(self, anchors):
        self.params_ = {}
        self.params_['period'] = default(self.period, 2 * np.pi)
        self.params_['period'] = tf.Variable(
            2 * np.pi / self.params_['period'],
            trainable=self.period_trainable,
            dtype=get_type(anchors))
        tf.add_to_collection("non_negative", self.params_['period'])
        self.kernel_sin = Map(self.kernel,
                              lambda X: tf.sin(self.params_['period'] * X))
        self.kernel_cos = Map(self.kernel,
                              lambda X: tf.cos(self.params_['period'] * X))
        self.kernel_periodic = self.kernel_sin * self.kernel_cos
        self.kernel_periodic.set_anchors(anchors)
        self.anchors_ = anchors
        return self


class KernelSum(Kernel):

    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    @scope('kernel_sum')
    def __call__(self, *args, **kwargs):
        if isinstance(self.kernel1, Kernel):
            lhs = self.kernel1
        else:
            lhs = Constant(self.kernel1, **kwargs)
        if isinstance(self.kernel2, Kernel):
            rhs = self.kernel2
        else:
            rhs = Constant(self.kernel2, **kwargs)
        kmat = lhs(*args, **kwargs) + rhs(*args, **kwargs)
        return kmat

    @scope('kernel_sum_Jacobian')
    def Jac(self, *args, comp_d=[1], **kwargs):
        if isinstance(self.kernel1, Kernel):
            lhs = self.kernel1
        else:
            lhs = Constant(self.kernel1, **kwargs)
        if isinstance(self.kernel2, Kernel):
            rhs = self.kernel2
        else:
            rhs = Constant(self.kernel2, **kwargs)
        return (lhs.Jac(*args, comp_d=comp_d, **kwargs) +
                rhs.Jac(*args, comp_d=comp_d, **kwargs))

    def set_anchors(self, *args):
        self.kernel1.set_anchors(*args)
        self.kernel2.set_anchors(*args)
        self.anchors_ = list(args)
        return self

    def n_basis(self, **kwargs):
        return int(self.kernel1.n_basis())

    def n_args(self):
        return len(self.get_anchors())


class KernelProduct(Kernel):

    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    @scope('kernel_product')
    def __call__(self, X):
        if isinstance(self.kernel1, Kernel):
            lhs = self.kernel1
        else:
            lhs = Constant(self.kernel1)
        if isinstance(self.kernel2, Kernel):
            rhs = self.kernel2
        else:
            rhs = Constant(self.kernel2)
        kmat = lhs(X) * rhs(X)
        variable_summaries('kernel_matrix', kmat)
        tf.summary.image('kernel_matrix',
                         tf.reshape(kmat,
                                    [1, kmat.shape[0], kmat.shape[1], 1]))
        return kmat

    @scope('kernel_product_Jacobian')
    def Jac(self, X):
        if isinstance(self.kernel1, Kernel):
            lhs = self.kernel1(X)
        else:
            lhs = Constant(self.kernel1)
        if isinstance(self.kernel2, Kernel):
            rhs = self.kernel2(X)
        else:
            rhs = Constant(self.kernel2)
        X_ = default(X, self.get_anchors())

        Jac_lhs_3d = lhs.Jac(X).reshape([X_.shape[0], X_.shape[1],
                                         self.get_anchors().shape[0]])
        Jac_rhs_3d = rhs.Jac(X).reshape([X_.shape[0], X_.shape[1],
                                         self.get_anchors().shape[0]])
        return (Jac_lhs_3d * lhs + Jac_rhs_3d * rhs).reshape(
            X_.shape[0] * X_.shape[1], self.get_anchors().shape[0])

    def set_anchors(self, anchors):
        self.kernel1.set_anchors(anchors)
        self.kernel2.set_anchors(anchors)
        self.anchors_ = anchors
        return self


class Gaussian(Kernel):

    def __init__(self, gamma=None, scale=None,
                 gamma_trainable=False,
                 scale_trainable=False):
        self.gamma = gamma
        self.scale = scale
        self.gamma_trainable = gamma_trainable
        self.scale_trainable = scale_trainable
        super(Gaussian, self).__init__()

    @scope('Gaussian_kernel')
    def __call__(self, X=None, **kwargs):
        dt = tf_type(get_env_precision())
        X_ = default(X, self.get_anchors())
        kmat = (self.get_param('scale') *
                tf.exp(-self.get_param('gamma') *
                       _fast_pairwise_euclidean(X_, self.get_anchors())))
        variable_summaries('kernel_matrix', kmat)
        tf.summary.image('kernel_matrix',
                         tf.reshape(kmat,
                                    [1, kmat.shape[0], kmat.shape[1], 1]))
        return kmat

    @scope('Gaussian_Jacobian')
    def Jac(self, X=None, side='l', **kwargs):
        dt = tf_type(get_env_precision())
        X_ = default(X, self.get_anchors())
        if side == 'l':
            Jac = (-2 * self.get_param('gamma') *
                   (tf.expand_dims(tf.cast(X_, dt), 1) -
                    tf.expand_dims(tf.cast(self.get_anchors(), dt), 0)) *
                   tf.expand_dims(self(X), 2))
            return tf.reshape(tf.transpose(Jac, [0, 2, 1]),
                              [X_.shape[0] * X_.shape[1],
                               self.n_basis()])
        elif side == 'r':
            Jac = (2 * self.get_param('gamma') *
                   (tf.expand_dims(tf.cast(X_, dt), 1) -
                    tf.expand_dims(tf.cast(self.get_anchors(), dt), 0)) *
                   tf.expand_dims(self(X), 2))
            return tf.reshape(tf.transpose(Jac, [0, 2, 1]),
                              [X_.shape[0] * X_.shape[1],
                               self.n_basis()])
        else:
            raise ValueError('side must be \'l\' or \'r\'')

    @scope('Gaussian_Hessian')
    def Hess(self, X, **kwargs):
        dt = tf_type(get_env_precision())
        X_ = default(X, self.get_anchors())
        delta = tf.expand_dims((tf.expand_dims(X_, 1) -
                                 tf.expand_dims(self.get_anchors(), 0)), 3)
        delta = tf.cast(delta, dt)
        Hess = (2 * tf.cast(self.get_param('gamma'), dt) *
                (2 * tf.cast(self.get_param('gamma'), dt) *
                 delta * tf.transpose(delta, [0, 1, 3, 2]) -
                 tf.eye(self.n_dims(), batch_shape=[X_.shape[0],
                                                    X_.shape[0]],
                        dtype=dt)) *
               tf.cast(tf.expand_dims(tf.expand_dims(self(X), 2), 3), dt))
        return tf.reshape(tf.transpose(-Hess, [0, 2, 1, 3]),
                          [X_.shape[0] * X_.shape[1],
                           self.n_basis() * self.n_dims()])

    def set_anchors(self, anchors):
        dt = tf_type(get_env_precision())
        self.anchors_ = anchors
        self.params_ = {}
        self.params_['gamma'] = default(self.gamma, 1)
        self.params_['gamma'] = tf.Variable(self.params_['gamma'],
                                            trainable=self.gamma_trainable,
                                            dtype=dt)
        tf.add_to_collection("non_negative", self.params_['gamma'])
        self.params_['scale'] = default(self.scale, 1.)
        self.params_['scale'] = tf.Variable(self.params_['scale'],
                                            trainable=self.scale_trainable,
                                            dtype=dt)
        tf.add_to_collection("scales", self.params_['scale'])
        tf.add_to_collection("non_negative", self.params_['scale'])
        return self


class Laplacian(Kernel):

    def __init__(self, gamma=None, scale=None,
                 gamma_trainable=False, scale_trainable=False):
        self.gamma = gamma
        self.scale = scale
        self.gamma_trainable = gamma_trainable
        self.scale_trainable = scale_trainable
        super(Laplacian, self).__init__()

    @scope('Laplacian_kernel')
    def __call__(self, X=None, **kwargs):
        X_ = default(X, self.get_anchors())
        kmat = (self.get_param('scale') *
                tf.exp(-self.get_param('gamma') *
                       _pairwise_manhattan(X_, self.get_anchors())))
        variable_summaries('kernel_matrix', kmat)
        tf.summary.image('kernel_matrix',
                         tf.reshape(kmat,
                                    [1, kmat.shape[0], kmat.shape[1], 1]))
        return kmat

    @scope('Laplacian_Jacobian')
    def Jac(self, X=None, side='l', **kwargs):
        if side == 'l':
            X_ = default(X, self.get_anchors())
            Jac = (-self.get_param('gamma') *
                   tf.sign(tf.expand_dims(X_, 1) -
                           tf.expand_dims(self.get_anchors(), 0)) *
                   tf.expand_dims(self(X), 2))
            return tf.reshape(tf.transpose(Jac, [0, 2, 1]),
                              [X_.shape[0] * X_.shape[1],
                               self.get_anchors().shape[0]])
        if side == 'r':
            return -tf.transpose(self.Jac(X, 'l', **kwargs))
        else:
            raise ValueError('side must be \'l\' or \'r\'')

    @scope('Laplacian_Hessian')
    def Hess(self, X, **kwargs):
        raise ValueError('Laplacian kernel is not twice differentiable')

    def set_anchors(self, anchors):
        self.anchors_ = anchors
        self.params_ = {}
        self.params_['gamma'] = default(self.gamma, 1.)
        self.params_['gamma'] = tf.Variable(self.params_['gamma'],
                                            trainable=self.gamma_trainable,
                                            dtype=get_type(anchors))
        tf.add_to_collection("non_negative", self.params_['gamma'])
        self.params_['scale'] = default(self.scale, 1.)
        self.params_['scale'] = tf.Variable(self.params_['scale'],
                                            trainable=self.scale_trainable,
                                            dtype=get_type(anchors),
                                            name='scale')
        tf.add_to_collection("scales", self.params_['scale'])
        tf.add_to_collection("non_negative", self.params_['scale'])
        return self


class ExponentiatedChi2(Kernel):

    def __init__(self, gamma=None, scale=None, eps=None,
                 gamma_trainable=False, scale_trainable=False):
        self.gamma = gamma
        self.scale = scale
        self.eps = eps
        self.gamma_trainable = gamma_trainable
        self.scale_trainable = scale_trainable
        super(ExponentiatedChi2, self).__init__()

    @scope('Exponentitated_Chi2_kernel')
    def __call__(self, X=None):
        X_ = default(X, self.get_anchors())
        kmat = (self.get_param('scale') *
                tf.exp(-self.get_param('gamma') *
                       _pairwise_chi2(X_, self.get_anchors(),
                                      self.get_param('eps'))))
        variable_summaries('kernel_matrix', kmat)
        tf.summary.image('kernel_matrix',
                         tf.reshape(kmat,
                                    [1, kmat.shape[0], kmat.shape[1], 1]))
        return kmat

    @scope('Exponentitated_Chi2_Jacobian')
    def Jac(self, X=None):
        X_ = default(X, self.get_anchors())
        diff1 = (tf.expand_dims(X_, 1) -
                 tf.expand_dims(self.get_anchors(), 0))
        diff3 = (tf.expand_dims(X_, 1) -
                 3 * tf.expand_dims(self.get_anchors(), 0))
        summ = (tf.expand_dims(X_, 1) +
                tf.expand_dims(self.get_anchors(), 0))
        Jac = (-self.get_param('gamma') * ((diff1 * diff3) / (summ * summ)) *
               tf.expand_dims(self(X), 2))
        return tf.reshape(tf.transpose(Jac, [0, 2, 1]),
                          [X_.shape[0] * X_.shape[1],
                           self.get_anchors().shape[0]])

    def set_anchors(self, anchors):
        self.anchors_ = anchors
        self.params_ = {}
        self.params_['gamma'] = default(self.gamma, 1.)
        self.params_['gamma'] = tf.Variable(self.params_['gamma'],
                                            trainable=self.gamma_trainable,
                                            dtype=get_type(anchors))
        tf.add_to_collection("non_negative", self.params_['gamma'])
        self.params_['scale'] = default(self.scale, 1.)
        self.params_['scale'] = tf.Variable(self.params_['scale'],
                                            trainable=self.scale_trainable,
                                            dtype=get_type(anchors),
                                            name='scale')
        self.params_['eps'] = default(self.eps,
                                      np.finfo(get_type(anchors)).eps)
        tf.add_to_collection("scales", self.params_['scale'])
        tf.add_to_collection("non_negative", self.params_['scale'])
        return self


class Constant(Kernel):

    def __init__(self, scale=None, scale_trainable=False):
        self.scale = scale
        self.scale_trainable = scale_trainable
        super(Constant, self).__init__()

    @scope('Constant_kernel')
    def __call__(self, X=None, **kwargs):
        X_ = default(X, self.get_anchors())
        dt = tf_type(get_env_precision())
        return tf.fill((self.n_basis(), self.n_basis()),
                       tf.cast(self.get_param('scale'), dt))

    @scope('Constant_Jacobian')
    def Jac(self, X=None, side='l', **kwargs):
        dt = tf_type(get_env_precision())
        if side == 'l':
            return linalg.LinearOperatorZeros(self.n_basis(),
                                              self.n_basis(),
                                              dtype=dt, is_square=True,
                                              is_self_adjoint=True,
                                              is_positive_definite=False,
                                              is_non_singular=False)
        elif side == 'r':
            return linalg.LinearOperatorZeros(self.n_basis(),
                                              self.n_basis_Jac(),
                                              dtype=dt, is_square=True,
                                              is_self_adjoint=True,
                                              is_positive_definite=False,
                                              is_non_singular=False)
        else:
            raise ValueError('side must be \'l\' or \'r\'')

    @scope('Constant_Hessian')
    def Hess(self, X, **kwargs):
        dt = tf_type(get_env_precision())
        return linalg.LinearOperatorZeros(self.n_basis_Jac(),
                                          self.n_basis_Jac(),
                                          dtype=dt, is_square=True,
                                          is_self_adjoint=True,
                                          is_positive_definite=False,
                                          is_non_singular=False)

    def n_basis(self, **kwargs):
        return int(1)

    def n_basis_Jac(self, **kwargs):
        return int(1)

    def set_anchors(self, anchors):
        self.anchors_ = anchors
        self.params_ = {}
        self.params_['scale'] = default(self.scale, 1.)
        self.params_['scale'] = tf.Variable(self.params_['scale'],
                                            trainable=self.scale_trainable,
                                            dtype=get_type(anchors),
                                            name='scale')
        tf.add_to_collection("scales", self.params_['scale'])
        tf.add_to_collection("non_negative", self.params_['scale'])
        return self


class Linear(Kernel):

    def __init__(self, shift=1, scale=None,
                 shift_trainable=False, scale_trainable=False):
        self.shift = shift
        self.scale = scale
        self.shift_trainable = shift_trainable
        self.scale_trainable = scale_trainable
        super(Linear, self).__init__()

    @scope('Linear_kernel')
    def __call__(self, X=None):
        X_ = default(X, self.get_anchors())
        return (self.get_param('scale') *
                (X_ @ tf.transpose(self.get_anchors()) +
                 self.get_param('shift')))

    @scope('Linear_Jacobian')
    def Jac(self, X=None):
        X_ = default(X, self.get_anchors())
        return (self.get_param('scale') *
                tf.reshape(tf.tile(tf.expand_dims(self.get_anchors(), 0),
                                   [X_.shape[0], 1, 1]),
                           [X_.shape[0] * X_.shape[1],
                            self.get_anchors().shape[0]]))

    def set_anchors(self, anchors):
        self.anchors_ = anchors
        self.params_ = {}
        self.params_['shift'] = default(self.shift, 1.)
        self.params_['shift'] = tf.Variable(self.params_['shift'],
                                            trainable=self.shift_trainable,
                                            dtype=get_type(anchors))
        tf.add_to_collection("non_negative", self.params_['shift'])
        self.params_['scale'] = default(self.scale, 1.)
        self.params_['scale'] = tf.Variable(self.params_['scale'],
                                            trainable=self.scale_trainable,
                                            dtype=get_type(anchors),
                                            name='scale')
        tf.add_to_collection("scales", self.params_['scale'])
        tf.add_to_collection("non_negative", self.params_['scale'])
        return self


class Impulse(Kernel):

    def __init__(self, scale=None, eps=None,
                 scale_trainable=False):
        self.scale = scale
        self.eps = eps
        self.scale_trainable = scale_trainable
        super(Impulse, self).__init__()

    @scope('Impulse_kernel')
    def __call__(self, X=None, **kwargs):
        X_ = default(X, self.get_anchors())
        if X is not self.get_anchors():
            kmat = (self.get_param('scale') *
                    _pairwise_equal(X_, self.get_anchors(),
                                    self.get_param('eps')))
            variable_summaries('kernel_matrix', kmat)
            tf.summary.image('kernel_matrix',
                             tf.reshape(kmat,
                                        [1, kmat.shape[0], kmat.shape[1], 1]))
        else:
            if tf.abs(self.get_param('scale') - 1) < self.get_param('eps'):
                kmat = tf.linalg.LinearOperatorIdentity(
                    self.get_anchors().shape[0])
            else:
                kmat = tf.linalg.LinearOperatorScaledIdentity(
                    self.get_anchors().shape[0], self.get_param('scale'))
        return kmat

    @scope('Impulse_Jacobian')
    def Jac(self, X=None, side='l', **kwargs):
        raise ValueError('Impulse kernel is not differentiable')

    @scope('Impulse_Jacobian')
    def Jac(self, X=None, **kwargs):
        raise ValueError('Impulse kernel is not twice differentiable')

    def set_anchors(self, anchors):
        self.anchors_ = anchors
        self.params_ = {}
        self.params_['scale'] = default(self.scale, 1.)
        self.params_['scale'] = tf.Variable(self.params_['scale'],
                                            trainable=self.scale_trainable,
                                            dtype=get_type(anchors),
                                            name='scale')
        self.params_['eps'] = default(self.scale,
                                      np.finfo(get_type(anchors)).eps)
        tf.add_to_collection("scales", self.params_['scale'])
        tf.add_to_collection("non_negative", self.params_['scale'])
        return self


class Intersection(Kernel):

    def __init__(self, scale=None, offset=None,
                 scale_trainable=False):
        self.scale = scale
        self.scale_trainable = scale_trainable
        self.offset = offset
        super(Intersection, self).__init__()

    @scope('Impulse_kernel')
    def __call__(self, X=None, **kwargs):
        X_ = default(X, self.get_anchors())
        dt = tf_type(get_env_precision())
        X_ = tf.concat([tf.where(X_ >= -self.offset,
                                 X_,
                                 tf.zeros(X_.shape, dtype=dt)),
                        tf.where(X_ < -self.offset,
                                 -X_,
                                 tf.zeros(X_.shape, dtype=dt))], axis=1)
        anchors = tf.concat([tf.where(self.get_anchors() >= -self.offset,
                                      self.get_anchors(),
                                      tf.zeros(self.get_anchors().shape,
                                               dtype=dt)),
                             tf.where(self.get_anchors() < -self.offset,
                                      -self.get_anchors(),
                                      tf.zeros(self.get_anchors().shape,
                                               dtype=dt))],
                             axis=1)
        kmat = self.get_param('scale') * _pairwise_min(X_ + self.offset,
                                                       anchors + self.offset)
        return kmat

    @scope('Impulse_Jacobian')
    def Jac(self, X=None, side='l', **kwargs):
        raise ValueError('Intersection kernel is not differentiable')

    @scope('Impulse_Jacobian')
    def Jac(self, X=None, **kwargs):
        raise ValueError('Intersection kernel is not twice differentiable')

    def set_anchors(self, anchors):
        self.anchors_ = anchors
        self.params_ = {}
        self.params_['offset'] = default(self.offset, 0.)
        self.params_['scale'] = default(self.scale, 1.)
        self.params_['scale'] = tf.Variable(self.params_['scale'],
                                            trainable=self.scale_trainable,
                                            dtype=get_type(anchors),
                                            name='scale')
        tf.add_to_collection("scales", self.params_['scale'])
        tf.add_to_collection("non_negative", self.params_['scale'])
        return self


class Polynomial(Kernel):

    def __init__(self, power=2, shift=1., scale=None,
                 power_trainable=False,
                 shift_trainable=False,
                 scale_trainable=False):
        self.power = power
        self.shift = shift
        self.scale = scale
        self.power_trainable = power_trainable
        self.shift_trainable = shift_trainable
        self.scale_trainable = scale_trainable
        super(Polynomial, self).__init__()

    @scope('Polynomial_kernel')
    def __call__(self, X=None):
        X_ = default(X, self.get_anchors())
        kmat = (self.get_param('scale') *
                (X_ @ tf.transpose(self.get_anchors()) +
                 self.get_param('shift')) ** self.get_param('power'))
        variable_summaries('kernel_matrix', kmat)
        tf.summary.image('kernel_matrix',
                         tf.reshape(kmat,
                                    [1, kmat.shape[0], kmat.shape[1], 1]))
        return kmat

    @scope('Polynomial_Jacobian')
    def Jac(self, X=None):
        X_ = default(X, self.get_anchors())
        if self.get_param('power') == 0:
            return np.full((X_.shape[0] * X_.shape[1],
                            self.get_anchors().shape[0]), 0)
        elif self.get_param('power') == 1:
            return (self.get_param('scale') *
                    tf.reshape(tf.tile(tf.expand_dims(self.get_anchors(), 0),
                                       [X_.shape[0], 1, 1]),
                               [X_.shape[0] * X_.shape[1],
                                self.get_anchors().shape[0]]))
        else:
            tmp = ((self.get_param('power') - 1) *
                   (self.get_param('scale') *
                   (X_ @ tf.transpose(self.get_anchors()) +
                    self.get_param('shift')) ** self.get_param('power')))
            return tf.reshape(tf.transpose(tf.expand_dims(tmp, 2) *
                                           tf.expand_dims(self.get_anchors(),
                                                          0),
                                           [0, 2, 1]),
                              [X_.shape[0] * X_.shape[1],
                               self.get_anchors().shape[0]])

    def set_anchors(self, anchors):
        self.anchors_ = anchors
        self.params_ = {}
        self.params_['power'] = max(default(self.power, 2), 0)
        self.params_['power'] = tf.Variable(self.params_['power'],
                                            trainable=self.power_trainable,
                                            dtype=get_type(anchors))
        self.params_['shift'] = default(self.shift, 0)
        self.params_['shift'] = tf.Variable(self.params_['shift'],
                                            trainable=self.shift_trainable,
                                            dtype=get_type(anchors))
        tf.add_to_collection("non_negative", self.params_['shift'])
        self.params_['scale'] = default(self.scale, 1)
        self.params_['scale'] = tf.Variable(self.params_['scale'],
                                            trainable=self.scale_trainable,
                                            dtype=get_type(anchors),
                                            name='scale')
        tf.add_to_collection("scales", self.params_['scale'])
        tf.add_to_collection("non_negative", self.params_['scale'])
        return self


class _KroneckerLinop(linalg.LinearOperatorKronecker):

    @scope('init_decomposable_kernel')
    def __init__(self, *args,
                 is_non_singular=None, is_positive_definite=None,
                 is_self_adjoint=None, is_square=None,
                 name=None, shape_o=None):

        ops = []
        arg_list = list(*args)
        dt = tf_type(get_env_precision())
        for i, op in enumerate(arg_list):
            if isinstance(op, int):
                ops.append(linalg.LinearOperatorIdentity(op, dtype=dt))
            elif isinstance(op, linalg.LinearOperator):
                ops.append(op)
            else:
                ops.append(linalg.LinearOperatorFullMatrix(
                    tf.cast(op, dtype=dt)))
        self.shape_o = shape_o
        super(_KroneckerLinop, self).__init__(ops,
                                              is_non_singular,
                                              is_positive_definite,
                                              is_self_adjoint,
                                              is_square,
                                              name)

    def __matmul__(self, other):
        return tf.reshape(self._matmul(other), self.shape_o)

    def __rmatmul__(self, other):
        return tf.transpose(tf.reshape(self._matmul(other,
                                                    adjoint=True,
                                                    adjoint_arg=True),
                                       self.shape_o))


class Decomposable(Kernel):

    def __init__(self, *args):
        self.kernels = list(args)

    @scope('Decomposable')
    def __call__(self, *args, **kwargs):
        X = list(args) + list([None] * (len(self.kernels) - len(args)))
        try:
            comp_p = kwargs['comp_p']
        except KeyError:
            comp_p = np.arange(len(X))
        shape = []
        for i in range(len(X)):
            if self.kernels[i].n_basis() == 1:
                shape += [1]
            elif X[i] is not None:
                shape += [X[i].shape[0]]
            else:
                shape += [self.kernels[i].n_basis()]
        kernels = []
        for (i, ker) in enumerate(self.kernels):
            if i in comp_p:
                kernels += [ker(X[i], **kwargs)]
            else:
                kernels += [self.kernels[i].n_basis()]
        return _KroneckerLinop(kernels, shape_o=shape)

    @scope('Decomposable_Jacobian')
    def Jac(self, *args, **kwargs):
        X = list(args) + list([None] * (len(self.kernels) - len(args)))
        try:
            comp_d = kwargs['comp_d']
        except KeyError:
            comp_d = np.arange(len(X))
        try:
            comp_p = kwargs['comp_p']
        except KeyError:
            comp_p = np.arange(len(X))
        shape = []
        for i in range(len(X)):
            aug = (kwargs['side'] == 'l') and (i in comp_d) and (i in comp_p)
            if self.kernels[i].n_basis() == 1:
                shape += [1]
            elif X[i] is not None:
                shape += [X[i].shape[0] * (X[i].shape[1] if aug else 1)]
            else:
                shape += [self.kernels[i].n_basis() *
                          (self.n_dims(idx=i) if aug else 1)]
        kernels = []
        for (i, ker) in enumerate(self.kernels):
            if( i in comp_d) and (i in comp_p):
                kernels += [ker.Jac(X[i], **kwargs)]
            elif i in comp_p:
                kernels += [ker(X[i], **kwargs)]
            else:
                kernels += [self.kernels[i].n_basis()]
        return _KroneckerLinop(kernels, shape_o=shape)

    @scope('Decomposable_Hessian')
    def Hess(self, *args, **kwargs):
        X = list(args) + list([None] * (len(self.kernels) - len(args)))
        try:
            comp_d = kwargs['comp_d']
        except KeyError:
            comp_d = np.arange(len(X))
        try:
            comp_p = kwargs['comp_p']
        except KeyError:
            comp_p = np.arange(len(X))
        shape = []
        for i in range(len(X)):
            aug = (i in comp_d) and (i in comp_p)
            if self.kernels[i].n_basis() == 1:
                shape += [1]
            elif X[i] is not None:
                shape += [X[i].shape[0] * (X[i].shape[1] if aug else 1)]
            else:
                shape += [self.kernels[i].n_basis() *
                          (self.n_dims(idx=i) if aug else 1)]
        kernels = []
        for (i, ker) in enumerate(self.kernels):
            if (i in comp_d) and (i in comp_p):
                kernels += [ker.Hess(X[i], **kwargs)]
            elif i in comp_p:
                kernels += [ker(X[i], **kwargs)]
            else:
                kernels += [self.kernels[i].n_basis()]
        return _KroneckerLinop(kernels, shape_o=shape)

    def n_basis(self, **kwargs):
        r = 1
        for i in range(len(self.kernels)):
            r *= self.kernels[i].n_basis(**kwargs)
        return int(r)

    def n_basis_Jac(self, **kwargs):
        if len(kwargs) == 0:
            comp = np.arange(self.n_args())
        else:
            comp = kwargs['comp_d']
        r = 1
        for i in range(len(self.kernels)):
            if i in comp:
                r *= self.kernels[i].n_basis_Jac(**kwargs)
            else:
                r *= self.kernels[i].n_basis(**kwargs)
        return int(r)

    def n_args(self):
        return int(len(self.kernels))

    def set_anchors(self, *args):
        self.anchors_ = list(args)
        for i, anchors in enumerate(self.anchors_):
            self.kernels[i].set_anchors(anchors)
        return self


KERNEL_LIST = {
    'Gaussian': Gaussian,
    'Laplacian': Laplacian,
    'linear': Linear,
    'polynomial': Polynomial,
    'constant': Constant,
    'impulse': Impulse,
    'exponentitatedChi2': ExponentiatedChi2
}
