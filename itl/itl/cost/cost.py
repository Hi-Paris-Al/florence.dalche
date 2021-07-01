from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from math import isclose

from ..utils import quadratic_infimal_convolution, Problem, data_array


__all__ = ['SquaredL2',
           'L1', 'HuberL1',
           'HuberHinge', 'Hinge', 'SquaredHinge',
           'HuberPinball', 'Pinball', 'SquaredPinball',
           'ploss', 'closs']


def ploss(y_true, y_pred, probs):
    # pylint: disable=E1101
    """Compute the pinball loss.
    Parameters
    ----------
    pred : {array-like}, shape = [n_quantiles, n_samples] or [n_samples]
        Predictions.
    y : {array-like}, shape = [n_samples]
        Targets.
    Returns
    -------
    l : {array}, shape = [n_quantiles]
        Average loss for each quantile level.
    """
    probs = np.asarray(probs).reshape(-1)
    residual = y_true - y_pred
    loss = np.sum([np.fmax(prob * res, (prob - 1) * res) for (res, prob) in
                   zip(residual.T, probs)], axis=1)
    return loss / y_true.shape[0]


def closs(y_true, y_pred, probs):
    # pylint: disable=E1101
    """Compute the crossing loss.
    Parameters
    ----------
    pred : {array-like}, shape = [n_quantiles, n_samples] or [n_samples]
        Predictions.
    y : {array-like}, shape = [n_samples]
        Targets.
    Returns
    -------
    l : {array}, shape = [n_quantiles]
        Average loss for each quantile level.
    """
    probs = np.asarray(probs).reshape(-1)
    residual = y_true - y_pred
    loss = np.sum([np.fmax(prob * res, (prob - 1) * res) for (res, prob) in
                   zip(residual.T, probs)], axis=1)
    return loss / y_true.shape[0]


class Cost(object):

    def __init__(self, kappa, problem):
        self.problem = problem
        self.kappa = kappa

    def __call__(self, true, pred):
        if not hasattr(self, 'kappa'):
            raise NotImplementedError('A class attribute kappa must be '
                                      'provided for infimal convolution')
        with tf.name_scope('Cost'):
            cost = quadratic_infimal_convolution(
                self.signature(self._residual(true, pred, self.problem)),
                self.kappa)
        return cost

    def construct(self):
        pass

    def signature(self, res):
        raise NotImplementedError('Abstract class \'Cost\' '
                                  'cannot be instanciated')

    @staticmethod
    def _residual(true, pred, problem):
        if problem == Problem.Regression:
            return tf.subtract(true, pred, name='diff_residual')
        elif problem == Problem.Classification:
            if isinstance(true, (float, int)) and true == 1:
                return pred
            else:
                return tf.multiply(true, pred, name='mult_residual')
        else:
            NotImplementedError('Unsupported problem')


class SquaredL2(Cost):

    def __init__(self, problem=Problem.Regression):
        super(L2_squared, self).__init__(kappa=0, problem=problem)

    def signature(self, res):
        val = .5 * res * res
        return val


class HuberL1(Cost):

    def __init__(self, kappa=.5, problem=Problem.Regression):
        super(HuberL1, self).__init__(kappa=kappa, problem=problem)

    def signature(self, res):
        return tf.abs(res)


class L1(HuberL1):

    def __init__(self, problem=Problem.Regression):
        super(L1, self).__init__(kappa=0, problem=problem)


class HuberHinge(Cost):

    def __init__(self, kappa=.5, problem=Problem.Classification):
        super(HuberHinge, self).__init__(kappa=kappa, problem=problem)

    def construct(self, margin=1.):
        super(HuberHinge, self).construct()
        self.margin_ = data_array(margin)

    def get_margin(self):
        if not hasattr(self, 'margin_'):
            raise BaseException('Cost must be constructed first')
        return self.margin_

    def signature(self, res):
        return tf.maximum(self.get_margin() - res, 0)


class Hinge(HuberHinge):

    def __init__(self, problem=Problem.Classification):
        super(Hinge, self).__init__(kappa=0, problem=problem)


class SquaredHinge(Hinge):

    def __init__(self, problem=Problem.Classification):
        super(SquaredHinge, self).__init__(problem=problem)

    def signature(self, res):
        return super(SquaredHinge, self).signature(res) ** 2


class HuberPinball(Cost):

    def __init__(self, kappa=.5, problem=Problem.Regression):
        super(HuberPinball, self).__init__(kappa=kappa, problem=problem)

    def construct(self, tilt=.5):
        super(HuberPinball, self).construct()
        self.tilt_ = data_array(tilt)

    def get_tilt(self):
        if not hasattr(self, 'tilt_'):
            raise BaseException('Cost must be constructed first')
        return self.tilt_

    def signature(self, res):
        return tf.maximum(tf.reshape(self.get_tilt(), [1, -1]) * res,
                          (tf.reshape(self.get_tilt(), [1, -1]) - 1) * res)


class Pinball(HuberPinball):

    def __init__(self, problem=Problem.Regression):
        super(Pinball, self).__init__(kappa=0, problem=problem)


class SquaredPinball(Pinball):

    def __init__(self, problem=Problem.Regression):
        super(SquaredPinball, self).__init__(problem=problem)

    def signature(self, res):
        return super(SquaredPinball, self).signature(res) ** 2
