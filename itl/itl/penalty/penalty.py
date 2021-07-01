from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from ..utils import quadratic_infimal_convolution


__all__ = ['HuberHinge', 'Hinge', 'SquaredHinge']


class Penalty(object):

    def __init__(self, kappa):
        self.kappa = kappa

    def __call__(self, X=None, t=None):
        if not hasattr(self, 'kappa'):
            raise NotImplementedError('A class attribute kappa must be '
                                      'provided for infimal convolution')
        val = quadratic_infimal_convolution(
            self.signature(self.get_model().Jac(X, t,
                                                comp_p=[1],
                                                comp_d=[1])), self.kappa)
        return val


    def construct(self, model, sign):
        self.model_ = model
        self.sign_ = sign

    def get_model(self):
        if not hasattr(self, 'model_'):
            raise BaseException('Penalty must be constructed first')
        return self.model_

    def signature(self, Jac):
        raise('Abstract class cannot be instanciated')


class HuberHinge(Penalty):

    def __init__(self, kappa=.5):
        super(HuberHinge, self).__init__(kappa)

    def construct(self, model, margin=0, sign=1):
        super(HuberHinge, self).construct(model, sign)
        self.margin_ = margin

    def get_margin(self):
        if not hasattr(self, 'margin_'):
            raise BaseException('Penalty must be constructed first')
        return self.margin_

    def signature(self, Jac):
        if (isinstance(self.get_margin(), (int, float)) and
            np.isclose(self.get_margin(), 0)):
            if (isinstance(self.sign_, (int, float)) and
                np.isclose(self.sign_, 1)):
                return -tf.minimum(Jac, 0)
            else:
                return -tf.minimum(Jac * self.sign_, 0)
        else:
            if (isinstance(self.sign_, (int, float)) and
                np.isclose(self.sign_, 1)):
                return -tf.minimum((Jac - self.get_margin()), 0)
            else:
                return -tf.minimum((Jac - self.get_margin()) * self.sign_, 0)


class Hinge(HuberHinge):

    def __init__(self):
        super(Hinge, self).__init__(kappa=0)


class SquaredHinge(Hinge):

    def __init__(self):
        super(SquaredHinge, self).__init__()

    def signature(self, Jac):
        return .5 * super(SquaredHinge, self).signature(Jac) ** 2

