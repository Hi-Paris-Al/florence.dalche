from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .utils import default, data_array


__all__ = ['Risk']


class Risk(object):

    def __init__(self, X):
        self.X = X

    def __call__(self):
        raise NotImplementedError('Abstract class \'Risk\' '
                                  'cannot be instanciated')

    def construct(self, anchors_weights_pair, weights_X):
        self.anchors_t_ = data_array(anchors_weights_pair[0])
        self.weights_t_ = default(anchors_weights_pair[1],
                                  1. / anchors_weights_pair[0].shape[0])
        self.weights_t_ = data_array(self.weights_t_)
        self.weights_X_ = default(weights_X, 1. / self.X.shape[0])
        self.weights_X_ = data_array(self.weights_X_)

    def _fast_reduce(self, ex_val):
        with tf.variable_scope('fast_reduce'):
            if self.get_weights_X().shape[1] == 1:
                if (max(self.get_weights_t().shape[0],
                        self.get_weights_t().shape[1]) == 1):
                    reduce_t = (tf.reduce_sum(ex_val, 1, keepdims=True) *
                                self.get_weights_t())
                else:
                    reduce_t = tf.reduce_sum(ex_val *
                         tf.transpose(self.get_weights_t()), 1,
                         keepdims=True)
                if self.get_weights_X().shape[0] == 1:
                    val = (self.get_weights_X() *
                           tf.reduce_sum(reduce_t, 0, keepdims=True))
                else:
                    val = tf.reduce_sum(self.get_weights_X() * reduce_t, 0,
                                        keep_dims=True)
            else:
                if self.get_weights_X().shape[0] == 1:
                    reduce_x = (self.get_weights_X() *
                                tf.reduce_sum(ex_val, 0, keepdims=True))
                else:
                    reduce_x = tf.reduce_sum(self.get_weights_X() * ex_val, 0,
                                             keep_dims=True)
                if (max(self.get_weights_t().shape[0],
                        self.get_weights_t().shape[1]) == 1):
                    val = (tf.reduce_sum(reduce_x, 1, keepdims=True) *
                           self.get_weights_t())
                else:
                    val = tf.reduce_sum(reduce_x *
                                        tf.reshape(self.get_weights_t(),
                                                   [1, -1]), 1,
                                        keepdims=True)

        return tf.reshape(val, [])

    def signature(self):
        raise NotImplementedError('Abstract class \'Risk\' '
                                  'cannot be instanciated')

    def get_anchors_t(self):
        if not hasattr(self, 'anchors_t_'):
            raise BaseException('Risk must be constructed.')
        return self.anchors_t_

    def get_weights_t(self):
        if not hasattr(self, 'weights_t_'):
            raise BaseException('Risk must be constructed first')
        return self.weights_t_

    def get_weights_X(self):
        if not hasattr(self, 'weights_t_'):
            raise BaseException('Risk must be constructed first')
        return self.weights_X_

    def set_weight_X(self, val):
        if not hasattr(self, 'weights_X_'):
            raise BaseException('Risk must be constructed first')
        self.weights_X_ = val
        return self

