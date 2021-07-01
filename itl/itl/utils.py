"""Utility functions and classes for ITL."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from math import isclose
from enum import Enum, unique, auto

__all__ = ['Problem', 'quadratic_infimal_convolution', 'default',
           'data_array', 'get_type', 'tf_type', 'np_type',
           'variable_summaries']


def default(val, alt):
    """Set a default value to a variable.

    Parameters
    ----------
    val : Any
        The variable to which to set a default value.

    alt : Any
        The default value.

    Returns
    -------
    res : Any
        res is set to alt if val is 'None', otherwise res is set to val.

    """
    if val is None:
        return alt
    else:
        return val


def get_env_precision():
    """Get the environment precision.

    Returns
    -------
    precision : str
        returns the environment variable 'ITL_PRECISION'. If the environment
        variable doesn't exists return 'fp32'.

    Raises
    ------
    NotImplementedError
        The environment variable 'ITL_PRECISION' is set to something different
        from 'fp32' or 'fp64'.

    """
    precision = default(os.environ.get('ITL_PRECISION'), 'fp32')
    if not ((precision == 'fp32') or (precision == 'fp64')):
        raise NotImplementedError('Only Floating point precision \'fp32\' or '
                                  '\'fp64\' are supported. {} is unsupported.'
                                  .format(precision))
    return precision


def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor.

    Parameters
    ----------
    name : str
        The name of the summary scope.

    var : tf.Tensor
        The tensor to which to attach the summary.

    Returns
    -------
    var : tf.Tensor
        The (unmodified) tensor passed in the arguments.
    """
    precision = get_env_precision()
    if precision == 'fp32':
        var = tf.cast(var, tf.float32)
    elif precision == 'fp64':
        var == tf.cast(var, tf.float64)
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)
    return var


@unique
class Problem(Enum):
    """An enumeration containing the two kind of problems currently supported
    by ITL: Regression and Classification. This help to construct the residuals
    in the class Cost.

    See Also
    --------
    Cost
    """

    Regression = auto()
    Classification = auto()


def default(val, alt):
    if val is None:
        return alt
    else:
        return val


def tf_type(fp):
    """Convert a string to a tensorflow type.

    Parameters
    ----------
    fp : str
        The precision string with value 'fp32' or 'fp64'.

    Returns
    -------
    t : type
        The tensorflow type corresponding to the 'fp32' or 'fp64' precision.

    Raises
    ------
    NotImplementedError
        fp is not 'fp32' or 'fp64'.
    """
    if fp == 'fp32':
        return tf.float32
    elif fp == 'fp64':
        return tf.float64
    else:
        raise NotImplementedError('Only Floating point precision \'fp32\' or '
                                  '\'fp64\' are supported')


def np_type(fp):
    """Convert a string to a numpy type.

    Parameters
    ----------
    fp : str
        The precision string with value 'fp32' or 'fp64'.

    Returns
    -------
    t : type
        The numpy type corresponding to the 'fp32' or 'fp64' precision.

    Raises
    ------
    NotImplementedError
        fp is not 'fp32' or 'fp64'.
    """
    if fp == 'fp32':
        return np.float32
    elif fp == 'fp64':
        return np.float64
    else:
        raise NotImplementedError('Only Floating point precision \'fp32\' or '
                                  '\'fp64\' are supported')


def get_type(array):
    """Returns the data type associated to a numpy or a tensorflow array.

    Parameters
    ----------
    array : {np.ndarray, tf.Tensor}
        The input array from which extract the dtype. It can be either a numpy
        or a Tensorflow array.

    Returns
    -------
    t : type
        the data type of 'array'
    """
    if isinstance(array, np.ndarray):
        return array.dtype.type
    else:
        return array.dtype


def data_array(array):
    """Convert any type of data to a type supported by ITL.

    Parameters
    ----------
    array : Any
        The input data, can be any (supported) type.

    Returns
    -------
    array : {np.ndarray}
        A two dimensional array containing the data and compatible with ITL.

    Raises
    ------
    BaseException
        The conversion to an ITL compatible type is impossible.
    """
    precision = get_env_precision()
    dtype = np_type(precision)
    if array is None:
        return None

    if isinstance(array, tf.Tensor):
        if len(array.shape) == 0:
            return tf.reshape(tf.cast(array, tf_type(precision)), [1, 1])
        else:
            return tf.reshape(tf.cast(array, tf_type(precision)),
                              [array.shape[0], -1])
    elif isinstance(array, (float, int)):
        N = 1
        d = 1
    elif isinstance(array, (list, tuple)):
        N = len(array)
        d = 1
    else:
        if hasattr(array, 'shape'):
            shape = array.shape
        else:
            raise BaseException('Invalid array conversion')
        if len(shape) == 0:
            N = 1
            d = 1
        elif len(shape) == 1:
            N = shape[0]
            d = 1
        elif len(shape) == 2:
            N = shape[0]
            d = shape[1]
        else:
            N = shape[0]
            d = np.prod(shape[1:])
    return np.asarray(array, dtype=dtype).reshape(N, d)


def quadratic_infimal_convolution(ex_val, kappa):
    if isclose(kappa, 0.):
        val = ex_val
    else:
        quadratic = tf.minimum(ex_val, kappa)
        linear = (ex_val - quadratic)
        val = quadratic * quadratic / kappa + linear
    return val
