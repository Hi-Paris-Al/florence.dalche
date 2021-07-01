""" Test itl module utils. """

import os
import pytest

import itl.utils as utils
import tensorflow as tf
import numpy as np


def test_default_1():
    val = 1
    assert(utils.default(val, None) == val)


def test_default_None():
    val = None
    assert(utils.default(val, 1) is not None)


def test_get_env_precision_float32():
    os.environ['ITL_PRECISION'] = 'fp32'
    precision = utils.get_env_precision()
    del os.environ['ITL_PRECISION']
    assert(precision == 'fp32')


def test_get_env_precision_float64():
    os.environ['ITL_PRECISION'] = 'fp64'
    precision = utils.get_env_precision()
    del os.environ['ITL_PRECISION']
    assert(precision == 'fp64')


def test_get_env_precision_failure():
    os.environ['ITL_PRECISION'] = 'int32'
    with pytest.raises(NotImplementedError):
        precision = utils.get_env_precision()
    del os.environ['ITL_PRECISION']

def test_variable_summaries_int():
    dummy = tf.constant(1)
    utils.variable_summaries("dummy", dummy)
    with tf.Session() as sess:
        test = sess.run(dummy)
    assert(test == 1)


def test_variable_summaries_float32():
    os.environ['ITL_PRECISION'] = 'fp32'
    dummy = tf.Variable(tf.zeros(shape=[2, 3], dtype=tf.float32),
                        trainable=True, name='dummy')
    utils.variable_summaries("dummy", dummy)
    tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test = sess.run(dummy)
    del os.environ['ITL_PRECISION']
    assert(np.allclose(test, np.zeros((2, 3))))


def test_variable_summaries_float64():
    os.environ['ITL_PRECISION'] = 'fp64'
    dummy = tf.Variable(tf.zeros(shape=[2, 3], dtype=tf.float64),
                        trainable=True, name='dummy')
    utils.variable_summaries("dummy", dummy)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test = sess.run(dummy)
    del os.environ['ITL_PRECISION']
    assert(np.allclose(test, np.zeros((2, 3))))


def test_Problem_Regression():
    pb = utils.Problem
    assert(hasattr(pb, 'Regression'))


def test_Problem_Classification():
    pb = utils.Problem
    assert(hasattr(pb, 'Classification'))


def test_tf_type_float32():
    fp = 'fp32'
    assert(utils.tf_type(fp) is tf.float32)


def test_tf_type_float64():
    fp = 'fp64'
    assert(utils.tf_type(fp) is tf.float64)


def test_tf_type_int32():
    fp = 'int64'
    with pytest.raises(NotImplementedError):
        dummy = utils.tf_type(fp)


def test_np_type_float32():
    fp = 'fp32'
    assert(utils.np_type(fp) is np.float32)


def test_np_type_float64():
    fp = 'fp64'
    assert(utils.np_type(fp) is np.float64)


def test_np_type_int32():
    fp = 'int64'
    with pytest.raises(NotImplementedError):
        dummy = utils.np_type(fp)


def test_get_type_numpy():
    dummy = np.zeros([], dtype=np.float32)
    assert(utils.get_type(dummy) is np.float32)


def test_get_type_tensorflow():
    dummy = tf.zeros([], dtype=np.float32)
    assert(utils.get_type(dummy) is tf.float32)


@pytest.mark.parametrize("array",
                         [1, 0., [1.], (0, 1.), [[1.], [1]], ([0.], [1]),
                          np.zeros([]), np.zeros((1, 1)), np.zeros((1,)),
                          np.zeros([1, 2, 3]), None, tf.constant(0),
                          tf.constant([0, 1]), 'toto'])
def test_data_array(array):
    if isinstance(array, str):
        with pytest.raises(BaseException):
            convert = utils.data_array(array)
        return
    convert = utils.data_array(array)
    assert(isinstance(convert, np.ndarray) or
           isinstance(convert, tf.Tensor) or
           (convert is None))
    if array is not None:
        if isinstance(array, tf.Tensor):
            tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                assert(len(sess.run(convert).shape) == 2)
        else:
            assert(len(convert.shape) == 2)


@pytest.mark.parametrize("ex_val", np.random.rand(10))
def test_quadratic_infinimal_convolution_0(ex_val):
    tf_ex_val = tf.constant(ex_val)
    val = utils.quadratic_infinimal_convolution(tf_ex_val, 0)
    tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        val = sess.run(val)
        assert(np.isclose(val, ex_val))

@pytest.mark.parametrize("ex_val", np.random.rand(10))
def test_quadratic_infinimal_convolution_1(ex_val):
    tf_ex_val = tf.constant(ex_val)
    val = utils.quadratic_infinimal_convolution(tf_ex_val, 1)
    tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        val = sess.run(val)
        assert(np.isclose(val, ex_val ** 2))
