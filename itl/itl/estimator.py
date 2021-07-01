from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
import numpy as np

from tensorflow.python import debug

from .kernel import *
from .model import *
from .solver import *
from .sampler import *
from .risk import *
from .utils import *
from .scoped_lru_cache import *
from .summary import *


__all__ = ['Estimator']


class Estimator(object):

    def __init__(self, model, lbda, sampler, cost, penalty,
                 device, solver, solver_param, config_param, summary,
                 debug):
        self.model = model
        self.lbda = lbda
        self.sampler = sampler
        self.cost = cost
        self.penalty = penalty
        self.device = device
        self.solver = solver
        self.solver_param = solver_param
        self.config_param = config_param
        self.summary = summary
        self.debug = debug

    def __del__(self):
        self.close_session().finalise()

    def finalise(self):
        clear_all_cached_functions()
        try:
            tf.get_default_session().close()
        except:
            pass
        try:
            tf.reset_default_graph()
        except:
            pass
        return self

    def close_session(self):
        if hasattr(self, 'sess_'):
            self.sess_.close()
        return self

    def _construct(self):
        raise NotImplementedError('Abstract class Estimator cannot be'
                                  'instanciated.')

    @property
    def optimizer(self):
        if not hasattr(self, 'optimizer_'):
            raise BaseException('Estimator must be constructed.')
        return self.optimizer_

    def _fit_bfgs(self, X, y, weights=None):
        self._construct(data_array(X), data_array(y), weights)
        self.solver_param_ = {
            'maxiter': 1000000,
            'disp': -1,
            'maxfun': 1000000,
            'gtol': 1e-6,
            'ftol': 2.220446049250313e-9,
            'maxls': 20,
            'maxcor': 20,
            'M': 1
        }
        self.solver_param_.update(self.solver_param)
        nn = tf.get_collection('non_negative')
        with tf.variable_scope('optimizer'):
            self.optimizer_ = ITLOptimizerInterface(
                self.risk_,
                method=self.solver,
                options=self.solver_param_,
                var_to_bounds=dict(zip(nn, ((0, np.inf), ) * len(nn))))
        start_time = time.time()
        if self.solver_param_['disp'] > 0:
            print('Construction time: {}s'.format(time.time() - start_time))
        construct_time = time.time()
        self.config_param_ = {
            'allow_soft_placement': True,
            'gpu_options': tf.GPUOptions(allow_growth=True),
            'intra_op_parallelism_threads': 0,
            'inter_op_parallelism_threads': 0
        }
        self.config_param_.update(self.config_param)
        config = tf.ConfigProto(**self.config_param_)
        self.sess_ = tf.Session(config=config)
        self.summary_ = {
            'path': '',
            'graph': 100,
            'step': 100,
            'time': np.inf,
        }
        self.summary_.update(self.summary)
        self.sess_.run(tf.global_variables_initializer())
        self.debug_ = {
            'tfdbg': False,
            'tensorboard_debug_address': '',
            'ui_type': 'curses'
        }
        self.debug_.update(self.debug)
        if ((self.debug_['tfdbg']) and
            (len(self.debug_['tensorboard_debug_address']) > 0)):
            raise ValueError(
                "The --debug and --tensorboard_debug_address flags are "
                "mutually exclusive.")
        elif self.debug_['tfdbg']:
            self.sess_ = debug.LocalCLIDebugWrapperSession(
                self.sess_, ui_type=self.debug_['ui_type'])
        elif len(self.debug_['tensorboard_debug_address']) > 0:
            self.sess_ = debug.TensorBoardDebugWrapperSession(
                self.sess_, self.debug_['tensorboard_debug_address'])
        if self.solver_param_['disp'] > 0:
            print('Variables initialization '
                  'time: {}s'.format(time.time() - construct_time))
        var_init_time = time.time()
        loss_callback = Summary(self.sess_, self.summary_).construct('train')
        self.optimizer.minimize(self.sess_, loss_callback=loss_callback)
        loss_callback.dump(
            self.sess_.run(loss_callback.merged_summaries))
        if self.solver_param_['disp'] > 0:
            print('Minimization time: {}s'.format(time.time() -
                                                  var_init_time))
            print("Total time: {}s".format(time.time() - start_time))

        return self

    def decision_function(self, X, tau, scope=''):
        with tf.variable_scope(scope):
            if self.device == 'cpu':
                device = '/device:CPU:0'
            elif self.device == 'gpu':
                device = '/device:GPU:0'
            else:
                device = self.device
            with tf.device(device):
                Summary(self.sess_, self.summary_).construct('test')
                pred = self.sess_.run(self.model(
                    data_array(X), data_array(tau)))
        return pred
